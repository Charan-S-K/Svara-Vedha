from django.shortcuts import render, redirect
from django.contrib import messages
from django.contrib.auth.models import User
from django.contrib.auth import authenticate
from django.contrib.auth import login as auth_login
from django.http import JsonResponse
from django.contrib.auth import logout as auth_logout
from django.views.decorators.http import require_POST
from django.db.models import Q
from django.conf import settings
from django.utils import timezone
import json
import logging
import numpy as np

logger = logging.getLogger(__name__)

from .models import Profile, AttendanceRecord
from mail import send_mail
from .services import (
    FaceValidationError,
    NoFaceDetectedError,
    MultipleFacesDetectedError,
    FaceRecognitionError,
    extract_face_embedding,
    compare_face_embeddings,
    get_face_storage,
    StorageNotConfiguredError,
    DuplicateEnrollmentError,
    InactiveTemplateError,
    verify_college_geofence,
    GeofenceValidationError,
)


@require_POST
def logout_view(request):
    auth_logout(request)
    return redirect("login")


def login(request):
    if request.method == "POST":
        email = request.POST.get("email", "").strip()
        password = request.POST.get("password", "")

        try:
            user_record = User.objects.get(email__iexact=email)
        except User.DoesNotExist:
            messages.error(request, "Invalid email or password.")
            return render(request, "accounts/login.html")
        except User.MultipleObjectsReturned:
            messages.error(
                request,
                "Multiple accounts use this email. Please contact support."
            )
            return render(request, "accounts/login.html")

        user = authenticate(
            request,
            username=user_record.username,
            password=password
        )

        if user is not None:
            auth_login(request, user)
            return redirect("dashboard")

        messages.error(request, "Invalid email or password.")

    return render(request, "accounts/login.html")



def register(request):

    if request.method == "POST":

        name = request.POST.get("name")
        phone = request.POST.get("phone")
        email = request.POST.get("email")
        password = request.POST.get("password")

        # Basic validation
        if not name or not phone or not email or not password:

            messages.error(
                request,
                "Please fill all fields."
            )

            return render(
                request,
                "accounts/register.html"
            )

        # Check existing account
        if User.objects.filter(
            username=email
        ).exists():

            messages.error(
                request,
                "An account with this email already exists."
            )

            return render(
                request,
                "accounts/register.html"
            )

        # Generate and send OTP
        otp = send_mail(email)

        # Temporarily store registration data
        request.session["register_name"] = name
        request.session["register_phone"] = phone
        request.session["register_email"] = email
        request.session["register_password"] = password
        request.session["register_otp"] = str(otp)

        return redirect("/verify-otp/")

    return render(
        request,
        "accounts/register.html"
    )


def verify_otp(request):

    # Make sure registration exists
    if "register_email" not in request.session:

        messages.error(
            request,
            "Registration session expired. Please register again."
        )

        return redirect("/register/")

    if request.method == "POST":

        entered_otp = request.POST.get("otp")

        stored_otp = request.session.get(
            "register_otp"
        )

        if entered_otp == stored_otp:

            name = request.session.get(
                "register_name"
            )

            phone = request.session.get(
                "register_phone"
            )

            email = request.session.get(
                "register_email"
            )

            password = request.session.get(
                "register_password"
            )

            # Create Django user
            user = User.objects.create_user(
                username=email,
                email=email,
                password=password,
                first_name=name
            )

            # Create profile
            Profile.objects.create(
                user=user,
                phone=phone
            )

            # Clear registration session
            request.session.pop(
                "register_name",
                None
            )

            request.session.pop(
                "register_phone",
                None
            )

            request.session.pop(
                "register_email",
                None
            )

            request.session.pop(
                "register_password",
                None
            )

            request.session.pop(
                "register_otp",
                None
            )

            # Authenticate user and navigate to biometric face enrollment
            auth_login(request, user, backend="django.contrib.auth.backends.ModelBackend")

            messages.success(
                request,
                "Registration verified successfully! Please complete your biometric face enrollment."
            )

            return redirect("/face-enroll/")

        else:

            messages.error(
                request,
                "Invalid OTP."
            )

    return render(
        request,
        "accounts/verify_otp.html"
    )

def resend(request):
    if  "register_email" not in request.session:
        messages.error(request,"Registration session expired. Please register again.")
        return redirect('/register/')
    email = request.session.get("register_email")
    otp = send_mail(email)

    request.session["register_otp"] = str(otp)

    messages.success(request,"A new OTP has been sent to your email.")

    return redirect("/verify-otp/")

def forgot(request):

    if request.method == "POST":

        email = request.POST.get("email")

        if not User.objects.filter(email=email).exists():

            return JsonResponse({
                "success": False,
                "message": "You are not registered. Please register first."
            })

        otp = send_mail(email)

        request.session["forgot_email"] = email
        request.session["forgot_otp"] = str(otp)
        request.session.pop("forgot_otp_verified", None)

        return JsonResponse({
            "success": True,
            "message": "OTP sent successfully."
        })

    return redirect("/")

def verify_forgot_otp(request):

    if "forgot_email" not in request.session:
        return JsonResponse({
            "success": False,
            "message": "Password reset session expired."
        })

    if request.method == "POST":

        entered_otp = request.POST.get("otp")

        stored_otp = request.session.get("forgot_otp")

        if entered_otp == stored_otp:

            request.session["forgot_otp_verified"] = True

            return JsonResponse({
                "success": True,
                "message": "OTP verified successfully."
            })

        return JsonResponse({
            "success": False,
            "message": "Invalid OTP. Please try again."
        })

    return JsonResponse({
        "success": False,
        "message": "Invalid request."
    })

def create_reset_password(request):

    if "forgot_email" not in request.session:

        messages.error(
            request,
            "Password reset session expired. Please try again."
        )

        return redirect("/")

    if not request.session.get("forgot_otp_verified"):

        messages.error(
            request,
            "Please verify the OTP first."
        )

        return redirect("/")

    email = request.session.get("forgot_email")

    if not User.objects.filter(username=email).exists():

        messages.error(
            request,
            "Account not found. Please create an account first."
        )

        return redirect("/")

    if request.method == "POST":

        new = request.POST.get("new")
        confirm = request.POST.get("confirm")

        if not new or not confirm:

            messages.error(
                request,
                "Please fill both password fields."
            )

            return redirect("/create-reset-password/")

        if len(new) < 6:

            messages.error(
                request,
                "Password must be at least 6 characters long."
            )

            return redirect("/create-reset-password/")

        if new != confirm:

            messages.error(
                request,
                "Both passwords should be same."
            )

            return redirect("/create-reset-password/")

        user = User.objects.get(username=email)

        user.set_password(new)
        user.save()

        request.session.pop("forgot_email", None)
        request.session.pop("forgot_otp", None)
        request.session.pop("forgot_otp_verified", None)

        messages.success(
            request,
            "Password reset successfully. Please login."
        )

        return redirect("/")

    return render(
        request,
        "accounts/create_new_password.html"
    )

def _parse_request_data(request):
    """Helper to extract parameters from POST form-data or JSON body."""
    data = {}
    if request.content_type and "application/json" in request.content_type:
        try:
            data = json.loads(request.body.decode("utf-8"))
        except Exception:
            data = {}
    for k, v in request.POST.items():
        data[k] = v
    return data


def face_enroll_api(request):
    """
    Backend face enrollment endpoint (Phase 4).
    Enrolls a fresh 512-D ArcFace template for the authenticated Django user in MongoDB Atlas.
    Supports authenticated re-enrollment when explicitly requested.
    """
    logger.info(
        "[ENROLLMENT STAGE 1] Enrollment request received: method=%s, path=%s, remote_ip=%s",
        request.method,
        request.path,
        request.META.get("REMOTE_ADDR", "unknown"),
    )

    if request.method != "POST":
        return JsonResponse({"success": False, "error": "Method not allowed. Use POST."}, status=405)

    if not request.user.is_authenticated:
        logger.warning("[ENROLLMENT STAGE 2 FAILED] Unauthenticated face enrollment attempt rejected.")
        return JsonResponse(
            {"success": False, "error": "Authentication required. You must be logged into an existing account to enroll your face."},
            status=401,
        )

    logger.info(
        "[ENROLLMENT STAGE 2] User authentication validated: user_id=%s, username=%s, email=%s",
        request.user.id,
        request.user.username,
        request.user.email,
    )

    data = _parse_request_data(request)
    face_image = data.get("face_image")
    re_enroll = str(data.get("re_enroll", "")).lower() in ("true", "1", "yes") or str(data.get("overwrite", "")).lower() in ("true", "1", "yes")

    if not face_image:
        logger.warning("[ENROLLMENT STAGE 3 FAILED] No face image payload provided for enrollment.")
        return JsonResponse({"success": False, "error": "No face image provided for enrollment."}, status=400)

    img_format = "base64_data_url" if isinstance(face_image, str) and face_image.startswith("data:") else "raw_string"
    logger.info(
        "[ENROLLMENT STAGE 3] Face image received: format=%s, payload_length=%d characters, re_enroll=%s",
        img_format,
        len(face_image),
        re_enroll,
    )

    try:
        extraction_result = extract_face_embedding(face_image)
        embedding = extraction_result["embedding"]

        storage = get_face_storage()
        persisted = storage.save_face_template(
            user_id=request.user.id,
            embedding=embedding,
            metadata={
                "username": request.user.username,
                "email": request.user.email,
                "det_score": extraction_result.get("det_score", 1.0),
            },
            overwrite=re_enroll,
        )

        if not persisted:
            logger.error("[ENROLLMENT STAGE 10 FAILED] Storage backend returned false for persistence.")
            return JsonResponse(
                {"success": False, "error": "Failed to persist face template in storage."},
                status=500,
            )

        logger.info("[ENROLLMENT COMPLETE] Biometric face template successfully enrolled for user_id=%s.", request.user.id)
        return JsonResponse({
            "success": True,
            "message": "Biometric face template enrolled successfully.",
            "redirect_url": "/dashboard/",
        })

    except DuplicateEnrollmentError as e:
        logger.info("[ENROLLMENT DUPLICATE] Duplicate enrollment detected for user_id=%s.", request.user.id)
        return JsonResponse({
            "success": False,
            "error": "Face already enrolled for this account. To update your face template, please confirm re-enrollment.",
            "code": "already_enrolled",
        }, status=409)
    except StorageNotConfiguredError as e:
        logger.error("[ENROLLMENT STORAGE ERROR] Storage not configured: %s", str(e))
        return JsonResponse({
            "success": False,
            "error": "Biometric template storage is not configured. Face template cannot be saved until MongoDB is connected.",
            "code": "storage_not_configured",
        }, status=503)
    except FaceValidationError as e:
        logger.warning("[ENROLLMENT VALIDATION ERROR] Face validation failed: %s (code=%s)", str(e), getattr(e, "code", "validation_error"))
        return JsonResponse({"success": False, "error": str(e), "code": getattr(e, "code", "validation_error")}, status=400)
    except Exception as e:
        logger.exception("[ENROLLMENT UNEXPECTED ERROR] Face enrollment failed for user_id=%s: %s", request.user.id, str(e))
        return JsonResponse({"success": False, "error": f"Face enrollment failed: {str(e)}"}, status=500)


def face_login_api(request):
    """
    Backend face login endpoint (Phase 5).
    Verifies probe face embedding against stored reference template in MongoDB and authenticates Django session.
    Cycles session key on successful authentication.
    """
    if request.method != "POST":
        return JsonResponse({"success": False, "error": "Method not allowed. Use POST."}, status=405)

    data = _parse_request_data(request)
    identifier = (data.get("identifier") or data.get("email") or data.get("username") or "").strip()
    face_image = data.get("face_image")

    if not identifier and request.user.is_authenticated:
        identifier = request.user.email or request.user.username

    if not identifier:
        return JsonResponse({
            "success": False,
            "error": "Account identifier (email or username) is required for face sign in.",
            "code": "missing_identifier",
        }, status=400)

    if not face_image:
        return JsonResponse({"success": False, "error": "No face image provided for verification."}, status=400)

    # Rate limiting: maximum 5 failed attempts per session window
    failed_attempts = request.session.get("face_login_failed_attempts", 0)
    if failed_attempts >= 5:
        return JsonResponse({
            "success": False,
            "error": "Too many failed face login attempts. Please sign in using your account password.",
            "code": "rate_limited",
        }, status=429)

    user = User.objects.filter(Q(email__iexact=identifier) | Q(username__iexact=identifier)).first()
    if not user:
        return JsonResponse({"success": False, "error": "No account found matching this identifier.", "code": "user_not_found"}, status=404)

    try:
        extraction_result = extract_face_embedding(face_image)
        probe_embedding = extraction_result["embedding"]

        storage = get_face_storage()
        template_record = storage.get_face_template(user.id)

        if template_record is None or "embedding" not in template_record:
            return JsonResponse({
                "success": False,
                "error": "No registered face template found for this account. Please enroll your face first.",
                "code": "template_not_found",
            }, status=404)

        # Check if template is active
        is_active = template_record.get("enrollment", {}).get("is_active", True)
        if not is_active:
            return JsonResponse({
                "success": False,
                "error": "Face template is inactive or archived. Please contact an administrator or re-enroll.",
                "code": "template_inactive",
            }, status=403)

        stored_embedding = template_record["embedding"]
        sim_result = compare_face_embeddings(probe_embedding, stored_embedding)

        if not sim_result["is_match"]:
            request.session["face_login_failed_attempts"] = failed_attempts + 1
            return JsonResponse({
                "success": False,
                "error": "Face verification failed.",
                "code": "face_mismatch",
            }, status=401)

        # Successful face verification -> reset counter, authenticate Django session, cycle session key
        request.session.pop("face_login_failed_attempts", None)
        auth_login(request, user, backend="django.contrib.auth.backends.ModelBackend")
        request.session.cycle_key()

        return JsonResponse({
            "success": True,
            "message": f"Welcome back, {user.first_name or user.username}! Identity verified.",
            "redirect_url": "/dashboard/",
        })

    except StorageNotConfiguredError:
        return JsonResponse({
            "success": False,
            "error": "Biometric template storage is not configured. Please connect your MongoDB implementation to enable face sign-in.",
            "code": "storage_not_configured",
        }, status=503)
    except FaceValidationError as e:
        return JsonResponse({"success": False, "error": str(e), "code": getattr(e, "code", "validation_error")}, status=400)
    except Exception as e:
        return JsonResponse({"success": False, "error": f"Face verification failed: {str(e)}"}, status=500)


def face_attendance_api(request):
    """
    Separate attendance verification endpoint (Phase 6).
    Verifies biometric identity and college geofence before recording attendance in both
    Django AttendanceRecord ORM and MongoDB attendance_records collection.
    Attendance remains strictly separate from dashboard login session.
    """
    if request.method != "POST":
        return JsonResponse({"success": False, "error": "Method not allowed. Use POST."}, status=405)

    data = _parse_request_data(request)
    face_image = data.get("face_image")
    latitude = data.get("latitude")
    longitude = data.get("longitude")
    accuracy = data.get("accuracy")

    if not face_image:
        return JsonResponse({"success": False, "error": "No face image provided for attendance punch."}, status=400)

    # 1. Resolve user
    if request.user.is_authenticated:
        user = request.user
    else:
        identifier = (data.get("identifier") or data.get("email") or data.get("username") or "").strip()
        if not identifier:
            return JsonResponse({
                "success": False,
                "error": "Account identifier (email or username) is required to punch attendance.",
                "code": "missing_identifier",
            }, status=400)
        user = User.objects.filter(Q(email__iexact=identifier) | Q(username__iexact=identifier)).first()
        if not user:
            return JsonResponse({"success": False, "error": "User account not found.", "code": "user_not_found"}, status=404)

    # 2. Server-side GPS & College Geofence Verification
    try:
        geo_result = verify_college_geofence(latitude, longitude, accuracy)
    except GeofenceValidationError as ge:
        return JsonResponse({"success": False, "error": str(ge), "code": getattr(ge, "code", "geofence_error")}, status=400)

    # 3. Server-side Biometric Face Verification
    try:
        extraction_result = extract_face_embedding(face_image)
        probe_embedding = extraction_result["embedding"]

        storage = get_face_storage()
        template_record = storage.get_face_template(user.id)

        if template_record is None or "embedding" not in template_record:
            return JsonResponse({
                "success": False,
                "error": "No registered face template found for this account. Please enroll your face first.",
                "code": "template_not_found",
            }, status=404)

        # Check if template is active
        is_active = template_record.get("enrollment", {}).get("is_active", True)
        if not is_active:
            return JsonResponse({
                "success": False,
                "error": "Face template is inactive. Attendance cannot be verified.",
                "code": "template_inactive",
            }, status=403)

        stored_embedding = template_record["embedding"]
        sim_result = compare_face_embeddings(probe_embedding, stored_embedding)

        if not sim_result["is_match"]:
            return JsonResponse({
                "success": False,
                "error": "Face verification failed.",
                "code": "face_mismatch",
            }, status=401)

        today = timezone.localdate() if hasattr(timezone, "localdate") else timezone.now().date()

        # 4. Duplicate daily punch prevention
        already_punched = AttendanceRecord.objects.filter(user=user, date=today).exists()
        if not already_punched and hasattr(storage, "has_punched_today"):
            try:
                already_punched = storage.has_punched_today(user.id)
            except Exception:
                pass

        if already_punched:
            return JsonResponse({
                "success": True,
                "attendance_status": "already_recorded",
                "message": f"Attendance already recorded for today ({today.isoformat()}).",
                "data": {
                    "user": user.username,
                    "date": str(today),
                    "status": "already_recorded",
                    "distance_meters": geo_result["distance_meters"],
                },
            })

        # 5. Record attendance in Django ORM
        AttendanceRecord.objects.create(
            user=user,
            date=today,
            status="present",
            latitude=geo_result["latitude"],
            longitude=geo_result["longitude"],
            distance_meters=geo_result["distance_meters"],
            accuracy_meters=geo_result["accuracy_meters"],
            similarity=sim_result["similarity"],
            verification_method="arcface_buffalo_l",
        )

        # 6. Record attendance in MongoDB collection if supported
        if hasattr(storage, "record_attendance"):
            try:
                storage.record_attendance(
                    user_id=user.id,
                    latitude=geo_result["latitude"],
                    longitude=geo_result["longitude"],
                    distance_meters=geo_result["distance_meters"],
                    accuracy_meters=geo_result["accuracy_meters"],
                    similarity=sim_result["similarity"],
                )
            except Exception:
                pass

        return JsonResponse({
            "success": True,
            "attendance_status": "recorded",
            "message": f"Attendance verified and recorded successfully ({geo_result['distance_meters']}m from campus center).",
            "data": {
                "user": user.username,
                "distance_meters": geo_result["distance_meters"],
                "accuracy_meters": geo_result["accuracy_meters"],
                "status": "present",
                "date": str(today),
            },
        })

    except StorageNotConfiguredError:
        return JsonResponse({
            "success": False,
            "error": "Face template storage is not configured. Attendance cannot be verified until MongoDB is connected.",
            "code": "storage_not_configured",
        }, status=503)
    except FaceValidationError as e:
        return JsonResponse({"success": False, "error": str(e), "code": getattr(e, "code", "validation_error")}, status=400)
    except Exception as e:
        return JsonResponse({"success": False, "error": f"Attendance verification failed: {str(e)}"}, status=500)


def face_login(request):
    """
    Unified face view.
    - On GET: Renders face.html with camera interface and MediaPipe guidance.
    - On POST: Dispatches to enroll, punch, or signin according to action_type.
    """
    if request.method == "POST":
        data = _parse_request_data(request)
        action_type = str(data.get("action_type", "signin")).strip().lower()
        if action_type == "enroll":
            return face_enroll_api(request)
        elif action_type == "punch":
            return face_attendance_api(request)
        else:
            return face_login_api(request)

    mode = request.GET.get("mode", "login")
    return render(request, "accounts/face.html", {"mode": mode})


def face_enroll_page(request):
    """Dedicated face enrollment page view."""
    return render(request, "accounts/face.html", {"mode": "enroll"})
