from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.contrib import messages
from django.contrib.auth.hashers import make_password
from django.shortcuts import render, redirect
from django.contrib.auth import authenticate
from django.contrib.auth import login as auth_login
from django.http import JsonResponse
from django.contrib.auth import logout as auth_logout
from django.views.decorators.http import require_POST


from mail import send_mail


def dashboard(request):
    return render(request, "dashboard/dashboard.html")

@login_required(login_url="login")
def profile(request):
    user = request.user

    # Fetch biometric face registration status & weekly update cooldown
    has_face_id = False
    can_update_face = True
    days_remaining = 0
    last_face_update = None

    try:
        from accounts.services import get_face_storage
        from datetime import datetime, timezone, timedelta
        storage = get_face_storage()
        template = storage.get_face_template(user.id)
        if template:
            enrollment = template.get("enrollment", {})
            is_active = enrollment.get("is_active", True)
            if is_active:
                has_face_id = True
                updated_at_raw = enrollment.get("updated_at") or enrollment.get("created_at")
                if updated_at_raw:
                    last_dt = None
                    if isinstance(updated_at_raw, str):
                        clean_str = updated_at_raw.replace("Z", "+00:00")
                        try:
                            last_dt = datetime.fromisoformat(clean_str)
                        except Exception:
                            last_dt = None
                    elif isinstance(updated_at_raw, datetime):
                        last_dt = updated_at_raw

                    if last_dt:
                        if last_dt.tzinfo is None:
                            last_dt = last_dt.replace(tzinfo=timezone.utc)
                        now_utc = datetime.now(timezone.utc)
                        diff = now_utc - last_dt
                        cooldown = timedelta(days=7)
                        if diff < cooldown:
                            can_update_face = False
                            remaining = cooldown - diff
                            days_remaining = max(1, remaining.days + (1 if remaining.seconds > 0 else 0))
                        else:
                            can_update_face = True
                            days_remaining = 0
                        last_face_update = last_dt.strftime("%b %d, %Y")
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning("Error fetching face template for profile: %s", e)

    context = {
        "first_name": user.first_name,
        "last_name": user.last_name,
        "email": user.email,
        "has_face_id": has_face_id,
        "can_update_face": can_update_face,
        "days_remaining": days_remaining,
        "last_face_update": last_face_update,
    }

    return render(request, "dashboard/profile.html", context)

@login_required(login_url="login")
def change_password(request):
    if request.method != "POST":
        return redirect("profile")

    new_password = request.POST.get("new_password", "")
    confirm_password = request.POST.get("confirm_password", "")

    if not new_password or not confirm_password:
        messages.error(request, "Both password fields are required.")
        return redirect("profile")

    if new_password != confirm_password:
        messages.error(request, "New and Confirm Password must match.")
        return redirect("profile")

    if len(new_password) < 6:
        messages.error(request, "Password must contain at least 6 characters.")
        return redirect("profile")

    user = request.user

    if not user.email:
        messages.error(request, "Your account has no registered email address.")
        return redirect("profile")

    otp = send_mail(user.email)

    request.session["password_change_user_id"] = user.id
    request.session["password_change_otp"] = str(otp)
    request.session["password_change_hash"] = make_password(new_password)
    request.session["password_change_verified"] = False

    return redirect("verify_password_otp")


@login_required(login_url="login")
def verify_password_otp(request):
    user_id = request.session.get("password_change_user_id")
    stored_otp = request.session.get("password_change_otp")

    if user_id != request.user.id or not stored_otp:
        messages.error(
            request,
            "Password change session expired. Please try again."
        )
        return redirect("profile")

    if request.method == "POST":
        entered_otp = request.POST.get("otp", "").strip()

        if entered_otp != stored_otp:
            messages.error(request, "Invalid OTP. Please try again.")
            return render(request, "dashboard/verify_password_otp.html")

        password_hash = request.session.get("password_change_hash")

        if not password_hash:
            messages.error(request, "Password change session expired.")
            return redirect("profile")

        user = request.user
        user.password = password_hash
        user.save(update_fields=["password"])

        for key in (
            "password_change_user_id",
            "password_change_otp",
            "password_change_hash",
            "password_change_verified",
        ):
            request.session.pop(key, None)

        messages.success(request, "Password changed successfully.")
        return redirect("profile")

    return render(request, "dashboard/verify_password_otp.html")


def resend(request):
    if  "register_email" not in request.session:
        messages.error(request,"Registration session expired. Please register again.")
        return redirect('/register/')
    email = request.session.get("register_email")
    otp = send_mail(email)

    request.session["register_otp"] = str(otp)

    messages.success(request,"A new OTP has been sent to your email.")

    return redirect("/verify-otp/")


def timetable_api(request):
    """
    Weekly Academic Timetable API endpoint for MongoDB persistence.
    GET: Retrieve current user's timetable document.
    POST / PUT: Upsert current user's timetable document.
    """
    if not request.user.is_authenticated:
        return JsonResponse({
            "status": "error",
            "message": "Authentication required. Please log in."
        }, status=401)

    user = request.user
    user_name = (user.get_full_name() or user.username or "Student").strip()

    if request.method == "GET":
        from .services.timetable_service import get_user_timetable
        res = get_user_timetable(user.id, user_name)
        return JsonResponse(res)

    elif request.method in ["POST", "PUT"]:
        import json
        from .services.timetable_service import validate_timetable_payload, save_user_timetable

        try:
            body = json.loads(request.body.decode("utf-8"))
        except Exception:
            return JsonResponse({
                "status": "error",
                "message": "Invalid JSON payload."
            }, status=400)

        is_valid, err_msg, cleaned_data = validate_timetable_payload(body)
        if not is_valid:
            return JsonResponse({
                "status": "error",
                "message": err_msg
            }, status=400)

        res = save_user_timetable(user.id, user_name, cleaned_data)
        return JsonResponse(res)

    return JsonResponse({
        "status": "error",
        "message": f"Method '{request.method}' not allowed."
    }, status=405)


def download_project_zip(request):
    """Serve the generated Voice_Ai.zip archive directly to the browser."""
    import os
    from django.http import FileResponse, Http404

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    zip_path = os.path.join(base_dir, "Voice_Ai.zip")

    if not os.path.exists(zip_path):
        # Fallback to parent dir
        parent_zip = os.path.join(os.path.dirname(base_dir), "Voice_Ai.zip")
        if os.path.exists(parent_zip):
            zip_path = parent_zip
        else:
            raise Http404("Voice_Ai.zip archive not found.")

    response = FileResponse(open(zip_path, "rb"), as_attachment=True, filename="Voice_Ai.zip")
    response["Content-Type"] = "application/zip"
    return response


