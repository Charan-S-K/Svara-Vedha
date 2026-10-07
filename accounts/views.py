from django.shortcuts import render, redirect
from django.contrib import messages
from django.contrib.auth.models import User
from django.contrib.auth import authenticate
from django.contrib.auth import login as auth_login
from django.http import JsonResponse

from .models import Profile
from mail import send_mail


def login(request):

    if request.method == "POST":

        email = request.POST.get("email")
        password = request.POST.get("password")

        user = authenticate(
            request,
            username=email,
            password=password
        )

        if user is not None:

            auth_login(request, user)

            return redirect("/")

        messages.error(
            request,
            "Invalid email or password."
        )

    return render(
        request,
        "accounts/login.html"
    )


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

            messages.success(
                request,
                "Registration successful. Please login."
            )

            return redirect("/")

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