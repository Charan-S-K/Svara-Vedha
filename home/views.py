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


@login_required(login_url="login")
def dashboard(request):
    return render(request, "dashboard/dashboard.html")

@login_required(login_url="login")
def profile(request):
    user = request.user

    context = {
        "first_name": user.first_name,
        "last_name": user.last_name,
        "email": user.email,
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
