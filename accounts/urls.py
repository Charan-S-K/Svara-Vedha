from django.urls import path
from . import views


urlpatterns = [

    path(
        "",
        views.login,
        name="login"
    ),

    path(
        "register/",
        views.register,
        name="register"
    ),

    path(
        "verify-otp/",
        views.verify_otp,
        name="verify_otp"
    ),
    path(
        "resend-otp/",
        views.resend,
        name="resend"
    ),
    path(
    "forgot-password/",
    views.forgot,
    name="forgot"
),
path(
    "verify-forgot-otp/",
    views.verify_forgot_otp,
    name="verify_forgot_otp"
),
path(
    "create-reset-password/",
    views.create_reset_password,
    name="create_reset_password"
),
]