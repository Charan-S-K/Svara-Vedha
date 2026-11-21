from django.urls import path
from . import views


urlpatterns = [

    path(
        "",
        views.login,
        name="login"
    ),
    path(
        "login/",
        views.login,
        name="login_direct"
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
path("logout/", views.logout_view, name="logout"),
path(
    "face-login/",
    views.face_login,
    name="face_login"
),
path(
    "face-enroll/",
    views.face_enroll_page,
    name="face_enroll"
),
path(
    "api/face/enroll/",
    views.face_enroll_api,
    name="face_enroll_api"
),
path(
    "api/face/login/",
    views.face_login_api,
    name="face_login_api"
),
path(
    "api/face/attendance/",
    views.face_attendance_api,
    name="face_attendance_api"
),
]