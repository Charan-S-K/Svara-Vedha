from django.urls import path
from . import views

urlpatterns =[
    path("",views.dashboard,name='dashboard'),
    path("profile/",views.profile,name='profile'),
    path(
    "change-password/",
    views.change_password,
    name="change_password"
),
path(
    "verify-password-otp/",
    views.verify_password_otp,
    name="verify_password_otp"
),
]