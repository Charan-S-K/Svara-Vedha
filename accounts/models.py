from django.db import models
from django.contrib.auth.models import User


class Profile(models.Model):
    user = models.OneToOneField(
        User,
        on_delete=models.CASCADE
    )

    phone = models.CharField(
        max_length=15
    )

    def __str__(self):
        return self.user.email


class AttendanceRecord(models.Model):
    """
    Biometric + Geofence Attendance Record.
    Stores verified attendance punches strictly separated from login sessions.
    Enforces one punch per user per calendar day.
    """
    user = models.ForeignKey(
        User,
        on_delete=models.CASCADE,
        related_name="attendance_records"
    )
    timestamp = models.DateTimeField(auto_now_add=True, db_index=True)
    date = models.DateField(auto_now_add=True, db_index=True)
    status = models.CharField(max_length=20, default="PRESENT")
    latitude = models.FloatField()
    longitude = models.FloatField()
    distance_meters = models.FloatField()
    accuracy_meters = models.FloatField()
    similarity = models.FloatField()
    verification_method = models.CharField(max_length=50, default="ArcFace+Geofence")

    class Meta:
        ordering = ["-timestamp"]
        constraints = [
            models.UniqueConstraint(fields=["user", "date"], name="unique_user_daily_attendance")
        ]

    def __str__(self):
        return f"{self.user.username} - {self.date} - {self.status}"
