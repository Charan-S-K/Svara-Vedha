"""
Geofence verification service for SVARAABHAVA Attendance.

Validates client browser GPS coordinates and verifies distance from the
designated college campus boundary using the Haversine formula.
"""

import math
import os
from typing import Any, Dict, Tuple
from django.conf import settings


class GeofenceValidationError(Exception):
    """Raised when GPS coordinates or accuracy fail validation."""
    def __init__(self, message: str, code: str = "invalid_gps"):
        super().__init__(message)
        self.message = message
        self.code = code


def haversine_distance_meters(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the great-circle distance between two GPS coordinate points
    on the Earth's surface using the Haversine formula.

    Returns:
        float: Distance in meters.
    """
    R = 6371000.0  # Earth's mean radius in meters

    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = (
        math.sin(delta_phi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2
    )
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return R * c


def get_geofence_config() -> Tuple[float, float, float, float]:
    """
    Retrieve college geofence coordinates, radius, and accuracy limits
    from Django settings or environment variables.
    """
    college_lat = float(getattr(settings, "COLLEGE_GEOFENCE_LATITUDE", 17.385044))
    college_lon = float(getattr(settings, "COLLEGE_GEOFENCE_LONGITUDE", 78.486671))
    radius = float(getattr(settings, "COLLEGE_GEOFENCE_RADIUS_METERS", 500.0))
    max_accuracy = float(getattr(settings, "MAX_ALLOWED_GPS_ACCURACY_METERS", 100.0))
    return college_lat, college_lon, radius, max_accuracy


def validate_gps_coordinates(lat_val: Any, lon_val: Any, accuracy_val: Any) -> Tuple[float, float, float]:
    """
    Validate and parse GPS coordinates from request parameters.

    Raises:
        GeofenceValidationError: If coordinates are missing, malformed, or out of range.
    """
    if lat_val is None or lon_val is None:
        raise GeofenceValidationError(
            "GPS coordinates (latitude and longitude) are mandatory for attendance verification.",
            code="missing_coordinates",
        )

    try:
        lat = float(lat_val)
        lon = float(lon_val)
    except (ValueError, TypeError) as e:
        raise GeofenceValidationError(
            "Malformed GPS coordinates. Latitude and longitude must be valid floating point numbers.",
            code="malformed_coordinates",
        ) from e

    if not (-90.0 <= lat <= 90.0):
        raise GeofenceValidationError(
            f"Latitude ({lat}) out of bounds. Must be between -90 and 90 degrees.",
            code="latitude_out_of_bounds",
        )

    if not (-180.0 <= lon <= 180.0):
        raise GeofenceValidationError(
            f"Longitude ({lon}) out of bounds. Must be between -180 and 180 degrees.",
            code="longitude_out_of_bounds",
        )

    accuracy = 0.0
    if accuracy_val is not None and str(accuracy_val).strip() != "":
        try:
            accuracy = float(accuracy_val)
        except (ValueError, TypeError) as e:
            raise GeofenceValidationError(
                "Malformed GPS accuracy reading.",
                code="malformed_accuracy",
            ) from e

        if accuracy < 0.0:
            raise GeofenceValidationError(
                "GPS accuracy radius cannot be negative.",
                code="negative_accuracy",
            )

    _, _, _, max_allowed_accuracy = get_geofence_config()
    if accuracy > max_allowed_accuracy:
        raise GeofenceValidationError(
            f"GPS signal accuracy (±{accuracy:.1f}m) is too poor. Maximum allowed uncertainty is ±{max_allowed_accuracy:.1f}m. Please move outdoors or enable precise location.",
            code="poor_accuracy",
        )

    return lat, lon, accuracy


def verify_college_geofence(lat_val: Any, lon_val: Any, accuracy_val: Any) -> Dict[str, Any]:
    """
    Verify whether the provided coordinates are within the configured college geofence.

    Returns:
        Dict[str, Any]: Geofence validation result containing distance, accuracy, and match flag.

    Raises:
        GeofenceValidationError: If coordinates are invalid or outside boundary.
    """
    lat, lon, accuracy = validate_gps_coordinates(lat_val, lon_val, accuracy_val)
    college_lat, college_lon, radius, max_accuracy = get_geofence_config()

    distance = haversine_distance_meters(lat, lon, college_lat, college_lon)
    is_inside = distance <= radius

    if not is_inside:
        raise GeofenceValidationError(
            f"Outside permitted campus area: You are {distance:.1f}m away from college campus (maximum allowed radius is {radius:.1f}m).",
            code="outside_geofence",
        )

    return {
        "is_inside": True,
        "latitude": lat,
        "longitude": lon,
        "distance_meters": round(distance, 2),
        "accuracy_meters": round(accuracy, 2),
        "radius_meters": radius,
        "college_coordinates": (college_lat, college_lon),
    }
