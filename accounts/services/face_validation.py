"""
Face Validation Service for SVARAABHAVA.

Provides comprehensive server-side validation for face images prior to biometric
feature extraction:
- Format and MIME validation.
- Payload and dimensions bounds checking.
- Safe OpenCV image decoding.
- Face count validation (strictly 1 face required).
- Face bounding box size validation.
- Basic image quality checks (blurriness and exposure/lighting).

SECURITY NOTICE:
Face detection alone does NOT provide liveness detection or presentation attack
detection (PAD). Do not claim that face detection guarantees a live human subject.
"""

import base64
import re
from typing import Any, Dict, List, Optional, Tuple, Union
import cv2
import numpy as np
from django.conf import settings


class FaceValidationError(Exception):
    """Base exception for face validation failures."""
    def __init__(self, message: str, code: Optional[str] = None):
        super().__init__(message)
        self.message = message
        self.code = code or "validation_error"


class InvalidImageFormatError(FaceValidationError):
    """Raised when the uploaded image format or encoding is unsupported."""
    def __init__(self, message: str = "Unsupported or invalid image format."):
        super().__init__(message, code="invalid_format")


class ImageDecodeError(FaceValidationError):
    """Raised when the image bytes cannot be decoded into a valid image matrix."""
    def __init__(self, message: str = "Failed to decode image data into a valid picture."):
        super().__init__(message, code="decode_error")


class NoFaceDetectedError(FaceValidationError):
    """Raised when no face is found in the image."""
    def __init__(self, message: str = "No face detected in the image. Please align your face inside the frame."):
        super().__init__(message, code="no_face_detected")


class MultipleFacesDetectedError(FaceValidationError):
    """Raised when more than one face is found in the image."""
    def __init__(self, count: int = 2):
        message = f"Multiple faces detected ({count}). Exactly one person must be visible in the camera frame."
        super().__init__(message, code="multiple_faces_detected")
        self.count = count


class FaceTooSmallError(FaceValidationError):
    """Raised when the detected face is too small or too distant from the camera."""
    def __init__(self, message: str = "Face is too small or too far away. Please move closer to the camera."):
        super().__init__(message, code="face_too_small")


class ImageQualityError(FaceValidationError):
    """Raised when the image quality (blurriness or lighting) is unacceptable."""
    def __init__(self, message: str, code: str = "poor_quality"):
        super().__init__(message, code=code)


# Default validation constraints (can be overridden via Django settings)
DEFAULT_MAX_IMAGE_BYTES = 10 * 1024 * 1024  # 10 MB
DEFAULT_MIN_IMAGE_BYTES = 512              # 512 Bytes
DEFAULT_MIN_DIMENSION = 128                # 128 px
DEFAULT_MAX_DIMENSION = 4096               # 4096 px
DEFAULT_MIN_FACE_SIZE = 60                 # 60x60 px face bbox minimum
DEFAULT_BLUR_THRESHOLD = 15.0              # Laplacian variance threshold
DEFAULT_MIN_BRIGHTNESS = 25.0              # Mean pixel intensity
DEFAULT_MAX_BRIGHTNESS = 235.0             # Mean pixel intensity


def get_validation_setting(name: str, default: Any) -> Any:
    """Retrieve validation parameter from Django settings with a safe fallback."""
    return getattr(settings, name, default)


def extract_raw_image_bytes(image_data: Union[str, bytes]) -> bytes:
    """
    Extract raw binary bytes from a base64 Data URL, standard base64 string,
    or raw bytes.

    Args:
        image_data: base64 Data URL, base64 string, or raw bytes.

    Returns:
        bytes: Raw decoded image binary data.

    Raises:
        InvalidImageFormatError: If encoding is invalid or corrupted.
    """
    if isinstance(image_data, bytes):
        return image_data

    if not isinstance(image_data, str) or not image_data.strip():
        raise InvalidImageFormatError("Empty or invalid image data received.")

    data_str = image_data.strip()

    # Handle data URL pattern: data:image/<format>;base64,<payload>
    if data_str.startswith("data:"):
        match = re.match(r"^data:image\/(jpeg|jpg|png|webp);base64,(.+)$", data_str, re.IGNORECASE)
        if not match:
            # Check if it has a data URL prefix with an unsupported mime type
            if "base64," in data_str:
                header, base64_payload = data_str.split("base64,", 1)
                data_str = base64_payload.strip()
            else:
                raise InvalidImageFormatError("Invalid base64 image Data URL scheme.")
        else:
            data_str = match.group(2).strip()

    try:
        raw_bytes = base64.b64decode(data_str, validate=True)
    except Exception as e:
        raise InvalidImageFormatError(f"Base64 decoding failed: {str(e)}") from e

    if not raw_bytes:
        raise InvalidImageFormatError("Decoded image buffer is empty.")

    return raw_bytes


def validate_image_bytes(raw_bytes: bytes) -> None:
    """
    Validate raw image byte buffer size and format magic numbers.

    Args:
        raw_bytes: Raw binary bytes of image file.

    Raises:
        InvalidImageFormatError: If size or format constraints are violated.
    """
    max_bytes = get_validation_setting("MAX_FACE_IMAGE_SIZE_BYTES", DEFAULT_MAX_IMAGE_BYTES)
    min_bytes = get_validation_setting("MIN_FACE_IMAGE_SIZE_BYTES", DEFAULT_MIN_IMAGE_BYTES)

    size = len(raw_bytes)
    if size < min_bytes:
        raise InvalidImageFormatError(f"Image file size ({size} bytes) is too small to contain valid face imagery.")
    if size > max_bytes:
        raise InvalidImageFormatError(f"Image file size ({size} bytes) exceeds maximum allowable limit of {max_bytes} bytes.")

    # Magic number header validation (JPEG, PNG, WEBP)
    is_jpeg = raw_bytes.startswith(b"\xff\xd8\xff")
    is_png = raw_bytes.startswith(b"\x89PNG\r\n\x1a\n")
    is_webp = len(raw_bytes) >= 12 and raw_bytes[:4] == b"RIFF" and raw_bytes[8:12] == b"WEBP"

    if not (is_jpeg or is_png or is_webp):
        raise InvalidImageFormatError("Image must be a valid JPEG, PNG, or WEBP file.")


def decode_image_safely(raw_bytes: bytes) -> np.ndarray:
    """
    Safely decode raw image bytes into an OpenCV BGR numpy array.

    Args:
        raw_bytes: Raw binary image bytes.

    Returns:
        np.ndarray: Decoded image in BGR format (height, width, 3).

    Raises:
        ImageDecodeError: If decoding fails or resulting image is corrupt.
    """
    try:
        nparr = np.frombuffer(raw_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except Exception as e:
        raise ImageDecodeError(f"OpenCV decoding encountered an error: {str(e)}") from e

    if img is None or not isinstance(img, np.ndarray) or img.size == 0:
        raise ImageDecodeError("Corrupted or unreadable image: OpenCV could not decode pixels.")

    h, w = img.shape[:2]
    min_dim = get_validation_setting("MIN_FACE_IMAGE_DIMENSION", DEFAULT_MIN_DIMENSION)
    max_dim = get_validation_setting("MAX_FACE_IMAGE_DIMENSION", DEFAULT_MAX_DIMENSION)

    if h < min_dim or w < min_dim:
        raise ImageDecodeError(f"Image dimensions ({w}x{h}) are too small. Minimum required is {min_dim}x{min_dim} pixels.")
    if h > max_dim or w > max_dim:
        raise ImageDecodeError(f"Image dimensions ({w}x{h}) exceed maximum allowable limit of {max_dim}x{max_dim} pixels.")

    return img


def check_image_quality(img: np.ndarray) -> Dict[str, Any]:
    """
    Perform basic quality checks: blurriness and lighting exposure.

    Args:
        img: BGR image matrix.

    Returns:
        Dict[str, Any]: Quality metrics including blur score and brightness.

    Raises:
        ImageQualityError: If image fails blur or brightness thresholds.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 1. Blur evaluation using Laplacian variance
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    blur_threshold = float(get_validation_setting("FACE_BLUR_THRESHOLD", DEFAULT_BLUR_THRESHOLD))
    if blur_score < blur_threshold:
        raise ImageQualityError(
            f"Image is too blurry (sharpness score {blur_score:.1f} < {blur_threshold:.1f}). Please capture a clear, steady image.",
            code="image_blurry",
        )

    # 2. Lighting / Exposure check
    mean_brightness = float(np.mean(gray))
    min_brightness = float(get_validation_setting("FACE_MIN_BRIGHTNESS", DEFAULT_MIN_BRIGHTNESS))
    max_brightness = float(get_validation_setting("FACE_MAX_BRIGHTNESS", DEFAULT_MAX_BRIGHTNESS))

    if mean_brightness < min_brightness:
        raise ImageQualityError(
            f"Image is too dark (brightness {mean_brightness:.1f} < {min_brightness:.1f}). Please ensure sufficient front lighting.",
            code="image_underexposed",
        )
    if mean_brightness > max_brightness:
        raise ImageQualityError(
            f"Image is severely overexposed (brightness {mean_brightness:.1f} > {max_brightness:.1f}). Please avoid strong backlighting.",
            code="image_overexposed",
        )

    return {
        "blur_score": blur_score,
        "mean_brightness": mean_brightness,
    }


def validate_face_count(faces: List[Any]) -> Any:
    """
    Validate that exactly one face is detected.

    Args:
        faces: List of face objects returned by InsightFace detector.

    Returns:
        The single detected face object.

    Raises:
        NoFaceDetectedError: If no face is detected.
        MultipleFacesDetectedError: If more than one face is detected.
    """
    count = len(faces)
    if count == 0:
        raise NoFaceDetectedError()
    if count > 1:
        raise MultipleFacesDetectedError(count=count)
    return faces[0]


def validate_face_size(bbox: Union[List[float], np.ndarray], img_shape: Tuple[int, ...]) -> None:
    """
    Validate that the detected face bounding box is adequately sized.

    Args:
        bbox: Bounding box coordinates [x1, y1, x2, y2].
        img_shape: Image shape tuple (height, width, ...).

    Raises:
        FaceTooSmallError: If face bbox width or height is below required thresholds.
    """
    x1, y1, x2, y2 = bbox[:4]
    face_w = max(0.0, float(x2 - x1))
    face_h = max(0.0, float(y2 - y1))

    min_face_size = float(get_validation_setting("MIN_FACE_BOUNDING_SIZE", DEFAULT_MIN_FACE_SIZE))
    if face_w < min_face_size or face_h < min_face_size:
        raise FaceTooSmallError(
            f"Face detected ({int(face_w)}x{int(face_h)} px) is smaller than required minimum of {int(min_face_size)}x{int(min_face_size)} px."
        )


def preprocess_and_validate_image(
    image_data: Union[str, bytes, np.ndarray],
    skip_quality_checks: bool = False,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    High-level entry point to decode, validate, and quality-check an incoming image.

    Args:
        image_data: base64 Data URL, base64 string, raw bytes, or pre-loaded np.ndarray.
        skip_quality_checks: If True, skips blur/brightness checks (e.g. for unit tests).

    Returns:
        Tuple[np.ndarray, Dict[str, Any]]: Decoded BGR image and validation metadata.

    Raises:
        FaceValidationError: If any validation rule is violated.
    """
    metadata: Dict[str, Any] = {}

    if isinstance(image_data, np.ndarray):
        img = image_data
        if img.ndim != 3 or img.shape[2] != 3:
            raise ImageDecodeError("Expected a 3-channel BGR image matrix.")
        h, w = img.shape[:2]
    else:
        raw_bytes = extract_raw_image_bytes(image_data)
        validate_image_bytes(raw_bytes)
        img = decode_image_safely(raw_bytes)
        h, w = img.shape[:2]
        metadata["payload_bytes"] = len(raw_bytes)

    metadata["width"] = w
    metadata["height"] = h

    if not skip_quality_checks:
        quality_info = check_image_quality(img)
        metadata.update(quality_info)

    return img, metadata
