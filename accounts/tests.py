"""
Unit and Integration Tests for SVARAABHAVA Face Feature Extraction & Recognition Backend.

Covers all 16 required test scenarios:
1. Valid image handling
2. Invalid image handling
3. No face detected
4. Multiple faces detected
5. Invalid embedding dimensions
6. NaN and infinite embeddings
7. Cosine similarity calculation
8. Matching embeddings
9. Nonmatching embeddings
10. Missing stored template
11. Unauthenticated enrollment
12. Failed face login
13. Successful face login with mocked storage
14. Attendance outside geofence
15. Poor GPS accuracy
16. Normal face login does not create attendance
"""

import base64
from unittest.mock import MagicMock, patch
import cv2
import numpy as np

from django.conf import settings
from django.contrib.auth.models import User
from django.test import Client, TestCase, override_settings
from django.urls import reverse, NoReverseMatch
from django.utils import timezone

from accounts.models import AttendanceRecord, Profile
from accounts.services import (
    EXPECTED_EMBEDDING_DIM,
    DEFAULT_SIMILARITY_THRESHOLD,
    FaceValidationError,
    InvalidImageFormatError,
    ImageDecodeError,
    NoFaceDetectedError,
    MultipleFacesDetectedError,
    FaceTooSmallError,
    ImageQualityError,
    InvalidEmbeddingDimensionError,
    InvalidEmbeddingValueError,
    FaceRecognitionError,
    preprocess_and_validate_image,
    validate_face_count,
    validate_face_size,
    extract_face_embedding,
    calculate_cosine_similarity,
    compare_face_embeddings,
    InMemoryFaceStorage,
    NotConfiguredFaceStorage,
    FaceStorageError,
    StorageNotConfiguredError,
    set_face_storage,
    get_face_storage,
    haversine_distance_meters,
    validate_gps_coordinates,
    verify_college_geofence,
    GeofenceValidationError,
    MongoFaceRepository,
    DuplicateEnrollmentError,
    InactiveTemplateError,
    TemplateNotFoundError,
    get_biometric_cipher,
)


def create_test_jpeg_b64(width: int = 320, height: int = 320) -> str:
    """Create a high-contrast valid synthetic image encoded as a JPEG data URL."""
    # Checkerboard / gradient pattern ensures sufficient sharpness and brightness
    img = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(height):
        for x in range(width):
            c = ((x // 32) + (y // 32)) % 2
            val = 80 + c * 100
            img[y, x] = [val, val, val]
    _, buf = cv2.imencode(".jpg", img)
    b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def create_mock_embedding(seed: int = 42) -> np.ndarray:
    """Generate a valid, normalized 512-dimensional float32 vector."""
    rng = np.random.RandomState(seed)
    vec = rng.randn(EXPECTED_EMBEDDING_DIM).astype(np.float32)
    norm = np.linalg.norm(vec)
    return (vec / norm).astype(np.float32)


def create_similar_embedding(base: np.ndarray, target_sim: float = 0.85, seed: int = 77) -> np.ndarray:
    """
    Construct a unit vector having mathematically exact cosine similarity with base.
    Uses Gram-Schmidt orthogonalization: target_sim * base + sqrt(1 - target_sim^2) * ortho.
    """
    rng = np.random.RandomState(seed)
    rand_vec = rng.randn(len(base)).astype(np.float32)
    ortho = rand_vec - np.dot(rand_vec, base) * base
    ortho = ortho / np.linalg.norm(ortho)
    probe = (target_sim * base) + (np.sqrt(1.0 - (target_sim ** 2)) * ortho)
    return (probe / np.linalg.norm(probe)).astype(np.float32)



class FaceRecognitionBackendTests(TestCase):
    """Suite of tests covering the face recognition pipeline, services, and endpoints."""

    def setUp(self):
        self.client = Client()
        self.username = "testuser@svaraabhava.ai"
        self.password = "Secr3tP@ssw0rd!"
        self.user = User.objects.create_user(
            username=self.username,
            email=self.username,
            first_name="Test",
            last_name="User",
            password=self.password,
        )
        # Use an isolated in-memory storage instance for tests
        self.mock_storage = InMemoryFaceStorage()
        set_face_storage(self.mock_storage)

        # Baseline genuine embedding for test user
        self.genuine_embedding = create_mock_embedding(seed=101)
        self.test_img_b64 = create_test_jpeg_b64()

        # College geofence coordinates from settings
        self.college_lat = float(getattr(settings, "COLLEGE_GEOFENCE_LATITUDE", 17.385044))
        self.college_lon = float(getattr(settings, "COLLEGE_GEOFENCE_LONGITUDE", 78.486671))

    def tearDown(self):
        # Reset storage
        set_face_storage(None)

    # -------------------------------------------------------------------------
    # 1. Valid Image
    # -------------------------------------------------------------------------
    def test_01_valid_image(self):
        """Test preprocessing and decoding of a valid image payload."""
        img, metadata = preprocess_and_validate_image(self.test_img_b64, skip_quality_checks=True)
        self.assertIsInstance(img, np.ndarray)
        self.assertEqual(img.ndim, 3)
        self.assertEqual(img.shape[2], 3)
        self.assertEqual(metadata["width"], 320)
        self.assertEqual(metadata["height"], 320)
        self.assertGreater(metadata["payload_bytes"], 0)

    # -------------------------------------------------------------------------
    # 2. Invalid Image
    # -------------------------------------------------------------------------
    def test_02_invalid_image(self):
        """Test rejection of corrupted bytes, empty strings, and non-image payloads."""
        # A. Empty string
        with self.assertRaises(InvalidImageFormatError):
            preprocess_and_validate_image("")

        # B. Corrupt base64 string
        with self.assertRaises(InvalidImageFormatError):
            preprocess_and_validate_image("data:image/jpeg;base64,ThisIsNotValidBase64@@@!!!")

        # C. Raw bytes of non-image file
        with self.assertRaises(InvalidImageFormatError):
            preprocess_and_validate_image(b"THIS_IS_A_TEXT_FILE_NOT_AN_IMAGE_HEADER")

        # D. Image smaller than minimum allowed dimensions (64x64 < 128x128)
        tiny_img = np.zeros((64, 64, 3), dtype=np.uint8)
        _, tiny_buf = cv2.imencode(".jpg", tiny_img)
        tiny_b64 = f"data:image/jpeg;base64,{base64.b64encode(tiny_buf.tobytes()).decode('utf-8')}"
        with self.assertRaises(ImageDecodeError):
            preprocess_and_validate_image(tiny_b64, skip_quality_checks=True)

    # -------------------------------------------------------------------------
    # 3. No Face Detected
    # -------------------------------------------------------------------------
    def test_03_no_face_detected(self):
        """Test rejection when zero faces are detected in the image."""
        faces_list = []
        with self.assertRaises(NoFaceDetectedError):
            validate_face_count(faces_list)

        # Verify pipeline raises NoFaceDetectedError when detector finds 0 faces
        with patch("accounts.services.face_recognition.FaceRecognitionService._load_model"):
            from accounts.services.face_recognition import FaceRecognitionService
            srv = FaceRecognitionService.get_instance()
            orig_app = srv._app
            mock_app = MagicMock()
            mock_app.get.return_value = []
            srv._app = mock_app
            try:
                with self.assertRaises(NoFaceDetectedError):
                    extract_face_embedding(self.test_img_b64, skip_quality_checks=True)
            finally:
                srv._app = orig_app

    # -------------------------------------------------------------------------
    # 4. Multiple Faces Detected
    # -------------------------------------------------------------------------
    def test_04_multiple_faces_detected(self):
        """Test rejection when multiple faces (>1) are present in the frame."""
        face1 = MagicMock()
        face2 = MagicMock()
        faces_list = [face1, face2]
        with self.assertRaises(MultipleFacesDetectedError) as ctx:
            validate_face_count(faces_list)
        self.assertEqual(ctx.exception.count, 2)

        # Verify pipeline rejects multi-face inputs
        with patch("accounts.services.face_recognition.FaceRecognitionService._load_model"):
            from accounts.services.face_recognition import FaceRecognitionService
            srv = FaceRecognitionService.get_instance()
            orig_app = srv._app
            mock_app = MagicMock()
            mock_app.get.return_value = [face1, face2]
            srv._app = mock_app
            try:
                with self.assertRaises(MultipleFacesDetectedError):
                    extract_face_embedding(self.test_img_b64, skip_quality_checks=True)
            finally:
                srv._app = orig_app

    # -------------------------------------------------------------------------
    # 5. Invalid Embedding Dimensions
    # -------------------------------------------------------------------------
    def test_05_invalid_embedding_dimensions(self):
        """Test rejection of embeddings with dimension != 512."""
        # 128-D vector
        wrong_128 = np.ones(128, dtype=np.float32)
        with self.assertRaises(InvalidEmbeddingDimensionError) as ctx:
            calculate_cosine_similarity(wrong_128, self.genuine_embedding)
        self.assertEqual(ctx.exception.actual_dim, 128)
        self.assertEqual(ctx.exception.expected_dim, 512)

        # 1024-D vector
        wrong_1024 = np.ones(1024, dtype=np.float32)
        with self.assertRaises(InvalidEmbeddingDimensionError):
            calculate_cosine_similarity(self.genuine_embedding, wrong_1024)

    # -------------------------------------------------------------------------
    # 6. NaN and Infinite Embeddings
    # -------------------------------------------------------------------------
    def test_06_nan_and_infinite_embeddings(self):
        """Test rejection of embeddings containing NaN, +Inf, or -Inf values."""
        # NaN embedding
        nan_vec = np.copy(self.genuine_embedding)
        nan_vec[10] = np.nan
        with self.assertRaises(InvalidEmbeddingValueError):
            calculate_cosine_similarity(nan_vec, self.genuine_embedding)

        # Infinite embedding
        inf_vec = np.copy(self.genuine_embedding)
        inf_vec[25] = np.inf
        with self.assertRaises(InvalidEmbeddingValueError):
            calculate_cosine_similarity(self.genuine_embedding, inf_vec)

        # All zeros / degenerate embedding
        zero_vec = np.zeros(512, dtype=np.float32)
        with self.assertRaises(InvalidEmbeddingValueError):
            calculate_cosine_similarity(zero_vec, self.genuine_embedding)

    # -------------------------------------------------------------------------
    # 7. Cosine Similarity
    # -------------------------------------------------------------------------
    def test_07_cosine_similarity(self):
        """Test mathematical accuracy of cosine similarity calculation."""
        v1 = self.genuine_embedding
        # Identity vector similarity must be 1.0
        sim_ident = calculate_cosine_similarity(v1, v1)
        self.assertAlmostEqual(sim_ident, 1.0, places=4)

        # Opposite vector similarity must be -1.0
        v_opp = -v1
        sim_opp = calculate_cosine_similarity(v1, v_opp)
        self.assertAlmostEqual(sim_opp, -1.0, places=4)

        # Orthogonal vectors: construct orthogonal vector via Gram-Schmidt
        rng = np.random.RandomState(99)
        v_rand = rng.randn(512).astype(np.float32)
        v_ortho = v_rand - np.dot(v_rand, v1) * v1
        v_ortho = v_ortho / np.linalg.norm(v_ortho)
        sim_ortho = calculate_cosine_similarity(v1, v_ortho)
        self.assertAlmostEqual(sim_ortho, 0.0, places=3)

    # -------------------------------------------------------------------------
    # 8. Matching Embeddings
    # -------------------------------------------------------------------------
    def test_08_matching_embeddings(self):
        """Test that same-person embeddings with similarity >= threshold match."""
        # Construct mathematically exact probe having similarity 0.85 (> 0.50 threshold)
        probe = create_similar_embedding(self.genuine_embedding, target_sim=0.85, seed=12)

        result = compare_face_embeddings(probe, self.genuine_embedding, threshold=0.50)
        self.assertTrue(result["is_match"])
        self.assertGreaterEqual(result["similarity"], 0.50)
        self.assertGreater(result["confidence_percentage"], 50.0)

    # -------------------------------------------------------------------------
    # 9. Nonmatching Embeddings
    # -------------------------------------------------------------------------
    def test_09_nonmatching_embeddings(self):
        """Test that different-person embeddings with similarity < threshold do not match."""
        impostor = create_mock_embedding(seed=999)
        result = compare_face_embeddings(impostor, self.genuine_embedding, threshold=0.50)
        self.assertFalse(result["is_match"])
        self.assertLess(result["similarity"], 0.50)

    # -------------------------------------------------------------------------
    # 10. Missing Stored Template
    # -------------------------------------------------------------------------
    def test_10_missing_stored_template(self):
        """Test face login when user has no face template stored."""
        # Ensure user has no template in storage
        self.mock_storage.delete_face_template(self.user.id)

        with patch("accounts.views.extract_face_embedding") as mock_extract:
            mock_extract.return_value = {
                "embedding": self.genuine_embedding,
                "bbox": [10, 10, 100, 100],
                "det_score": 0.98,
            }
            resp = self.client.post(
                reverse("face_login_api"),
                data={
                    "identifier": self.username,
                    "face_image": self.test_img_b64,
                },
            )
            self.assertEqual(resp.status_code, 404)
            data = resp.json()
            self.assertFalse(data["success"])
            self.assertEqual(data["code"], "template_not_found")

    # -------------------------------------------------------------------------
    # 11. Unauthenticated Enrollment
    # -------------------------------------------------------------------------
    def test_11_unauthenticated_enrollment(self):
        """Test that unauthenticated users are rejected from enrolling face templates."""
        resp = self.client.post(
            reverse("face_enroll_api"),
            data={"face_image": self.test_img_b64},
        )
        self.assertEqual(resp.status_code, 401)
        data = resp.json()
        self.assertFalse(data["success"])

        # Also test unified /face-login/ endpoint with action_type='enroll'
        resp2 = self.client.post(
            reverse("face_login"),
            data={"action_type": "enroll", "face_image": self.test_img_b64},
        )
        self.assertEqual(resp2.status_code, 401)
        self.assertFalse(resp2.json()["success"])

    # -------------------------------------------------------------------------
    # 12. Failed Face Login
    # -------------------------------------------------------------------------
    def test_12_failed_face_login(self):
        """Test that an impostor face probe is rejected and does not authenticate a session."""
        # Enroll genuine template
        self.mock_storage.save_face_template(self.user.id, self.genuine_embedding)

        # Probe with different face
        impostor_embedding = create_mock_embedding(seed=555)
        with patch("accounts.views.extract_face_embedding") as mock_extract:
            mock_extract.return_value = {
                "embedding": impostor_embedding,
                "bbox": [10, 10, 100, 100],
                "det_score": 0.99,
            }
            resp = self.client.post(
                reverse("face_login_api"),
                data={
                    "identifier": self.username,
                    "face_image": self.test_img_b64,
                },
            )
            self.assertEqual(resp.status_code, 401)
            data = resp.json()
            self.assertFalse(data["success"])
            self.assertEqual(data["code"], "face_mismatch")

            # Verify no Django session was created
            self.assertNotIn("_auth_user_id", self.client.session)

    # -------------------------------------------------------------------------
    # 13. Successful Face Login with Mocked Storage
    # -------------------------------------------------------------------------
    def test_13_successful_face_login_with_mocked_storage(self):
        """Test successful face verification creates session and redirects to dashboard."""
        self.mock_storage.save_face_template(self.user.id, self.genuine_embedding)

        # Probe with matching embedding having similarity 0.88 (> 0.50 threshold)
        probe = create_similar_embedding(self.genuine_embedding, target_sim=0.88, seed=7)

        with patch("accounts.views.extract_face_embedding") as mock_extract:
            mock_extract.return_value = {
                "embedding": probe,
                "bbox": [20, 20, 150, 150],
                "det_score": 0.98,
            }
            resp = self.client.post(
                reverse("face_login_api"),
                data={
                    "identifier": self.username,
                    "face_image": self.test_img_b64,
                },
            )
            self.assertEqual(resp.status_code, 200, f"Error: {resp.content}")
            data = resp.json()
            self.assertTrue(data["success"])
            self.assertEqual(data["redirect_url"], "/dashboard/")

            # Verify user session is authenticated in Django
            self.assertEqual(int(self.client.session["_auth_user_id"]), self.user.id)

    # -------------------------------------------------------------------------
    # 14. Attendance Outside Geofence
    # -------------------------------------------------------------------------
    def test_14_attendance_outside_geofence(self):
        """Test that attendance punch is rejected when coordinates fall outside campus boundary."""
        self.mock_storage.save_face_template(self.user.id, self.genuine_embedding)

        # Remote location: Mumbai coordinates (~600 km away from Hyderabad campus)
        mumbai_lat = 19.0760
        mumbai_lon = 72.8777
        resp = self.client.post(
            reverse("face_attendance_api"),
            data={
                "identifier": self.username,
                "face_image": self.test_img_b64,
                "latitude": mumbai_lat,
                "longitude": mumbai_lon,
                "accuracy": 15.0,
            },
        )
        self.assertEqual(resp.status_code, 400)
        data = resp.json()
        self.assertFalse(data["success"])
        self.assertEqual(data["code"], "outside_geofence")

    # -------------------------------------------------------------------------
    # 15. Poor GPS Accuracy
    # -------------------------------------------------------------------------
    def test_15_poor_gps_accuracy(self):
        """Test that attendance punch is rejected when GPS accuracy exceeds threshold."""
        self.mock_storage.save_face_template(self.user.id, self.genuine_embedding)

        # Accurate center of campus, but high uncertainty radius (±350m > 100m limit)
        resp = self.client.post(
            reverse("face_attendance_api"),
            data={
                "identifier": self.username,
                "face_image": self.test_img_b64,
                "latitude": self.college_lat,
                "longitude": self.college_lon,
                "accuracy": 350.0,
            },
        )
        self.assertEqual(resp.status_code, 400)
        data = resp.json()
        self.assertFalse(data["success"])
        self.assertEqual(data["code"], "poor_accuracy")

    # -------------------------------------------------------------------------
    # 16. Normal Face Login Does Not Create Attendance
    # -------------------------------------------------------------------------
    def test_16_normal_face_login_does_not_create_attendance(self):
        """Verify that standard face login does not touch attendance logic or return attendance data."""
        self.mock_storage.save_face_template(self.user.id, self.genuine_embedding)

        with patch("accounts.views.extract_face_embedding") as mock_extract, \
             patch("accounts.views.verify_college_geofence") as mock_geo:

            mock_extract.return_value = {
                "embedding": self.genuine_embedding,
                "bbox": [20, 20, 150, 150],
                "det_score": 0.99,
            }

            resp = self.client.post(
                reverse("face_login_api"),
                data={
                    "identifier": self.username,
                    "face_image": self.test_img_b64,
                    # Even if client erroneously sent coordinates to login endpoint:
                    "latitude": self.college_lat,
                    "longitude": self.college_lon,
                    "accuracy": 10.0,
                },
            )
            self.assertEqual(resp.status_code, 200, f"Error: {resp.content}")
            data = resp.json()
            self.assertTrue(data["success"])
            self.assertNotIn("attendance_status", data)
            # Geofence service must never have been called during login
            mock_geo.assert_not_called()

    # -------------------------------------------------------------------------
    # 17. Storage Not Configured (MongoDB Separation Requirement)
    # -------------------------------------------------------------------------
    def test_17_storage_not_configured_clean_error(self):
        """Verify clean 503 response when storage is NotConfiguredFaceStorage."""
        # Switch to default unconfigured storage
        set_face_storage(NotConfiguredFaceStorage())

        self.client.force_login(self.user)
        with patch("accounts.views.extract_face_embedding") as mock_extract:
            mock_extract.return_value = {
                "embedding": self.genuine_embedding,
                "bbox": [20, 20, 150, 150],
                "det_score": 0.99,
            }
            # Enrollment with unconfigured storage
            resp_enroll = self.client.post(
                reverse("face_enroll_api"),
                data={"face_image": self.test_img_b64},
            )
            self.assertEqual(resp_enroll.status_code, 503)
            data_enroll = resp_enroll.json()
            self.assertFalse(data_enroll["success"])
            self.assertEqual(data_enroll["code"], "storage_not_configured")

            # Login with unconfigured storage
            self.client.logout()
            resp_login = self.client.post(
                reverse("face_login_api"),
                data={"identifier": self.username, "face_image": self.test_img_b64},
            )
            self.assertEqual(resp_login.status_code, 503)
            data_login = resp_login.json()
            self.assertFalse(data_login["success"])
            self.assertEqual(data_login["code"], "storage_not_configured")

    # -------------------------------------------------------------------------
    # 18. Successful Attendance Location & Identity Recording
    # -------------------------------------------------------------------------
    def test_18_successful_attendance_verification(self):
        """Verify successful identity + location records attendance in AttendanceRecord model."""
        self.mock_storage.save_face_template(self.user.id, self.genuine_embedding)

        with patch("accounts.views.extract_face_embedding") as mock_extract:
            mock_extract.return_value = {
                "embedding": self.genuine_embedding,
                "bbox": [20, 20, 150, 150],
                "det_score": 0.99,
            }
            resp = self.client.post(
                reverse("face_attendance_api"),
                data={
                    "identifier": self.username,
                    "face_image": self.test_img_b64,
                    "latitude": self.college_lat,
                    "longitude": self.college_lon,
                    "accuracy": 12.0,
                },
            )
            self.assertEqual(resp.status_code, 200, f"Error: {resp.content}")
            data = resp.json()
            self.assertTrue(data["success"])
            self.assertEqual(data["attendance_status"], "recorded")
            self.assertIn("verified and recorded", data["message"])
            self.assertTrue(AttendanceRecord.objects.filter(user=self.user).exists())

    # -------------------------------------------------------------------------
    # 19. Magic Link Button Navigation
    # -------------------------------------------------------------------------
    def test_19_magic_link_button_renders_face_login_url(self):
        """Verify that the login page renders the Magic Link button pointing to face_login URL."""
        resp = self.client.get(reverse("login"))
        self.assertEqual(resp.status_code, 200)
        face_login_url = reverse("face_login")
        self.assertContains(resp, f'href="{face_login_url}"')
        self.assertContains(resp, "Magic Link")

    # -------------------------------------------------------------------------
    # 20. ArcFace Dedicated Extraction Endpoints Removed
    # -------------------------------------------------------------------------
    def test_20_face_extract_api_endpoint_removed(self):
        """Verify that dedicated /api/face/extract/ and alias endpoints have been removed (404)."""
        resp1 = self.client.post("/api/face/extract/", data={"face_image": self.test_img_b64})
        self.assertEqual(resp1.status_code, 404)

        resp2 = self.client.post("/accounts/api/face/extract/", data={"face_image": self.test_img_b64})
        self.assertEqual(resp2.status_code, 404)

        with self.assertRaises(NoReverseMatch):
            reverse("face_extract_api")

        with self.assertRaises(NoReverseMatch):
            reverse("face_extract_api_alias")

    # -------------------------------------------------------------------------
    # 21. Face Login Page Completely Removes Extraction Card & Button UI
    # -------------------------------------------------------------------------
    def test_21_face_template_ui_removes_extraction_elements(self):
        """Verify GET /face-login/ has completely removed extraction button, cards, and vector viewer."""
        resp = self.client.get(reverse("face_login"))
        self.assertEqual(resp.status_code, 200)
        self.assertNotContains(resp, "btnFaceExtract")
        self.assertNotContains(resp, "faceFeatureExtractionCard")
        self.assertNotContains(resp, "Extract Face Features")
        self.assertNotContains(resp, "completeVectorText")
        self.assertNotContains(resp, "btnCopyVector")
        self.assertNotContains(resp, "featExtPreview")
        self.assertNotContains(resp, "Total Extracted Features")

    # -------------------------------------------------------------------------
    # 22. Face Login API Does Not Leak Raw Biometric Embeddings or Similarity
    # -------------------------------------------------------------------------
    def test_22_face_login_api_does_not_leak_raw_embeddings_or_similarity(self):
        """Verify face_login_api does not expose raw embeddings, vector stats, or similarity floats."""
        self.mock_storage.save_face_template(self.user.id, self.genuine_embedding)

        # 1. On successful match
        with patch("accounts.views.extract_face_embedding") as mock_extract:
            mock_extract.return_value = {
                "embedding": self.genuine_embedding,
                "bbox": [20, 20, 150, 150],
                "det_score": 0.99,
            }
            resp_success = self.client.post(
                reverse("face_login_api"),
                data={"identifier": self.username, "face_image": self.test_img_b64},
            )
            self.assertEqual(resp_success.status_code, 200)
            data_success = resp_success.json()
            self.assertTrue(data_success["success"])
            self.assertNotIn("embedding", data_success)
            self.assertNotIn("similarity", data_success)
            self.assertNotIn("complete_vector", data_success)
            self.assertNotIn("feature_preview", data_success)

        # 2. On mismatch
        impostor_embedding = create_mock_embedding(seed=999)
        with patch("accounts.views.extract_face_embedding") as mock_extract:
            mock_extract.return_value = {
                "embedding": impostor_embedding,
                "bbox": [20, 20, 150, 150],
                "det_score": 0.99,
            }
            resp_fail = self.client.post(
                reverse("face_login_api"),
                data={"identifier": self.username, "face_image": self.test_img_b64},
            )
            self.assertEqual(resp_fail.status_code, 401)
            data_fail = resp_fail.json()
            self.assertFalse(data_fail["success"])
            self.assertNotIn("similarity", data_fail)
            self.assertNotIn("embedding", data_fail)
            self.assertEqual(data_fail["error"], "Face verification failed.")

    # -------------------------------------------------------------------------
    # 23. Attendance API Does Not Leak Raw Biometric Embeddings or Similarity
    # -------------------------------------------------------------------------
    def test_23_face_attendance_api_does_not_leak_raw_embeddings_or_similarity(self):
        """Verify face_attendance_api never exposes raw embeddings or similarity scores to client."""
        self.mock_storage.save_face_template(self.user.id, self.genuine_embedding)

        # 1. On successful punch
        with patch("accounts.views.extract_face_embedding") as mock_extract:
            mock_extract.return_value = {
                "embedding": self.genuine_embedding,
                "bbox": [20, 20, 150, 150],
                "det_score": 0.99,
            }
            resp_punch = self.client.post(
                reverse("face_attendance_api"),
                data={
                    "identifier": self.username,
                    "face_image": self.test_img_b64,
                    "latitude": self.college_lat,
                    "longitude": self.college_lon,
                    "accuracy": 15.0,
                },
            )
            self.assertEqual(resp_punch.status_code, 200)
            data_punch = resp_punch.json()
            self.assertTrue(data_punch["success"])
            self.assertNotIn("similarity", data_punch)
            self.assertNotIn("similarity", data_punch.get("data", {}))
            self.assertNotIn("embedding", data_punch)

        # 2. On face mismatch
        impostor_embedding = create_mock_embedding(seed=888)
        with patch("accounts.views.extract_face_embedding") as mock_extract:
            mock_extract.return_value = {
                "embedding": impostor_embedding,
                "bbox": [20, 20, 150, 150],
                "det_score": 0.99,
            }
            resp_mismatch = self.client.post(
                reverse("face_attendance_api"),
                data={
                    "identifier": self.username,
                    "face_image": self.test_img_b64,
                    "latitude": self.college_lat,
                    "longitude": self.college_lon,
                    "accuracy": 15.0,
                },
            )
            self.assertEqual(resp_mismatch.status_code, 401)
            data_mismatch = resp_mismatch.json()
            self.assertFalse(data_mismatch["success"])
            self.assertNotIn("similarity", data_mismatch)
            self.assertEqual(data_mismatch["error"], "Face verification failed.")

    # -------------------------------------------------------------------------
    # 24. Face Login POST Dispatches Signin and Creates Django Session
    # -------------------------------------------------------------------------
    def test_24_face_login_dispatches_signin_and_creates_session(self):
        """Verify unified POST /face-login/ defaults to signin, authenticates user, and redirects to dashboard."""
        self.mock_storage.save_face_template(self.user.id, self.genuine_embedding)

        with patch("accounts.views.extract_face_embedding") as mock_extract:
            mock_extract.return_value = {
                "embedding": self.genuine_embedding,
                "bbox": [20, 20, 150, 150],
                "det_score": 0.99,
            }
            resp = self.client.post(
                reverse("face_login"),
                data={"action_type": "signin", "identifier": self.username, "face_image": self.test_img_b64},
            )
            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            self.assertTrue(data["success"])
            self.assertEqual(data["redirect_url"], "/dashboard/")
            self.assertEqual(int(self.client.session["_auth_user_id"]), self.user.id)

    # -------------------------------------------------------------------------
    # 25. Face Login POST Dispatches Attendance Punch Without Creating Session
    # -------------------------------------------------------------------------
    def test_25_face_attendance_does_not_create_session(self):
        """Verify unified POST /face-login/ with action_type=punch records attendance but leaves user logged out."""
        self.mock_storage.save_face_template(self.user.id, self.genuine_embedding)

        with patch("accounts.views.extract_face_embedding") as mock_extract:
            mock_extract.return_value = {
                "embedding": self.genuine_embedding,
                "bbox": [20, 20, 150, 150],
                "det_score": 0.99,
            }
            resp = self.client.post(
                reverse("face_login"),
                data={
                    "action_type": "punch",
                    "identifier": self.username,
                    "face_image": self.test_img_b64,
                    "latitude": self.college_lat,
                    "longitude": self.college_lon,
                    "accuracy": 10.0,
                },
            )
            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            self.assertTrue(data["success"])
            self.assertEqual(data["attendance_status"], "recorded")
            # Strictly no Django session created
            self.assertIsNone(self.client.session.get("_auth_user_id"))

    # -------------------------------------------------------------------------
    # 26. Attendance Geofence Rejection with Clear Message
    # -------------------------------------------------------------------------
    def test_26_attendance_rejects_with_outside_geofence_message(self):
        """Verify attendance rejected outside campus area provides Outside permitted campus area message."""
        self.mock_storage.save_face_template(self.user.id, self.genuine_embedding)

        resp = self.client.post(
            reverse("face_attendance_api"),
            data={
                "identifier": self.username,
                "face_image": self.test_img_b64,
                "latitude": 28.6139,  # New Delhi
                "longitude": 77.2090,
                "accuracy": 15.0,
            },
        )
        self.assertEqual(resp.status_code, 400)
        data = resp.json()
        self.assertFalse(data["success"])
        self.assertEqual(data["code"], "outside_geofence")
        self.assertIn("Outside permitted campus area", data["error"])

    # -------------------------------------------------------------------------
    # 27. Face Login Rejects Mismatch with Generic Error
    # -------------------------------------------------------------------------
    def test_27_face_login_rejects_mismatch_with_face_verification_failed(self):
        """Verify face login probe mismatch returns 401 with informative 'Face verification failed.' message."""
        self.mock_storage.save_face_template(self.user.id, self.genuine_embedding)

        impostor_embedding = create_mock_embedding(seed=777)
        with patch("accounts.views.extract_face_embedding") as mock_extract:
            mock_extract.return_value = {
                "embedding": impostor_embedding,
                "bbox": [20, 20, 150, 150],
                "det_score": 0.99,
            }
            resp = self.client.post(
                reverse("face_login_api"),
                data={"identifier": self.username, "face_image": self.test_img_b64},
            )
            self.assertEqual(resp.status_code, 401)
            data = resp.json()
            self.assertFalse(data["success"])
            self.assertEqual(data["code"], "face_mismatch")
            self.assertEqual(data["error"], "Face verification failed.")
            self.assertNotIn("similarity", data)

    # -------------------------------------------------------------------------
    # 28. MongoFaceRepository - Application-layer Encryption & CRUD
    # -------------------------------------------------------------------------
    def test_28_mongo_face_repository_crud_and_encryption(self):
        """Verify MongoFaceRepository encrypts embeddings with Fernet, handles CRUD, and prevents duplicate enrollment."""
        mock_templates_store = {}
        mock_templates_col = MagicMock()
        mock_attendance_col = MagicMock()

        def fake_find_one(query):
            uid = query.get("user_id")
            doc = mock_templates_store.get(uid)
            if not doc:
                return None
            if "enrollment.is_active" in query and not doc.get("enrollment", {}).get("is_active"):
                return None
            return doc.copy()

        def fake_insert_one(doc):
            uid = doc["user_id"]
            if uid in mock_templates_store:
                from pymongo.errors import DuplicateKeyError
                raise DuplicateKeyError("E11000 duplicate key error")
            mock_templates_store[uid] = doc.copy()
            res = MagicMock()
            res.inserted_id = "test_id_123"
            return res

        def fake_replace_one(query, doc, upsert=False):
            uid = query["user_id"]
            mock_templates_store[uid] = doc.copy()
            res = MagicMock()
            res.matched_count = 1
            res.modified_count = 1
            return res

        def fake_delete_one(query):
            uid = query["user_id"]
            had = uid in mock_templates_store
            if had:
                del mock_templates_store[uid]
            res = MagicMock()
            res.deleted_count = 1 if had else 0
            return res

        mock_templates_col.find_one.side_effect = fake_find_one
        mock_templates_col.insert_one.side_effect = fake_insert_one
        mock_templates_col.replace_one.side_effect = fake_replace_one
        mock_templates_col.delete_one.side_effect = fake_delete_one

        repo = MongoFaceRepository(
            collection=mock_templates_col,
            attendance_collection=mock_attendance_col,
        )

        # 1. Save new face template
        str_id = str(self.user.id)
        saved = repo.save_face_template(
            user_id=self.user.id,
            embedding=self.genuine_embedding,
            metadata={"email": self.user.email},
            overwrite=False,
        )
        self.assertTrue(saved)
        self.assertIn(str_id, mock_templates_store)

        # Verify application-layer encryption: stored embedding must be bytes (ciphertext), NOT raw float array
        stored_doc = mock_templates_store[str_id]
        self.assertIsInstance(stored_doc["embedding"], bytes)
        self.assertNotEqual(stored_doc["embedding"], self.genuine_embedding.tobytes())

        # 2. Retrieve face template and verify automatic decryption
        retrieved = repo.get_face_template(self.user.id)
        self.assertIsNotNone(retrieved)
        self.assertEqual(str(retrieved["user_id"]), str_id)
        self.assertIsInstance(retrieved["embedding"], np.ndarray)
        self.assertEqual(retrieved["embedding"].dtype, np.float32)
        self.assertEqual(retrieved["embedding"].shape, (512,))
        self.assertTrue(np.allclose(retrieved["embedding"], self.genuine_embedding, atol=1e-5))

        # 3. Duplicate enrollment prevention: Attempting to save again without overwrite raises DuplicateEnrollmentError
        with self.assertRaises(DuplicateEnrollmentError):
            repo.save_face_template(
                user_id=self.user.id,
                embedding=self.genuine_embedding,
                overwrite=False,
            )

        # 4. Authenticated re-enrollment with overwrite=True succeeds
        new_embedding = create_mock_embedding(seed=202)
        updated = repo.save_face_template(
            user_id=self.user.id,
            embedding=new_embedding,
            overwrite=True,
        )
        self.assertTrue(updated)
        retrieved_updated = repo.get_face_template(self.user.id)
        self.assertTrue(np.allclose(retrieved_updated["embedding"], new_embedding, atol=1e-5))

        # 5. Inactive template handling
        mock_templates_store[str_id]["enrollment"]["is_active"] = False
        with self.assertRaises(InactiveTemplateError):
            repo.get_face_template(self.user.id)

        # 6. Delete template
        deleted = repo.delete_face_template(self.user.id)
        self.assertTrue(deleted)
        self.assertIsNone(repo.get_face_template(self.user.id))

    # -------------------------------------------------------------------------
    # 29. MongoFaceRepository - Attendance Records & Daily Duplicate Punch Check
    # -------------------------------------------------------------------------
    def test_29_mongo_face_repository_attendance(self):
        """Verify MongoFaceRepository records attendance and enforces has_punched_today."""
        mock_attendance_store = []
        mock_attendance_col = MagicMock()

        def fake_att_insert(doc):
            mock_attendance_store.append(doc.copy())
            res = MagicMock()
            res.inserted_id = "att_123"
            return res

        def fake_att_find_one(query):
            uid = query.get("user_id")
            date_str = query.get("date")
            for doc in mock_attendance_store:
                if doc.get("user_id") == uid and doc.get("date") == date_str:
                    return doc.copy()
            return None

        mock_attendance_col.insert_one.side_effect = fake_att_insert
        mock_attendance_col.find_one.side_effect = fake_att_find_one

        repo = MongoFaceRepository(
            collection=MagicMock(),
            attendance_collection=mock_attendance_col,
        )

        self.assertFalse(repo.has_punched_today(self.user.id))

        # Record punch
        recorded = repo.record_attendance(
            user_id=self.user.id,
            latitude=17.385044,
            longitude=78.486671,
            distance_meters=10.5,
            accuracy_meters=8.0,
            similarity=0.88,
        )
        self.assertTrue(recorded)
        self.assertEqual(len(mock_attendance_store), 1)
        self.assertEqual(mock_attendance_store[0]["user_id"], str(self.user.id))
        self.assertEqual(mock_attendance_store[0]["verification"]["similarity"], 0.88)

        # Should now report true for today
        self.assertTrue(repo.has_punched_today(self.user.id))

    # -------------------------------------------------------------------------
    # 30. Face Enrollment API - Success, Duplicate Prevention & Re-Enrollment
    # -------------------------------------------------------------------------
    def test_30_face_enroll_api_duplicate_handling(self):
        """Verify face_enroll_api handles duplicate enrollment and re-enrollment."""
        test_storage = InMemoryFaceStorage()
        set_face_storage(test_storage)
        self.client.force_login(self.user)

        with patch("accounts.views.extract_face_embedding") as mock_extract:
            mock_extract.return_value = {
                "embedding": self.genuine_embedding,
                "bbox": [20, 20, 150, 150],
                "det_score": 0.99,
            }

            # 1. First enrollment: succeeds
            resp1 = self.client.post(
                reverse("face_enroll_api"),
                data={"face_image": self.test_img_b64},
            )
            self.assertEqual(resp1.status_code, 200)
            data1 = resp1.json()
            self.assertTrue(data1["success"])
            self.assertEqual(data1["redirect_url"], "/dashboard/")

            # 2. Second enrollment attempt with mock repository raising DuplicateEnrollmentError
            with patch.object(test_storage, "save_face_template", side_effect=DuplicateEnrollmentError("Face already enrolled")):
                resp2 = self.client.post(
                    reverse("face_enroll_api"),
                    data={"face_image": self.test_img_b64},
                )
                self.assertEqual(resp2.status_code, 409)
                data2 = resp2.json()
                self.assertFalse(data2["success"])
                self.assertEqual(data2["code"], "already_enrolled")

            # 3. Third enrollment attempt with re_enroll=true: succeeds
            resp3 = self.client.post(
                reverse("face_enroll_api"),
                data={"face_image": self.test_img_b64, "re_enroll": "true"},
            )
            self.assertEqual(resp3.status_code, 200)
            data3 = resp3.json()
            self.assertTrue(data3["success"])

    # -------------------------------------------------------------------------
    # 31. Face Login API - Session Cycle Key & Successful Authentication
    # -------------------------------------------------------------------------
    def test_31_face_login_api_session_cycle_key(self):
        """Verify face_login_api cycles session key and authenticates Django session."""
        self.mock_storage.save_face_template(self.user.id, self.genuine_embedding)

        with patch("accounts.views.extract_face_embedding") as mock_extract:
            mock_extract.return_value = {
                "embedding": self.genuine_embedding,
                "bbox": [20, 20, 150, 150],
                "det_score": 0.99,
            }
            session = self.client.session
            session["dummy_key"] = "test"
            session.save()

            resp = self.client.post(
                reverse("face_login_api"),
                data={"identifier": self.username, "face_image": self.test_img_b64},
            )
            self.assertEqual(resp.status_code, 200, f"Error: {resp.content}")
            data = resp.json()
            self.assertTrue(data["success"])
            self.assertEqual(data["redirect_url"], "/dashboard/")

            # User must be authenticated in session
            self.assertEqual(int(self.client.session.get("_auth_user_id")), self.user.id)

    # -------------------------------------------------------------------------
    # 32. Face Login API - Inactive Template Rejection
    # -------------------------------------------------------------------------
    def test_32_face_login_api_inactive_template(self):
        """Verify face_login_api rejects inactive templates with 403 template_inactive."""
        with patch.object(self.mock_storage, "get_face_template", return_value={
            "user_id": self.user.id,
            "embedding": self.genuine_embedding,
            "enrollment": {"is_active": False},
        }):
            with patch("accounts.views.extract_face_embedding") as mock_extract:
                mock_extract.return_value = {
                    "embedding": self.genuine_embedding,
                    "bbox": [20, 20, 150, 150],
                    "det_score": 0.99,
                }
                resp = self.client.post(
                    reverse("face_login_api"),
                    data={"identifier": self.username, "face_image": self.test_img_b64},
                )
                self.assertEqual(resp.status_code, 403)
                data = resp.json()
                self.assertFalse(data["success"])
                self.assertEqual(data["code"], "template_inactive")

    # -------------------------------------------------------------------------
    # 33. Face Login API - Rate Limiting After 5 Failures
    # -------------------------------------------------------------------------
    def test_33_face_login_api_rate_limiting(self):
        """Verify that 5 failed face login attempts locks out the user with HTTP 429."""
        self.mock_storage.save_face_template(self.user.id, self.genuine_embedding)
        impostor_embedding = create_mock_embedding(seed=999)

        with patch("accounts.views.extract_face_embedding") as mock_extract:
            mock_extract.return_value = {
                "embedding": impostor_embedding,
                "bbox": [20, 20, 150, 150],
                "det_score": 0.99,
            }
            # Perform 5 failed attempts
            for i in range(5):
                resp = self.client.post(
                    reverse("face_login_api"),
                    data={"identifier": self.username, "face_image": self.test_img_b64},
                )
                self.assertEqual(resp.status_code, 401)
                self.assertFalse(resp.json()["success"])

            # 6th attempt must be rate limited with HTTP 429
            resp_locked = self.client.post(
                reverse("face_login_api"),
                data={"identifier": self.username, "face_image": self.test_img_b64},
            )
            self.assertEqual(resp_locked.status_code, 429)
            data_locked = resp_locked.json()
            self.assertFalse(data_locked["success"])
            self.assertEqual(data_locked["code"], "rate_limited")

    # -------------------------------------------------------------------------
    # 34. Attendance API - Duplicate Daily Punch Prevention & Separation from Login
    # -------------------------------------------------------------------------
    def test_34_attendance_daily_duplicate_punch_prevention(self):
        """Verify attendance API prevents multiple punches on same day and does not log user in."""
        self.mock_storage.save_face_template(self.user.id, self.genuine_embedding)

        with patch("accounts.views.extract_face_embedding") as mock_extract:
            mock_extract.return_value = {
                "embedding": self.genuine_embedding,
                "bbox": [20, 20, 150, 150],
                "det_score": 0.99,
            }
            # First punch of the day: recorded
            resp1 = self.client.post(
                reverse("face_attendance_api"),
                data={
                    "identifier": self.username,
                    "face_image": self.test_img_b64,
                    "latitude": self.college_lat,
                    "longitude": self.college_lon,
                    "accuracy": 15.0,
                },
            )
            self.assertEqual(resp1.status_code, 200)
            data1 = resp1.json()
            self.assertTrue(data1["success"])
            self.assertEqual(data1["attendance_status"], "recorded")

            # Verify AttendanceRecord created in Django DB
            self.assertEqual(AttendanceRecord.objects.filter(user=self.user).count(), 1)

            # User must NOT be logged into Django session
            self.assertIsNone(self.client.session.get("_auth_user_id"))

            # Second punch of the day: already_recorded, no second record created
            resp2 = self.client.post(
                reverse("face_attendance_api"),
                data={
                    "identifier": self.username,
                    "face_image": self.test_img_b64,
                    "latitude": self.college_lat,
                    "longitude": self.college_lon,
                    "accuracy": 15.0,
                },
            )
            self.assertEqual(resp2.status_code, 200)
            data2 = resp2.json()
            self.assertTrue(data2["success"])
            self.assertEqual(data2["attendance_status"], "already_recorded")
            self.assertIn("already recorded for today", data2["message"])

            # Still only 1 record
            self.assertEqual(AttendanceRecord.objects.filter(user=self.user).count(), 1)

    # -------------------------------------------------------------------------
    # 35. Verify OTP Navigation to Face Enrollment
    # -------------------------------------------------------------------------
    def test_35_verify_otp_redirects_to_face_enroll(self):
        """Verify successful OTP verification logs in the user and redirects to /face-enroll/."""
        session = self.client.session
        session["register_name"] = "Alice Biometric"
        session["register_phone"] = "9876543210"
        session["register_email"] = "alice@svaraabhava.ai"
        session["register_password"] = "AliceSecurePass123!"
        session["register_otp"] = "654321"
        session.save()

        resp = self.client.post(
            reverse("verify_otp"),
            data={"otp": "654321"},
        )
        self.assertEqual(resp.status_code, 302)
        self.assertEqual(resp.url, "/face-enroll/")

        # Verify user was created in Django ORM and is logged in
        created_user = User.objects.filter(email="alice@svaraabhava.ai").first()
        self.assertIsNotNone(created_user)
        self.assertEqual(int(self.client.session.get("_auth_user_id")), created_user.id)

    # -------------------------------------------------------------------------
    # 36. Cryptographic Key Derivation & Cipher Fallback
    # -------------------------------------------------------------------------
    def test_36_biometric_cipher_generation_and_roundtrip(self):
        """Verify get_biometric_cipher encrypts and decrypts 512-D float32 embeddings accurately."""
        cipher = get_biometric_cipher()
        self.assertIsNotNone(cipher)

        raw_bytes = self.genuine_embedding.tobytes()
        token = cipher.encrypt(raw_bytes)

        # Ciphertext must differ from plaintext
        self.assertNotEqual(token, raw_bytes)

        decrypted_bytes = cipher.decrypt(token)
        reconstructed = np.frombuffer(decrypted_bytes, dtype=np.float32)

        self.assertTrue(np.allclose(self.genuine_embedding, reconstructed))

    # -------------------------------------------------------------------------
    # 37. ArcFace Model Failure during Enrollment
    # -------------------------------------------------------------------------
    def test_37_enrollment_arcface_model_failure(self):
        """Verify face_enroll_api handles ArcFace pipeline failures cleanly."""
        self.client.force_login(self.user)
        with patch("accounts.views.extract_face_embedding", side_effect=FaceRecognitionError("ArcFace recognition model failed to produce a feature embedding.")):
            resp = self.client.post(
                reverse("face_enroll_api"),
                data={"face_image": self.test_img_b64},
            )
            self.assertEqual(resp.status_code, 500)
            data = resp.json()
            self.assertFalse(data["success"])
            self.assertIn("ArcFace recognition model failed", data["error"])

    # -------------------------------------------------------------------------
    # 38. MongoDB Connection Failure during Enrollment
    # -------------------------------------------------------------------------
    def test_38_enrollment_mongodb_connection_failure(self):
        """Verify face_enroll_api handles MongoDB connection/SSL failures gracefully."""
        self.client.force_login(self.user)
        with patch("accounts.views.extract_face_embedding", return_value={"embedding": self.genuine_embedding, "det_score": 0.99}):
            with patch.object(self.mock_storage, "save_face_template", side_effect=FaceStorageError("MongoDB connection failed: SSL handshake failed")):
                resp = self.client.post(
                    reverse("face_enroll_api"),
                    data={"face_image": self.test_img_b64},
                )
                self.assertEqual(resp.status_code, 500)
                data = resp.json()
                self.assertFalse(data["success"])
                self.assertIn("MongoDB connection failed", data["error"])

    # -------------------------------------------------------------------------
    # 39. Biometric Encryption Failure during Enrollment
    # -------------------------------------------------------------------------
    def test_39_enrollment_encryption_failure(self):
        """Verify MongoFaceRepository raises FaceStorageError if encryption fails."""
        mock_coll = MagicMock()
        mock_coll.find_one.return_value = None
        repo = MongoFaceRepository(collection=mock_coll)

        with patch.object(repo, "_encrypt", side_effect=Exception("Cryptographic cipher failure")):
            with self.assertRaises(Exception):
                repo.save_face_template(
                    user_id=self.user.id,
                    embedding=self.genuine_embedding,
                )

    # -------------------------------------------------------------------------
    # 40. Successful MongoDB Insert with Real Fernet Encryption
    # -------------------------------------------------------------------------
    def test_40_successful_mongodb_insert_with_real_fernet_encryption(self):
        """Verify end-to-end MongoDB document creation with encrypted embedding and indexes."""
        mock_coll = MagicMock()
        mock_coll.name = "face_templates"
        mock_coll.find_one.return_value = None
        mock_coll.replace_one.return_value = MagicMock(upserted_id="mock_id_123")

        repo = MongoFaceRepository(collection=mock_coll)
        set_face_storage(repo)

        self.client.force_login(self.user)
        with patch("accounts.views.extract_face_embedding", return_value={"embedding": self.genuine_embedding, "det_score": 0.985}):
            resp = self.client.post(
                reverse("face_enroll_api"),
                data={"face_image": self.test_img_b64},
            )
            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            self.assertTrue(data["success"])
            self.assertEqual(data["redirect_url"], "/dashboard/")

            # Verify replace_one was called on MongoDB collection with encrypted embedding
            self.assertTrue(mock_coll.replace_one.called)
            args, kwargs = mock_coll.replace_one.call_args
            doc_filter = args[0]
            doc_body = args[1]
            self.assertEqual(doc_filter, {"user_id": str(self.user.id)})
            self.assertIn("embedding", doc_body)
            self.assertIsInstance(doc_body["embedding"], bytes)
            # Ensure raw embedding is not plaintext
            self.assertNotEqual(doc_body["embedding"], self.genuine_embedding.tobytes())
            self.assertEqual(doc_body["model"]["name"], "ArcFace")
            self.assertEqual(doc_body["model"]["dimensions"], 512)
            self.assertTrue(doc_body["enrollment"]["is_active"])

