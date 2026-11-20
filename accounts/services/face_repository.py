"""
Face Template Repository for MongoDB Atlas.

Implements persistent storage, retrieval, updating, and deletion of biometric
ArcFace face templates in the 'face_templates' collection inside 'svaraabhava_biometrics'.

Features:
- Application-layer authenticated encryption (Fernet / AES-GCM) of embeddings before persistence.
- Dynamic dimension detection (never hardcoded 512).
- Unique index on user_id to enforce one template per user.
- Compound index on (user_id, enrollment.is_active) for high-performance active queries.
- Zero raw biometric embeddings or secret keys written to application logs.
- Attendance logging and duplicate punch verification in 'attendance_records'.
"""

import base64
from datetime import datetime, timezone
import hashlib
import logging
import os
from typing import Any, Dict, Optional, Union
import numpy as np
from django.conf import settings
from pymongo.collection import Collection
from pymongo.errors import DuplicateKeyError, PyMongoError

from .face_storage_interface import FaceStorageInterface, FaceStorageError
from .mongo_connection import (
    get_face_templates_collection,
    get_attendance_collection,
    MongoConnectionError,
)

logger = logging.getLogger(__name__)


def get_biometric_cipher():
    """
    Initialize authenticated application-layer cipher using Fernet (AES-128-CBC + HMAC-SHA256).
    Key is read from BIOMETRIC_ENCRYPTION_KEY or FACE_ENCRYPTION_KEY.
    If not provided in development/testing, falls back to a key derived from Django SECRET_KEY.
    """
    from cryptography.fernet import Fernet

    raw_key = (
        os.getenv("BIOMETRIC_ENCRYPTION_KEY")
        or getattr(settings, "BIOMETRIC_ENCRYPTION_KEY", None)
        or os.getenv("FACE_ENCRYPTION_KEY")
        or getattr(settings, "FACE_ENCRYPTION_KEY", None)
    )

    if raw_key:
        if isinstance(raw_key, str):
            raw_key = raw_key.strip().encode("utf-8")
        try:
            return Fernet(raw_key)
        except Exception:
            # If a pass-phrase or raw 32-byte secret was given, derive a 32-byte urlsafe base64 key
            derived = base64.urlsafe_b64encode(hashlib.sha256(raw_key).digest())
            return Fernet(derived)

    # Fallback to key derived from Django SECRET_KEY (deterministic for local dev/testing)
    secret = getattr(settings, "SECRET_KEY", "default-insecure-svara-key").encode("utf-8")
    derived = base64.urlsafe_b64encode(hashlib.sha256(b"svara-biometrics-" + secret).digest())
    return Fernet(derived)


class DuplicateEnrollmentError(FaceStorageError):
    """Raised when attempting to enroll a face for a user that already has an active template."""
    pass


class TemplateNotFoundError(FaceStorageError):
    """Raised when a face template is not found in MongoDB."""
    pass


class InactiveTemplateError(FaceStorageError):
    """Raised when a retrieved face template is marked as inactive."""
    pass


class MongoFaceRepository(FaceStorageInterface):
    """
    MongoDB Atlas implementation of FaceStorageInterface.
    Persists encrypted ArcFace embedding documents in MongoDB Atlas.
    """

    def __init__(
        self,
        collection: Optional[Collection] = None,
        attendance_collection: Optional[Collection] = None,
    ):
        """
        Initialize the repository.

        Args:
            collection: Optional pre-configured PyMongo collection for face_templates.
            attendance_collection: Optional pre-configured PyMongo collection for attendance_records.
        """
        self._collection = collection
        self._attendance_collection = attendance_collection
        self._indexes_ensured = False

    def _get_collection(self) -> Collection:
        """Get the PyMongo face_templates collection and ensure indexes are created."""
        if self._collection is not None:
            if not self._indexes_ensured:
                self._ensure_indexes(self._collection)
            return self._collection

        try:
            coll = get_face_templates_collection()
            if not self._indexes_ensured:
                self._ensure_indexes(coll)
            return coll
        except MongoConnectionError as e:
            raise FaceStorageError(f"MongoDB connection failed: {str(e)}") from e

    def _ensure_indexes(self, coll: Collection) -> None:
        """Create required unique and compound indexes on face_templates."""
        try:
            # 1. Unique index on user_id
            coll.create_index("user_id", unique=True, name="uniq_user_id")
            # 2. Compound index on (user_id, enrollment.is_active)
            coll.create_index(
                [("user_id", 1), ("enrollment.is_active", 1)],
                name="idx_user_active",
            )
            self._indexes_ensured = True
        except Exception as e:
            logger.warning("Could not verify or create MongoDB indexes: %s", str(e))

    def _encrypt(self, embedding: np.ndarray) -> bytes:
        """Encrypt raw float32 embedding bytes using authenticated application-layer cipher."""
        cipher = get_biometric_cipher()
        raw_bytes = embedding.astype(np.float32).tobytes()
        encrypted = cipher.encrypt(raw_bytes)
        logger.info(
            "[ENROLLMENT STAGE 9] Encryption successful: algorithm=Fernet(AES-128-CBC+HMAC-SHA256), encrypted_size=%d bytes",
            len(encrypted),
        )
        return encrypted

    def _decrypt(self, encrypted_payload: Union[bytes, str, list]) -> np.ndarray:
        """Decrypt stored payload into a float32 NumPy embedding array."""
        if isinstance(encrypted_payload, (list, np.ndarray)):
            # Unencrypted fallback (e.g. legacy test data)
            return np.array(encrypted_payload, dtype=np.float32)

        cipher = get_biometric_cipher()
        if isinstance(encrypted_payload, str):
            encrypted_payload = encrypted_payload.encode("utf-8")

        try:
            raw_bytes = cipher.decrypt(encrypted_payload)
            return np.frombuffer(raw_bytes, dtype=np.float32)
        except Exception as e:
            raise FaceStorageError(f"Biometric embedding decryption failed: {str(e)}") from e

    # =========================================================================
    # FaceStorageInterface Implementation
    # =========================================================================

    def save_face_template(
        self,
        user_id: Union[int, str],
        embedding: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None,
        overwrite: bool = False,
    ) -> bool:
        """
        Persist a new face template embedding in MongoDB Atlas.

        Args:
            user_id: Unique Django User identifier.
            embedding: Normalized float32 NumPy array from ArcFace.
            metadata: Optional dictionary with auxiliary metadata.
            overwrite: If True, updates existing template; if False, prevents duplicate enrollment.

        Returns:
            bool: True if saved successfully.

        Raises:
            DuplicateEnrollmentError: If user is already enrolled and overwrite is False.
            FaceStorageError: If database persistence fails.
        """
        str_user_id = str(user_id)
        coll = self._get_collection()

        # Check for existing enrollment
        existing = coll.find_one({"user_id": str_user_id})
        if existing and not overwrite:
            raise DuplicateEnrollmentError(
                f"User {str_user_id} already has a registered face template. "
                "Use re-enrollment to update your biometric face identity."
            )

        dim = int(embedding.shape[0])
        encrypted_embedding = self._encrypt(embedding)
        now_utc = datetime.now(timezone.utc)

        doc = {
            "user_id": str_user_id,
            "embedding": encrypted_embedding,
            "model": {
                "name": "ArcFace",
                "version": getattr(settings, "INSIGHTFACE_MODEL_NAME", "buffalo_l"),
                "dimensions": dim,
            },
            "enrollment": {
                "created_at": existing["enrollment"]["created_at"] if existing else now_utc,
                "updated_at": now_utc,
                "is_active": True,
            },
            "metadata": {
                "det_score": float(metadata.get("det_score", 1.0)) if metadata else 1.0,
                "username": str(metadata.get("username", "")) if metadata else "",
                "email": str(metadata.get("email", "")) if metadata else "",
            },
        }

        try:
            coll.replace_one({"user_id": str_user_id}, doc, upsert=True)
            logger.info(
                "[ENROLLMENT STAGE 10] MongoDB write successful: collection='%s', user_id=%s, model=%s, dimensions=%d, updated_at=%s",
                coll.name,
                str_user_id,
                doc["model"]["name"],
                dim,
                now_utc.isoformat(),
            )
            return True
        except DuplicateKeyError as e:
            raise DuplicateEnrollmentError(f"Duplicate enrollment detected for user {str_user_id}") from e
        except PyMongoError as e:
            logger.error("MongoDB error while saving face template for user %s: %s", str_user_id, str(e))
            raise FaceStorageError(f"Database error saving face template: {str(e)}") from e

    def get_face_template(
        self,
        user_id: Union[int, str],
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve and decrypt face template from MongoDB Atlas by user ID.

        Args:
            user_id: Unique Django User identifier.

        Returns:
            Optional[Dict[str, Any]]: Dictionary containing:
                - 'user_id': User ID string.
                - 'embedding': Decrypted float32 NumPy array.
                - 'model': Model metadata dictionary.
                - 'enrollment': Enrollment timestamps and active status.
            Returns None if no template exists.

        Raises:
            InactiveTemplateError: If the template exists but is marked inactive.
            FaceStorageError: If retrieval or decryption fails.
        """
        str_user_id = str(user_id)
        coll = self._get_collection()

        try:
            doc = coll.find_one({"user_id": str_user_id})
            if not doc:
                return None

            # Verify active status
            enrollment = doc.get("enrollment", {})
            if not enrollment.get("is_active", True):
                raise InactiveTemplateError(f"Face template for user {str_user_id} is inactive or revoked.")

            decrypted_emb = self._decrypt(doc["embedding"])
            expected_dim = doc.get("model", {}).get("dimensions", len(decrypted_emb))

            if decrypted_emb.shape[0] != expected_dim:
                raise FaceStorageError(
                    f"Decrypted embedding dimension mismatch: expected {expected_dim}, got {decrypted_emb.shape[0]}"
                )

            return {
                "user_id": doc["user_id"],
                "embedding": decrypted_emb,
                "model": doc.get("model", {}),
                "enrollment": doc.get("enrollment", {}),
                "metadata": doc.get("metadata", {}),
            }

        except (InactiveTemplateError, FaceStorageError):
            raise
        except PyMongoError as e:
            logger.error("MongoDB error retrieving face template for user %s: %s", str_user_id, str(e))
            raise FaceStorageError(f"Database error retrieving face template: {str(e)}") from e

    def update_face_template(
        self,
        user_id: Union[int, str],
        embedding: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Update / re-enroll an existing face template in MongoDB Atlas.

        Args:
            user_id: Unique Django User identifier.
            embedding: Normalized float32 NumPy array.
            metadata: Optional dictionary with auxiliary metadata.

        Returns:
            bool: True if updated successfully.
        """
        return self.save_face_template(user_id, embedding, metadata=metadata, overwrite=True)

    def delete_face_template(
        self,
        user_id: Union[int, str],
    ) -> bool:
        """
        Delete a face template from MongoDB Atlas.

        Args:
            user_id: Unique Django User identifier.

        Returns:
            bool: True if deleted, False if not found.
        """
        str_user_id = str(user_id)
        coll = self._get_collection()

        try:
            result = coll.delete_one({"user_id": str_user_id})
            return result.deleted_count > 0
        except PyMongoError as e:
            raise FaceStorageError(f"Database error deleting face template: {str(e)}") from e

    def has_face_template(
        self,
        user_id: Union[int, str],
    ) -> bool:
        """
        Check if an active face template exists in MongoDB Atlas for a user.

        Args:
            user_id: Unique Django User identifier.

        Returns:
            bool: True if active face template exists, False otherwise.
        """
        str_user_id = str(user_id)
        coll = self._get_collection()

        try:
            doc = coll.find_one(
                {"user_id": str_user_id, "enrollment.is_active": True},
                projection={"_id": 1},
            )
            return doc is not None
        except PyMongoError as e:
            raise FaceStorageError(f"Database error checking face template: {str(e)}") from e

    # =========================================================================
    # Attendance Storage in MongoDB
    # =========================================================================

    def _get_attendance_collection(self) -> Collection:
        """Get the PyMongo attendance_records collection."""
        if self._attendance_collection is not None:
            return self._attendance_collection
        return get_attendance_collection()

    def record_attendance(
        self,
        user_id: Union[int, str],
        timestamp: Optional[datetime] = None,
        status: str = "PRESENT",
        location_data: Optional[Dict[str, Any]] = None,
        similarity: float = 1.0,
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
        distance_meters: Optional[float] = None,
        accuracy_meters: Optional[float] = None,
        date_str: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Store an attendance punch record in MongoDB 'attendance_records'.

        Args:
            user_id: Unique Django User identifier.
            timestamp: UTC datetime of the punch.
            status: Attendance status (e.g. 'PRESENT').
            location_data: Optional geofence verification details dict.
            similarity: Cosine similarity score achieved.
            latitude: GPS latitude.
            longitude: GPS longitude.
            distance_meters: Distance from college center.
            accuracy_meters: GPS accuracy.
            date_str: Date string YYYY-MM-DD.

        Returns:
            Dict[str, Any]: Stored record details including record ID.
        """
        str_user_id = str(user_id)
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)
        if date_str is None:
            date_str = timestamp.strftime("%Y-%m-%d")

        loc = location_data or {}
        lat = latitude if latitude is not None else float(loc.get("latitude", 0.0))
        lon = longitude if longitude is not None else float(loc.get("longitude", 0.0))
        acc = accuracy_meters if accuracy_meters is not None else float(loc.get("accuracy_meters", 0.0))
        dist = distance_meters if distance_meters is not None else float(loc.get("distance_meters", 0.0))

        try:
            coll = self._get_attendance_collection()
            # Ensure index on (user_id, date) for quick duplicate checking
            try:
                coll.create_index([("user_id", 1), ("date_str", 1)], name="idx_user_date")
            except Exception:
                pass

            record = {
                "user_id": str_user_id,
                "timestamp_utc": timestamp,
                "date_str": date_str,
                "date": date_str,
                "status": status,
                "location": {
                    "latitude": lat,
                    "longitude": lon,
                    "accuracy_meters": acc,
                    "distance_meters": dist,
                    "campus_verified": True,
                },
                "verification": {
                    "method": "ArcFace",
                    "similarity": round(float(similarity), 4),
                },
            }

            res = coll.insert_one(record)
            record["_id"] = str(res.inserted_id)
            return record

        except Exception as e:
            logger.warning("MongoDB attendance record logging failed: %s", str(e))
            return {
                "user_id": str_user_id,
                "status": status,
                "error": str(e),
            }

    def has_punched_today(
        self,
        user_id: Union[int, str],
        today_date_str: Optional[str] = None,
    ) -> bool:
        """
        Check if user has already punched attendance today.

        Args:
            user_id: User identifier.
            today_date_str: 'YYYY-MM-DD' UTC date string.

        Returns:
            bool: True if an attendance record exists for today.
        """
        str_user_id = str(user_id)
        if not today_date_str:
            today_date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        try:
            coll = self._get_attendance_collection()
            doc = coll.find_one({"user_id": str_user_id, "date_str": today_date_str})
            if doc is not None:
                return True
            return coll.find_one({"user_id": str_user_id, "date": today_date_str}) is not None
        except Exception:
            return False
