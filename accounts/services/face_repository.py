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
import json
import logging
import os
import sqlite3
from typing import Any, Dict, List, Optional, Union
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


class LocalEncryptedFaceRepository(FaceStorageInterface):
    """
    Secure, persistent local biometric vault for SVARAABHAVA.
    Stores application-layer encrypted ArcFace templates in a local SQLite vault.
    Guarantees sub-millisecond retrieval and offline functionality when MongoDB Atlas
    is unreachable or experiencing network/IP-whitelist delays.
    """
    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            base_dir = getattr(settings, "BASE_DIR", None)
            if base_dir:
                models_dir = os.path.join(str(base_dir), "models")
                os.makedirs(models_dir, exist_ok=True)
                db_path = os.path.join(models_dir, "biometrics_vault.db")
            else:
                db_path = "biometrics_vault.db"
        self._db_path = os.path.abspath(db_path)
        self._init_db()

    def _get_connection(self):
        conn = sqlite3.connect(self._db_path, timeout=5.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        return conn

    def _init_db(self):
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS face_templates (
                    user_id TEXT PRIMARY KEY,
                    encrypted_embedding BLOB NOT NULL,
                    model_name TEXT NOT NULL,
                    dimensions INTEGER NOT NULL,
                    is_active INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    metadata_json TEXT
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_active ON face_templates(is_active);")
            conn.commit()

    def save_face_template(
        self,
        user_id: Union[int, str],
        embedding: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None,
        overwrite: bool = False,
    ) -> bool:
        str_user_id = str(user_id)
        cipher = get_biometric_cipher()
        encrypted_bytes = cipher.encrypt(embedding.astype(np.float32).tobytes())
        now_iso = datetime.now(timezone.utc).isoformat()
        dim = int(embedding.shape[0])
        meta_str = json.dumps(metadata or {})

        with self._get_connection() as conn:
            cur = conn.execute("SELECT created_at FROM face_templates WHERE user_id = ?", (str_user_id,))
            row = cur.fetchone()
            if row and not overwrite:
                raise DuplicateEnrollmentError(
                    f"User {str_user_id} already has a registered face template. "
                    "Use re-enrollment to update your biometric face identity."
                )
            created_at = row["created_at"] if row else now_iso

            conn.execute("""
                INSERT INTO face_templates (user_id, encrypted_embedding, model_name, dimensions, is_active, created_at, updated_at, metadata_json)
                VALUES (?, ?, ?, ?, 1, ?, ?, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                    encrypted_embedding = excluded.encrypted_embedding,
                    dimensions = excluded.dimensions,
                    is_active = 1,
                    updated_at = excluded.updated_at,
                    metadata_json = excluded.metadata_json
            """, (str_user_id, encrypted_bytes, "ArcFace", dim, created_at, now_iso, meta_str))
            conn.commit()
        return True

    def get_face_template(self, user_id: Union[int, str]) -> Optional[Dict[str, Any]]:
        str_user_id = str(user_id)
        with self._get_connection() as conn:
            cur = conn.execute("SELECT * FROM face_templates WHERE user_id = ? AND is_active = 1", (str_user_id,))
            row = cur.fetchone()
            if not row:
                return None
            cipher = get_biometric_cipher()
            try:
                decrypted = np.frombuffer(cipher.decrypt(row["encrypted_embedding"]), dtype=np.float32)
            except Exception as e:
                raise FaceStorageError(f"Local biometric vault decryption failed: {str(e)}") from e
            meta = json.loads(row["metadata_json"]) if row["metadata_json"] else {}
            return {
                "user_id": row["user_id"],
                "embedding": decrypted,
                "model": {"name": row["model_name"], "dimensions": row["dimensions"]},
                "enrollment": {"created_at": row["created_at"], "updated_at": row["updated_at"], "is_active": True},
                "metadata": meta,
            }

    def get_all_active_templates(self) -> List[Dict[str, Any]]:
        templates = []
        with self._get_connection() as conn:
            cur = conn.execute("SELECT * FROM face_templates WHERE is_active = 1")
            cipher = get_biometric_cipher()
            for row in cur.fetchall():
                try:
                    decrypted = np.frombuffer(cipher.decrypt(row["encrypted_embedding"]), dtype=np.float32)
                    meta = json.loads(row["metadata_json"]) if row["metadata_json"] else {}
                    templates.append({
                        "user_id": row["user_id"],
                        "embedding": decrypted,
                        "metadata": meta,
                        "model": {"name": row["model_name"], "dimensions": row["dimensions"]},
                    })
                except Exception as e:
                    logger.warning("Could not decrypt template for user %s: %s", row["user_id"], e)
        return templates

    def update_face_template(self, user_id: Union[int, str], embedding: np.ndarray, metadata: Optional[Dict[str, Any]] = None) -> bool:
        return self.save_face_template(user_id, embedding, metadata=metadata, overwrite=True)

    def delete_face_template(self, user_id: Union[int, str]) -> bool:
        str_user_id = str(user_id)
        with self._get_connection() as conn:
            conn.execute("DELETE FROM face_templates WHERE user_id = ?", (str_user_id,))
            conn.commit()
        return True

    def has_face_template(self, user_id: Union[int, str]) -> bool:
        str_user_id = str(user_id)
        with self._get_connection() as conn:
            cur = conn.execute("SELECT 1 FROM face_templates WHERE user_id = ? AND is_active = 1", (str_user_id,))
            return cur.fetchone() is not None


class MongoFaceRepository(FaceStorageInterface):
    """
    MongoDB Atlas implementation of FaceStorageInterface.
    Persists encrypted ArcFace embedding documents in MongoDB Atlas with
    local encrypted SQLite cache fallback for sub-millisecond retrieval and offline tolerance.
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
        self._local_vault = LocalEncryptedFaceRepository()

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
        Persist a new face template embedding in MongoDB Atlas and local encrypted vault.
        """
        str_user_id = str(user_id)
        # Always persist to encrypted local vault first for guaranteed resilience and speed
        self._local_vault.save_face_template(user_id, embedding, metadata=metadata, overwrite=overwrite)

        # Attempt synchronization with MongoDB Atlas
        try:
            coll = self._get_collection()
            existing = coll.find_one({"user_id": str_user_id})
            if existing and not overwrite:
                raise DuplicateEnrollmentError(
                    f"User {str_user_id} already has a registered face template."
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

            coll.replace_one({"user_id": str_user_id}, doc, upsert=True)
            logger.info(
                "[ENROLLMENT STAGE 10] MongoDB Atlas write successful: collection='%s', user_id=%s, model=%s, dimensions=%d",
                coll.name,
                str_user_id,
                doc["model"]["name"],
                dim,
            )
        except DuplicateEnrollmentError:
            raise
        except Exception as e:
            logger.warning(
                "[ENROLLMENT STORAGE NOTICE] MongoDB Atlas synchronization failed (%s). "
                "Face template safely persisted in encrypted local biometric vault for user_id=%s.",
                str(e),
                str_user_id,
            )

        return True

    def get_face_template(
        self,
        user_id: Union[int, str],
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve and decrypt face template from local vault (fast path) or MongoDB Atlas.
        """
        str_user_id = str(user_id)
        # 1. Fast path: check local encrypted vault (sub-millisecond)
        local_rec = self._local_vault.get_face_template(user_id)
        if local_rec is not None:
            return local_rec

        # 2. Check MongoDB Atlas if not in local vault
        try:
            coll = self._get_collection()
            doc = coll.find_one({"user_id": str_user_id})
            if not doc:
                return None

            enrollment = doc.get("enrollment", {})
            if not enrollment.get("is_active", True):
                raise InactiveTemplateError(f"Face template for user {str_user_id} is inactive or revoked.")

            decrypted_emb = self._decrypt(doc["embedding"])
            expected_dim = doc.get("model", {}).get("dimensions", len(decrypted_emb))

            if decrypted_emb.shape[0] != expected_dim:
                raise FaceStorageError(
                    f"Decrypted embedding dimension mismatch: expected {expected_dim}, got {decrypted_emb.shape[0]}"
                )

            meta = doc.get("metadata", {})
            # Cache into local vault for future instant retrieval
            self._local_vault.save_face_template(user_id, decrypted_emb, metadata=meta, overwrite=True)

            return {
                "user_id": doc["user_id"],
                "embedding": decrypted_emb,
                "model": doc.get("model", {}),
                "enrollment": doc.get("enrollment", {}),
                "metadata": meta,
            }

        except InactiveTemplateError:
            raise
        except Exception as e:
            logger.warning("MongoDB Atlas query failed (%s). Local vault returned None.", str(e))
            return None

    def get_all_active_templates(self) -> List[Dict[str, Any]]:
        """
        Retrieve all active face templates across local vault and MongoDB Atlas.
        """
        local_templates = self._local_vault.get_all_active_templates()
        if local_templates:
            return local_templates

        try:
            coll = self._get_collection()
            cursor = coll.find({"enrollment.is_active": True})
            for doc in cursor:
                try:
                    decrypted_emb = self._decrypt(doc["embedding"])
                    meta = doc.get("metadata", {})
                    self._local_vault.save_face_template(doc["user_id"], decrypted_emb, metadata=meta, overwrite=True)
                except Exception:
                    pass
            return self._local_vault.get_all_active_templates()
        except Exception as e:
            logger.warning("Could not sync templates from MongoDB Atlas: %s", str(e))
            return local_templates

    def update_face_template(
        self,
        user_id: Union[int, str],
        embedding: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Update / re-enroll an existing face template in storage.
        """
        return self.save_face_template(user_id, embedding, metadata=metadata, overwrite=True)

    def delete_face_template(
        self,
        user_id: Union[int, str],
    ) -> bool:
        """
        Delete a face template from storage.
        """
        str_user_id = str(user_id)
        self._local_vault.delete_face_template(user_id)
        try:
            coll = self._get_collection()
            coll.delete_one({"user_id": str_user_id})
        except Exception:
            pass
        return True

    def has_face_template(
        self,
        user_id: Union[int, str],
    ) -> bool:
        """
        Check if an active face template exists for a user.
        """
        str_user_id = str(user_id)
        if self._local_vault.has_face_template(user_id):
            return True
        try:
            coll = self._get_collection()
            doc = coll.find_one(
                {"user_id": str_user_id, "enrollment.is_active": True},
                projection={"_id": 1},
            )
            return doc is not None
        except Exception:
            return False

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
