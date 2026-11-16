"""
Face Storage Interface for SVARAABHAVA.

Provides a clean abstraction layer for persisting and retrieving biometric
ArcFace face templates.

IMPORTANT NOTE:
MongoDB connection, collections, and queries are intentionally omitted.
The developer will connect their own MongoDB repository implementation.
Until a concrete storage backend is configured, this interface raises
StorageNotConfiguredError or returns storage unconfigured status.
"""

from abc import ABC, abstractmethod
import base64
import os
from typing import Any, Dict, Optional, Union
import numpy as np
from django.conf import settings
from django.utils.module_loading import import_string


class FaceStorageError(Exception):
    """Base exception for face template storage operations."""
    pass


class StorageNotConfiguredError(FaceStorageError):
    """Raised when face template storage backend is not configured."""
    pass


class FaceStorageInterface(ABC):
    """
    Abstract Interface for Face Template Storage.

    Implementations must define persistence, retrieval, updating,
    deletion, and existence verification of ArcFace face embeddings.
    """

    @abstractmethod
    def save_face_template(
        self,
        user_id: Union[int, str],
        embedding: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None,
        overwrite: bool = False,
        **kwargs,
    ) -> bool:
        """
        Persist a new face template embedding for a user.

        Args:
            user_id: Unique Django User identifier.
            embedding: 512-dimensional float32 NumPy array.
            metadata: Optional dictionary with auxiliary metadata (e.g. timestamp).

        Returns:
            bool: True if persisted successfully, False otherwise.

        Raises:
            StorageNotConfiguredError: If storage backend is not connected.
            FaceStorageError: If persistence fails.
        """
        pass

    @abstractmethod
    def get_face_template(
        self,
        user_id: Union[int, str],
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve stored face template by user ID.

        Args:
            user_id: Unique Django User identifier.

        Returns:
            Optional[Dict[str, Any]]: Dictionary containing 'embedding' (numpy array)
            and any stored metadata, or None if template does not exist.

        Raises:
            StorageNotConfiguredError: If storage backend is not connected.
            FaceStorageError: If retrieval fails.
        """
        pass

    @abstractmethod
    def update_face_template(
        self,
        user_id: Union[int, str],
        embedding: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Update an existing face template embedding for a user.

        Args:
            user_id: Unique Django User identifier.
            embedding: 512-dimensional float32 NumPy array.
            metadata: Optional dictionary with auxiliary metadata.

        Returns:
            bool: True if updated successfully, False otherwise.

        Raises:
            StorageNotConfiguredError: If storage backend is not connected.
            FaceStorageError: If update fails.
        """
        pass

    @abstractmethod
    def delete_face_template(
        self,
        user_id: Union[int, str],
    ) -> bool:
        """
        Delete stored face template for a user.

        Args:
            user_id: Unique Django User identifier.

        Returns:
            bool: True if deleted successfully, False otherwise.

        Raises:
            StorageNotConfiguredError: If storage backend is not connected.
            FaceStorageError: If deletion fails.
        """
        pass

    @abstractmethod
    def has_face_template(
        self,
        user_id: Union[int, str],
    ) -> bool:
        """
        Check whether a face template exists for a user.

        Args:
            user_id: Unique Django User identifier.

        Returns:
            bool: True if face template exists, False otherwise.

        Raises:
            StorageNotConfiguredError: If storage backend is not connected.
            FaceStorageError: If query fails.
        """
        pass

    # =========================================================================
    # Application-Level Encryption Hooks
    # =========================================================================
    def encrypt_embedding(self, embedding: np.ndarray) -> bytes:
        """
        Hook prepared for application-level encryption of face embeddings.

        When MongoDB is connected with an encryption secret (e.g. FACE_ENCRYPTION_KEY),
        this hook can encrypt the raw embedding bytes before writing to the database.
        Default implementation returns raw float32 bytes or applies simple envelope.
        """
        raw_bytes = embedding.astype(np.float32).tobytes()
        enc_key = getattr(settings, "FACE_ENCRYPTION_KEY", None) or os.getenv("FACE_ENCRYPTION_KEY")
        if enc_key:
            # Prepare for symmetric encryption (e.g. Fernet or AES-GCM)
            try:
                from cryptography.fernet import Fernet
                f = Fernet(enc_key.encode() if isinstance(enc_key, str) else enc_key)
                return f.encrypt(raw_bytes)
            except Exception as e:
                raise FaceStorageError(f"Application-level encryption failed: {str(e)}") from e
        return raw_bytes

    def decrypt_embedding(self, payload: Union[bytes, list, np.ndarray]) -> np.ndarray:
        """
        Hook prepared for application-level decryption of face embeddings.
        """
        if isinstance(payload, np.ndarray):
            return payload.astype(np.float32)
        if isinstance(payload, list):
            return np.array(payload, dtype=np.float32)

        enc_key = getattr(settings, "FACE_ENCRYPTION_KEY", None) or os.getenv("FACE_ENCRYPTION_KEY")
        raw_bytes = payload
        if enc_key:
            try:
                from cryptography.fernet import Fernet
                f = Fernet(enc_key.encode() if isinstance(enc_key, str) else enc_key)
                raw_bytes = f.decrypt(raw_bytes)
            except Exception as e:
                raise FaceStorageError(f"Application-level decryption failed: {str(e)}") from e

        return np.frombuffer(raw_bytes, dtype=np.float32)


class NotConfiguredFaceStorage(FaceStorageInterface):
    """
    Default face template storage implementation.

    Clearly indicates that the storage backend is not configured until
    the user connects their own MongoDB repository implementation.
    """

    ERROR_MESSAGE = (
        "Face template storage is not configured. "
        "Please connect your MongoDB repository implementation to enable biometric persistence."
    )

    def save_face_template(
        self,
        user_id: Union[int, str],
        embedding: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None,
        overwrite: bool = False,
        **kwargs,
    ) -> bool:
        raise StorageNotConfiguredError(self.ERROR_MESSAGE)

    def get_face_template(
        self,
        user_id: Union[int, str],
    ) -> Optional[Dict[str, Any]]:
        raise StorageNotConfiguredError(self.ERROR_MESSAGE)

    def update_face_template(
        self,
        user_id: Union[int, str],
        embedding: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        raise StorageNotConfiguredError(self.ERROR_MESSAGE)

    def delete_face_template(
        self,
        user_id: Union[int, str],
    ) -> bool:
        raise StorageNotConfiguredError(self.ERROR_MESSAGE)

    def has_face_template(
        self,
        user_id: Union[int, str],
    ) -> bool:
        raise StorageNotConfiguredError(self.ERROR_MESSAGE)


class InMemoryFaceStorage(FaceStorageInterface):
    """
    In-Memory face storage backend for automated unit testing and mock verification.
    Not intended for production deployment.
    """

    def __init__(self):
        self._templates: Dict[str, Dict[str, Any]] = {}

    def _key(self, user_id: Union[int, str]) -> str:
        return str(user_id)

    def save_face_template(
        self,
        user_id: Union[int, str],
        embedding: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None,
        overwrite: bool = False,
        **kwargs,
    ) -> bool:
        key = self._key(user_id)
        if key in self._templates and not overwrite:
            from accounts.services.face_repository import DuplicateEnrollmentError
            raise DuplicateEnrollmentError(f"User {user_id} already has a face template enrolled.")
        self._templates[key] = {
            "user_id": user_id,
            "embedding": np.copy(embedding).astype(np.float32),
            "metadata": metadata or {},
        }
        return True

    def get_face_template(
        self,
        user_id: Union[int, str],
    ) -> Optional[Dict[str, Any]]:
        key = self._key(user_id)
        record = self._templates.get(key)
        if record is None:
            return None
        return {
            "user_id": record["user_id"],
            "embedding": np.copy(record["embedding"]),
            "metadata": dict(record["metadata"]),
        }

    def update_face_template(
        self,
        user_id: Union[int, str],
        embedding: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        key = self._key(user_id)
        if key not in self._templates:
            return False
        self._templates[key] = {
            "user_id": user_id,
            "embedding": np.copy(embedding).astype(np.float32),
            "metadata": metadata or {},
        }
        return True

    def delete_face_template(
        self,
        user_id: Union[int, str],
    ) -> bool:
        key = self._key(user_id)
        if key in self._templates:
            del self._templates[key]
            return True
        return False

    def has_face_template(
        self,
        user_id: Union[int, str],
    ) -> bool:
        return self._key(user_id) in self._templates

    def clear(self):
        """Helper to clear test storage."""
        self._templates.clear()


# Active storage backend instance cache
_ACTIVE_STORAGE_INSTANCE: Optional[FaceStorageInterface] = None


def get_face_storage() -> FaceStorageInterface:
    """
    Get the configured face storage interface instance.

    Looks up settings.FACE_STORAGE_BACKEND if set (dotted path to class).
    If not explicitly set but MONGODB_URI is available, automatically initializes
    MongoFaceRepository(). Otherwise returns NotConfiguredFaceStorage().
    """
    global _ACTIVE_STORAGE_INSTANCE
    if _ACTIVE_STORAGE_INSTANCE is not None:
        return _ACTIVE_STORAGE_INSTANCE

    backend_path = getattr(settings, "FACE_STORAGE_BACKEND", None)
    if backend_path:
        try:
            storage_cls = import_string(backend_path)
            _ACTIVE_STORAGE_INSTANCE = storage_cls()
            return _ACTIVE_STORAGE_INSTANCE
        except Exception as e:
            raise FaceStorageError(f"Failed to initialize FACE_STORAGE_BACKEND '{backend_path}': {str(e)}") from e

    # Check if MongoDB is configured via environment or settings
    mongo_uri = os.getenv("MONGODB_URI") or getattr(settings, "MONGODB_URI", None)
    if mongo_uri:
        try:
            from .face_repository import MongoFaceRepository
            _ACTIVE_STORAGE_INSTANCE = MongoFaceRepository()
            return _ACTIVE_STORAGE_INSTANCE
        except Exception as e:
            raise FaceStorageError(f"Failed to initialize MongoDB face repository: {str(e)}") from e

    return NotConfiguredFaceStorage()


def set_face_storage(backend: Optional[FaceStorageInterface]) -> None:
    """
    Explicitly set or reset the active face storage backend.
    Primarily used for testing and dynamic dependency injection.
    """
    global _ACTIVE_STORAGE_INSTANCE
    _ACTIVE_STORAGE_INSTANCE = backend
