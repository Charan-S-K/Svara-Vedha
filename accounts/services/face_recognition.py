"""
Face Recognition Service for SVARAABHAVA.

Provides ArcFace feature extraction using InsightFace (buffalo_l model pack):
- Thread-safe singleton model loader (loads once and reuses).
- Server-side face detection and alignment.
- Rejection of zero-face and multiple-face frames.
- Extraction of 512-dimensional ArcFace identity embeddings.
- Rigorous embedding L2 normalization and numerical validity checks (NaN/Inf rejection).
- Returns verified embedding array and detection metadata.

CRITICAL SECURITY RULE:
Never trust or accept facial embeddings or landmarks sent by the client browser.
All biometric feature extraction MUST be performed strictly on the backend.
"""

import os
import threading
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from django.conf import settings

from .face_validation import (
    FaceValidationError,
    FaceTooSmallError,
    NoFaceDetectedError,
    MultipleFacesDetectedError,
    preprocess_and_validate_image,
    validate_face_count,
    validate_face_size,
)

EXPECTED_EMBEDDING_DIM = 512


class FaceRecognitionError(Exception):
    """Base exception for face recognition pipeline failures."""
    pass


class InvalidEmbeddingDimensionError(FaceRecognitionError):
    """Raised when extracted embedding dimension does not match expected dimension (512)."""
    def __init__(self, actual_dim: int, expected_dim: int = EXPECTED_EMBEDDING_DIM):
        super().__init__(f"Invalid embedding dimension: expected {expected_dim}, got {actual_dim}.")
        self.actual_dim = actual_dim
        self.expected_dim = expected_dim


class InvalidEmbeddingValueError(FaceRecognitionError):
    """Raised when extracted embedding contains NaN, infinite, or degenerate values."""
    def __init__(self, message: str = "Face embedding contains invalid numerical values (NaN or Inf)."):
        super().__init__(message)


class FaceRecognitionService:
    """
    Singleton service wrapper around InsightFace FaceAnalysis (ArcFace).
    Loads the buffalo_l model once into memory and reuses it across requests.
    """

    _instance: Optional["FaceRecognitionService"] = None
    _lock = threading.Lock()

    def __init__(self):
        if hasattr(self, "_initialized") and self._initialized:
            return

        self._app = None
        self._load_model()
        self._initialized = True

    @classmethod
    def get_instance(cls) -> "FaceRecognitionService":
        """Thread-safe access to the singleton FaceRecognitionService instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def _resolve_model_root(self) -> str:
        """Resolve the InsightFace model directory from settings or defaults."""
        configured_root = getattr(settings, "INSIGHTFACE_ROOT", None) or os.getenv("INSIGHTFACE_ROOT")
        if configured_root and os.path.exists(configured_root):
            return os.path.abspath(configured_root)

        # Check BASE_DIR / models / .insightface
        base_dir = getattr(settings, "BASE_DIR", None)
        if base_dir:
            workspace_root = os.path.abspath(os.path.join(str(base_dir), "models", ".insightface"))
            if os.path.exists(workspace_root):
                return workspace_root

        # Fallback to standard user home ~/.insightface
        user_home_root = os.path.expanduser("~/.insightface")
        return user_home_root

    def _load_model(self) -> None:
        """Initialize and prepare the InsightFace FaceAnalysis buffalo_l pipeline."""
        from insightface.app import FaceAnalysis

        model_name = getattr(settings, "INSIGHTFACE_MODEL_NAME", "buffalo_l")
        model_root = self._resolve_model_root()
        providers = getattr(settings, "INSIGHTFACE_PROVIDERS", ["CPUExecutionProvider"])
        det_size = getattr(settings, "INSIGHTFACE_DET_SIZE", (640, 640))

        app = FaceAnalysis(name=model_name, root=model_root, providers=providers)
        # Prepare the model pipeline (detection + ArcFace recognition)
        app.prepare(ctx_id=-1, det_size=det_size)

        # Verify recognition model is loaded
        rec_model = app.models.get("recognition")
        if rec_model is None:
            raise FaceRecognitionError("InsightFace failed to initialize the 'recognition' (ArcFace) module.")

        self._app = app
        self._model_root = model_root
        self._model_name = model_name

    @property
    def app(self):
        return self._app

    def normalize_embedding(self, embedding: np.ndarray) -> np.ndarray:
        """
        Normalize a face embedding vector to unit L2 length.

        Args:
            embedding: 1D NumPy array representing the embedding.

        Returns:
            np.ndarray: Unit L2 normalized float32 embedding vector.

        Raises:
            InvalidEmbeddingValueError: If vector has zero norm or contains NaN/Inf.
        """
        if embedding is None or not isinstance(embedding, np.ndarray):
            raise InvalidEmbeddingValueError("Embedding must be a valid NumPy array.")

        if np.isnan(embedding).any() or np.isinf(embedding).any():
            raise InvalidEmbeddingValueError("Raw embedding contains NaN or Inf values.")

        norm = float(np.linalg.norm(embedding))
        if norm == 0.0 or np.isnan(norm) or np.isinf(norm):
            raise InvalidEmbeddingValueError(f"Invalid embedding L2 norm: {norm}.")

        normalized = (embedding / norm).astype(np.float32)
        return normalized

    def validate_embedding(self, embedding: np.ndarray) -> None:
        """
        Validate dimensions and numerical validity of an ArcFace embedding.

        Args:
            embedding: NumPy array to validate.

        Raises:
            InvalidEmbeddingDimensionError: If embedding dimension != 512.
            InvalidEmbeddingValueError: If embedding contains NaN, Inf, or zero values.
        """
        if embedding is None or not isinstance(embedding, np.ndarray):
            raise InvalidEmbeddingValueError("Embedding must be a non-null NumPy array.")

        flat = embedding.flatten()
        actual_dim = flat.shape[0]
        if actual_dim != EXPECTED_EMBEDDING_DIM:
            raise InvalidEmbeddingDimensionError(actual_dim=actual_dim, expected_dim=EXPECTED_EMBEDDING_DIM)

        if np.isnan(flat).any() or np.isinf(flat).any():
            raise InvalidEmbeddingValueError("Embedding contains NaN or infinite values.")

        # Ensure embedding is not degenerate (all zeros)
        if np.all(flat == 0):
            raise InvalidEmbeddingValueError("Degenerate embedding: all components are zero.")

    def extract_face_embedding(
        self,
        image_data: Union[str, bytes, np.ndarray],
        skip_quality_checks: bool = False,
    ) -> Dict[str, Any]:
        """
        Full backend biometric extraction pipeline:
        1. Decode and validate image payload.
        2. Detect faces using SCRFD backend detector.
        3. Enforce exactly 1 face present.
        4. Validate face bounding box size.
        5. Extract 512-D ArcFace embedding.
        6. L2 normalize embedding.
        7. Validate embedding dimension (512) and numerical integrity.

        Args:
            image_data: base64 Data URL, base64 string, raw bytes, or BGR ndarray.
            skip_quality_checks: Whether to skip blur/exposure validation (useful for tests).

        Returns:
            Dict[str, Any]: Dictionary containing:
                - 'embedding': 512-D normalized float32 numpy array.
                - 'bbox': [x1, y1, x2, y2] bounding box coordinates.
                - 'det_score': Detection confidence score.
                - 'embedding_dim': 512.
                - 'validation_metadata': Image and quality metrics.

        Raises:
            FaceValidationError: If image format, size, or quality is invalid.
            NoFaceDetectedError: If zero faces are detected.
            MultipleFacesDetectedError: If >1 faces are detected.
            FaceTooSmallError: If detected face is below minimum size.
            FaceRecognitionError: If extraction or embedding validation fails.
        """
        # 1. Image preprocessing and format validation
        img, img_metadata = preprocess_and_validate_image(
            image_data,
            skip_quality_checks=skip_quality_checks,
        )

        # 2. Run backend face detection & recognition pipeline
        faces = self._app.get(img)

        # 3. Reject zero-face or multi-face images
        face = validate_face_count(faces)

        # 4. Validate face size
        validate_face_size(face.bbox, img.shape)

        # 5. Extract ArcFace embedding
        raw_embedding = getattr(face, "embedding", None)
        if raw_embedding is None:
            raise FaceRecognitionError("ArcFace recognition model failed to produce a feature embedding.")

        # 6. Normalize embedding
        normed_embedding = self.normalize_embedding(raw_embedding)

        # 7. Validate embedding dimensions and numerical validity
        self.validate_embedding(normed_embedding)

        return {
            "embedding": normed_embedding,
            "bbox": [float(c) for c in face.bbox[:4]],
            "det_score": float(face.det_score) if hasattr(face, "det_score") else 1.0,
            "embedding_dim": int(normed_embedding.shape[0]),
            "validation_metadata": img_metadata,
        }


def extract_face_embedding(
    image_data: Union[str, bytes, np.ndarray],
    skip_quality_checks: bool = False,
) -> Dict[str, Any]:
    """
    Convenience function for backend face feature extraction.
    Reuses the singleton FaceRecognitionService instance.
    """
    service = FaceRecognitionService.get_instance()
    return service.extract_face_embedding(image_data, skip_quality_checks=skip_quality_checks)
