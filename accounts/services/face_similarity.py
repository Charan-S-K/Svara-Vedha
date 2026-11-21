"""
Face Similarity Service for SVARAABHAVA.

Calculates cosine similarity between ArcFace face embeddings and determines
biometric match results against a configurable threshold.

THRESHOLD CALIBRATION DOCUMENTATION:
==================================
The default threshold configured here (0.50) is an INITIAL BASELINE for development
and testing. It is NOT an empirical production-ready threshold.

In production biometric systems, the similarity threshold MUST be calibrated using
a formal operational dataset containing paired genuine (same-person) and impostor
(different-person) probe images. The operating threshold should be selected
based on acceptable False Match Rate (FMR / FAR) and False Non-Match Rate (FNMR / FRR)
trade-offs matching organizational security policy.

DO NOT HASH EMBEDDINGS:
Face embeddings are continuous high-dimensional vectors. Direct cryptographic hashing
(e.g., SHA-256) destroys the spatial geometry required for distance/similarity metrics.
Embeddings must be compared via vector dot product / cosine similarity.
"""

import os
from typing import Any, Dict, List, Optional, Union
import numpy as np
from django.conf import settings

from .face_recognition import (
    EXPECTED_EMBEDDING_DIM,
    InvalidEmbeddingDimensionError,
    InvalidEmbeddingValueError,
)

# Baseline threshold for cosine similarity comparison
# ArcFace cosine similarity range is [-1.0, 1.0], with identical faces approaching 1.0.
DEFAULT_SIMILARITY_THRESHOLD = 0.50


def get_similarity_threshold() -> float:
    """
    Retrieve configured face similarity threshold from Django settings
    or environment variables.
    """
    configured = getattr(settings, "FACE_SIMILARITY_THRESHOLD", None)
    if configured is None:
        env_val = os.getenv("FACE_SIMILARITY_THRESHOLD")
        if env_val:
            try:
                configured = float(env_val)
            except ValueError:
                configured = None

    if configured is None:
        return DEFAULT_SIMILARITY_THRESHOLD

    return float(configured)


def validate_and_normalize_vector(
    embedding: Any,
    expected_dim: Optional[int] = None,
    name: str = "Embedding",
) -> np.ndarray:
    """
    Validate that an input vector is a valid float32 NumPy array with expected dimensions
    and normalize it to unit L2 norm.

    Args:
        embedding: List, array-like, or NumPy array.
        expected_dim: Expected vector dimension (defaults to settings.ARCFACE_EMBEDDING_DIM or 512).
        name: Variable name for error diagnostics.

    Returns:
        np.ndarray: L2 normalized 1D float32 array.

    Raises:
        InvalidEmbeddingDimensionError: If vector dimension does not match.
        InvalidEmbeddingValueError: If vector has NaN, Inf, or zero length.
    """
    if embedding is None:
        raise InvalidEmbeddingValueError(f"{name} cannot be None.")

    target_dim = expected_dim or int(getattr(settings, "ARCFACE_EMBEDDING_DIM", EXPECTED_EMBEDDING_DIM))

    if not isinstance(embedding, np.ndarray):
        try:
            arr = np.array(embedding, dtype=np.float32)
        except Exception as e:
            raise InvalidEmbeddingValueError(f"{name} could not be converted to a NumPy array: {str(e)}") from e
    else:
        arr = embedding.astype(np.float32)

    flat = arr.flatten()
    if flat.shape[0] != target_dim:
        raise InvalidEmbeddingDimensionError(actual_dim=flat.shape[0], expected_dim=target_dim)

    if np.isnan(flat).any() or np.isinf(flat).any():
        raise InvalidEmbeddingValueError(f"{name} contains NaN or infinite values.")

    norm = float(np.linalg.norm(flat))
    if norm == 0.0 or np.isnan(norm) or np.isinf(norm):
        raise InvalidEmbeddingValueError(f"{name} has zero or degenerate L2 norm.")

    return (flat / norm).astype(np.float32)


def calculate_cosine_similarity(
    embedding1: Any,
    embedding2: Any,
    expected_dim: Optional[int] = None,
) -> float:
    """
    Calculate the cosine similarity between two face embeddings:
        cos_sim = dot(u, v) / (||u|| * ||v||)

    For unit L2 normalized vectors, this equals the vector dot product.

    Returns:
        float: Cosine similarity score clipped between -1.0 and 1.0.

    Raises:
        InvalidEmbeddingDimensionError: If vector dimensions do not match.
        InvalidEmbeddingValueError: If vectors have NaN, Inf, or zero norm.
    """
    # Expected dimension defaults to configured ARCFACE_EMBEDDING_DIM (512)
    target_dim = expected_dim or int(getattr(settings, "ARCFACE_EMBEDDING_DIM", EXPECTED_EMBEDDING_DIM))

    vec1 = validate_and_normalize_vector(embedding1, expected_dim=target_dim, name="Probe Embedding")
    vec2 = validate_and_normalize_vector(embedding2, expected_dim=target_dim, name="Stored Embedding")

    # Compute dot product
    sim = float(np.dot(vec1, vec2))

    # Guard against float rounding exceeding bounds
    clipped_sim = max(-1.0, min(1.0, sim))
    return clipped_sim


def compare_face_embeddings(
    probe_embedding: Any,
    stored_embedding: Any,
    threshold: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Compare a captured (probe) face embedding against a stored reference template.

    Args:
        probe_embedding: The newly captured face embedding.
        stored_embedding: The reference template embedding from storage.
        threshold: Optional custom similarity threshold. If None, uses configured threshold.

    Returns:
        Dict[str, Any]:
            - 'is_match': bool (True if similarity >= threshold)
            - 'similarity': float (cosine similarity value)
            - 'threshold': float (comparison threshold applied)
            - 'confidence_percentage': float (normalized indicator for UI diagnostics)
    """
    if threshold is None:
        threshold = get_similarity_threshold()

    similarity = calculate_cosine_similarity(probe_embedding, stored_embedding)
    is_match = similarity >= threshold

    # Approximate diagnostic confidence percentage (0 to 100%)
    confidence = max(0.0, min(100.0, ((similarity + 1.0) / 2.0) * 100.0))

    return {
        "is_match": bool(is_match),
        "similarity": round(float(similarity), 4),
        "threshold": round(float(threshold), 4),
        "confidence_percentage": round(confidence, 2),
    }


def verify_user_face(
    user_id: Union[int, str],
    probe_embedding: np.ndarray,
    storage: Optional[Any] = None,
    threshold: Optional[float] = None,
) -> Dict[str, Any]:
    """
    High-level face verification service:
    1. Retrieves enrolled face template from storage repository.
    2. Decrypts and validates reference embedding.
    3. Calculates cosine similarity against probe embedding.
    4. Evaluates against calibrated threshold.

    Args:
        user_id: Django User ID.
        probe_embedding: Newly extracted face embedding.
        storage: Optional FaceStorageInterface instance (defaults to get_face_storage()).
        threshold: Optional comparison threshold.

    Returns:
        Dict[str, Any]: Verification outcome including is_match, similarity, and threshold.

    Raises:
        TemplateNotFoundError: If user has no registered face template.
        InactiveTemplateError: If face template is revoked or inactive.
        FaceStorageError: If storage operation fails.
    """
    from .face_storage_interface import get_face_storage

    if storage is None:
        storage = get_face_storage()

    template_record = storage.get_face_template(user_id)
    if template_record is None or "embedding" not in template_record:
        return {
            "is_match": False,
            "error": "No registered face template found for this user.",
            "code": "template_not_found",
            "user_id": str(user_id),
        }

    stored_embedding = template_record["embedding"]
    result = compare_face_embeddings(probe_embedding, stored_embedding, threshold=threshold)
    result["user_id"] = str(user_id)
    return result


def identify_face_candidate(
    probe_embedding: Any,
    candidates: List[Dict[str, Any]],
    threshold: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Search a pool of registered biometric face templates to identify a matching candidate.

    Enforces strict security rules:
    - Cosine similarity: dot(embedding_1, embedding_2) for normalized embeddings.
    - Never accept a candidate whose similarity is below the threshold.
    - If multiple registered users exist, identifies the highest genuine match.
    - Ensures that a different person cannot be accepted merely because they are the highest score.
    """
    if threshold is None:
        threshold = get_similarity_threshold()

    if not candidates:
        return {
            "matched": False,
            "user_id": None,
            "similarity": 0.0,
            "threshold": round(float(threshold), 4),
            "message": "No registered face templates available for identification.",
            "code": "no_templates",
        }

    target_dim = int(getattr(settings, "ARCFACE_EMBEDDING_DIM", EXPECTED_EMBEDDING_DIM))
    probe_vec = validate_and_normalize_vector(probe_embedding, expected_dim=target_dim, name="Probe Embedding")

    best_match: Optional[Dict[str, Any]] = None
    best_sim: float = -1.0
    second_sim: float = -1.0

    for cand in candidates:
        stored_raw = cand.get("embedding")
        if stored_raw is None:
            continue
        try:
            stored_vec = validate_and_normalize_vector(stored_raw, expected_dim=target_dim, name="Stored Candidate")
            sim = float(np.dot(probe_vec, stored_vec))
            sim = max(-1.0, min(1.0, sim))

            if sim > best_sim:
                second_sim = best_sim
                best_sim = sim
                best_match = cand
            elif sim > second_sim:
                second_sim = sim
        except Exception:
            continue

    if best_match is None or best_sim < threshold:
        return {
            "matched": False,
            "user_id": None,
            "similarity": round(max(0.0, best_sim), 4),
            "threshold": round(float(threshold), 4),
            "message": "Face did not match any registered identity above the security threshold.",
            "code": "face_mismatch",
        }

    return {
        "matched": True,
        "user_id": str(best_match.get("user_id")),
        "similarity": round(float(best_sim), 4),
        "threshold": round(float(threshold), 4),
        "second_similarity": round(float(second_sim), 4) if second_sim > -1.0 else None,
        "candidate": best_match,
    }
