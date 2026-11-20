"""
MongoDB Connection Module for SVARAABHAVA Biometric Authentication.

Provides clean connection management, pooling, and collection access for
MongoDB Atlas using PyMongo. Credentials and URIs are loaded strictly
from environment variables or Django settings.
"""

import logging
import os
from typing import Optional
from django.conf import settings
import pymongo
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError, PyMongoError

logger = logging.getLogger(__name__)

# Singleton client instance
_MONGO_CLIENT: Optional[MongoClient] = None


class MongoConnectionError(Exception):
    """Raised when connecting to MongoDB fails or times out."""
    pass


def get_mongo_uri() -> Optional[str]:
    """Retrieve the MongoDB connection URI from environment or Django settings."""
    return (
        os.getenv("MONGODB_URI")
        or getattr(settings, "MONGODB_URI", None)
        or os.getenv("MONGO_URI")
    )


def get_database_name() -> str:
    """Retrieve the target biometric database name."""
    return (
        os.getenv("MONGODB_DATABASE")
        or getattr(settings, "MONGODB_DATABASE", None)
        or "svaraabhava_biometrics"
    )


def get_templates_collection_name() -> str:
    """Retrieve the target face templates collection name."""
    return (
        os.getenv("MONGODB_COLLECTION")
        or getattr(settings, "MONGODB_COLLECTION", None)
        or "face_templates"
    )


def get_attendance_collection_name() -> str:
    """Retrieve the attendance records collection name."""
    return (
        os.getenv("MONGODB_ATTENDANCE_COLLECTION")
        or getattr(settings, "MONGODB_ATTENDANCE_COLLECTION", None)
        or "attendance_records"
    )


def get_mongo_client(
    timeout_ms: int = 5000,
    force_new: bool = False,
) -> MongoClient:
    """
    Get or initialize the shared PyMongo MongoClient instance.

    Args:
        timeout_ms: Server selection timeout in milliseconds (default 5000ms).
        force_new: If True, closes any existing client and creates a new one.

    Returns:
        MongoClient: Connected PyMongo client.

    Raises:
        MongoConnectionError: If MONGODB_URI is not set or server is unreachable.
    """
    global _MONGO_CLIENT

    if force_new and _MONGO_CLIENT is not None:
        try:
            _MONGO_CLIENT.close()
        except Exception:
            pass
        _MONGO_CLIENT = None

    if _MONGO_CLIENT is not None:
        return _MONGO_CLIENT

    uri = get_mongo_uri()
    if not uri:
        raise MongoConnectionError(
            "MONGODB_URI is not configured. Please set MONGODB_URI in your environment or .env file."
        )

    try:
        client_kwargs = {
            "serverSelectionTimeoutMS": timeout_ms,
            "connectTimeoutMS": timeout_ms,
            "socketTimeoutMS": 10000,
            "maxPoolSize": 50,
            "minPoolSize": 5,
            "retryWrites": True,
            "appname": "SvaraabhavaFaceAuth",
        }
        try:
            import certifi
            ca_path = certifi.where()
            if os.path.exists(ca_path):
                client_kwargs["tlsCAFile"] = ca_path
        except ImportError:
            pass

        # Initialize client with connection pooling and timeouts
        client = MongoClient(uri, **client_kwargs)
        # Test connection immediately
        client.admin.command("ping")
        _MONGO_CLIENT = client
        logger.info("Successfully connected to MongoDB Atlas.")
        return _MONGO_CLIENT

    except (ServerSelectionTimeoutError, ConnectionFailure) as e:
        err_msg = str(e)
        logger.error("Failed to connect to MongoDB: %s", err_msg)
        if "client" in locals():
            try:
                client.close()
            except Exception:
                pass
        if "TLSV1_ALERT_INTERNAL_ERROR" in err_msg or "tlsv1 alert internal error" in err_msg:
            raise MongoConnectionError(
                "MongoDB Atlas TLS connection rejected (TLSV1_ALERT_INTERNAL_ERROR). "
                "Your IP address is not whitelisted in MongoDB Atlas Network Access. "
                "Please add your current IP address (or 0.0.0.0/0) to the IP Access List in the MongoDB Atlas dashboard."
            ) from e
        raise MongoConnectionError(
            f"Failed to connect to MongoDB server within {timeout_ms}ms. Verify MONGODB_URI, network, and IP access list."
        ) from e
    except PyMongoError as e:
        logger.error("PyMongo error during client initialization: %s", str(e))
        if "client" in locals():
            try:
                client.close()
            except Exception:
                pass
        raise MongoConnectionError(f"MongoDB connection error: {str(e)}") from e


def get_biometrics_db(client: Optional[MongoClient] = None) -> Database:
    """
    Get the SVARAABHAVA biometrics database instance.

    Args:
        client: Optional pre-existing MongoClient. If None, uses get_mongo_client().

    Returns:
        Database: The 'svaraabhava_biometrics' database.
    """
    if client is None:
        client = get_mongo_client()
    return client[get_database_name()]


def get_face_templates_collection(
    db: Optional[Database] = None,
) -> Collection:
    """
    Get the 'face_templates' collection.

    Args:
        db: Optional pre-existing Database instance.

    Returns:
        Collection: PyMongo collection for face templates.
    """
    if db is None:
        db = get_biometrics_db()
    return db[get_templates_collection_name()]


def get_attendance_collection(
    db: Optional[Database] = None,
) -> Collection:
    """
    Get the 'attendance_records' collection.

    Args:
        db: Optional pre-existing Database instance.

    Returns:
        Collection: PyMongo collection for attendance records.
    """
    if db is None:
        db = get_biometrics_db()
    return db[get_attendance_collection_name()]


def close_mongo_connection() -> None:
    """Gracefully close the global PyMongo client connection."""
    global _MONGO_CLIENT
    if _MONGO_CLIENT is not None:
        try:
            _MONGO_CLIENT.close()
            logger.info("Closed MongoDB client connection.")
        except Exception as e:
            logger.warning("Error closing MongoDB client: %s", str(e))
        finally:
            _MONGO_CLIENT = None


def ping_mongo(timeout_ms: int = 3000) -> bool:
    """
    Ping MongoDB server to test connectivity.

    Returns:
        bool: True if connection is alive and healthy, False otherwise.
    """
    try:
        client = get_mongo_client(timeout_ms=timeout_ms)
        client.admin.command("ping")
        return True
    except Exception:
        return False
