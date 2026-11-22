"""
User Synchronization Service: PostgreSQL -> MongoDB.

Synchronizes user identity records (userid, name) from the primary PostgreSQL
relational database into MongoDB ('user' database, 'users' collection).

Guarantees:
- Add-on feature only: PostgreSQL remains the authoritative source of truth.
- Non-blocking: failures in MongoDB synchronization are safely caught, logged,
  and never disrupt or rollback PostgreSQL registration or authentication.
- Idempotent: uses bulk upserts with $setOnInsert and unique index on 'userid'
  to completely prevent duplicate documents or unnecessary overwrites.
"""

import logging
import os
from typing import Any, Dict, List, Optional
from django.conf import settings
from pymongo import MongoClient, UpdateOne
from pymongo.collection import Collection
from pymongo.database import Database
from pymongo.errors import PyMongoError

from .mongo_connection import get_mongo_client, MongoConnectionError

logger = logging.getLogger(__name__)


def get_user_sync_db_name() -> str:
    """
    Retrieve the target MongoDB database name for user synchronization.
    Defaults strictly to 'user' as specified in requirements.
    """
    env_dedicated = os.getenv("MONGODB_USER_DATABASE") or getattr(settings, "MONGODB_USER_DATABASE", None)
    if env_dedicated:
        return env_dedicated

    env_main = os.getenv("MONGODB_DATABASE") or getattr(settings, "MONGODB_DATABASE", None)
    if env_main and env_main.lower() not in ("svaraabhava_biometrics", "biometrics"):
        return env_main

    return "user"


def get_user_sync_collection_name() -> str:
    """
    Retrieve the target MongoDB collection name for user synchronization.
    Defaults strictly to 'users' as specified in requirements.
    """
    env_dedicated = (
        os.getenv("MONGODB_USER_COLLECTION")
        or os.getenv("MONGODB_USERS_COLLECTION")
        or getattr(settings, "MONGODB_USER_COLLECTION", None)
    )
    if env_dedicated:
        return env_dedicated

    env_main = os.getenv("MONGODB_COLLECTION") or getattr(settings, "MONGODB_COLLECTION", None)
    if env_main and env_main.lower() not in ("face_templates", "attendance_records"):
        return env_main

    return "users"


def get_user_sync_database(client: Optional[MongoClient] = None) -> Database:
    """
    Get the target MongoDB Database instance (default 'user').
    """
    if client is None:
        client = get_mongo_client()
    return client[get_user_sync_db_name()]


def get_user_sync_collection(client: Optional[MongoClient] = None) -> Collection:
    """
    Get the target MongoDB Collection instance (default 'user.users').
    """
    db = get_user_sync_database(client=client)
    return db[get_user_sync_collection_name()]


def ensure_user_sync_indexes(collection: Optional[Collection] = None) -> None:
    """
    Ensures a unique index on 'userid' exists in the 'users' collection to enforce
    data integrity and avoid duplicate user documents.
    """
    try:
        if collection is None:
            collection = get_user_sync_collection()
        collection.create_index("userid", unique=True, name="uniq_userid_index", background=True)
    except Exception as e:
        logger.warning("Could not ensure unique index on 'userid' in MongoDB: %s", e)


def sync_user_to_mongo(user_id: int, name: str, client: Optional[MongoClient] = None) -> bool:
    """
    Synchronizes a newly registered PostgreSQL user into MongoDB under the
    'user.users' collection.

    Document format:
        {
            "userid": <user_id>,
            "name": "<name>"
        }

    Uses $setOnInsert with upsert=True:
    - If document with 'userid' does not exist, it inserts it.
    - If document already exists, it leaves it untouched without duplicates.
    - Preserves PostgreSQL integer userid data type.
    - Never raises unhandled exceptions to caller (non-blocking).

    Args:
        user_id: The primary key (id) from the PostgreSQL auth_user table.
        name: The corresponding user's name from PostgreSQL.
        client: Optional pre-existing MongoClient.

    Returns:
        bool: True if synchronization succeeded or document already exists;
              False if MongoDB was unavailable or encountered an error.
    """
    if user_id is None:
        logger.warning("sync_user_to_mongo called with None user_id. Skipping.")
        return False

    clean_name = str(name).strip() if name is not None else ""

    try:
        coll = get_user_sync_collection(client=client)
        ensure_user_sync_indexes(coll)

        # Build clean document with ONLY userid and name
        filter_doc = {"userid": int(user_id)}
        update_doc = {
            "$setOnInsert": {
                "userid": int(user_id),
                "name": clean_name,
            }
        }

        result = coll.update_one(filter_doc, update_doc, upsert=True)
        if result.upserted_id:
            logger.info("MongoDB sync: inserted new user document: userid=%s, name='%s'", user_id, clean_name)
        else:
            logger.info("MongoDB sync: user document already present: userid=%s, name='%s'", user_id, clean_name)
        return True
    except MongoConnectionError as conn_err:
        logger.warning(
            "MongoDB unavailable during user synchronization (userid=%s, name='%s'): %s. "
            "PostgreSQL registration remains successful. User can be synchronized later.",
            user_id, clean_name, conn_err
        )
        return False
    except PyMongoError as mongo_err:
        logger.error(
            "PyMongo error during user synchronization (userid=%s, name='%s'): %s",
            user_id, clean_name, mongo_err, exc_info=True
        )
        return False
    except Exception as ex:
        logger.error(
            "Unexpected error during user synchronization (userid=%s, name='%s'): %s",
            user_id, clean_name, ex, exc_info=True
        )
        return False


def sync_all_existing_users(batch_size: int = 500, client: Optional[MongoClient] = None) -> Dict[str, Any]:
    """
    Initial one-time or recurring synchronization process:
    1. Connects to PostgreSQL via Django ORM.
    2. Reads the existing users table.
    3. Retrieves 'userid' and corresponding 'name' for every user.
    4. Connects to MongoDB (database 'user', collection 'users').
    5. Inserts documents using bulk upsert with $setOnInsert to prevent duplicates.

    Returns:
        dict: Summary statistics {
            "total_pg_users": int,
            "upserted": int,
            "already_existing": int,
            "success": bool,
            "error": Optional[str]
        }
    """
    from django.contrib.auth.models import User

    try:
        coll = get_user_sync_collection(client=client)
        ensure_user_sync_indexes(coll)

        users_qs = User.objects.all().order_by("id")
        total_pg_users = users_qs.count()

        if total_pg_users == 0:
            logger.info("No PostgreSQL users found to synchronize.")
            return {
                "total_pg_users": 0,
                "upserted": 0,
                "already_existing": 0,
                "success": True,
                "error": None,
            }

        operations: List[UpdateOne] = []
        for u in users_qs:
            # Determine name: first_name if available, else full_name, else username
            user_name = (u.first_name or "").strip()
            if not user_name:
                user_name = u.get_full_name().strip() or u.username

            operations.append(
                UpdateOne(
                    {"userid": int(u.id)},
                    {"$setOnInsert": {"userid": int(u.id), "name": user_name}},
                    upsert=True,
                )
            )

        total_upserted = 0
        total_matched = 0

        for i in range(0, len(operations), batch_size):
            batch = operations[i : i + batch_size]
            res = coll.bulk_write(batch, ordered=False)
            total_upserted += res.upserted_count
            total_matched += res.matched_count

        logger.info(
            "Bulk user sync to MongoDB completed: %d total in PG, %d inserted, %d already existed.",
            total_pg_users, total_upserted, total_matched
        )

        return {
            "total_pg_users": total_pg_users,
            "upserted": total_upserted,
            "already_existing": total_matched,
            "success": True,
            "error": None,
        }
    except Exception as e:
        logger.error("Error during bulk user synchronization to MongoDB: %s", str(e), exc_info=True)
        return {
            "total_pg_users": 0,
            "upserted": 0,
            "already_existing": 0,
            "success": False,
            "error": str(e),
        }


def get_synced_mongo_users(client: Optional[MongoClient] = None) -> List[Dict[str, Any]]:
    """
    Utility to fetch all synchronized user documents from MongoDB 'user.users',
    excluding MongoDB '_id' and returning only 'userid' and 'name'.
    """
    try:
        coll = get_user_sync_collection(client=client)
        cursor = coll.find({}, {"_id": 0, "userid": 1, "name": 1}).sort("userid", 1)
        return list(cursor)
    except Exception as e:
        logger.error("Failed to fetch synced users from MongoDB: %s", e)
        return []
