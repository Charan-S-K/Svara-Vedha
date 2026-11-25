"""
Weekly Academic Timetable Service for MongoDB integration.

Database: time_table
Collection: table

Document schema:
{
  "_id": ObjectId("..."),
  "name": "...",
  "mon": [1, 2, 3],
  "tues": [1, 2, 3],
  "wed": [1, 2, 3, 4],
  "thurs": [1, 2, 3, 4, 5],
  "fri": [1, 2],
  "sat": [],
  "user_id": 9
}
"""

import json
import logging
import os
import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Union
from django.conf import settings

logger = logging.getLogger(__name__)

VALID_DAYS = ["mon", "tues", "wed", "thurs", "fri", "sat"]
MIN_PERIOD = 1
MAX_PERIOD = 7


def _get_local_db_path() -> str:
    """Get path to local SQLite cache for timetable resilience."""
    base_dir = getattr(settings, "BASE_DIR", os.getcwd())
    db_name = getattr(settings, "TIMETABLE_CACHE_DB", "local_timetable.sqlite3")
    return os.path.join(base_dir, db_name)


def _init_local_db():
    """Ensure local timetable cache table exists."""
    db_path = _get_local_db_path()
    try:
        with sqlite3.connect(db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS timetable_cache (
                    user_id INTEGER PRIMARY KEY,
                    doc_json TEXT NOT NULL,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()
    except Exception as e:
        logger.warning("Failed to initialize local timetable SQLite cache: %s", e)


def _save_local_cache(user_id: int, doc_dict: Dict[str, Any]):
    """Persist copy to local SQLite cache."""
    _init_local_db()
    db_path = _get_local_db_path()
    try:
        clean_doc = dict(doc_dict)
        clean_doc.pop("_id", None)
        if isinstance(clean_doc.get("updated_at"), datetime):
            clean_doc["updated_at"] = clean_doc["updated_at"].isoformat()

        doc_json = json.dumps(clean_doc)
        with sqlite3.connect(db_path) as conn:
            conn.execute("""
                INSERT INTO timetable_cache (user_id, doc_json, updated_at)
                VALUES (?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(user_id) DO UPDATE SET
                    doc_json = excluded.doc_json,
                    updated_at = CURRENT_TIMESTAMP
            """, (int(user_id), doc_json))
            conn.commit()
    except Exception as e:
        logger.warning("Error saving to local timetable cache: %s", e)


def _get_local_cache(user_id: int) -> Optional[Dict[str, Any]]:
    """Retrieve document from local SQLite cache."""
    _init_local_db()
    db_path = _get_local_db_path()
    try:
        with sqlite3.connect(db_path) as conn:
            cur = conn.execute(
                "SELECT doc_json FROM timetable_cache WHERE user_id = ?",
                (int(user_id),)
            )
            row = cur.fetchone()
            if row and row[0]:
                return json.loads(row[0])
    except Exception as e:
        logger.warning("Error reading from local timetable cache: %s", e)
    return None


def _format_doc(doc: Dict[str, Any]) -> Dict[str, Any]:
    """Clean MongoDB document for JSON response."""
    result = {
        "user_id": int(doc.get("user_id", 0)),
        "name": str(doc.get("name", "")),
        "mon": list(doc.get("mon", [])),
        "tues": list(doc.get("tues", [])),
        "wed": list(doc.get("wed", [])),
        "thurs": list(doc.get("thurs", [])),
        "fri": list(doc.get("fri", [])),
        "sat": list(doc.get("sat", [])),
    }
    if "_id" in doc:
        result["_id"] = str(doc["_id"])
    return result


def validate_timetable_payload(data: Any) -> Tuple[bool, Optional[str], Optional[Dict[str, List[int]]]]:
    """
    Validate timetable payload according to strict rules:
    - Must be a dictionary.
    - Day keys must be a subset of VALID_DAYS.
    - No unknown keys allowed.
    - Each day's value must be a list of integers between 1 and 7.
    - No duplicates.
    - Sorted in ascending order.
    - Empty arrays supported.
    """
    if not isinstance(data, dict):
        return False, "Payload must be a JSON object.", None

    cleaned: Dict[str, List[int]] = {}

    for k in data.keys():
        if k not in VALID_DAYS:
            return False, f"Unknown day key '{k}'. Allowed keys: {', '.join(VALID_DAYS)}.", None

    for day in VALID_DAYS:
        val = data.get(day, [])
        if not isinstance(val, list):
            return False, f"Day '{day}' must be an array of integers.", None

        int_list: List[int] = []
        for item in val:
            if not isinstance(item, int) or isinstance(item, bool):
                return False, f"Invalid period '{item}' in day '{day}'. Periods must be integers.", None
            if item < MIN_PERIOD or item > MAX_PERIOD:
                return False, f"Period '{item}' in day '{day}' is out of range. Must be between {MIN_PERIOD} and {MAX_PERIOD}.", None
            if item in int_list:
                return False, f"Duplicate period '{item}' found in day '{day}'.", None
            int_list.append(item)

        sorted_list = sorted(int_list)
        if int_list != sorted_list:
            return False, f"Periods for day '{day}' must be sorted in ascending order.", None

        cleaned[day] = sorted_list

    return True, None, cleaned


def get_user_timetable(user_id: int, user_name: str = "") -> Dict[str, Any]:
    """
    Fetch the timetable document for the given user_id from MongoDB 'time_table.table'.
    Falls back to local cache or default empty timetable.
    """
    int_user_id = int(user_id)
    mongo_attempted = False
    mongo_error_msg = None

    try:
        from accounts.services.mongo_connection import get_mongo_client
        mongo_attempted = True
        client = get_mongo_client(timeout_ms=2500)
        db = client["time_table"]
        col = db["table"]

        doc = col.find_one({"user_id": int_user_id})
        if doc:
            _save_local_cache(int_user_id, doc)
            formatted = _format_doc(doc)
            return {
                "status": "success",
                "source": "mongodb",
                "data": formatted,
            }
    except Exception as e:
        mongo_error_msg = str(e)
        logger.warning("MongoDB read for user %s failed: %s", int_user_id, e)

    # Check local cache
    cached = _get_local_cache(int_user_id)
    if cached:
        formatted = _format_doc(cached)
        res = {
            "status": "success",
            "source": "local_cache",
            "data": formatted,
        }
        if mongo_attempted and mongo_error_msg:
            res["mongo_status"] = f"Offline/unreachable: {mongo_error_msg}"
        return res

    # Default empty structure
    default_doc = {
        "user_id": int_user_id,
        "name": user_name,
        "mon": [],
        "tues": [],
        "wed": [],
        "thurs": [],
        "fri": [],
        "sat": [],
    }
    return {
        "status": "success",
        "source": "default",
        "data": default_doc,
    }


def save_user_timetable(
    user_id: int,
    user_name: str,
    timetable_data: Dict[str, List[int]],
) -> Dict[str, Any]:
    """
    Update or create the user's timetable document in MongoDB 'time_table.table'.
    Always maintains ascending order, no duplicates, and mirrors to local cache.
    """
    int_user_id = int(user_id)
    now_utc = datetime.now(timezone.utc)

    doc_payload = {
        "user_id": int_user_id,
        "name": user_name,
        "mon": sorted(list(set(timetable_data.get("mon", [])))),
        "tues": sorted(list(set(timetable_data.get("tues", [])))),
        "wed": sorted(list(set(timetable_data.get("wed", [])))),
        "thurs": sorted(list(set(timetable_data.get("thurs", [])))),
        "fri": sorted(list(set(timetable_data.get("fri", [])))),
        "sat": sorted(list(set(timetable_data.get("sat", [])))),
        "updated_at": now_utc,
    }

    # Always persist locally for offline guarantee
    _save_local_cache(int_user_id, doc_payload)

    # Attempt MongoDB persistence
    try:
        from accounts.services.mongo_connection import get_mongo_client
        client = get_mongo_client(timeout_ms=3000)
        db = client["time_table"]
        col = db["table"]

        # Ensure index on user_id
        col.create_index("user_id", unique=True, background=True)

        res = col.update_one(
            {"user_id": int_user_id},
            {"$set": doc_payload},
            upsert=True,
        )

        # Retrieve saved document
        updated_doc = col.find_one({"user_id": int_user_id}) or doc_payload

        return {
            "status": "success",
            "source": "mongodb",
            "matched_count": res.matched_count,
            "modified_count": res.modified_count,
            "upserted_id": str(res.upserted_id) if res.upserted_id else None,
            "data": _format_doc(updated_doc),
            "message": "Timetable updated successfully in MongoDB.",
        }

    except Exception as e:
        logger.warning("MongoDB update for user %s failed: %s", int_user_id, e)
        return {
            "status": "success",
            "source": "local_cache",
            "warning": f"Saved to local vault. MongoDB Atlas connection rejected/pending IP whitelist: {str(e)}",
            "data": _format_doc(doc_payload),
            "message": "Timetable saved successfully.",
        }
