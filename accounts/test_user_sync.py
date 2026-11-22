"""
Test Suite for MongoDB User Synchronization (PostgreSQL -> MongoDB user.users).

Covers all 5 verification scenarios specified in requirements:
Test 1 - Existing Users Synchronization
Test 2 - New User Registration Integration
Test 3 - Duplicate Prevention & Idempotency
Test 4 - MongoDB Unavailable (Non-blocking & Graceful Fallback)
Test 5 - Preservation of Existing Functionality
"""

from unittest.mock import MagicMock, patch
from django.contrib.auth.models import User
from django.test import Client, TestCase
from pymongo.errors import ServerSelectionTimeoutError

from accounts.models import Profile
from accounts.services.user_sync import (
    sync_user_to_mongo,
    sync_all_existing_users,
    get_synced_mongo_users,
    get_user_sync_db_name,
    get_user_sync_collection_name,
)
from accounts.services.mongo_connection import MongoConnectionError


class MockMongoCollection:
    """Mock MongoDB Collection for isolated in-memory unit testing."""

    def __init__(self):
        self.docs = []  # list of dicts

    def create_index(self, key, **kwargs):
        pass

    def update_one(self, filter_doc, update_doc, upsert=False):
        userid = filter_doc.get("userid")
        existing = next((d for d in self.docs if d.get("userid") == userid), None)
        res = MagicMock()
        if existing:
            res.matched_count = 1
            res.upserted_id = None
        elif upsert:
            new_doc = dict(update_doc.get("$setOnInsert", {}))
            self.docs.append(new_doc)
            res.matched_count = 0
            res.upserted_id = f"mock_id_{userid}"
        return res

    def bulk_write(self, operations, ordered=False):
        upserted_count = 0
        matched_count = 0
        for op in operations:
            filter_doc = op._filter
            update_doc = op._doc
            userid = filter_doc.get("userid")
            existing = next((d for d in self.docs if d.get("userid") == userid), None)
            if existing:
                matched_count += 1
            else:
                new_doc = dict(update_doc.get("$setOnInsert", {}))
                self.docs.append(new_doc)
                upserted_count += 1
        res = MagicMock()
        res.upserted_count = upserted_count
        res.matched_count = matched_count
        return res

    def find(self, filter_doc=None, projection=None):
        out = []
        for d in self.docs:
            doc_copy = {k: v for k, v in d.items() if k != "_id"}
            out.append(doc_copy)
        cursor = MagicMock()
        cursor.sort = MagicMock(return_value=out)
        return cursor


class MongoUserSyncTests(TestCase):
    def setUp(self):
        self.client = Client()
        self.mock_coll = MockMongoCollection()

    def test_01_existing_users_synchronization(self):
        """
        Test 1 - Existing Users:
        Reads all existing PostgreSQL users, runs sync_all_existing_users,
        and verifies every user has a corresponding MongoDB document with
        exact 'userid' and 'name'.
        """
        u1 = User.objects.create_user(username="u1@test.com", email="u1@test.com", password="pass", first_name="Charan")
        u2 = User.objects.create_user(username="u2@test.com", email="u2@test.com", password="pass", first_name="Rahul")

        with patch("accounts.services.user_sync.get_user_sync_collection", return_value=self.mock_coll):
            result = sync_all_existing_users()
            self.assertTrue(result["success"])
            self.assertEqual(result["total_pg_users"], 2)
            self.assertEqual(result["upserted"], 2)

            synced = get_synced_mongo_users()
            self.assertEqual(len(synced), 2)
            self.assertEqual(synced[0], {"userid": u1.id, "name": "Charan"})
            self.assertEqual(synced[1], {"userid": u2.id, "name": "Rahul"})

    def test_02_new_registration_workflow(self):
        """
        Test 2 - New Registration:
        Completes the OTP verification registration step. Verifies the user record
        is created in PostgreSQL and the same userid and name are inserted into MongoDB.
        """
        session = self.client.session
        session["register_name"] = "Alice Wonder"
        session["register_phone"] = "9876543210"
        session["register_email"] = "alice@example.com"
        session["register_password"] = "SecurePass123!"
        session["register_otp"] = "123456"
        session.save()

        with patch("accounts.services.user_sync.get_user_sync_collection", return_value=self.mock_coll):
            resp = self.client.post("/verify-otp/", {"otp": "123456"}, follow=False)
            self.assertEqual(resp.status_code, 302)
            self.assertEqual(resp["Location"], "/face-enroll/")

            # Verify PostgreSQL record
            pg_user = User.objects.get(email="alice@example.com")
            self.assertEqual(pg_user.first_name, "Alice Wonder")

            # Verify MongoDB document
            self.assertEqual(len(self.mock_coll.docs), 1)
            mongo_doc = self.mock_coll.docs[0]
            self.assertEqual(mongo_doc["userid"], pg_user.id)
            self.assertEqual(mongo_doc["name"], "Alice Wonder")

    def test_03_duplicate_prevention_and_idempotency(self):
        """
        Test 3 - Duplicate Prevention:
        Re-running synchronization on existing users must not create duplicates
        or overwrite existing records unnecessarily.
        """
        u = User.objects.create_user(username="dup@test.com", email="dup@test.com", password="pass", first_name="Duplicate Tester")

        with patch("accounts.services.user_sync.get_user_sync_collection", return_value=self.mock_coll):
            # First run
            res1 = sync_all_existing_users()
            self.assertEqual(res1["upserted"], 1)
            self.assertEqual(len(self.mock_coll.docs), 1)

            # Second run
            res2 = sync_all_existing_users()
            self.assertEqual(res2["upserted"], 0)
            self.assertEqual(res2["already_existing"], 1)
            self.assertEqual(len(self.mock_coll.docs), 1)

    def test_04_mongodb_unavailable_non_blocking_registration(self):
        """
        Test 4 - MongoDB Unavailable:
        When MongoDB connection fails, PostgreSQL registration MUST succeed,
        the redirect to /face-enroll/ must proceed normally, and no error
        must be displayed to the user.
        """
        session = self.client.session
        session["register_name"] = "Offline Bob"
        session["register_phone"] = "9123456789"
        session["register_email"] = "bob@offline.com"
        session["register_password"] = "OfflinePass123!"
        session["register_otp"] = "999888"
        session.save()

        # Simulate MongoDB connection failure
        with patch("accounts.services.user_sync.get_user_sync_collection", side_effect=MongoConnectionError("Atlas timeout")):
            resp = self.client.post("/verify-otp/", {"otp": "999888"}, follow=False)
            # Registration succeeds and redirects normally
            self.assertEqual(resp.status_code, 302)
            self.assertEqual(resp["Location"], "/face-enroll/")

            # User exists in PostgreSQL
            pg_user = User.objects.filter(email="bob@offline.com").first()
            self.assertIsNotNone(pg_user)
            self.assertEqual(pg_user.first_name, "Offline Bob")

        # Verify that running sync later successfully syncs the missed user
        with patch("accounts.services.user_sync.get_user_sync_collection", return_value=self.mock_coll):
            res = sync_all_existing_users()
            self.assertTrue(res["success"])
            self.assertIn({"userid": pg_user.id, "name": "Offline Bob"}, self.mock_coll.docs)

    def test_05_existing_functionality_preserved(self):
        """
        Test 5 - Existing Functionality:
        Standard Django login, authentication, and profile page continue to work.
        """
        user = User.objects.create_user(username="loginuser@test.com", email="loginuser@test.com", password="Password123!", first_name="Login User")
        Profile.objects.create(user=user, phone="9999999999")

        # Test login view
        resp = self.client.post("/", {"email": "loginuser@test.com", "password": "Password123!"}, follow=False)
        self.assertEqual(resp.status_code, 302)
        self.assertEqual(resp["Location"], "/dashboard/")

        # Test profile access
        self.client.force_login(user)
        resp_profile = self.client.get("/dashboard/profile/")
        self.assertEqual(resp_profile.status_code, 200)

    def test_06_database_and_collection_names(self):
        """Verify default database is 'user' and collection is 'users'."""
        self.assertEqual(get_user_sync_db_name(), "user")
        self.assertEqual(get_user_sync_collection_name(), "users")
