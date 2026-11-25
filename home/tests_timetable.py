import json
import os
from django.conf import settings
from django.test import TestCase, Client
from django.contrib.auth.models import User
from home.services.timetable_service import (
    validate_timetable_payload,
    get_user_timetable,
    save_user_timetable,
)

class TimetableBackendTests(TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls._orig_db = getattr(settings, "TIMETABLE_CACHE_DB", "local_timetable.sqlite3")
        settings.TIMETABLE_CACHE_DB = "test_timetable_cache.sqlite3"

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()
        settings.TIMETABLE_CACHE_DB = cls._orig_db
        test_db_path = os.path.join(getattr(settings, "BASE_DIR", os.getcwd()), "test_timetable_cache.sqlite3")
        if os.path.exists(test_db_path):
            try:
                os.remove(test_db_path)
            except Exception:
                pass

    def setUp(self):
        self.user1 = User.objects.create_user(
            username="charan_test",
            email="charan@example.com",
            first_name="Charan",
            last_name="Tej S K"
        )
        self.user2 = User.objects.create_user(
            username="dhyan_test",
            email="dhyan@example.com",
            first_name="Dhyan",
            last_name="K"
        )
        self.client1 = Client()
        self.client1.force_login(self.user1)

        self.client2 = Client()
        self.client2.force_login(self.user2)

    def test_validation_success(self):
        valid_payload = {
            "mon": [1, 2, 3],
            "tues": [1, 2, 3],
            "wed": [2, 4],
            "thurs": [1, 2, 3, 4, 5],
            "fri": [1, 2],
            "sat": []
        }
        is_valid, err, cleaned = validate_timetable_payload(valid_payload)
        self.assertTrue(is_valid)
        self.assertIsNone(err)
        self.assertEqual(cleaned["wed"], [2, 4])
        self.assertEqual(cleaned["sat"], [])

    def test_validation_rejects_unknown_days(self):
        payload = {"monday": [1, 2], "mon": [1, 2]}
        is_valid, err, _ = validate_timetable_payload(payload)
        self.assertFalse(is_valid)
        self.assertIn("Unknown day key", err)

    def test_validation_rejects_out_of_range_period(self):
        payload = {"mon": [0, 1, 2]}
        is_valid, err, _ = validate_timetable_payload(payload)
        self.assertFalse(is_valid)
        self.assertIn("out of range", err)

        payload2 = {"mon": [1, 8]}
        is_valid2, err2, _ = validate_timetable_payload(payload2)
        self.assertFalse(is_valid2)
        self.assertIn("out of range", err2)

    def test_validation_rejects_duplicates(self):
        payload = {"mon": [1, 2, 2]}
        is_valid, err, _ = validate_timetable_payload(payload)
        self.assertFalse(is_valid)
        self.assertIn("Duplicate period", err)

    def test_validation_rejects_unsorted_arrays(self):
        payload = {"mon": [3, 1, 2]}
        is_valid, err, _ = validate_timetable_payload(payload)
        self.assertFalse(is_valid)
        self.assertIn("must be sorted in ascending order", err)

    def test_save_and_get_user_timetable(self):
        timetable_data = {
            "mon": [1, 2, 3],
            "tues": [1, 2, 3],
            "wed": [1, 2, 3, 4],
            "thurs": [1, 2, 3, 4, 5],
            "fri": [1, 2],
            "sat": []
        }
        # Save for user 1
        save_res = save_user_timetable(self.user1.id, "Charan Tej S K", timetable_data)
        self.assertEqual(save_res["status"], "success")
        self.assertEqual(save_res["data"]["user_id"], self.user1.id)
        self.assertEqual(save_res["data"]["mon"], [1, 2, 3])
        self.assertEqual(save_res["data"]["thurs"], [1, 2, 3, 4, 5])

        # Fetch for user 1
        fetch_res = get_user_timetable(self.user1.id, "Charan Tej S K")
        self.assertEqual(fetch_res["status"], "success")
        self.assertEqual(fetch_res["data"]["user_id"], self.user1.id)
        self.assertEqual(fetch_res["data"]["name"], "Charan Tej S K")
        self.assertEqual(fetch_res["data"]["wed"], [1, 2, 3, 4])
        self.assertEqual(fetch_res["data"]["sat"], [])

    def test_update_existing_document_without_duplicate(self):
        timetable_v1 = {"mon": [1, 2], "tues": [], "wed": [], "thurs": [], "fri": [], "sat": []}
        save_user_timetable(self.user1.id, "Charan Tej S K", timetable_v1)

        # Update with new slots
        timetable_v2 = {"mon": [1, 2, 3, 5], "tues": [4], "wed": [], "thurs": [], "fri": [], "sat": []}
        save_res = save_user_timetable(self.user1.id, "Charan Tej S K", timetable_v2)
        self.assertEqual(save_res["data"]["mon"], [1, 2, 3, 5])
        self.assertEqual(save_res["data"]["tues"], [4])

        # Verify updated fetch
        fetch_res = get_user_timetable(self.user1.id, "Charan Tej S K")
        self.assertEqual(fetch_res["data"]["mon"], [1, 2, 3, 5])

    def test_users_have_separate_timetables(self):
        t1 = {"mon": [1, 2], "tues": [], "wed": [], "thurs": [], "fri": [], "sat": []}
        t2 = {"mon": [5, 6, 7], "tues": [1], "wed": [], "thurs": [], "fri": [], "sat": []}

        save_user_timetable(self.user1.id, "Charan", t1)
        save_user_timetable(self.user2.id, "Dhyan", t2)

        f1 = get_user_timetable(self.user1.id)
        f2 = get_user_timetable(self.user2.id)

        self.assertEqual(f1["data"]["mon"], [1, 2])
        self.assertEqual(f2["data"]["mon"], [5, 6, 7])
        self.assertNotEqual(f1["data"]["user_id"], f2["data"]["user_id"])

    def test_api_get_and_post(self):
        # GET initial
        resp = self.client1.get("/dashboard/api/timetable/")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "success")

        # POST payload
        payload = {
            "mon": [1, 2],
            "tues": [3],
            "wed": [4, 5],
            "thurs": [6],
            "fri": [7],
            "sat": []
        }
        post_resp = self.client1.post(
            "/dashboard/api/timetable/",
            data=json.dumps(payload),
            content_type="application/json"
        )
        self.assertEqual(post_resp.status_code, 200)
        post_data = post_resp.json()
        self.assertEqual(post_data["status"], "success")
        self.assertEqual(post_data["data"]["wed"], [4, 5])

        # Re-fetch via API to verify persistence
        resp_after = self.client1.get("/dashboard/api/timetable/")
        self.assertEqual(resp_after.status_code, 200)
        data_after = resp_after.json()
        self.assertEqual(data_after["data"]["mon"], [1, 2])
        self.assertEqual(data_after["data"]["fri"], [7])
