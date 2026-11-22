"""
Django management command to synchronize existing PostgreSQL users into MongoDB.

Usage:
    python manage.py sync_users_to_mongo [--batch-size 500]

Behavior:
    1. Connects to PostgreSQL via Django ORM.
    2. Reads the existing auth_user table.
    3. Retrieves 'userid' (id) and 'name' (first_name or full_name/username) for every user.
    4. Connects to MongoDB (database 'user', collection 'users').
    5. Inserts documents using bulk upsert with $setOnInsert to prevent duplicates.
    6. Ensures unique index on 'userid'.
"""

from django.core.management.base import BaseCommand
from accounts.services.user_sync import (
    sync_all_existing_users,
    get_synced_mongo_users,
    get_user_sync_db_name,
    get_user_sync_collection_name,
)


class Command(BaseCommand):
    help = "Synchronizes all existing PostgreSQL users into MongoDB ('user.users' collection)."

    def add_arguments(self, parser):
        parser.add_argument(
            "--batch-size",
            type=int,
            default=500,
            help="Number of user documents to upsert per MongoDB bulk_write batch (default: 500).",
        )

    def handle(self, *args, **options):
        batch_size = options.get("batch_size", 500)
        db_name = get_user_sync_db_name()
        coll_name = get_user_sync_collection_name()

        self.stdout.write(
            self.style.NOTICE(
                f"Starting PostgreSQL -> MongoDB user synchronization into '{db_name}.{coll_name}'..."
            )
        )

        result = sync_all_existing_users(batch_size=batch_size)

        if result.get("success"):
            total = result.get("total_pg_users", 0)
            upserted = result.get("upserted", 0)
            existing = result.get("already_existing", 0)

            self.stdout.write(
                self.style.SUCCESS(
                    f"Synchronization completed successfully!\n"
                    f"  - Total PostgreSQL users: {total}\n"
                    f"  - Newly inserted to MongoDB: {upserted}\n"
                    f"  - Already existing in MongoDB: {existing}"
                )
            )

            # Display synced documents for verification
            synced_docs = get_synced_mongo_users()
            self.stdout.write(self.style.SUCCESS(f"Total documents currently in '{db_name}.{coll_name}': {len(synced_docs)}"))
            for doc in synced_docs[:10]:
                self.stdout.write(f"  {doc}")
            if len(synced_docs) > 10:
                self.stdout.write(f"  ... and {len(synced_docs) - 10} more documents.")
        else:
            err = result.get("error", "Unknown error")
            self.stdout.write(self.style.ERROR(f"Synchronization failed: {err}"))
