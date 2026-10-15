"""MongoDB persistence for encrypted Google OAuth credentials."""

from datetime import datetime, timezone

from pymongo import MongoClient


class MongoGoogleOAuthStore:
    def __init__(self, uri, database_name):
        self.client = MongoClient(uri, serverSelectionTimeoutMS=3000)
        self.collection = self.client[database_name]["google_oauth_credentials"]

    def save_encrypted_credentials(self, user_id, encrypted_credentials):
        self.collection.create_index("user_id", unique=True)
        self.collection.replace_one(
            {"user_id": str(user_id)},
            {
                "user_id": str(user_id),
                "encrypted_credentials": encrypted_credentials,
                "updated_at": datetime.now(timezone.utc),
            },
            upsert=True,
        )

    def has_credentials(self, user_id):
        return self.collection.find_one(
            {"user_id": str(user_id)}, {"_id": 1}
        ) is not None

    def get_encrypted_credentials(self, user_id):
        document = self.collection.find_one(
            {"user_id": str(user_id)}, {"encrypted_credentials": 1}
        )
        return document["encrypted_credentials"] if document else None

    def delete_credentials(self, user_id):
        self.collection.delete_one({"user_id": str(user_id)})