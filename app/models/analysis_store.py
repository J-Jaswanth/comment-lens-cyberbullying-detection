"""MongoDB persistence for comment analysis results."""

from datetime import datetime, timezone
from pymongo import MongoClient


class MongoAnalysisStore:
    def __init__(self, uri, database_name):
        self.client = MongoClient(uri, serverSelectionTimeoutMS=3000)
        self.collection = self.client[database_name]["saved_analyses"]

    def save_analysis(self, user_id, video_id, summary_data):
        self.collection.create_index([("user_id", 1), ("video_id", 1)], unique=True)
        self.collection.replace_one(
            {"user_id": str(user_id), "video_id": str(video_id)},
            {
                "user_id": str(user_id),
                "video_id": str(video_id),
                "summary": summary_data,
                "updated_at": datetime.now(timezone.utc),
            },
            upsert=True,
        )

    def get_analysis(self, user_id, video_id):
        return self.collection.find_one(
            {"user_id": str(user_id), "video_id": str(video_id)},
            {"_id": 0, "summary": 1, "updated_at": 1},
        )

    def get_analyses_for_videos(self, user_id, video_ids):
        if not video_ids:
            return {}
        cursor = self.collection.find(
            {"user_id": str(user_id), "video_id": {"$in": [str(v) for v in video_ids]}},
            {"_id": 0, "video_id": 1, "summary": 1, "updated_at": 1},
        )
        return {item["video_id"]: item["summary"] for item in cursor}
