"""MongoDB persistence for user accounts."""

from bson import ObjectId
from bson.errors import InvalidId
from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError

from app.models.user import User


class UserAlreadyExists(Exception):
    """Raised when a username or email is already registered."""


class MongoUserStore:
    def __init__(self, uri, database_name):
        self.client = MongoClient(uri, serverSelectionTimeoutMS=3000)
        self.collection = self.client[database_name]["users"]

    def create_user(self, user):
        self.collection.create_index("username", unique=True)
        self.collection.create_index("email", unique=True)
        document = {
            "username": user.username.casefold(),
            "email": user.email.casefold(),
            "password_hash": user.password_hash,
        }
        try:
            result = self.collection.insert_one(document)
        except DuplicateKeyError as error:
            raise UserAlreadyExists from error
        return User.from_document({**document, "_id": result.inserted_id})

    def get_by_login(self, identifier):
        normalized = identifier.strip().casefold()
        document = self.collection.find_one(
            {"$or": [{"username": normalized}, {"email": normalized}]}
        )
        return User.from_document(document) if document else None

    def get_by_id(self, user_id):
        try:
            object_id = ObjectId(user_id)
        except (InvalidId, TypeError):
            return None
        document = self.collection.find_one({"_id": object_id})
        return User.from_document(document) if document else None