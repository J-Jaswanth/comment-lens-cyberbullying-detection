import os
import uuid
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from bson import ObjectId
from pymongo.errors import DuplicateKeyError

from app.models.google_oauth_store import MongoGoogleOAuthStore
from app.models.user import User
from app.models.user_store import MongoUserStore, UserAlreadyExists


@patch("app.models.user_store.MongoClient")
def test_create_user_normalizes_unique_account_fields(mongo_client):
    client = MagicMock()
    collection = client.__getitem__.return_value.__getitem__.return_value
    inserted_id = ObjectId()
    collection.insert_one.return_value = SimpleNamespace(inserted_id=inserted_id)
    mongo_client.return_value = client
    store = MongoUserStore("mongodb://example.test", "test_db")
    user = User("", "Student_1", "STUDENT@example.com", "hashed-password")

    created = store.create_user(user)

    assert created.id == str(inserted_id)
    assert created.username == "student_1"
    assert created.email == "student@example.com"
    collection.create_index.assert_any_call("username", unique=True)
    collection.create_index.assert_any_call("email", unique=True)
    collection.insert_one.assert_called_once_with(
        {
            "username": "student_1",
            "email": "student@example.com",
            "password_hash": "hashed-password",
        }
    )


@patch("app.models.user_store.MongoClient")
def test_duplicate_account_is_reported_as_domain_error(mongo_client):
    client = MagicMock()
    collection = client.__getitem__.return_value.__getitem__.return_value
    collection.insert_one.side_effect = DuplicateKeyError("duplicate")
    mongo_client.return_value = client
    store = MongoUserStore("mongodb://example.test", "test_db")
    user = User("", "student_1", "student@example.com", "hashed-password")

    with pytest.raises(UserAlreadyExists):
        store.create_user(user)


@patch("app.models.user_store.MongoClient")
def test_login_lookup_normalizes_identifier(mongo_client):
    client = MagicMock()
    collection = client.__getitem__.return_value.__getitem__.return_value
    document = {
        "_id": ObjectId(),
        "username": "student_1",
        "email": "student@example.com",
        "password_hash": "hashed-password",
    }
    collection.find_one.return_value = document
    mongo_client.return_value = client
    store = MongoUserStore("mongodb://example.test", "test_db")

    found = store.get_by_login(" STUDENT@example.com ")

    assert found.id == str(document["_id"])
    collection.find_one.assert_called_once_with(
        {"$or": [{"username": "student@example.com"}, {"email": "student@example.com"}]}
    )


@patch("app.models.user_store.MongoClient")
def test_invalid_user_id_does_not_query_database(mongo_client):
    client = MagicMock()
    collection = client.__getitem__.return_value.__getitem__.return_value
    mongo_client.return_value = client
    store = MongoUserStore("mongodb://example.test", "test_db")

    assert store.get_by_id("not-an-object-id") is None
    collection.find_one.assert_not_called()


@pytest.fixture
def live_user_store():
    uri = os.getenv("MONGODB_TEST_URI")
    if os.getenv("RUN_MONGODB_INTEGRATION") != "1" or not uri:
        pytest.skip("Set RUN_MONGODB_INTEGRATION=1 and MONGODB_TEST_URI to opt in.")

    database_name = f"cyberbullying_test_{uuid.uuid4().hex}"
    store = MongoUserStore(uri, database_name)
    try:
        yield store
    finally:
        store.client.drop_database(database_name)
        store.client.close()


def test_live_mongodb_account_persistence(live_user_store):
    user = User("", "live_test_user", "live-test@example.invalid", "test-hash")

    created = live_user_store.create_user(user)
    found_by_login = live_user_store.get_by_login("LIVE_TEST_USER")
    found_by_id = live_user_store.get_by_id(created.id)

    assert found_by_login.id == created.id
    assert found_by_id.email == "live-test@example.invalid"

    with pytest.raises(UserAlreadyExists):
        live_user_store.create_user(user)


def test_live_mongodb_keeps_oauth_credentials_separate_by_user(live_user_store):
    database_name = live_user_store.collection.database.name
    oauth_store = MongoGoogleOAuthStore(
        os.environ["MONGODB_TEST_URI"],
        database_name,
    )
    try:
        oauth_store.save_encrypted_credentials("user-one", "ciphertext-one")
        oauth_store.save_encrypted_credentials("user-two", "ciphertext-two")

        assert oauth_store.has_credentials("user-one")
        assert oauth_store.get_encrypted_credentials("user-one") == "ciphertext-one"
        assert oauth_store.get_encrypted_credentials("user-two") == "ciphertext-two"

        oauth_store.delete_credentials("user-one")
        assert not oauth_store.has_credentials("user-one")
        assert oauth_store.has_credentials("user-two")
    finally:
        oauth_store.client.close()