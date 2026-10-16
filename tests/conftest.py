import pytest
from cryptography.fernet import Fernet

from app import create_app
from app.models.user import User
from app.models.user_store import UserAlreadyExists


class FakeUserStore:
    def __init__(self):
        self.users = {}

    def create_user(self, user):
        if any(
            existing.username == user.username or existing.email == user.email
            for existing in self.users.values()
        ):
            raise UserAlreadyExists
        stored_user = User(
            user_id=str(len(self.users) + 1),
            username=user.username,
            email=user.email,
            password_hash=user.password_hash,
        )
        self.users[stored_user.id] = stored_user
        return stored_user

    def get_by_login(self, identifier):
        normalized = identifier.strip().casefold()
        return next(
            (
                user
                for user in self.users.values()
                if user.username == normalized or user.email == normalized
            ),
            None,
        )

    def get_by_id(self, user_id):
        return self.users.get(user_id)


class FakeOAuthStore:
    def __init__(self):
        self.credentials = {}

    def save_encrypted_credentials(self, user_id, encrypted_credentials):
        self.credentials[str(user_id)] = encrypted_credentials

    def has_credentials(self, user_id):
        return str(user_id) in self.credentials

    def get_encrypted_credentials(self, user_id):
        return self.credentials.get(str(user_id))

    def delete_credentials(self, user_id):
        self.credentials.pop(str(user_id), None)


class FakeAnalysisStore:
    def __init__(self):
        self.analyses = {}

    def save_analysis(self, user_id, video_id, summary_data):
        self.analyses[(str(user_id), str(video_id))] = summary_data

    def get_analysis(self, user_id, video_id):
        data = self.analyses.get((str(user_id), str(video_id)))
        return {"summary": data} if data else None

    def get_analyses_for_videos(self, user_id, video_ids):
        result = {}
        for vid in video_ids:
            key = (str(user_id), str(vid))
            if key in self.analyses:
                result[str(vid)] = self.analyses[key]
        return result


@pytest.fixture
def account_store():
    return FakeUserStore()


@pytest.fixture
def oauth_store():
    return FakeOAuthStore()


@pytest.fixture
def analysis_store():
    return FakeAnalysisStore()


@pytest.fixture
def test_encryption_key():
    return Fernet.generate_key().decode("ascii")


@pytest.fixture
def app(account_store, oauth_store, analysis_store, test_encryption_key):
    return create_app(
        {
            "TESTING": True,
            "SECRET_KEY": "test-only-secret-key",
            "WTF_CSRF_ENABLED": False,
            "RATELIMIT_ENABLED": False,
            "USER_STORE": account_store,
            "OAUTH_STORE": oauth_store,
            "ANALYSIS_STORE": analysis_store,
            "GOOGLE_CLIENT_ID": None,
            "GOOGLE_CLIENT_SECRET": None,
            "GOOGLE_REDIRECT_URI": "http://127.0.0.1:5000/auth/google/callback",
            "GOOGLE_TOKEN_ENCRYPTION_KEY": test_encryption_key,
        }
    )


@pytest.fixture
def client(app):
    return app.test_client()


@pytest.fixture
def csrf_client(account_store, oauth_store, test_encryption_key):
    app = create_app(
        {
            "SECRET_KEY": "test-only-secret-key",
            "RATELIMIT_ENABLED": False,
            "USER_STORE": account_store,
            "OAUTH_STORE": oauth_store,
            "GOOGLE_CLIENT_ID": None,
            "GOOGLE_CLIENT_SECRET": None,
            "GOOGLE_TOKEN_ENCRYPTION_KEY": test_encryption_key,
        }
    )
    return app.test_client()