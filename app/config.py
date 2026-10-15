"""Environment-backed Flask configuration."""

import os
from datetime import timedelta

from dotenv import load_dotenv

load_dotenv()


def configure_oauthlib_for_environment(environment):
    oauthlib_flags = (
        "OAUTHLIB_INSECURE_TRANSPORT",
        "OAUTHLIB_RELAX_TOKEN_SCOPE",
    )
    if environment.casefold() == "development":
        for flag in oauthlib_flags:
            os.environ[flag] = "1"
    else:
        for flag in oauthlib_flags:
            os.environ.pop(flag, None)


APP_ENV = os.getenv("APP_ENV", "development").strip().casefold()
configure_oauthlib_for_environment(APP_ENV)


class Config:
    APP_ENV = APP_ENV
    SECRET_KEY = os.getenv("SECRET_KEY")
    MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
    MONGODB_DB_NAME = os.getenv("MONGODB_DB_NAME", "cyberbullying_analytics")
    GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
    GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
    GOOGLE_REDIRECT_URI = os.getenv(
        "GOOGLE_REDIRECT_URI",
        "http://127.0.0.1:5000/auth/google/callback",
    )
    GOOGLE_TOKEN_ENCRYPTION_KEY = os.getenv("GOOGLE_TOKEN_ENCRYPTION_KEY")
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SECURE = os.getenv("SESSION_COOKIE_SECURE", "false").lower() == "true"
    SESSION_COOKIE_SAMESITE = "Lax"
    PERMANENT_SESSION_LIFETIME = timedelta(minutes=60)
    SESSION_REFRESH_EACH_REQUEST = False
    WTF_CSRF_ENABLED = True
    WTF_CSRF_TIME_LIMIT = 3600
    RATELIMIT_STORAGE_URI = os.getenv("RATE_LIMIT_STORAGE_URI", "memory://")