"""Google OAuth flow creation and encryption for saved token payloads."""

from cryptography.fernet import Fernet, InvalidToken
from google_auth_oauthlib.flow import Flow

YOUTUBE_READONLY_SCOPE = "https://www.googleapis.com/auth/youtube.readonly"


class OAuthConfigurationError(ValueError):
    """Raised when required Google OAuth or token-encryption settings are absent."""


def validate_token_encryption_key(encryption_key):
    if not isinstance(encryption_key, str) or not encryption_key.strip():
        raise OAuthConfigurationError(
            "GOOGLE_TOKEN_ENCRYPTION_KEY is missing. Set a valid Fernet key in .env before starting Flask."
        )
    try:
        Fernet(encryption_key.encode("ascii"))
    except (UnicodeEncodeError, ValueError) as error:
        raise OAuthConfigurationError(
            "GOOGLE_TOKEN_ENCRYPTION_KEY is invalid. Set a valid Fernet key in .env before starting Flask."
        ) from error


def build_google_flow(config, state=None, code_verifier=None):
    required = ("GOOGLE_CLIENT_ID", "GOOGLE_CLIENT_SECRET", "GOOGLE_REDIRECT_URI")
    if any(not config.get(key) for key in required):
        raise OAuthConfigurationError("Google OAuth client settings are incomplete.")

    client_id = config["GOOGLE_CLIENT_ID"]
    if (
        not isinstance(client_id, str)
        or client_id != client_id.strip()
        or any(character.isspace() for character in client_id)
        or not client_id.endswith(".apps.googleusercontent.com")
    ):
        raise OAuthConfigurationError(
            "GOOGLE_CLIENT_ID must be the Google Web OAuth client ID ending in "
            ".apps.googleusercontent.com."
        )

    client_config = {
        "web": {
            "client_id": config["GOOGLE_CLIENT_ID"],
            "client_secret": config["GOOGLE_CLIENT_SECRET"],
            "auth_uri": "https://accounts.google.com/o/oauth2/v2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "redirect_uris": [config["GOOGLE_REDIRECT_URI"]],
        }
    }
    return Flow.from_client_config(
        client_config,
        scopes=[YOUTUBE_READONLY_SCOPE],
        redirect_uri=config["GOOGLE_REDIRECT_URI"],
        state=state,
        code_verifier=code_verifier,
        autogenerate_code_verifier=code_verifier is None,
    )


def encrypt_credentials(credentials_json, encryption_key):
    validate_token_encryption_key(encryption_key)
    try:
        cipher = Fernet(encryption_key.encode("ascii"))
        return cipher.encrypt(credentials_json.encode("utf-8")).decode("ascii")
    except (UnicodeEncodeError, ValueError) as error:
        raise OAuthConfigurationError("Google token encryption key is invalid.") from error


def decrypt_credentials(encrypted_credentials, encryption_key):
    if not encryption_key:
        raise OAuthConfigurationError("Google token encryption is not configured.")
    try:
        cipher = Fernet(encryption_key.encode("ascii"))
        return cipher.decrypt(encrypted_credentials.encode("ascii")).decode("utf-8")
    except (UnicodeEncodeError, InvalidToken, ValueError) as error:
        raise OAuthConfigurationError("Saved Google credentials could not be decrypted.") from error