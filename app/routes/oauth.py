"""Google OAuth authorization and callback routes."""

from hmac import compare_digest

from flask import Blueprint, abort, current_app, flash, redirect, request, session, url_for
from flask_login import current_user, login_required
from google.auth.exceptions import GoogleAuthError
from oauthlib.oauth2 import OAuth2Error
from oauthlib.oauth2.rfc6749.errors import (
    InvalidTokenError,
    MissingTokenError,
    MissingTokenTypeError,
)
from pymongo.errors import PyMongoError
from requests.exceptions import RequestException

from app.services.google_oauth import (
    OAuthConfigurationError,
    build_google_flow,
    encrypt_credentials,
)

oauth = Blueprint("oauth", __name__)
STATE_SESSION_KEY = "google_oauth_state"
VERIFIER_SESSION_KEY = "google_oauth_code_verifier"


def _log_oauth_failure(stage, category, description):
    current_app.logger.warning(
        "Google OAuth failure stage=%s category=%s description=%s",
        stage,
        category,
        description,
    )
    print(f"OAUTH-DEBUG stage={stage} category={category}", flush=True)


@oauth.route("/oauth/google/connect")
@login_required
def connect_google():
    try:
        flow = build_google_flow(current_app.config)
        authorization_url, state = flow.authorization_url(
            prompt="consent",
        )
    except OAuthConfigurationError as error:
        _log_oauth_failure(
            "before_google_redirect",
            type(error).__name__,
            "Local OAuth configuration validation failed.",
        )
        flash(str(error), "error")
        return redirect(url_for("auth.home"))
    except (GoogleAuthError, OAuth2Error, RequestException, ValueError) as error:
        _log_oauth_failure(
            "before_google_redirect",
            type(error).__name__,
            "Could not create the Google authorization request.",
        )
        flash("Google authorization could not be started. Check the sanitized server diagnostic.", "error")
        return redirect(url_for("auth.home"))

    session[STATE_SESSION_KEY] = state
    session[VERIFIER_SESSION_KEY] = flow.code_verifier
    return redirect(authorization_url)


@oauth.route("/oauth/google/callback", endpoint="google_callback_legacy")
@oauth.route("/auth/google/callback", endpoint="google_callback")
@login_required
def google_callback():
    expected_state = session.pop(STATE_SESSION_KEY, None)
    code_verifier = session.pop(VERIFIER_SESSION_KEY, None)
    returned_state = request.args.get("state", "")
    if not expected_state or not code_verifier or not compare_digest(expected_state, returned_state):
        _log_oauth_failure(
            "google_redirect",
            "OAuthStateValidationError",
            "Callback state or PKCE data did not match the session.",
        )
        abort(400, description="Google OAuth state could not be verified.")

    if request.args.get("error"):
        _log_oauth_failure(
            "google_redirect",
            "GoogleAuthorizationError",
            "Google returned an authorization error before token exchange.",
        )
        flash("Google authorization was not completed.", "info")
        return redirect(url_for("auth.home"))

    try:
        flow = build_google_flow(
            current_app.config,
            state=expected_state,
            code_verifier=code_verifier,
        )
    except OAuthConfigurationError as error:
        _log_oauth_failure(
            "before_token_exchange",
            type(error).__name__,
            "Local OAuth configuration validation failed.",
        )
        flash("Google OAuth or token encryption is not configured correctly.", "error")
        return redirect(url_for("auth.home"))

    try:
        flow.fetch_token(authorization_response=request.url)
    except (InvalidTokenError, MissingTokenError, MissingTokenTypeError, ValueError) as error:
        _log_oauth_failure(
            "token_response_parsing",
            type(error).__name__,
            "Google returned a token response that could not be parsed or validated.",
        )
        flash(
            f"Google authorization could not be completed ({type(error).__name__})",
            "error",
        )
        return redirect(url_for("auth.home"))
    except Exception as error:
        _log_oauth_failure(
            "token_exchange",
            type(error).__name__,
            "Google token endpoint exchange failed or was rejected.",
        )
        flash(
            f"Google authorization could not be completed ({type(error).__name__})",
            "error",
        )
        return redirect(url_for("auth.home"))

    try:
        credentials = flow.credentials
    except Exception as error:
        _log_oauth_failure(
            "token_response_parsing",
            type(error).__name__,
            "Parsed token response could not be converted to Google credentials.",
        )
        flash(
            f"Google authorization could not be completed ({type(error).__name__})",
            "error",
        )
        return redirect(url_for("auth.home"))

    try:
        credentials_json = credentials.to_json()
    except Exception as error:
        _log_oauth_failure(
            "credential_serialization",
            type(error).__name__,
            "Google credentials could not be serialized for secure storage.",
        )
        flash("Google authorization could not be completed. Please try again.", "error")
        return redirect(url_for("auth.home"))

    try:
        encrypted_credentials = encrypt_credentials(
            credentials_json,
            current_app.config.get("GOOGLE_TOKEN_ENCRYPTION_KEY"),
        )
    except Exception as error:
        _log_oauth_failure(
            "token_encryption",
            type(error).__name__,
            "Google credentials could not be encrypted for storage.",
        )
        flash("Google OAuth or token encryption is not configured correctly.", "error")
        return redirect(url_for("auth.home"))

    try:
        current_app.config["OAUTH_STORE"].save_encrypted_credentials(
            current_user.id,
            encrypted_credentials,
        )
    except PyMongoError as error:
        _log_oauth_failure(
            "token_storage",
            type(error).__name__,
            "Encrypted credentials could not be saved to account storage.",
        )
        flash("Google connected, but secure token storage is unavailable.", "error")
        return redirect(url_for("auth.home"))

    flash("YouTube access was connected and the credentials were encrypted for this account.", "success")
    return redirect(url_for("auth.home"))