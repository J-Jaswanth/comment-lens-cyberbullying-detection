import logging
from unittest.mock import MagicMock, patch
from urllib.parse import parse_qs, urlparse

from cryptography.fernet import Fernet
from oauthlib.oauth2.rfc6749.errors import MissingTokenError
import pytest
from requests.exceptions import ConnectionError

from app.config import configure_oauthlib_for_environment
from app import create_app
from app.routes.oauth import STATE_SESSION_KEY, VERIFIER_SESSION_KEY
from app.services.google_oauth import (
    OAuthConfigurationError,
    build_google_flow,
    decrypt_credentials,
    encrypt_credentials,
)


def sign_in(client):
    client.post(
        "/auth/register",
        data={
            "username": "oauth_student",
            "email": "oauth@example.com",
            "password": "strong-password-123",
            "confirm_password": "strong-password-123",
        },
    )
    client.post(
        "/auth/login",
        data={"identifier": "oauth_student", "password": "strong-password-123"},
    )


def test_development_enables_oauthlib_local_flags(monkeypatch):
    monkeypatch.delenv("OAUTHLIB_INSECURE_TRANSPORT", raising=False)
    monkeypatch.delenv("OAUTHLIB_RELAX_TOKEN_SCOPE", raising=False)

    configure_oauthlib_for_environment("development")

    assert __import__("os").environ["OAUTHLIB_INSECURE_TRANSPORT"] == "1"
    assert __import__("os").environ["OAUTHLIB_RELAX_TOKEN_SCOPE"] == "1"


def test_production_does_not_enable_oauthlib_local_flags(monkeypatch):
    import os

    monkeypatch.delenv("OAUTHLIB_INSECURE_TRANSPORT", raising=False)
    monkeypatch.delenv("OAUTHLIB_RELAX_TOKEN_SCOPE", raising=False)

    configure_oauthlib_for_environment("production")

    assert "OAUTHLIB_INSECURE_TRANSPORT" not in os.environ
    assert "OAUTHLIB_RELAX_TOKEN_SCOPE" not in os.environ


@pytest.mark.parametrize(
    "encryption_key",
    [None, "generate-a-fernet-key"],
    ids=["missing", "placeholder"],
)
def test_app_startup_rejects_missing_or_placeholder_encryption_key(encryption_key):
    with pytest.raises(RuntimeError, match="GOOGLE_TOKEN_ENCRYPTION_KEY"):
        create_app(
            {
                "SECRET_KEY": "test-only-secret-key",
                "GOOGLE_TOKEN_ENCRYPTION_KEY": encryption_key,
                "GOOGLE_CLIENT_ID": None,
                "GOOGLE_CLIENT_SECRET": None,
            }
        )


def test_oauth_connect_requires_an_application_login(client):
    response = client.get("/oauth/google/connect")

    assert response.status_code == 302
    assert "/auth/login" in response.headers["Location"]


def test_oauth_connect_reports_missing_google_configuration(client):
    sign_in(client)

    response = client.get("/oauth/google/connect", follow_redirects=True)

    assert response.status_code == 200
    assert b"Google OAuth client settings are incomplete" in response.data


@patch("app.routes.oauth.build_google_flow")
def test_oauth_connect_shows_safe_configuration_error(build_flow, client):
    sign_in(client)
    build_flow.side_effect = OAuthConfigurationError(
        "GOOGLE_CLIENT_ID must use Google's Web client format."
    )

    response = client.get("/oauth/google/connect", follow_redirects=True)

    assert response.status_code == 200
    assert b"GOOGLE_CLIENT_ID must use Google" in response.data


def test_official_flow_builds_state_and_pkce_authorization_url():
    config = {
        "GOOGLE_CLIENT_ID": "test-client.apps.googleusercontent.com",
        "GOOGLE_CLIENT_SECRET": "test-client-secret",
        "GOOGLE_REDIRECT_URI": "http://127.0.0.1:5000/auth/google/callback",
    }
    flow = build_google_flow(config)

    authorization_url, state = flow.authorization_url(prompt="consent")
    query = parse_qs(urlparse(authorization_url).query)
    callback_flow = build_google_flow(
        config,
        state=state,
        code_verifier=flow.code_verifier,
    )

    assert query["state"] == [state]
    assert query["code_challenge_method"] == ["S256"]
    assert flow.code_verifier
    assert callback_flow.oauth2session.state == state


def test_official_flow_rejects_malformed_client_id():
    config = {
        "GOOGLE_CLIENT_ID": "invalid-client-id",
        "GOOGLE_CLIENT_SECRET": "test-client-secret",
        "GOOGLE_REDIRECT_URI": "http://127.0.0.1:5000/auth/google/callback",
    }

    try:
        build_google_flow(config)
    except OAuthConfigurationError as error:
        assert "GOOGLE_CLIENT_ID" in str(error)
    else:
        raise AssertionError("Malformed Google client ID should be rejected.")


def test_encrypt_credentials_rejects_documented_placeholder_key():
    with pytest.raises(OAuthConfigurationError):
        encrypt_credentials("synthetic-test-payload", "generate-a-fernet-key")


def test_both_configured_and_legacy_callback_routes_are_registered(client):
    configured = client.get("/auth/google/callback")
    legacy = client.get("/oauth/google/callback")

    assert configured.status_code == 302
    assert legacy.status_code == 302
    assert "/auth/login" in configured.headers["Location"]
    assert "/auth/login" in legacy.headers["Location"]


@patch("app.routes.oauth.build_google_flow")
def test_connect_stores_single_use_state_and_pkce_verifier(build_flow, client):
    sign_in(client)
    flow = MagicMock()
    flow.authorization_url.return_value = (
        "https://accounts.google.com/o/oauth2/v2/auth?state=test-state",
        "test-state",
    )
    flow.code_verifier = "test-code-verifier"
    build_flow.return_value = flow

    response = client.get("/oauth/google/connect")

    assert response.status_code == 302
    assert response.headers["Location"].startswith("https://accounts.google.com/")
    flow.authorization_url.assert_called_once_with(prompt="consent")
    with client.session_transaction() as flask_session:
        assert flask_session[STATE_SESSION_KEY] == "test-state"
        assert flask_session[VERIFIER_SESSION_KEY] == "test-code-verifier"


def test_callback_rejects_mismatched_state(client):
    sign_in(client)
    with client.session_transaction() as flask_session:
        flask_session[STATE_SESSION_KEY] = "expected-state"
        flask_session[VERIFIER_SESSION_KEY] = "test-code-verifier"

    response = client.get("/oauth/google/callback?state=attacker-state&code=unused")

    assert response.status_code == 400


def test_callback_logs_state_failure_without_query_values(client, caplog):
    sign_in(client)
    with caplog.at_level(logging.WARNING, logger="app"):
        response = client.get(
            "/auth/google/callback?state=private-state&code=private-auth-code"
        )

    assert response.status_code == 400
    assert "stage=google_redirect" in caplog.text
    assert "category=OAuthStateValidationError" in caplog.text
    assert "private-state" not in caplog.text
    assert "private-auth-code" not in caplog.text


def test_google_redirect_error_is_logged_without_provider_query_values(client, caplog):
    sign_in(client)
    with client.session_transaction() as flask_session:
        flask_session[STATE_SESSION_KEY] = "expected-state"
        flask_session[VERIFIER_SESSION_KEY] = "test-code-verifier"

    with caplog.at_level(logging.WARNING, logger="app"):
        response = client.get(
            "/auth/google/callback?state=expected-state&error=access_denied&error_description=private-detail"
        )

    assert response.status_code == 302
    assert "category=GoogleAuthorizationError" in caplog.text
    assert "access_denied" not in caplog.text
    assert "private-detail" not in caplog.text


@patch("app.routes.oauth.build_google_flow")
def test_token_exchange_logs_exception_category_without_secrets(
    build_flow,
    client,
    caplog,
    capsys,
):
    sign_in(client)
    with client.session_transaction() as flask_session:
        flask_session[STATE_SESSION_KEY] = "expected-state"
        flask_session[VERIFIER_SESSION_KEY] = "test-code-verifier"
    flow = MagicMock()
    raw_exception_message = "private-client-secret private-access-token private-auth-code"
    flow.fetch_token.side_effect = ConnectionError(raw_exception_message)
    build_flow.return_value = flow

    with caplog.at_level(logging.WARNING, logger="app"):
        response = client.get(
            "/auth/google/callback?state=expected-state&code=private-auth-code",
            follow_redirects=True,
        )

    assert response.status_code == 200
    assert "stage=token_exchange" in caplog.text
    assert "category=ConnectionError" in caplog.text
    assert "token endpoint exchange failed or was rejected" in caplog.text
    assert b"Google authorization could not be completed (ConnectionError)" in response.data
    assert raw_exception_message not in caplog.text
    assert "/auth/google/callback" not in caplog.text
    assert "state=" not in caplog.text
    assert "code=" not in caplog.text
    assert "private-client-secret" not in caplog.text
    assert "private-access-token" not in caplog.text
    assert "private-auth-code" not in caplog.text
    console = capsys.readouterr().out
    assert "OAUTH-DEBUG stage=token_exchange category=ConnectionError" in console
    assert raw_exception_message not in console
    assert "private-auth-code" not in console


@patch("app.routes.oauth.build_google_flow")
def test_token_response_parse_failure_logs_sanitized_stage(
    build_flow,
    client,
    caplog,
    capsys,
):
    sign_in(client)
    with client.session_transaction() as flask_session:
        flask_session[STATE_SESSION_KEY] = "expected-state"
        flask_session[VERIFIER_SESSION_KEY] = "test-code-verifier"
    flow = MagicMock()
    flow.fetch_token.side_effect = MissingTokenError(
        description="private-access-token private-auth-code"
    )
    build_flow.return_value = flow

    with caplog.at_level(logging.WARNING, logger="app"):
        response = client.get(
            "/auth/google/callback?state=expected-state&code=private-auth-code",
            follow_redirects=True,
        )

    assert response.status_code == 200
    assert "stage=token_response_parsing" in caplog.text
    assert "category=MissingTokenError" in caplog.text
    assert "could not be parsed or validated" in caplog.text
    assert b"Google authorization could not be completed (MissingTokenError)" in response.data
    assert "private-access-token" not in caplog.text
    assert "private-auth-code" not in caplog.text
    assert "/auth/google/callback" not in caplog.text
    assert "code=" not in caplog.text
    console = capsys.readouterr().out
    assert "OAUTH-DEBUG stage=token_response_parsing category=MissingTokenError" in console
    assert "private-access-token" not in console
    assert "private-auth-code" not in console


@patch("app.routes.oauth.build_google_flow")
def test_credential_serialization_failure_logs_sanitized_stage(
    build_flow,
    client,
    caplog,
):
    sign_in(client)
    with client.session_transaction() as flask_session:
        flask_session[STATE_SESSION_KEY] = "expected-state"
        flask_session[VERIFIER_SESSION_KEY] = "test-code-verifier"
    flow = MagicMock()
    flow.credentials.to_json.side_effect = ValueError(
        "private-client-secret private-refresh-token"
    )
    build_flow.return_value = flow

    with caplog.at_level(logging.WARNING, logger="app"):
        response = client.get(
            "/auth/google/callback?state=expected-state&code=private-code"
        )

    assert response.status_code == 302
    assert "stage=credential_serialization" in caplog.text
    assert "category=ValueError" in caplog.text
    assert "private-client-secret" not in caplog.text
    assert "private-refresh-token" not in caplog.text
    assert "private-code" not in caplog.text
    assert "/auth/google/callback" not in caplog.text


@patch("app.routes.oauth.encrypt_credentials")
@patch("app.routes.oauth.build_google_flow")
def test_token_encryption_failure_logs_sanitized_stage(
    build_flow,
    encrypt_credentials_mock,
    client,
    caplog,
):
    sign_in(client)
    with client.session_transaction() as flask_session:
        flask_session[STATE_SESSION_KEY] = "expected-state"
        flask_session[VERIFIER_SESSION_KEY] = "test-code-verifier"
    flow = MagicMock()
    flow.credentials.to_json.return_value = "synthetic-credentials-json"
    build_flow.return_value = flow
    encrypt_credentials_mock.side_effect = ValueError("private-token")

    with caplog.at_level(logging.WARNING, logger="app"):
        response = client.get(
            "/auth/google/callback?state=expected-state&code=private-code"
        )

    assert response.status_code == 302
    assert "stage=token_encryption" in caplog.text
    assert "category=ValueError" in caplog.text
    assert "private-token" not in caplog.text
    assert "private-code" not in caplog.text
    assert "/auth/google/callback" not in caplog.text


@patch("app.routes.oauth.build_google_flow")
def test_callback_encrypts_and_stores_credentials_for_signed_in_user(
    build_flow,
    client,
    oauth_store,
    app,
):
    sign_in(client)
    encryption_key = Fernet.generate_key().decode("ascii")
    app.config["GOOGLE_CLIENT_ID"] = "test-client-id"
    app.config["GOOGLE_CLIENT_SECRET"] = "test-client-secret"
    app.config["GOOGLE_TOKEN_ENCRYPTION_KEY"] = encryption_key
    with client.session_transaction() as flask_session:
        flask_session[STATE_SESSION_KEY] = "expected-state"
        flask_session[VERIFIER_SESSION_KEY] = "test-code-verifier"

    flow = MagicMock()
    flow.credentials.to_json.return_value = '{"token":"test-access-token"}'
    build_flow.return_value = flow

    response = client.get("/oauth/google/callback?state=expected-state&code=test-code")

    assert response.status_code == 302
    flow.fetch_token.assert_called_once()
    encrypted = oauth_store.get_encrypted_credentials("1")
    assert encrypted != '{"token":"test-access-token"}'
    assert decrypt_credentials(encrypted, encryption_key) == '{"token":"test-access-token"}'


@patch("app.routes.oauth.build_google_flow")
def test_callback_restores_session_pkce_verifier_before_token_exchange(
    build_flow,
    client,
    app,
    oauth_store,
    test_encryption_key,
):
    sign_in(client)
    app.config["GOOGLE_TOKEN_ENCRYPTION_KEY"] = test_encryption_key
    with client.session_transaction() as flask_session:
        flask_session[STATE_SESSION_KEY] = "expected-state"
        flask_session[VERIFIER_SESSION_KEY] = "stored-pkce-verifier"

    flow = MagicMock()
    flow.credentials.to_json.return_value = "synthetic-credentials-json"
    build_flow.return_value = flow

    response = client.get(
        "/auth/google/callback?state=expected-state&code=synthetic-code"
    )

    assert response.status_code == 302
    assert build_flow.call_args.kwargs["state"] == "expected-state"
    assert build_flow.call_args.kwargs["code_verifier"] == "stored-pkce-verifier"
    flow.fetch_token.assert_called_once()
    assert oauth_store.has_credentials("1")
    with client.session_transaction() as flask_session:
        assert STATE_SESSION_KEY not in flask_session
        assert VERIFIER_SESSION_KEY not in flask_session