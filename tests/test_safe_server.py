from unittest.mock import MagicMock

from app.safe_server import OAuthSafeRequestHandler, is_oauth_callback_path


def test_oauth_callback_request_targets_are_recognized_without_query_string():
    assert is_oauth_callback_path(
        "/auth/google/callback?code=private-code&state=private-state"
    )
    assert is_oauth_callback_path("/oauth/google/callback?code=private-code")
    assert not is_oauth_callback_path("/auth/register?next=/")


def test_oauth_callback_access_log_never_includes_query_string():
    handler = object.__new__(OAuthSafeRequestHandler)
    handler.path = "/auth/google/callback?code=private-code&state=private-state"
    handler.log = MagicMock()

    handler.log_request(302)

    handler.log.assert_called_once_with(
        "info",
        "OAuth callback request completed with status %s",
        302,
    )
    assert "private-code" not in str(handler.log.call_args)
    assert "private-state" not in str(handler.log.call_args)