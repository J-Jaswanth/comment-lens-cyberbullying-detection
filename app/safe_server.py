"""Development-server request logging that protects OAuth callback parameters."""

from werkzeug.serving import WSGIRequestHandler

OAUTH_CALLBACK_PATHS = {
    "/auth/google/callback",
    "/oauth/google/callback",
}


def is_oauth_callback_path(request_target):
    return request_target.split("?", 1)[0] in OAUTH_CALLBACK_PATHS


class OAuthSafeRequestHandler(WSGIRequestHandler):
    def log_request(self, code="-", size="-"):
        if is_oauth_callback_path(self.path):
            self.log("info", "OAuth callback request completed with status %s", code)
            return
        super().log_request(code, size)