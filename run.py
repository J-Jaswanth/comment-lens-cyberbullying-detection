"""Local development entry point."""

from app import create_app
from app.safe_server import OAuthSafeRequestHandler

app = create_app()


if __name__ == "__main__":
    app.run(request_handler=OAuthSafeRequestHandler)