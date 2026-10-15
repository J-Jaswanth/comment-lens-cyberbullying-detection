"""Flask application factory and extension setup."""

from flask import Flask
from flask_login import LoginManager
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_wtf.csrf import CSRFProtect
from pymongo.errors import PyMongoError

from app.config import Config
from app.models.user import User
from app.models.google_oauth_store import MongoGoogleOAuthStore
from app.models.user_store import MongoUserStore
from app.services.google_oauth import OAuthConfigurationError, validate_token_encryption_key

csrf = CSRFProtect()
login_manager = LoginManager()
limiter = Limiter(key_func=get_remote_address, default_limits=[])


def create_app(test_config=None):
	app = Flask(__name__)
	app.config.from_object(Config)
	if test_config:
		app.config.update(test_config)

	if not app.config.get("SECRET_KEY"):
		raise RuntimeError("Set SECRET_KEY in your local .env before starting Flask.")
	try:
		validate_token_encryption_key(app.config.get("GOOGLE_TOKEN_ENCRYPTION_KEY"))
	except OAuthConfigurationError as error:
		raise RuntimeError(str(error)) from error

	if "USER_STORE" not in app.config:
		app.config["USER_STORE"] = MongoUserStore(
			app.config["MONGODB_URI"], app.config["MONGODB_DB_NAME"]
		)
	if "OAUTH_STORE" not in app.config:
		app.config["OAUTH_STORE"] = MongoGoogleOAuthStore(
			app.config["MONGODB_URI"], app.config["MONGODB_DB_NAME"]
		)

	csrf.init_app(app)
	login_manager.init_app(app)
	limiter.init_app(app)
	login_manager.login_view = "auth.login"
	login_manager.login_message = "Please sign in to view that page."
	login_manager.login_message_category = "info"

	@login_manager.user_loader
	def load_user(user_id):
		try:
			return app.config["USER_STORE"].get_by_id(user_id)
		except PyMongoError:
			app.logger.exception("Could not load the signed-in account.")
			return None

	from app.routes.auth import auth
	from app.routes.oauth import oauth
	from app.routes.youtube import youtube
	from app.routes.analysis import analysis

	app.register_blueprint(auth)
	app.register_blueprint(oauth)
	app.register_blueprint(youtube)
	app.register_blueprint(analysis)
	return app