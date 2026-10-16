"""Registration, login, logout, and authenticated home routes."""

from urllib.parse import urlsplit

from flask import Blueprint, current_app, flash, redirect, render_template, request, session, url_for
from flask_login import current_user, login_required, login_user, logout_user
from werkzeug.security import check_password_hash, generate_password_hash
from pymongo.errors import PyMongoError

from app import limiter
from app.forms import LoginForm, RegistrationForm
from app.models.user import User
from app.models.user_store import UserAlreadyExists

auth = Blueprint("auth", __name__)


def _mongo_storage_error_message(error):
    detail = f"{type(error).__name__}: {error}".strip()
    if not detail or detail == ":":
        detail = "MongoDB connection failed."
    base = "Account storage is unavailable. Check your MongoDB setup."
    if "SSL" in detail or "TLS" in detail or "handshake" in detail.lower():
        return f"{base} The database connection failed during TLS/SSL setup: {detail}"
    if "timeout" in detail.lower() or "selection" in detail.lower():
        return f"{base} The database did not respond in time: {detail}"
    return f"{base} Details: {detail}"


def _safe_next_url(candidate):
    if not candidate:
        return None
    parsed = urlsplit(candidate)
    if (
        parsed.scheme
        or parsed.netloc
        or "\\" in candidate
        or not candidate.startswith("/")
        or candidate.startswith("//")
    ):
        return None
    return candidate


@auth.route("/")
@login_required
def home():
    try:
        youtube_connected = current_app.config["OAUTH_STORE"].has_credentials(current_user.id)
    except PyMongoError as error:
        current_app.logger.exception("Could not load the YouTube connection status.")
        return _mongo_storage_error_message(error), 503
    return render_template("home.html", youtube_connected=youtube_connected)


@auth.route("/auth/register", methods=["GET", "POST"])
@limiter.limit("5 per hour")
def register():
    form = RegistrationForm()
    if form.validate_on_submit():
        username = form.username.data.strip().casefold()
        email = form.email.data.strip().casefold()
        user = User(
            user_id="",
            username=username,
            email=email,
            password_hash=generate_password_hash(form.password.data, method="scrypt"),
        )
        try:
            current_app.config["USER_STORE"].create_user(user)
        except UserAlreadyExists:
            flash("An account may already exist for those details.", "error")
            return render_template("auth/register.html", form=form), 409
        except PyMongoError as error:
            current_app.logger.exception("Account registration failed.")
            flash(_mongo_storage_error_message(error), "error")
            return render_template("auth/register.html", form=form), 503
        flash("Your account is ready. Please sign in.", "success")
        return redirect(url_for("auth.login"))
    return render_template("auth/register.html", form=form)


@auth.route("/auth/login", methods=["GET", "POST"])
@limiter.limit("10 per minute")
def login():
    if current_user.is_authenticated:
        return redirect(url_for("auth.home"))
    form = LoginForm()
    if form.validate_on_submit():
        try:
            user = current_app.config["USER_STORE"].get_by_login(form.identifier.data)
        except PyMongoError as error:
            current_app.logger.exception("Account sign-in lookup failed.")
            flash(_mongo_storage_error_message(error), "error")
            return render_template("auth/login.html", form=form), 503
        if user and check_password_hash(user.password_hash, form.password.data):
            session.clear()
            login_user(user, fresh=True)
            session.permanent = True
            return redirect(_safe_next_url(request.args.get("next")) or url_for("auth.home"))
        flash("The username/email or password is incorrect.", "error")
    return render_template("auth/login.html", form=form)


@auth.route("/auth/logout", methods=["POST"])
@login_required
def logout():
    logout_user()
    session.clear()
    flash("You have signed out.", "info")
    return redirect(url_for("auth.login"))