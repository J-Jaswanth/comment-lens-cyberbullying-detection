"""Authenticated JSON endpoints for YouTube videos and comments."""

from flask import Blueprint, abort, current_app, jsonify, request
from flask_login import current_user, login_required
from google.auth.exceptions import GoogleAuthError
from pymongo.errors import PyMongoError
from requests.exceptions import RequestException

from app.services.google_oauth import OAuthConfigurationError
from app.services.youtube_api import (
    YouTubeAPIError,
    YouTubeDataService,
    YouTubeNotConnected,
    YouTubeQuotaExceeded,
    YouTubeResourceUnavailable,
    create_youtube_client,
)

youtube = Blueprint("youtube", __name__, url_prefix="/api/youtube")


def _get_service():
    try:
        api = create_youtube_client(
            current_app.config,
            current_app.config["OAUTH_STORE"],
            current_user.id,
        )
    except YouTubeNotConnected:
        return None, (jsonify(error="Connect a YouTube account first."), 409)
    except OAuthConfigurationError:
        current_app.logger.exception("Google token configuration is invalid.")
        return None, (jsonify(error="Google token configuration is invalid."), 503)
    except (GoogleAuthError, PyMongoError, RequestException, YouTubeAPIError):
        current_app.logger.exception("Could not initialize YouTube access.")
        return None, (jsonify(error="YouTube access is temporarily unavailable."), 503)
    return YouTubeDataService(api), None


def _page_options(default, maximum):
    max_results_value = request.args.get("maxResults", str(default))
    try:
        max_results = int(max_results_value)
    except ValueError:
        abort(400, description="maxResults must be a whole number.")
    if not 1 <= max_results <= maximum:
        abort(400, description=f"maxResults must be between 1 and {maximum}.")
    return request.args.get("pageToken"), max_results


def _run_page(operation):
    try:
        return jsonify(operation())
    except YouTubeQuotaExceeded:
        return jsonify(error="YouTube API quota is exhausted. Try again after quota resets."), 429
    except YouTubeResourceUnavailable:
        return jsonify(error="This video or its comments are unavailable to this account."), 404
    except YouTubeAPIError:
        current_app.logger.exception("YouTube API request failed.")
        return jsonify(error="YouTube could not complete the request. Try again later."), 502


@youtube.get("/videos")
@login_required
def list_videos():
    page_token, max_results = _page_options(default=25, maximum=50)
    service, error_response = _get_service()
    if error_response:
        return error_response
    return _run_page(lambda: service.list_videos(page_token, max_results))


@youtube.get("/videos/<video_id>/comments")
@login_required
def list_comments(video_id):
    page_token, max_results = _page_options(default=100, maximum=100)
    service, error_response = _get_service()
    if error_response:
        return error_response
    return _run_page(lambda: service.list_comments(video_id, page_token, max_results))


@youtube.get("/comments/<comment_id>/replies")
@login_required
def list_replies(comment_id):
    page_token, max_results = _page_options(default=100, maximum=100)
    service, error_response = _get_service()
    if error_response:
        return error_response
    return _run_page(lambda: service.list_replies(comment_id, page_token, max_results))