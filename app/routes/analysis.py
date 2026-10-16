"""
Dashboard and comment-analysis routes.

/dashboard          – list authenticated user's YouTube videos
/analyze/<video_id> – fetch comments, run ML pipeline, show results
"""

from flask import (
    Blueprint,
    current_app,
    jsonify,
    redirect,
    render_template,
    request,
    url_for,
)
from flask_login import current_user, login_required
from google.auth.exceptions import GoogleAuthError
from pymongo.errors import PyMongoError
from requests.exceptions import RequestException

from app.services.analysis_service import AnalysisSummary, analyze_comments, load_models
from app.services.google_oauth import OAuthConfigurationError
from app.services.youtube_api import (
    YouTubeAPIConfigurationError,
    YouTubeAPIError,
    YouTubeAuthenticationError,
    YouTubeCommentsDisabled,
    YouTubeDataService,
    YouTubeInsufficientPermissions,
    YouTubeNotConnected,
    YouTubeQuotaExceeded,
    YouTubeResourceUnavailable,
    YouTubeVideoNotFound,
    create_youtube_client,
)

analysis = Blueprint("analysis", __name__)


# ---------------------------------------------------------------------------
# Shared helper: build an authorized YouTubeDataService for current_user
# ---------------------------------------------------------------------------

def _get_youtube_service():
    """
    Return ``(service, error_tuple)`` where exactly one of the two is None.
    ``error_tuple`` is ``(template_name, context_dict)`` for render_template.
    """
    try:
        api = create_youtube_client(
            current_app.config,
            current_app.config["OAUTH_STORE"],
            current_user.id,
        )
        return YouTubeDataService(api), None
    except YouTubeNotConnected:
        return None, ("error_state.html", {
            "title": "YouTube not connected",
            "message": "Connect your YouTube account first.",
            "action_url": url_for("oauth.connect_google"),
            "action_label": "Connect YouTube with Google",
        })
    except YouTubeAuthenticationError:
        current_app.logger.warning("YouTube authentication expired or invalid.")
        return None, ("error_state.html", {
            "title": "Authentication expired",
            "message": "Your YouTube access has expired or was revoked. Please reconnect your account.",
            "action_url": url_for("oauth.connect_google"),
            "action_label": "Reconnect YouTube with Google",
        })
    except OAuthConfigurationError:
        current_app.logger.exception("Google token configuration invalid.")
        return None, ("error_state.html", {
            "title": "Configuration error",
            "message": "Google OAuth token configuration is invalid. Check server settings.",
            "action_url": url_for("auth.home"),
            "action_label": "Back to home",
        })
    except (GoogleAuthError, PyMongoError, RequestException, YouTubeAPIError):
        current_app.logger.exception("Could not initialize YouTube service.")
        return None, ("error_state.html", {
            "title": "YouTube unavailable",
            "message": "YouTube access is temporarily unavailable. Try again shortly.",
            "action_url": url_for("analysis.dashboard"),
            "action_label": "Retry",
        })


# ---------------------------------------------------------------------------
# Dashboard – video listing
# ---------------------------------------------------------------------------

@analysis.route("/dashboard")
@login_required
def dashboard():
    service, error = _get_youtube_service()
    if error:
        template, ctx = error
        return render_template(template, **ctx), 503

    page_token = request.args.get("pageToken")
    try:
        result = service.list_videos(page_token=page_token, max_results=25)
    except YouTubeQuotaExceeded:
        return render_template(
            "error_state.html",
            title="Quota exhausted",
            message="YouTube API daily quota is exhausted. Try again after midnight Pacific time.",
            action_url=url_for("auth.home"),
            action_label="Back to home",
        ), 429
    except YouTubeInsufficientPermissions:
        return render_template(
            "error_state.html",
            title="Permission needed",
            message="Your connected YouTube account lacks required permissions. Please reconnect your account.",
            action_url=url_for("oauth.connect_google"),
            action_label="Reconnect YouTube with Google",
        ), 403
    except YouTubeAuthenticationError:
        return render_template(
            "error_state.html",
            title="Authentication expired",
            message="Your YouTube session has expired or been revoked. Please reconnect your account.",
            action_url=url_for("oauth.connect_google"),
            action_label="Reconnect YouTube with Google",
        ), 401
    except YouTubeAPIConfigurationError:
        return render_template(
            "error_state.html",
            title="API configuration issue",
            message="YouTube Data API v3 is not enabled in your Google Cloud Console project.",
            action_url=url_for("auth.home"),
            action_label="Back to home",
        ), 503
    except YouTubeResourceUnavailable:
        return render_template(
            "error_state.html",
            title="Channel unavailable",
            message="Your YouTube channel or its uploads playlist is not accessible with this account.",
            action_url=url_for("auth.home"),
            action_label="Back to home",
        ), 404
    except YouTubeAPIError:
        current_app.logger.exception("YouTube video listing failed.")
        return render_template(
            "error_state.html",
            title="YouTube error",
            message="YouTube could not complete the request. Try again later.",
            action_url=url_for("analysis.dashboard"),
            action_label="Retry",
        ), 502

    videos = result.get("items", [])
    next_page_token = result.get("next_page_token")

    # Attach saved analyses if available
    video_ids = [v["video_id"] for v in videos]
    analysis_store = current_app.config.get("ANALYSIS_STORE")
    saved_analyses = {}
    if analysis_store:
        try:
            saved_analyses = analysis_store.get_analyses_for_videos(current_user.id, video_ids)
        except Exception:
            current_app.logger.warning("Could not retrieve saved analyses for dashboard.")

    for v in videos:
        v["saved_analysis"] = saved_analyses.get(v["video_id"])

    return render_template(
        "dashboard.html",
        videos=videos,
        next_page_token=next_page_token,
    )



# ---------------------------------------------------------------------------
# Comment analysis – fetch, predict, display
# ---------------------------------------------------------------------------

# Maximum comments fetched per analysis request (YouTube max = 100 per page,
# we fetch up to 2 pages to stay within reasonable quota usage).
_MAX_COMMENTS_PER_ANALYSIS = 100


@analysis.route("/analyze/<video_id>")
@login_required
def analyze_video(video_id):
    """
    Fetch up to ``_MAX_COMMENTS_PER_ANALYSIS`` top-level comments for
    ``video_id``, run sentiment + cyberbullying classification on each,
    and render the results page.  Empty-comment and error states are
    handled explicitly.
    """
    if not video_id or len(video_id) > 64:
        return render_template(
            "error_state.html",
            title="Invalid video",
            message="The video ID is missing or invalid.",
            action_url=url_for("analysis.dashboard"),
            action_label="Back to dashboard",
        ), 400

    service, error = _get_youtube_service()
    if error:
        template, ctx = error
        return render_template(template, **ctx), 503

    # Fetch comments (up to _MAX_COMMENTS_PER_ANALYSIS in one API call)
    try:
        result = service.list_comments(
            video_id,
            page_token=None,
            max_results=_MAX_COMMENTS_PER_ANALYSIS,
        )
    except YouTubeQuotaExceeded:
        return render_template(
            "error_state.html",
            title="Quota exhausted",
            message="YouTube API daily quota is exhausted. Try again after midnight Pacific time.",
            action_url=url_for("analysis.dashboard"),
            action_label="Back to dashboard",
        ), 429
    except YouTubeInsufficientPermissions:
        current_app.logger.warning("Insufficient permissions to fetch comments for video %s", video_id)
        return render_template(
            "error_state.html",
            title="Permission needed",
            message=(
                "Your connected YouTube account does not have permission to retrieve comments. "
                "Please reconnect your YouTube account to grant comment access."
            ),
            action_url=url_for("oauth.connect_google"),
            action_label="Reconnect YouTube with Google",
        ), 403
    except YouTubeAuthenticationError:
        current_app.logger.warning("YouTube authentication invalid or expired for video %s", video_id)
        return render_template(
            "error_state.html",
            title="Authentication expired",
            message="Your YouTube session has expired or been revoked. Please reconnect your account.",
            action_url=url_for("oauth.connect_google"),
            action_label="Reconnect YouTube with Google",
        ), 401
    except YouTubeCommentsDisabled:
        return render_template(
            "error_state.html",
            title="Comments disabled",
            message="Comments are disabled on this video.",
            action_url=url_for("analysis.dashboard"),
            action_label="Back to dashboard",
        ), 403
    except YouTubeVideoNotFound:
        return render_template(
            "error_state.html",
            title="Video unavailable",
            message="The requested video was not found or is private.",
            action_url=url_for("analysis.dashboard"),
            action_label="Back to dashboard",
        ), 404
    except YouTubeAPIConfigurationError:
        current_app.logger.error("YouTube Data API v3 is not configured in Google Cloud for video %s", video_id)
        return render_template(
            "error_state.html",
            title="API configuration issue",
            message="YouTube Data API v3 is not enabled in your Google Cloud Console project.",
            action_url=url_for("analysis.dashboard"),
            action_label="Back to dashboard",
        ), 503
    except YouTubeResourceUnavailable:
        return render_template(
            "error_state.html",
            title="Comments unavailable",
            message=(
                "Comments are unavailable for this video or restricted for this account."
            ),
            action_url=url_for("analysis.dashboard"),
            action_label="Back to dashboard",
        ), 404
    except YouTubeAPIError:
        current_app.logger.exception("Comment fetch failed for video %s", video_id)
        return render_template(
            "error_state.html",
            title="YouTube error",
            message="YouTube could not retrieve comments for this video. Try again later.",
            action_url=url_for("analysis.dashboard"),
            action_label="Back to dashboard",
        ), 502


    comment_items = result.get("items", [])

    if not comment_items:
        return render_template(
            "analysis.html",
            video_id=video_id,
            summary=None,
            empty=True,
        )

    # Load models (production if available, demo otherwise) and analyse
    try:
        models = load_models()
        summary = analyze_comments(comment_items, models=models)
        analysis_store = current_app.config.get("ANALYSIS_STORE")
        if analysis_store:
            try:
                analysis_store.save_analysis(
                    current_user.id,
                    video_id,
                    {
                        "total": summary.total,
                        "harmful_count": summary.harmful_count,
                        "harmful_pct": summary.harmful_pct,
                        "sentiment_counts": summary.sentiment_counts,
                        "bullying_counts": summary.bullying_counts,
                        "is_demo": summary.is_demo,
                    },
                )
            except Exception:
                current_app.logger.warning("Could not persist analysis summary for video %s", video_id)
    except Exception:
        current_app.logger.exception(
            "Analysis pipeline failed for video %s", video_id
        )
        return render_template(
            "error_state.html",
            title="Analysis error",
            message="The comment analysis pipeline failed. Check the server log for details.",
            action_url=url_for("analysis.dashboard"),
            action_label="Back to dashboard",
        ), 500

    return render_template(
        "analysis.html",
        video_id=video_id,
        summary=summary,
        empty=False,
    )

