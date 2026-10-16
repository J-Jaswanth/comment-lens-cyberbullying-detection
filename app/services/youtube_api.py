"""Authenticated, paginated access to the official YouTube Data API."""

import json

from google.auth.exceptions import GoogleAuthError, RefreshError
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from app.services.google_oauth import (
    OAuthConfigurationError,
    YOUTUBE_READONLY_SCOPE,
    YOUTUBE_SCOPES,
    decrypt_credentials,
    encrypt_credentials,
)


class YouTubeAPIError(Exception):
    """Raised for safe-to-report YouTube API failures."""


class YouTubeNotConnected(Exception):
    """Raised when the current account has no saved YouTube authorization."""


class YouTubeAuthenticationError(YouTubeAPIError):
    """Raised when stored credentials are invalid, expired, or revoked."""


class YouTubeInsufficientPermissions(YouTubeAPIError):
    """Raised when the OAuth token lacks required scopes (e.g. comment access)."""


class YouTubeAPIConfigurationError(YouTubeAPIError):
    """Raised when the YouTube Data API is disabled or not configured in Google Cloud Console."""


class YouTubeQuotaExceeded(YouTubeAPIError):
    """Raised when Google reports an exhausted YouTube API quota or rate limit."""


class YouTubeResourceUnavailable(YouTubeAPIError):
    """Raised when a requested resource or its comments are unavailable."""


class YouTubeCommentsDisabled(YouTubeResourceUnavailable):
    """Raised when comments are disabled on a video."""


class YouTubeVideoNotFound(YouTubeResourceUnavailable):
    """Raised when the requested video cannot be found or is private."""


def create_youtube_client(config, oauth_store, user_id):
    encrypted_credentials = oauth_store.get_encrypted_credentials(user_id)
    if not encrypted_credentials:
        raise YouTubeNotConnected

    credentials_json = decrypt_credentials(
        encrypted_credentials,
        config.get("GOOGLE_TOKEN_ENCRYPTION_KEY"),
    )
    try:
        credentials_info = json.loads(credentials_json)
        credentials = Credentials.from_authorized_user_info(
            credentials_info,
            scopes=credentials_info.get("scopes") or YOUTUBE_SCOPES,
        )
    except (json.JSONDecodeError, TypeError, ValueError) as error:
        raise OAuthConfigurationError("Saved Google credentials are invalid.") from error

    if credentials.expired:
        if not credentials.refresh_token:
            raise YouTubeAuthenticationError("YouTube access has expired and cannot be refreshed.")
        try:
            credentials.refresh(Request())
        except RefreshError as error:
            raise YouTubeAuthenticationError("YouTube session refresh failed. Please reconnect your account.") from error
        updated_credentials = encrypt_credentials(
            credentials.to_json(),
            config.get("GOOGLE_TOKEN_ENCRYPTION_KEY"),
        )
        oauth_store.save_encrypted_credentials(user_id, updated_credentials)

    try:
        return build(
            "youtube",
            "v3",
            credentials=credentials,
            cache_discovery=False,
        )
    except (GoogleAuthError, OSError, ValueError) as error:
        raise YouTubeAPIError("Could not initialize YouTube access.") from error


def _execute(request):
    try:
        return request.execute()
    except HttpError as error:
        status_code = getattr(error.resp, "status", None)
        try:
            status_code = int(status_code)
        except (TypeError, ValueError):
            status_code = getattr(error, "status_code", 500)

        reason = ""
        message = ""
        try:
            payload = json.loads(error.content.decode("utf-8"))
            err_dict = payload.get("error", {})
            api_errors = err_dict.get("errors", [])
            if api_errors:
                reason = api_errors[0].get("reason", "")
                message = api_errors[0].get("message", "")
            if not message:
                message = err_dict.get("message", "")
        except (AttributeError, UnicodeDecodeError, json.JSONDecodeError):
            pass

        message_lower = message.lower()

        # Quota and rate limits
        if (
            reason in {"quotaExceeded", "dailyLimitExceeded", "rateLimitExceeded", "userRateLimitExceeded"}
            or status_code == 429
        ):
            raise YouTubeQuotaExceeded("YouTube API quota exceeded.") from error

        # Insufficient scopes / permissions
        if (
            reason in {"insufficientPermissions"}
            or "insufficient authentication scopes" in message_lower
            or "insufficient permission" in message_lower
        ):
            raise YouTubeInsufficientPermissions(
                "YouTube authorization has insufficient scopes to access comments. Please reconnect your account."
            ) from error

        # API not configured / enabled in Google Cloud Console
        if reason in {"accessNotConfigured", "serviceDisabled"}:
            raise YouTubeAPIConfigurationError(
                "YouTube Data API v3 is not enabled in your Google Cloud Console project."
            ) from error

        # Authentication / token expired or invalid
        if status_code == 401 or reason in {"authError", "invalidCredentials", "tokenExpired", "required"}:
            raise YouTubeAuthenticationError(
                "YouTube authentication is expired or invalid. Please reconnect your account."
            ) from error

        # Comments specifically disabled by the video owner
        if reason == "commentsDisabled":
            raise YouTubeCommentsDisabled("Comments are disabled for this video.") from error

        # Video or resource not found
        if reason in {"videoNotFound", "playlistNotFound", "notFound"} or status_code == 404:
            raise YouTubeVideoNotFound("The requested video or resource could not be found.") from error

        # Forbidden resource (e.g. private video)
        if reason == "forbidden" or status_code == 403:
            raise YouTubeResourceUnavailable("This video or resource is private or inaccessible.") from error

        raise YouTubeAPIError("YouTube could not complete this request.") from error


def parse_iso8601_duration(duration_str):
    """Parse ISO 8601 duration (e.g. PT1M15S, PT45S, PT1H2M) into total seconds."""
    import re
    if not duration_str or not isinstance(duration_str, str):
        return None
    match = re.match(
        r"^PT(?:(?P<hours>\d+)H)?(?:(?P<minutes>\d+)M)?(?:(?P<seconds>\d+)S)?$",
        duration_str,
    )
    if not match:
        return None
    hours = int(match.group("hours") or 0)
    minutes = int(match.group("minutes") or 0)
    seconds = int(match.group("seconds") or 0)
    return hours * 3600 + minutes * 60 + seconds


def is_short_video(duration_str, title="", description=""):
    """Identify if a video is a YouTube Short (duration <= 60s or tagged #shorts)."""
    seconds = parse_iso8601_duration(duration_str)
    if seconds is not None and 0 < seconds <= 60:
        return True
    combined_text = f"{title} {description}".lower()
    return "#shorts" in combined_text or "#short" in combined_text


class YouTubeDataService:
    def __init__(self, api):
        self.api = api

    def list_videos(self, page_token=None, max_results=25):
        channel_response = _execute(
            self.api.channels()
            .list(part="contentDetails", mine=True, maxResults=1)
        )
        channels = channel_response.get("items", [])
        if not channels:
            return {"items": [], "next_page_token": None}

        uploads_playlist_id = channels[0]["contentDetails"]["relatedPlaylists"]["uploads"]
        request_parameters = {
            "part": "snippet,contentDetails",
            "playlistId": uploads_playlist_id,
            "maxResults": max_results,
        }
        if page_token:
            request_parameters["pageToken"] = page_token
        response = _execute(self.api.playlistItems().list(**request_parameters))
        videos = []
        for item in response.get("items", []):
            snippet = item.get("snippet", {})
            content_details = item.get("contentDetails", {})
            video_id = content_details.get("videoId") or snippet.get("resourceId", {}).get("videoId")
            if not video_id:
                continue
            thumbs = snippet.get("thumbnails", {})
            thumbnail_url = (
                thumbs.get("medium", {}).get("url")
                or thumbs.get("high", {}).get("url")
                or thumbs.get("default", {}).get("url")
            )
            videos.append(
                {
                    "video_id": video_id,
                    "title": snippet.get("title", "Untitled video"),
                    "description": snippet.get("description", ""),
                    "published_at": snippet.get("publishedAt"),
                    "thumbnail_url": thumbnail_url,
                    "view_count": None,
                    "like_count": None,
                    "comment_count": None,
                    "is_short": False,
                    "video_type": "video",
                }
            )

        # Batch-enrich statistics, duration, and high-res thumbnails in one call
        video_ids = [v["video_id"] for v in videos]
        if video_ids and hasattr(self.api, "videos"):
            try:
                details_response = _execute(
                    self.api.videos().list(
                        part="snippet,contentDetails,statistics",
                        id=",".join(video_ids[:50]),
                        maxResults=len(video_ids[:50]),
                    )
                )
                items = details_response.get("items", [])
                details_map = {item.get("id"): item for item in items if isinstance(item, dict) and item.get("id")}
                for v in videos:
                    details = details_map.get(v["video_id"])
                    if not details:
                        continue
                    snip = details.get("snippet", {})
                    stats = details.get("statistics", {})
                    c_details = details.get("contentDetails", {})

                    if snip.get("description"):
                        v["description"] = snip["description"]
                    d_thumbs = snip.get("thumbnails", {})
                    if "medium" in d_thumbs:
                        v["thumbnail_url"] = d_thumbs["medium"]["url"]
                    elif "high" in d_thumbs:
                        v["thumbnail_url"] = d_thumbs["high"]["url"]

                    try:
                        if "viewCount" in stats:
                            v["view_count"] = int(stats["viewCount"])
                        if "likeCount" in stats:
                            v["like_count"] = int(stats["likeCount"])
                        if "commentCount" in stats:
                            v["comment_count"] = int(stats["commentCount"])
                    except (ValueError, TypeError):
                        pass

                    duration = c_details.get("duration", "")
                    v["duration"] = duration
                    v["is_short"] = is_short_video(duration, v.get("title", ""), v.get("description", ""))
                    v["video_type"] = "short" if v["is_short"] else "video"
            except Exception:
                # Graceful fallback: maintain basic video metadata from playlistItems
                pass

        return {"items": videos, "next_page_token": response.get("nextPageToken")}


    def list_comments(self, video_id, page_token=None, max_results=100):
        request_parameters = {
            "part": "snippet",
            "videoId": video_id,
            "maxResults": max_results,
            "textFormat": "plainText",
        }
        if page_token:
            request_parameters["pageToken"] = page_token
        response = _execute(self.api.commentThreads().list(**request_parameters))
        comments = []
        for thread in response.get("items", []):
            thread_snippet = thread.get("snippet", {})
            comment = thread_snippet.get("topLevelComment", {})
            comment_snippet = comment.get("snippet", {})
            comments.append(
                {
                    "comment_id": comment.get("id"),
                    "thread_id": thread.get("id"),
                    "text": comment_snippet.get("textDisplay", ""),
                    "published_at": comment_snippet.get("publishedAt"),
                    "reply_count": thread_snippet.get("totalReplyCount", 0),
                }
            )
        return {"items": comments, "next_page_token": response.get("nextPageToken")}

    def list_replies(self, parent_comment_id, page_token=None, max_results=100):
        request_parameters = {
            "part": "snippet",
            "parentId": parent_comment_id,
            "maxResults": max_results,
            "textFormat": "plainText",
        }
        if page_token:
            request_parameters["pageToken"] = page_token
        response = _execute(self.api.comments().list(**request_parameters))
        comments = [
            {
                "comment_id": item.get("id"),
                "text": item.get("snippet", {}).get("textDisplay", ""),
                "published_at": item.get("snippet", {}).get("publishedAt"),
            }
            for item in response.get("items", [])
        ]
        return {"items": comments, "next_page_token": response.get("nextPageToken")}