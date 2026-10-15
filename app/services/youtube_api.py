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
    decrypt_credentials,
    encrypt_credentials,
)


class YouTubeNotConnected(Exception):
    """Raised when the current account has no saved YouTube authorization."""


class YouTubeQuotaExceeded(Exception):
    """Raised when Google reports an exhausted YouTube API quota."""


class YouTubeResourceUnavailable(Exception):
    """Raised when a requested video or its comments are unavailable."""


class YouTubeAPIError(Exception):
    """Raised for other safe-to-report YouTube API failures."""


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
            scopes=[YOUTUBE_READONLY_SCOPE],
        )
    except (json.JSONDecodeError, TypeError, ValueError) as error:
        raise OAuthConfigurationError("Saved Google credentials are invalid.") from error

    if credentials.expired:
        if not credentials.refresh_token:
            raise YouTubeNotConnected
        try:
            credentials.refresh(Request())
        except RefreshError as error:
            raise YouTubeNotConnected from error
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
        reason = ""
        try:
            payload = json.loads(error.content.decode("utf-8"))
            api_errors = payload.get("error", {}).get("errors", [])
            if api_errors:
                reason = api_errors[0].get("reason", "")
        except (AttributeError, UnicodeDecodeError, json.JSONDecodeError):
            pass

        if reason in {"quotaExceeded", "dailyLimitExceeded", "rateLimitExceeded"}:
            raise YouTubeQuotaExceeded from error
        if reason in {"commentsDisabled", "forbidden", "videoNotFound", "playlistNotFound"}:
            raise YouTubeResourceUnavailable from error
        raise YouTubeAPIError("YouTube could not complete this request.") from error


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
            videos.append(
                {
                    "video_id": video_id,
                    "title": snippet.get("title", "Untitled video"),
                    "published_at": snippet.get("publishedAt"),
                    "thumbnail_url": snippet.get("thumbnails", {}).get("default", {}).get("url"),
                }
            )
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