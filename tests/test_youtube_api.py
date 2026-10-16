from unittest.mock import MagicMock

import pytest
from googleapiclient.errors import HttpError
from httplib2 import Response

from app.services.youtube_api import (
    YouTubeDataService,
    YouTubeQuotaExceeded,
    YouTubeResourceUnavailable,
    _execute,
)


def test_list_videos_follows_uploads_playlist_and_page_token():
    api = MagicMock()
    api.channels.return_value.list.return_value.execute.return_value = {
        "items": [
            {"contentDetails": {"relatedPlaylists": {"uploads": "uploads-123"}}}
        ]
    }
    api.playlistItems.return_value.list.return_value.execute.return_value = {
        "items": [
            {
                "snippet": {
                    "title": "Demo video",
                    "publishedAt": "2026-01-01T00:00:00Z",
                    "thumbnails": {"default": {"url": "https://example.test/thumb.jpg"}},
                },
                "contentDetails": {"videoId": "video-123"},
            }
        ],
        "nextPageToken": "next-videos",
    }

    result = YouTubeDataService(api).list_videos("previous-videos", 12)

    assert result["items"][0]["video_id"] == "video-123"
    assert result["next_page_token"] == "next-videos"
    api.channels.return_value.list.assert_called_once_with(
        part="contentDetails", mine=True, maxResults=1
    )
    api.playlistItems.return_value.list.assert_called_once_with(
        part="snippet,contentDetails",
        playlistId="uploads-123",
        maxResults=12,
        pageToken="previous-videos",
    )


def test_list_comments_and_replies_use_official_comment_endpoints():
    api = MagicMock()
    api.commentThreads.return_value.list.return_value.execute.return_value = {
        "items": [
            {
                "id": "thread-1",
                "snippet": {
                    "totalReplyCount": 2,
                    "topLevelComment": {
                        "id": "comment-1",
                        "snippet": {
                            "textDisplay": "A top-level comment",
                            "publishedAt": "2026-01-01T00:00:00Z",
                        },
                    },
                },
            }
        ],
        "nextPageToken": "next-comments",
    }
    api.comments.return_value.list.return_value.execute.return_value = {
        "items": [{"id": "reply-1", "snippet": {"textDisplay": "A reply"}}]
    }
    service = YouTubeDataService(api)

    comments = service.list_comments("video-1", "older", 50)
    replies = service.list_replies("comment-1", max_results=20)

    assert comments["items"][0]["text"] == "A top-level comment"
    assert comments["items"][0]["reply_count"] == 2
    assert comments["next_page_token"] == "next-comments"
    assert replies["items"][0]["text"] == "A reply"
    api.commentThreads.return_value.list.assert_called_once_with(
        part="snippet",
        videoId="video-1",
        maxResults=50,
        textFormat="plainText",
        pageToken="older",
    )
    api.comments.return_value.list.assert_called_once_with(
        part="snippet",
        parentId="comment-1",
        maxResults=20,
        textFormat="plainText",
    )


from app.services.google_oauth import YOUTUBE_FORCE_SSL_SCOPE, YOUTUBE_READONLY_SCOPE, YOUTUBE_SCOPES
from app.services.youtube_api import (
    YouTubeAPIConfigurationError,
    YouTubeAPIError,
    YouTubeAuthenticationError,
    YouTubeCommentsDisabled,
    YouTubeDataService,
    YouTubeInsufficientPermissions,
    YouTubeQuotaExceeded,
    YouTubeResourceUnavailable,
    YouTubeVideoNotFound,
    _execute,
)


def test_scopes_include_readonly_and_force_ssl():
    assert YOUTUBE_READONLY_SCOPE in YOUTUBE_SCOPES
    assert YOUTUBE_FORCE_SSL_SCOPE in YOUTUBE_SCOPES
    assert YOUTUBE_FORCE_SSL_SCOPE == "https://www.googleapis.com/auth/youtube.force-ssl"


@pytest.mark.parametrize(
    ("status", "reason", "exception"),
    [
        ("403", "quotaExceeded", YouTubeQuotaExceeded),
        ("429", "rateLimitExceeded", YouTubeQuotaExceeded),
        ("403", "commentsDisabled", YouTubeCommentsDisabled),
        ("403", "commentsDisabled", YouTubeResourceUnavailable),
        ("403", "insufficientPermissions", YouTubeInsufficientPermissions),
        ("403", "accessNotConfigured", YouTubeAPIConfigurationError),
        ("401", "authError", YouTubeAuthenticationError),
        ("404", "videoNotFound", YouTubeVideoNotFound),
        ("404", "videoNotFound", YouTubeResourceUnavailable),
        ("403", "forbidden", YouTubeResourceUnavailable),
        ("500", "backendError", YouTubeAPIError),
    ],
)
def test_api_error_reasons_become_safe_domain_errors(status, reason, exception):
    response = Response({"status": status})
    response.reason = "Error"
    content = f'{{"error":{{"errors":[{{"reason":"{reason}"}}]}}}}'.encode()
    request = MagicMock()
    request.execute.side_effect = HttpError(response, content)

    with pytest.raises(exception):
        _execute(request)


def test_insufficient_scopes_in_error_message_raises_permission_error():
    response = Response({"status": "403"})
    response.reason = "Forbidden"
    content = b'{"error":{"message":"Request had insufficient authentication scopes.","errors":[{"message":"Insufficient Permission","reason":"insufficientPermissions"}]}}'
    request = MagicMock()
    request.execute.side_effect = HttpError(response, content)

    with pytest.raises(YouTubeInsufficientPermissions):
        _execute(request)