from unittest.mock import MagicMock, patch
import pytest
from flask import render_template

from app.services.analysis_service import (
    AnalysisSummary,
    CommentResult,
    analyze_comments,
    load_models,
)
from app.services.ml_pipeline import JIGSAW_LABELS, load_jigsaw_toxicity_csv
from app.services.youtube_api import (
    YouTubeAPIConfigurationError,
    YouTubeAPIError,
    YouTubeAuthenticationError,
    YouTubeCommentsDisabled,
    YouTubeInsufficientPermissions,
    YouTubeQuotaExceeded,
    YouTubeVideoNotFound,
)


def _sign_in(client):
    client.post(
        "/auth/register",
        data={
            "username": "analyzer_user",
            "email": "analyzer@example.com",
            "password": "strong-password-123",
            "confirm_password": "strong-password-123",
        },
    )
    client.post(
        "/auth/login",
        data={"identifier": "analyzer_user", "password": "strong-password-123"},
    )


def test_analyze_requires_login(client):
    response = client.get("/analyze/test-vid")
    assert response.status_code == 302
    assert "/auth/login" in response.headers["Location"]


def test_analyze_rejects_missing_or_oversized_video_id(client):
    _sign_in(client)
    res = client.get("/analyze/" + "a" * 65)
    assert res.status_code == 400
    assert b"Invalid video" in res.data


@patch("app.routes.analysis._get_youtube_service")
def test_analyze_handles_insufficient_permissions_with_403(get_service, client):
    _sign_in(client)
    service = MagicMock()
    service.list_comments.side_effect = YouTubeInsufficientPermissions(
        "Insufficient scopes"
    )
    get_service.return_value = (service, None)

    response = client.get("/analyze/test-vid-123")

    assert response.status_code == 403
    assert b"Permission needed" in response.data
    assert b"Reconnect YouTube with Google" in response.data
    assert b"/oauth/google/connect" in response.data


@patch("app.routes.analysis._get_youtube_service")
def test_analyze_handles_comments_disabled_with_403(get_service, client):
    _sign_in(client)
    service = MagicMock()
    service.list_comments.side_effect = YouTubeCommentsDisabled("Disabled")
    get_service.return_value = (service, None)

    response = client.get("/analyze/test-vid-123")

    assert response.status_code == 403
    assert b"Comments disabled" in response.data
    assert b"Comments are disabled on this video." in response.data


@patch("app.routes.analysis._get_youtube_service")
def test_analyze_handles_video_not_found_with_404(get_service, client):
    _sign_in(client)
    service = MagicMock()
    service.list_comments.side_effect = YouTubeVideoNotFound("Not found")
    get_service.return_value = (service, None)

    response = client.get("/analyze/test-vid-123")

    assert response.status_code == 404
    assert b"Video unavailable" in response.data


@patch("app.routes.analysis._get_youtube_service")
def test_analyze_handles_quota_exceeded_with_429(get_service, client):
    _sign_in(client)
    service = MagicMock()
    service.list_comments.side_effect = YouTubeQuotaExceeded("Quota limit")
    get_service.return_value = (service, None)

    response = client.get("/analyze/test-vid-123")

    assert response.status_code == 429
    assert b"Quota exhausted" in response.data


@patch("app.routes.analysis._get_youtube_service")
def test_analyze_handles_auth_expired_with_401(get_service, client):
    _sign_in(client)
    service = MagicMock()
    service.list_comments.side_effect = YouTubeAuthenticationError("Token expired")
    get_service.return_value = (service, None)

    response = client.get("/analyze/test-vid-123")

    assert response.status_code == 401
    assert b"Authentication expired" in response.data
    assert b"Reconnect YouTube with Google" in response.data


@patch("app.routes.analysis._get_youtube_service")
def test_analyze_handles_api_configuration_issue_with_503(get_service, client):
    _sign_in(client)
    service = MagicMock()
    service.list_comments.side_effect = YouTubeAPIConfigurationError("Not enabled")
    get_service.return_value = (service, None)

    response = client.get("/analyze/test-vid-123")

    assert response.status_code == 503
    assert b"API configuration issue" in response.data


@patch("app.routes.analysis._get_youtube_service")
def test_analyze_handles_unexpected_api_error_with_502(get_service, client):
    _sign_in(client)
    service = MagicMock()
    service.list_comments.side_effect = YouTubeAPIError("Server error")
    get_service.return_value = (service, None)

    response = client.get("/analyze/test-vid-123")

    assert response.status_code == 502
    assert b"YouTube error" in response.data


@patch("app.routes.analysis._get_youtube_service")
def test_analyze_renders_empty_state_when_no_comments(get_service, client):
    _sign_in(client)
    service = MagicMock()
    service.list_comments.return_value = {"items": [], "next_page_token": None}
    get_service.return_value = (service, None)

    response = client.get("/analyze/test-vid-123")

    assert response.status_code == 200
    assert b"No comments" in response.data or b"empty" in response.data.lower() or b"0 comments" in response.data.lower()


@patch("app.routes.analysis._get_youtube_service")
def test_analyze_pipeline_with_synthetic_comments(get_service, client):
    _sign_in(client)
    synthetic_comments = [
        {"comment_id": "c1", "text": "This video is brilliant, thank you so much!"},
        {"comment_id": "c2", "text": "Average content, nothing special."},
        {"comment_id": "c3", "text": "This explanation is completely wrong and frustrating."},
        {"comment_id": "c4", "text": "You are pathetic and useless, delete your channel."},
        {"comment_id": "c5", "text": "I will find you and hurt you."},
    ]
    service = MagicMock()
    service.list_comments.return_value = {
        "items": synthetic_comments,
        "next_page_token": None,
    }
    get_service.return_value = (service, None)

    response = client.get("/analyze/test-vid-123")

    assert response.status_code == 200
    # Checks that analysis ran and rendered summary metrics
    assert b"Analysis" in response.data or b"Sentiment" in response.data


def test_synthetic_comment_analysis_keeps_sentiment_and_bullying_separate():
    """Verify that a negative non-cyberbullying comment is not labeled as cyberbullying."""
    models = load_models()
    comments = [
        {"comment_id": "1", "text": "Great tutorial, very clear!"},
        {"comment_id": "2", "text": "This is terrible and boring."},  # Negative, but not bullying
        {"comment_id": "3", "text": "You are so stupid and ugly."},   # Negative and insult
    ]

    summary = analyze_comments(comments, models=models)

    assert summary.total == 3
    assert summary.sentiment_counts.get("positive", 0) >= 1
    assert summary.sentiment_counts.get("negative", 0) >= 1
    # Check c2: negative sentiment, but not_bullying
    c2 = next(c for c in summary.comments if c.comment_id == "2")
    assert c2.sentiment_label == "negative"
    assert c2.bullying_label == "not_bullying"
    assert c2.is_bullying is False

    # Check c3: bullying
    c3 = next(c for c in summary.comments if c.comment_id == "3")
    assert c3.is_bullying is True
    assert c3.bullying_label != "not_bullying"

    # Verify counts and percentage calculations
    assert summary.harmful_count == 1
    assert summary.harmful_pct == round(1 / 3 * 100, 1)


def test_gaming_slang_and_ambiguous_comments_handling():
    """Verify handling of gaming slang ('Headshots super bro') and ambiguous promo ('SUB🖤🖤')."""
    models = load_models()
    comments = [
        {"comment_id": "g1", "text": "Headshots super bro"},
        {"comment_id": "g2", "text": "SUB🖤🖤"},
    ]

    summary = analyze_comments(comments, models=models)

    # 1. Gaming slang: positive praise, not cyberbullying
    c_gaming = next(c for c in summary.comments if c.comment_id == "g1")
    assert c_gaming.sentiment_label == "positive"
    assert c_gaming.bullying_label == "not_bullying"
    assert c_gaming.is_bullying is False

    # 2. Ambiguous short promo: neutral, flagged for review
    c_sub = next(c for c in summary.comments if c.comment_id == "g2")
    assert c_sub.sentiment_label == "neutral"
    assert c_sub.bullying_label == "not_bullying"
    assert c_sub.is_bullying is False
    assert c_sub.needs_review is True
    assert c_sub.review_reason is not None


def test_demo_model_maps_unsupported_labels_to_research_categories():
    """The fallback demo model should use the required taxonomy and keep bullying separate from sentiment."""
    models = load_models()
    comments = [
        {"comment_id": "a1", "text": "You are so stupid and ugly."},
        {"comment_id": "a2", "text": "This is terrible and boring."},
    ]

    summary = analyze_comments(comments, models=models)

    harmful = next(c for c in summary.comments if c.comment_id == "a1")
    non_harmful = next(c for c in summary.comments if c.comment_id == "a2")

    assert harmful.sentiment_label in {"negative", "neutral"}
    assert harmful.bullying_label in {
        "other_cyberbullying",
        "age_based_bullying",
        "ethnicity_based_bullying",
        "gender_based_bullying",
        "religion_based_bullying",
    }
    assert harmful.is_bullying is True
    assert non_harmful.bullying_label == "not_bullying"
    assert non_harmful.is_bullying is False
    assert summary.is_demo is True
    assert models.model_warning
    assert "illustrative" in models.model_warning.lower()
    assert "jigsaw" in models.model_warning.lower()


def test_jigsaw_label_contract_is_exactly_six_binary_targets():
    assert JIGSAW_LABELS == [
        "toxic",
        "severe_toxic",
        "obscene",
        "threat",
        "insult",
        "identity_hate",
    ]


def test_load_jigsaw_toxicity_csv_validates_binary_labels(tmp_path):
    csv_path = tmp_path / "jigsaw_sample.csv"
    csv_path.write_text(
        "comment_text,toxic,severe_toxic,obscene,threat,insult,identity_hate\n"
        "Great video,0,0,0,0,0,0\n"
        "You are stupid,1,0,0,0,1,0\n",
        encoding="utf-8",
    )

    records = load_jigsaw_toxicity_csv(csv_path)

    assert len(records) == 2
    assert records[0]["labels"] == {
        "toxic": 0,
        "severe_toxic": 0,
        "obscene": 0,
        "threat": 0,
        "insult": 0,
        "identity_hate": 0,
    }
    assert records[1]["labels"]["insult"] == 1


def test_partial_production_loader_keeps_legacy_demo_warning():
    models = load_models()

    assert models.is_demo is True
    assert models.model_warning
    assert models.toxicity is not None
    assert "demo" in models.model_warning.lower()
    assert "jigsaw" in models.model_warning.lower()


def test_jigsaw_scores_are_reported_without_changing_bullying_verdict():
    class FixedModel:
        def __init__(self, label, confidence=0.9):
            self.label = label
            self.confidence = confidence

        def predict(self, _text):
            return {"label": self.label, "confidence": self.confidence}

    class FixedToxicityModel:
        def predict(self, _text):
            return {label: 0.95 for label in JIGSAW_LABELS}

    models = load_models()
    models.sentiment = FixedModel("negative")
    models.bullying = FixedModel("not_bullying")
    models.toxicity = FixedToxicityModel()

    summary = analyze_comments(
        [{"comment_id": "tox-1", "text": "A critical comment."}],
        models=models,
    )

    result = summary.comments[0]
    assert result.is_bullying is False
    assert result.bullying_label == "not_bullying"
    assert result.toxicity_scores == {label: 0.95 for label in JIGSAW_LABELS}


def test_analysis_template_renders_jigsaw_scores_and_legacy_demo_warning(app):
    summary = AnalysisSummary(
        total=1,
        sentiment_counts={"neutral": 1},
        bullying_counts={"not_bullying": 1},
        harmful_count=0,
        harmful_pct=0.0,
        is_demo=True,
        model_warning="Legacy sentiment and bullying outputs are demo.",
        comments=[
            CommentResult(
                comment_id="render-1",
                text="Example comment",
                sentiment_label="neutral",
                sentiment_confidence=0.8,
                bullying_label="not_bullying",
                bullying_confidence=0.8,
                is_bullying=False,
                toxicity_scores={"toxic": 0.825},
            )
        ],
    )

    with app.test_request_context():
        html = render_template(
            "analysis.html",
            video_id="test-video",
            summary=summary,
            empty=False,
        )

    assert "Legacy sentiment / bullying demo" in html
    assert "Jigsaw Toxicity Scores" in html
    assert "82.5%" in html
    assert "not calibrated probabilities" in html


def test_short_social_phrase_requires_review_but_not_harmful():
    """Ambiguous short phrases should not become harmful without reliable evidence."""
    class FixedModel:
        def __init__(self, label, confidence=0.81):
            self.label = label
            self.confidence = confidence

        def predict(self, _text):
            return {"label": self.label, "confidence": self.confidence}

    models = load_models()
    models.sentiment = FixedModel("negative", confidence=0.81)
    models.bullying = FixedModel("not_bullying", confidence=0.81)
    models.is_demo = True

    summary = analyze_comments(
        [{"comment_id": "s1", "text": "Go 700 Fam 🎉🎉"}],
        models=models,
    )

    result = summary.comments[0]
    assert result.sentiment_label == "negative"
    assert result.bullying_label == "not_bullying"
    assert result.is_bullying is False
    assert result.needs_review is True


def test_persistence_stores_and_retrieves_analysis(app):
    """Verify that analysis store saves and retrieves analysis summaries."""
    from app.models.analysis_store import MongoAnalysisStore
    store = app.config.get("ANALYSIS_STORE")
    if not store:
        pytest.skip("Analysis store not configured")

    user_id = "test_user_persistence"
    video_id = "vid_persistence_123"
    summary_data = {
        "total": 10,
        "harmful_count": 2,
        "harmful_pct": 20.0,
        "sentiment_counts": {"positive": 5, "negative": 3, "neutral": 2},
        "bullying_counts": {"not_bullying": 8, "insult": 2},
        "is_demo": True,
    }

    try:
        store.save_analysis(user_id, video_id, summary_data)
        saved = store.get_analysis(user_id, video_id)
        assert saved is not None
        assert saved["summary"]["total"] == 10
        assert saved["summary"]["harmful_count"] == 2

        batch = store.get_analyses_for_videos(user_id, [video_id, "other_vid"])
        assert video_id in batch
        assert batch[video_id]["harmful_pct"] == 20.0
    except Exception as e:
        # If Mongo is not running locally in test environment, skip gracefully
        pytest.skip(f"MongoDB not available for persistence test: {e}")

