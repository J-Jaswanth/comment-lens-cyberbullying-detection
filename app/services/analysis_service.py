"""
Comment analysis service.

Loads production models from the ``models/`` directory when they exist.
If neither model is found, a clearly-labelled DEMO model is trained at
import-time on a small synthetic corpus so the pipeline can be exercised
end-to-end.  All predictions carry an ``is_demo`` flag so the UI can
display an appropriate warning.

To train real production models, provide a labeled CSV and run::

    .venv\\Scripts\\python.exe -m app.services.ml_pipeline \\
        data/private/labeled_comments.csv --output-dir models
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_MODELS_DIR = Path(__file__).parent.parent.parent / "models"
_SENTIMENT_MODEL_PATH = _MODELS_DIR / "sentiment_model.joblib"
_BULLYING_MODEL_PATH = _MODELS_DIR / "bullying_category_model.joblib"

NON_BULLYING_LABEL = "not_bullying"

# ---------------------------------------------------------------------------
# Synthetic corpus for the demo model (30 hand-labeled examples).
# NOT training evidence – used only to make the pipeline runnable during
# development and software tests when no real dataset is available.
# ---------------------------------------------------------------------------
_DEMO_CORPUS = [
    # (text, sentiment, bullying_category)
    ("Great video, loved it!", "positive", "not_bullying"),
    ("Amazing content, keep it up!", "positive", "not_bullying"),
    ("This is wonderful, thank you!", "positive", "not_bullying"),
    ("Excellent work, very helpful.", "positive", "not_bullying"),
    ("I really enjoyed watching this.", "positive", "not_bullying"),
    ("Fantastic explanation, subscribed!", "positive", "not_bullying"),
    ("This made my day, brilliant.", "positive", "not_bullying"),
    ("Outstanding quality content here.", "positive", "not_bullying"),
    ("Okay video, nothing special.", "neutral", "not_bullying"),
    ("Fine, I guess.", "neutral", "not_bullying"),
    ("It was alright.", "neutral", "not_bullying"),
    ("Average content, not bad.", "neutral", "not_bullying"),
    ("Decent effort, could improve.", "neutral", "not_bullying"),
    ("Not great, not terrible either.", "neutral", "not_bullying"),
    ("So so, maybe next time better.", "neutral", "not_bullying"),
    ("Middle of the road content.", "neutral", "not_bullying"),
    ("This is terrible and boring.", "negative", "not_bullying"),
    ("Worst video I have seen today.", "negative", "not_bullying"),
    ("Disappointing, waste of time.", "negative", "not_bullying"),
    ("Very poor explanation here.", "negative", "not_bullying"),
    ("You are so stupid and ugly.", "negative", "insult"),
    ("What an idiot, go away forever.", "negative", "insult"),
    ("Nobody likes you, loser.", "negative", "insult"),
    ("You are worthless and pathetic.", "negative", "insult"),
    ("I will find you and hurt you.", "negative", "threat"),
    ("Watch out, you will regret this.", "negative", "threat"),
    ("Stop making videos, you are disgusting.", "negative", "harassment"),
    ("You should delete your channel, freak.", "negative", "harassment"),
    ("Everyone hates your content, quit.", "negative", "harassment"),
    ("Go die in a hole, nobody cares.", "negative", "harassment"),
]

# Module-level singletons so the demo model is only trained once per process.
_demo_sentiment_model = None
_demo_bullying_model = None


def _build_demo_models():
    """Fit tiny demo classifiers on the synthetic corpus (lazy, once per process)."""
    global _demo_sentiment_model, _demo_bullying_model
    if _demo_sentiment_model is not None:
        return

    from app.services.ml_pipeline import TextClassifier, _make_pipeline

    texts = [row[0] for row in _DEMO_CORPUS]
    sentiment_labels = [row[1] for row in _DEMO_CORPUS]
    bullying_labels = [row[2] for row in _DEMO_CORPUS]

    sent_pipe = _make_pipeline()
    sent_pipe.fit(texts, sentiment_labels)
    _demo_sentiment_model = TextClassifier(
        pipeline=sent_pipe,
        target_column="sentiment",
        labels=sorted(set(sentiment_labels)),
        evaluation={
            "note": (
                "DEMO MODEL – trained on 30 synthetic examples. "
                "Predictions are illustrative only and must not be "
                "interpreted as accurate or reliable."
            )
        },
    )

    bull_pipe = _make_pipeline()
    bull_pipe.fit(texts, bullying_labels)
    _demo_bullying_model = TextClassifier(
        pipeline=bull_pipe,
        target_column="bullying_category",
        labels=sorted(set(bullying_labels)),
        evaluation={
            "note": (
                "DEMO MODEL – trained on 30 synthetic examples. "
                "Predictions are illustrative only and must not be "
                "interpreted as accurate or reliable."
            )
        },
    )


# ---------------------------------------------------------------------------
# Public model loader
# ---------------------------------------------------------------------------

@dataclass
class LoadedModels:
    sentiment: object
    bullying: object
    is_demo: bool


def load_models() -> LoadedModels:
    """
    Return production models when both joblib files exist in ``models/``,
    otherwise return the synthetic demo model pair.
    """
    import joblib

    if _SENTIMENT_MODEL_PATH.exists() and _BULLYING_MODEL_PATH.exists():
        try:
            sentiment = joblib.load(_SENTIMENT_MODEL_PATH)
            bullying = joblib.load(_BULLYING_MODEL_PATH)
            logger.info("Loaded production models from %s", _MODELS_DIR)
            return LoadedModels(sentiment=sentiment, bullying=bullying, is_demo=False)
        except Exception:
            logger.exception(
                "Could not load production models; falling back to demo model."
            )

    _build_demo_models()
    return LoadedModels(
        sentiment=_demo_sentiment_model,
        bullying=_demo_bullying_model,
        is_demo=True,
    )


# ---------------------------------------------------------------------------
# Per-comment result and aggregate summary
# ---------------------------------------------------------------------------

@dataclass
class CommentResult:
    comment_id: str
    text: str
    sentiment_label: str
    sentiment_confidence: float
    bullying_label: str
    bullying_confidence: float
    is_bullying: bool


@dataclass
class AnalysisSummary:
    total: int
    sentiment_counts: dict
    bullying_counts: dict
    harmful_count: int
    harmful_pct: float
    is_demo: bool
    comments: list = field(default_factory=list)


def analyze_comments(
    comment_items: list,
    models: Optional[LoadedModels] = None,
) -> AnalysisSummary:
    """
    Run sentiment and cyberbullying predictions on a list of comment dicts.

    Each dict must contain at least ``text`` and ``comment_id`` keys (the
    shape returned by :class:`app.services.youtube_api.YouTubeDataService`).
    Empty or whitespace-only texts are skipped.

    Returns an :class:`AnalysisSummary` with per-comment predictions and
    aggregate counts.  Never fabricates comments, scores, or statistics.
    """
    if models is None:
        models = load_models()

    results: list[CommentResult] = []
    sentiment_counts: dict = {}
    bullying_counts: dict = {}
    harmful_count = 0

    for item in comment_items:
        text = (item.get("text") or "").strip()
        comment_id = item.get("comment_id") or ""
        if not text:
            continue

        sent = models.sentiment.predict(text)
        bull = models.bullying.predict(text)
        is_bullying = bull["label"] != NON_BULLYING_LABEL

        results.append(
            CommentResult(
                comment_id=comment_id,
                text=text,
                sentiment_label=sent["label"],
                sentiment_confidence=round(sent["confidence"], 4),
                bullying_label=bull["label"],
                bullying_confidence=round(bull["confidence"], 4),
                is_bullying=is_bullying,
            )
        )
        sentiment_counts[sent["label"]] = sentiment_counts.get(sent["label"], 0) + 1
        bullying_counts[bull["label"]] = bullying_counts.get(bull["label"], 0) + 1
        if is_bullying:
            harmful_count += 1

    total = len(results)
    harmful_pct = round(harmful_count / total * 100, 1) if total else 0.0

    return AnalysisSummary(
        total=total,
        sentiment_counts=sentiment_counts,
        bullying_counts=bullying_counts,
        harmful_count=harmful_count,
        harmful_pct=harmful_pct,
        is_demo=models.is_demo,
        comments=results,
    )
