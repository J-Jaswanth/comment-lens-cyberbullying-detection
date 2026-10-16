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
import json
import re
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
_TOXICITY_MODEL_PATH = _MODELS_DIR / "jigsaw_toxicity_model.joblib"
_TOXICITY_LABELS_PATH = _MODELS_DIR / "jigsaw_toxicity_labels.json"

NON_BULLYING_LABEL = "not_bullying"
REQUIRED_BULLYING_LABELS = {
    "not_bullying",
    "other_cyberbullying",
    "age_based_bullying",
    "ethnicity_based_bullying",
    "gender_based_bullying",
    "religion_based_bullying",
}

_LABEL_ALIASES = {
    "not_bullying": "not_bullying",
    "other": "other_cyberbullying",
    "other_cyberbullying": "other_cyberbullying",
    "insult": "other_cyberbullying",
    "threat": "other_cyberbullying",
    "harassment": "other_cyberbullying",
    "age": "age_based_bullying",
    "age_based_bullying": "age_based_bullying",
    "ethnicity": "ethnicity_based_bullying",
    "ethnicity_based_bullying": "ethnicity_based_bullying",
    "gender": "gender_based_bullying",
    "gender_based_bullying": "gender_based_bullying",
    "religion": "religion_based_bullying",
    "religion_based_bullying": "religion_based_bullying",
}


def normalize_bullying_label(label):
    """Map ad hoc demo labels to the supported research taxonomy."""
    normalized = str(label or "").strip().casefold().replace(" ", "_")
    if not normalized:
        return NON_BULLYING_LABEL
    return _LABEL_ALIASES.get(normalized, "other_cyberbullying")


def _short_ambiguous_text(text):
    words = re.findall(r"\b[\w']+\b", text.casefold())
    if len(words) > 5:
        return False
    social_tokens = {"sub", "subscribe", "bro", "gg", "clutch", "headshot", "headshots", "follow", "like"}
    return bool(words) and set(words).intersection(social_tokens)


# ---------------------------------------------------------------------------
# Synthetic corpus for the demo model.
# NOT training evidence – used only to make the pipeline runnable during
# development and software tests when no real dataset is available.
# ---------------------------------------------------------------------------
_DEMO_CORPUS = [
    # Positive sentiment, safe (gaming slang, general praise, emoji usage)
    ("Great video, loved it!", "positive", "not_bullying"),
    ("Great tutorial, very clear!", "positive", "not_bullying"),
    ("This is brilliant and very clear!", "positive", "not_bullying"),
    ("Amazing content, keep it up!", "positive", "not_bullying"),
    ("This is wonderful, thank you!", "positive", "not_bullying"),
    ("Excellent work, very helpful.", "positive", "not_bullying"),
    ("I really enjoyed watching this.", "positive", "not_bullying"),
    ("Fantastic explanation, subscribed!", "positive", "not_bullying"),
    ("This made my day, brilliant.", "positive", "not_bullying"),
    ("Outstanding quality content here.", "positive", "not_bullying"),
    ("Headshots super bro", "positive", "not_bullying"),
    ("Super bro keep going with these headshots", "positive", "not_bullying"),
    ("Insane headshot super bro clutch!", "positive", "not_bullying"),
    ("Great clutch and nice aim bro", "positive", "not_bullying"),
    ("Awesome gameplay, well played gg", "positive", "not_bullying"),
    ("Loved the video and gameplay bro 🔥", "positive", "not_bullying"),
    ("Super bro keep it up 👍", "positive", "not_bullying"),

    # Neutral sentiment, safe (inquiries, subscriptions, casual commentary)
    ("Okay video, nothing special.", "neutral", "not_bullying"),
    ("Fine, I guess.", "neutral", "not_bullying"),
    ("It was alright.", "neutral", "not_bullying"),
    ("Average content, not bad.", "neutral", "not_bullying"),
    ("Decent effort, could improve.", "neutral", "not_bullying"),
    ("Not great, not terrible either.", "neutral", "not_bullying"),
    ("So so, maybe next time better.", "neutral", "not_bullying"),
    ("Middle of the road content.", "neutral", "not_bullying"),
    ("What song did you use at 2:15?", "neutral", "not_bullying"),
    ("Can you make a tutorial on this?", "neutral", "not_bullying"),
    ("SUB🖤🖤", "neutral", "not_bullying"),
    ("sub for sub", "neutral", "not_bullying"),
    ("sub to my channel please", "neutral", "not_bullying"),
    ("Check out my channel too", "neutral", "not_bullying"),
    ("Timestamp at 03:45", "neutral", "not_bullying"),

    # Negative sentiment, safe (content criticism without cyberbullying)
    ("This is terrible and boring.", "negative", "not_bullying"),
    ("Worst video I have seen today.", "negative", "not_bullying"),
    ("Disappointing, waste of time.", "negative", "not_bullying"),
    ("Very poor explanation here.", "negative", "not_bullying"),
    ("I disagree with this method completely.", "negative", "not_bullying"),
    ("The audio quality in this clip is bad.", "negative", "not_bullying"),
    ("Bad strategy in that round.", "negative", "not_bullying"),
    ("Not a fan of this new format.", "negative", "not_bullying"),

    # Other cyberbullying
    ("You are so stupid and ugly.", "negative", "other_cyberbullying"),
    ("What an idiot, go away forever.", "negative", "other_cyberbullying"),
    ("Nobody likes you, loser.", "negative", "other_cyberbullying"),
    ("You are worthless and pathetic.", "negative", "other_cyberbullying"),
    ("Brainless clown stop talking.", "negative", "other_cyberbullying"),

    # Other cyberbullying: threat-based harassment
    ("I will find you and hurt you.", "negative", "other_cyberbullying"),
    ("Watch out, you will regret this.", "negative", "other_cyberbullying"),
    ("I am going to beat you up.", "negative", "other_cyberbullying"),
    ("I know where you live and you are dead.", "negative", "other_cyberbullying"),

    # Other cyberbullying: degradation and exclusion
    ("Stop making videos, you are disgusting.", "negative", "other_cyberbullying"),
    ("You should delete your channel, freak.", "negative", "other_cyberbullying"),
    ("Everyone hates your content, quit.", "negative", "other_cyberbullying"),
    ("Go die in a hole, nobody cares.", "negative", "other_cyberbullying"),
    ("We are going to mass report you until you get banned.", "negative", "other_cyberbullying"),

    # Age-based bullying
    ("You are too old to be on YouTube boomer get off.", "negative", "age_based_bullying"),
    ("Stupid toddler kid should not have internet access.", "negative", "age_based_bullying"),
    ("Go back to retirement home grandpa.", "negative", "age_based_bullying"),

    # Ethnicity-based bullying
    ("Go back to your country you do not belong here.", "negative", "ethnicity_based_bullying"),
    ("Filthy immigrant nobody wants your kind here.", "negative", "ethnicity_based_bullying"),
    ("Hate people of your race so much.", "negative", "ethnicity_based_bullying"),

    # Gender-based bullying
    ("Women should stay in kitchen stop gaming.", "negative", "gender_based_bullying"),
    ("Fake gamer girl just seeking attention.", "negative", "gender_based_bullying"),
    ("Dumb girl go learn how to play.", "negative", "gender_based_bullying"),

    # Religion-based bullying
    ("Your religious beliefs are wicked and evil.", "negative", "religion_based_bullying"),
    ("All people of your religion are terrorists.", "negative", "religion_based_bullying"),
    ("Disgusting cult followers should be expelled.", "negative", "religion_based_bullying"),
]

# Module-level singletons so the demo model is only trained once per process.
_demo_sentiment_model = None
_demo_bullying_model = None


def _build_demo_models():
    """Fit demo classifiers on the synthetic corpus (lazy, once per process)."""
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
                "DEMO MODEL – trained on synthetic examples. "
                "Predictions are illustrative only and must not be "
                "interpreted as calibrated real-world accuracy."
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
                "DEMO MODEL – trained on synthetic examples. "
                "Predictions are illustrative only and must not be "
                "interpreted as calibrated real-world accuracy."
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
    model_warning: str = ""
    toxicity: object = None
    toxicity_labels: list = field(default_factory=list)


def load_models() -> LoadedModels:
    """
    Return production models when both joblib files exist in ``models/``,
    otherwise return the synthetic demo model pair.
    """
    import joblib

    toxicity_model = None
    toxicity_labels = []
    sentiment = None
    bullying = None

    try:
        if _SENTIMENT_MODEL_PATH.exists():
            sentiment = joblib.load(_SENTIMENT_MODEL_PATH)
        if _BULLYING_MODEL_PATH.exists():
            bullying = joblib.load(_BULLYING_MODEL_PATH)
        if _TOXICITY_MODEL_PATH.exists():
            from app.services.ml_pipeline import JIGSAW_LABELS, JigsawToxicityModel

            toxicity_pipeline = joblib.load(_TOXICITY_MODEL_PATH)
            toxicity_labels = (
                json.loads(_TOXICITY_LABELS_PATH.read_text(encoding="utf-8"))
                if _TOXICITY_LABELS_PATH.exists()
                else JIGSAW_LABELS
            )
            toxicity_metadata_path = _MODELS_DIR / "jigsaw_toxicity_metadata.json"
            toxicity_evaluation = (
                json.loads(toxicity_metadata_path.read_text(encoding="utf-8"))
                if toxicity_metadata_path.exists()
                else {}
            )
            toxicity_model = JigsawToxicityModel(
                pipeline=toxicity_pipeline,
                labels=toxicity_labels,
                evaluation=toxicity_evaluation,
            )
        if sentiment is not None and bullying is not None:
            logger.info("Loaded production models from %s", _MODELS_DIR)
            return LoadedModels(
                sentiment=sentiment,
                bullying=bullying,
                is_demo=False,
                model_warning="",
                toxicity=toxicity_model,
                toxicity_labels=toxicity_labels,
            )
    except Exception:
        logger.exception(
            "Could not load production models; falling back to demo model."
        )

    _build_demo_models()
    return LoadedModels(
        sentiment=_demo_sentiment_model,
        bullying=_demo_bullying_model,
        is_demo=True,
        model_warning=(
            "Illustrative fallback model only for sentiment and bullying: the legacy classifier "
            "pair is still demo, while a trained Jigsaw toxicity model may be available separately "
            "for evaluation."
        ),
        toxicity=toxicity_model,
        toxicity_labels=toxicity_labels,
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
    toxicity_scores: dict = field(default_factory=dict)
    needs_review: bool = False
    review_reason: Optional[str] = None


@dataclass
class AnalysisSummary:
    total: int
    sentiment_counts: dict
    bullying_counts: dict
    harmful_count: int
    harmful_pct: float
    is_demo: bool
    model_warning: str = ""
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
    import re

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

        sentiment_label = str(sent.get("label", "neutral")).strip().casefold()
        bullying_label = normalize_bullying_label(bull.get("label"))
        sentiment_confidence = float(sent.get("confidence", 0.0))
        bullying_confidence = float(bull.get("confidence", 0.0))
        is_bullying = bullying_label != NON_BULLYING_LABEL
        toxicity_scores = (
            models.toxicity.predict(text)
            if models.toxicity is not None
            else {}
        )

        # Uncertainty & review heuristics. Keep sentiment and cyberbullying as
        # separate predictions; a negative sentiment score alone does not mean
        # bullying, and short social-media phrases often need human review.
        needs_review = False
        review_reason = None

        cleaned_text = text.strip()
        normalized_lower = cleaned_text.casefold()
        words = re.findall(r"\b\w+\b", normalized_lower)

        if _short_ambiguous_text(cleaned_text):
            needs_review = True
            review_reason = "Short or ambiguous social-media phrase"
        elif sentiment_confidence < 0.60 or (is_bullying and bullying_confidence < 0.60):
            needs_review = True
            min_conf = round(min(sentiment_confidence, bullying_confidence) * 100)
            review_reason = f"Low model confidence ({min_conf}%)"

        if models.is_demo and len(words) <= 5 and not is_bullying:
            social_tokens = {
                "bro",
                "gg",
                "fam",
                "sub",
                "subscribe",
                "headshot",
                "clutch",
                "support",
                "love",
                "like",
            }
            if set(words).intersection(social_tokens):
                needs_review = True
                review_reason = "Short social-media phrase in demo mode requires human review"

        results.append(
            CommentResult(
                comment_id=comment_id,
                text=text,
                sentiment_label=sentiment_label,
                sentiment_confidence=round(sentiment_confidence, 4),
                bullying_label=bullying_label,
                bullying_confidence=round(bullying_confidence, 4),
                is_bullying=is_bullying,
                toxicity_scores=toxicity_scores,
                needs_review=needs_review,
                review_reason=review_reason,
            )
        )
        sentiment_counts[sentiment_label] = sentiment_counts.get(sentiment_label, 0) + 1
        bullying_counts[bullying_label] = bullying_counts.get(bullying_label, 0) + 1
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
        model_warning=getattr(models, "model_warning", ""),
        comments=results,
    )
