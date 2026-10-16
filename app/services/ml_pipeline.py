"""Data-driven text models for sentiment and cyberbullying classification."""

import argparse
import csv
import html
import json
import re
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline

try:
    from iterative_stratification import MultilabelStratifiedShuffleSplit
except ImportError:  # pragma: no cover - optional dependency fallback
    MultilabelStratifiedShuffleSplit = None

SENTIMENT_LABELS = {"positive", "negative", "neutral"}
CYBERBULLYING_TARGET = "bullying_category"
NON_BULLYING_LABEL = "not_bullying"
REQUIRED_COLUMNS = {"text", "sentiment", CYBERBULLYING_TARGET, "group_id"}
JIGSAW_LABELS = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
]
JIGSAW_REQUIRED_COLUMNS = {"comment_text"}.union(JIGSAW_LABELS)


class DatasetValidationError(ValueError):
    """Raised when labeled comments do not meet the documented data contract."""


@dataclass
class TextClassifier:
    pipeline: Pipeline
    target_column: str
    labels: list[str]
    evaluation: dict

    def predict(self, text):
        processed_text = preprocess_comment(text)
        probabilities = self.pipeline.predict_proba([processed_text])[0]
        class_probabilities = zip(self.pipeline.classes_, probabilities)
        label, confidence = max(class_probabilities, key=lambda item: item[1])
        return {"label": str(label), "confidence": float(confidence)}


@dataclass
class JigsawToxicityModel:
    pipeline: Pipeline
    labels: list[str]
    evaluation: dict

    def predict(self, text):
        processed_text = preprocess_comment(text)
        output_probabilities = self.pipeline.predict_proba([processed_text])
        estimators = self.pipeline.named_steps["classifier"].estimators_
        scores = {}
        for label, estimator, probabilities in zip(
            self.labels, estimators, output_probabilities
        ):
            classes = list(estimator.classes_)
            positive_index = classes.index(1) if 1 in classes else None
            scores[label] = (
                float(probabilities[0][positive_index])
                if positive_index is not None
                else 0.0
            )
        return scores


EMOJI_MAP = {
    "🖤": " emoji_heart ",
    "❤️": " emoji_heart ",
    "💖": " emoji_heart ",
    "💕": " emoji_heart ",
    "💓": " emoji_heart ",
    "💗": " emoji_heart ",
    "💘": " emoji_heart ",
    "💙": " emoji_heart ",
    "💚": " emoji_heart ",
    "💛": " emoji_heart ",
    "💜": " emoji_heart ",
    "🤍": " emoji_heart ",
    "🤎": " emoji_heart ",
    "🔥": " emoji_fire ",
    "👍": " emoji_thumbsup ",
    "👏": " emoji_clap ",
    "😂": " emoji_laugh ",
    "🤣": " emoji_laugh ",
    "😹": " emoji_laugh ",
    "💀": " emoji_skull ",
    "☠️": " emoji_skull ",
    "💯": " emoji_hundred ",
    "😍": " emoji_love ",
    "🥰": " emoji_love ",
    "😘": " emoji_love ",
    "😊": " emoji_smile ",
    "😀": " emoji_smile ",
    "😃": " emoji_smile ",
    "😄": " emoji_smile ",
    "😁": " emoji_smile ",
    "😎": " emoji_cool ",
    "😢": " emoji_sad ",
    "😭": " emoji_sad ",
    "😞": " emoji_sad ",
    "😔": " emoji_sad ",
    "🙁": " emoji_sad ",
    "☹️": " emoji_sad ",
    "😡": " emoji_angry ",
    "😠": " emoji_angry ",
    "🤬": " emoji_angry ",
    "🤮": " emoji_disgust ",
    "🤢": " emoji_disgust ",
    "🖕": " emoji_middlefinger ",
    "💩": " emoji_poop ",
    "🤡": " emoji_clown ",
    "👑": " emoji_crown ",
    "🎯": " emoji_target ",
}


def preprocess_comment(text):
    """Normalize surface noise while retaining words, emojis, and punctuation cues."""
    text = html.unescape(text or "")
    text = re.sub(r"<[^>]*>", " ", text)
    text = re.sub(r"https?://\S+|www\.\S+", " URL ", text, flags=re.IGNORECASE)
    text = re.sub(r"(?<!\w)@[\w.-]+", " USER ", text)
    for char, replacement in EMOJI_MAP.items():
        if char in text:
            text = text.replace(char, replacement)
    text = re.sub(r"[\ufe0e\ufe0f\U0001f3fb-\U0001f3ff]", "", text)
    return re.sub(r"\s+", " ", text).strip().casefold()


def validate_records(records):
    if not records:
        raise DatasetValidationError("The dataset contains no comment rows.")

    cleaned = []
    for row_number, record in enumerate(records, start=2):
        row = {}
        for column in REQUIRED_COLUMNS:
            value = record.get(column)
            if not isinstance(value, str) or not value.strip():
                raise DatasetValidationError(
                    f"Row {row_number} has a missing or blank {column!r} value."
                )
            row[column] = value.strip()
        row["sentiment"] = row["sentiment"].casefold()
        row[CYBERBULLYING_TARGET] = row[CYBERBULLYING_TARGET].casefold()
        if row["sentiment"] not in SENTIMENT_LABELS:
            raise DatasetValidationError(
                f"Row {row_number} has an invalid sentiment label; use positive, negative, or neutral."
            )
        cleaned.append(row)

    sentiment_labels = {row["sentiment"] for row in cleaned}
    if sentiment_labels != SENTIMENT_LABELS:
        missing = ", ".join(sorted(SENTIMENT_LABELS - sentiment_labels))
        raise DatasetValidationError(f"The dataset is missing sentiment labels: {missing}.")

    bullying_labels = {row[CYBERBULLYING_TARGET] for row in cleaned}
    if NON_BULLYING_LABEL not in bullying_labels or len(bullying_labels) < 2:
        raise DatasetValidationError(
            "Bullying labels must include not_bullying and at least one harmful category."
        )
    return cleaned


def load_labeled_comments(csv_path):
    """Load and validate a UTF-8 CSV using the documented four-column schema."""
    with Path(csv_path).open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        fieldnames = set(reader.fieldnames or [])
        missing = REQUIRED_COLUMNS - fieldnames
        if missing:
            raise DatasetValidationError(
                f"CSV is missing required columns: {', '.join(sorted(missing))}."
            )
        return validate_records(list(reader))


def load_jigsaw_toxicity_csv(csv_path):
    """Validate and normalize the Jigsaw toxicity CSV into text + six binary targets."""
    with Path(csv_path).open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        fieldnames = set(reader.fieldnames or [])
        missing = JIGSAW_REQUIRED_COLUMNS - fieldnames
        if missing:
            raise DatasetValidationError(
                f"Jigsaw CSV is missing required columns: {', '.join(sorted(missing))}."
            )

        records = []
        for row_number, record in enumerate(reader, start=2):
            text = (record.get("comment_text") or "").strip()
            if not text:
                raise DatasetValidationError(
                    f"Row {row_number} has a missing or blank comment_text value."
                )
            labels = {}
            for label in JIGSAW_LABELS:
                raw_value = record.get(label, "0")
                if raw_value is None or str(raw_value).strip() == "":
                    value = 0
                else:
                    try:
                        value = int(float(str(raw_value).strip()))
                    except ValueError as error:
                        raise DatasetValidationError(
                            f"Row {row_number} has a non-integer value for {label!r}."
                        ) from error
                if value not in {0, 1}:
                    raise DatasetValidationError(
                        f"Row {row_number} has an invalid binary value for {label!r}; use 0 or 1."
                    )
                labels[label] = value
            records.append({"text": text, "group_id": record.get("id", f"row-{row_number}"), "labels": labels})

    if not records:
        raise DatasetValidationError("The Jigsaw dataset contains no rows.")
    return records


def _make_pipeline():
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=False,
                    ngram_range=(1, 2),
                    token_pattern=r"(?u)\b\w+\b",
                    min_df=1,
                    max_features=50000,
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=42,
                ),
            ),
        ]
    )


def _grouped_split(texts, labels, groups, test_size, random_state):
    if not 0 < test_size < 1:
        raise ValueError("test_size must be greater than 0 and less than 1.")
    split_count = max(2, round(1 / test_size))
    if len(set(groups)) < split_count:
        raise DatasetValidationError(
            f"Group-aware evaluation needs at least {split_count} distinct group_id values."
        )

    splitter = StratifiedGroupKFold(
        n_splits=split_count,
        shuffle=True,
        random_state=random_state,
    )
    all_labels = set(labels)
    for train_indices, test_indices in splitter.split(texts, labels, groups):
        train_labels = {labels[index] for index in train_indices}
        test_labels = {labels[index] for index in test_indices}
        if train_labels == all_labels and test_labels == all_labels:
            train_groups = {groups[index] for index in train_indices}
            test_groups = {groups[index] for index in test_indices}
            if train_groups.isdisjoint(test_groups):
                return train_indices, test_indices

    raise DatasetValidationError(
        "Could not make a group-separated train/test split containing every label. "
        "Add more independently grouped examples for each class."
    )


def train_classifier(records, target_column, test_size=0.2, random_state=42):
    """Evaluate one target on held-out groups, then fit its deployable model."""
    if target_column not in {"sentiment", CYBERBULLYING_TARGET}:
        raise ValueError("target_column must be sentiment or bullying_category.")
    rows = validate_records(records)
    texts = [preprocess_comment(row["text"]) for row in rows]
    labels = [row[target_column] for row in rows]
    groups = [row["group_id"] for row in rows]
    train_indices, test_indices = _grouped_split(
        texts, labels, groups, test_size, random_state
    )

    evaluation_model = _make_pipeline()
    evaluation_model.fit(
        [texts[index] for index in train_indices],
        [labels[index] for index in train_indices],
    )
    expected = [labels[index] for index in test_indices]
    predicted = evaluation_model.predict([texts[index] for index in test_indices])
    class_labels = sorted(set(labels))
    report = classification_report(
        expected,
        predicted,
        labels=class_labels,
        output_dict=True,
        zero_division=0,
    )
    per_class = {
        label: {
            metric: float(report[label][metric])
            for metric in ("precision", "recall", "f1-score", "support")
        }
        for label in class_labels
    }
    metrics = {
        "accuracy": float(report["accuracy"]),
        "macro": {
            metric: float(report["macro avg"][metric])
            for metric in ("precision", "recall", "f1-score")
        },
        "weighted": {
            metric: float(report["weighted avg"][metric])
            for metric in ("precision", "recall", "f1-score")
        },
        "per_class": per_class,
    }
    evaluation = {
        "target": target_column,
        "model": "tfidf_logistic_regression",
        "split": "StratifiedGroupKFold",
        "train_rows": len(train_indices),
        "test_rows": len(test_indices),
        "metrics": metrics,
        "confusion_matrix": {
            "labels": class_labels,
            "counts": confusion_matrix(expected, predicted, labels=class_labels).tolist(),
        },
        "confidence_note": "Predicted probabilities are model scores, not calibrated real-world certainty.",
    }

    final_model = _make_pipeline()
    final_model.fit(texts, labels)
    return TextClassifier(
        pipeline=final_model,
        target_column=target_column,
        labels=class_labels,
        evaluation=evaluation,
    )


def analyze_comment(sentiment_model, bullying_model, text):
    """Return independent sentiment and bullying predictions for one comment."""
    if sentiment_model.target_column != "sentiment":
        raise ValueError("sentiment_model must be trained for sentiment.")
    if bullying_model.target_column != CYBERBULLYING_TARGET:
        raise ValueError("bullying_model must be trained for bullying_category.")
    bullying = bullying_model.predict(text)
    bullying["is_bullying"] = bullying["label"] != NON_BULLYING_LABEL
    return {
        "sentiment": sentiment_model.predict(text),
        "cyberbullying": bullying,
    }


def _make_jigsaw_pipeline():
    base = TfidfVectorizer(
        lowercase=False,
        ngram_range=(1, 2),
        token_pattern=r"(?u)\b\w+\b",
        min_df=2,
        max_features=50000,
        sublinear_tf=True,
    )
    classifier = MultiOutputClassifier(
        LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            random_state=42,
        )
    )
    return Pipeline([("tfidf", base), ("classifier", classifier)])


def _make_jigsaw_train_test_split(texts, label_matrix, test_size=0.2, random_state=42):
    if MultilabelStratifiedShuffleSplit is not None:
        splitter = MultilabelStratifiedShuffleSplit(
            test_size=test_size,
            random_state=random_state,
        )
        train_index, test_index = next(splitter.split(texts, label_matrix))
        return train_index, test_index

    indices = np.arange(len(texts))
    train_index, test_index = train_test_split(
        indices,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
    )
    return train_index, test_index


def _evaluate_jigsaw_split(expected, predicted):
    per_label = {}
    confusion_matrices = {}
    for index, label in enumerate(JIGSAW_LABELS):
        expected_label = expected[:, index]
        predicted_label = predicted[:, index]
        per_label[label] = {
            "precision": float(
                precision_score(expected_label, predicted_label, zero_division=0)
            ),
            "recall": float(
                recall_score(expected_label, predicted_label, zero_division=0)
            ),
            "f1": float(f1_score(expected_label, predicted_label, zero_division=0)),
            "positive_support": int(expected_label.sum()),
        }
        confusion_matrices[label] = confusion_matrix(
            expected_label, predicted_label, labels=[0, 1]
        ).tolist()

    return {
        "macro_precision": float(
            np.mean([per_label[label]["precision"] for label in JIGSAW_LABELS])
        ),
        "macro_recall": float(
            np.mean([per_label[label]["recall"] for label in JIGSAW_LABELS])
        ),
        "macro_f1": float(np.mean([per_label[label]["f1"] for label in JIGSAW_LABELS])),
        "micro_precision": float(
            precision_score(expected, predicted, average="micro", zero_division=0)
        ),
        "micro_recall": float(
            recall_score(expected, predicted, average="micro", zero_division=0)
        ),
        "micro_f1": float(
            f1_score(expected, predicted, average="micro", zero_division=0)
        ),
        "per_label": per_label,
        "confusion_matrices": confusion_matrices,
    }


def train_jigsaw_toxicity_model(csv_path, output_dir, test_size=0.2, random_state=42):
    """Train and evaluate a Jigsaw multi-label toxicity model using the six independent labels."""
    records = load_jigsaw_toxicity_csv(csv_path)
    texts = [preprocess_comment(row["text"]) for row in records]
    label_matrix = np.array(
        [[row["labels"][label] for label in JIGSAW_LABELS] for row in records],
        dtype=np.int64,
    )

    train_index, test_index = _make_jigsaw_train_test_split(
        texts,
        label_matrix,
        test_size=test_size,
        random_state=random_state,
    )
    train_texts = [texts[idx] for idx in train_index]
    test_texts = [texts[idx] for idx in test_index]
    train_labels = label_matrix[train_index]
    test_labels = label_matrix[test_index]

    if len(train_texts) < 2:
        raise DatasetValidationError("The toxicity dataset is too small for a valid train/test split.")

    validation_texts, test_texts, validation_labels, test_labels = train_test_split(
        test_texts,
        test_labels,
        test_size=0.5,
        random_state=random_state,
        shuffle=True,
    )

    model = _make_jigsaw_pipeline()
    model.fit(train_texts, train_labels)
    validation_predictions = model.predict(validation_texts)
    test_predictions = model.predict(test_texts)
    validation_metrics = _evaluate_jigsaw_split(
        validation_labels, validation_predictions
    )
    test_metrics = _evaluate_jigsaw_split(test_labels, test_predictions)

    metrics = {
        "task": "jigsaw_toxicity_multi_label",
        "labels": JIGSAW_LABELS,
        "train_rows": len(train_texts),
        "validation_rows": len(validation_texts),
        "test_rows": len(test_texts),
        "macro_precision": validation_metrics["macro_precision"],
        "macro_recall": validation_metrics["macro_recall"],
        "macro_f1": validation_metrics["macro_f1"],
        "micro_precision": validation_metrics["micro_precision"],
        "micro_recall": validation_metrics["micro_recall"],
        "micro_f1": validation_metrics["micro_f1"],
        "per_label": validation_metrics["per_label"],
        "confusion_matrices": validation_metrics["confusion_matrices"],
        "test_metrics": test_metrics,
        "confidence_note": "These probability outputs are prediction scores and are not treated as calibrated probabilities.",
    }

    final_model = _make_jigsaw_pipeline()
    final_model.fit(texts, label_matrix)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    model_path = destination / "jigsaw_toxicity_model.joblib"
    metadata_path = destination / "jigsaw_toxicity_metadata.json"
    labels_path = destination / "jigsaw_toxicity_labels.json"
    joblib.dump(final_model, model_path)
    labels_path.write_text(json.dumps(JIGSAW_LABELS), encoding="utf-8")
    metadata_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    return JigsawToxicityModel(
        pipeline=final_model,
        labels=JIGSAW_LABELS,
        evaluation=metrics,
    )


def train_dataset(csv_path, output_dir):
    records = load_labeled_comments(csv_path)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    results = {}
    for target_column in ("sentiment", CYBERBULLYING_TARGET):
        model = train_classifier(records, target_column)
        joblib.dump(model, destination / f"{target_column}_model.joblib")
        report_path = destination / f"{target_column}_evaluation.json"
        report_path.write_text(
            json.dumps(model.evaluation, indent=2),
            encoding="utf-8",
        )
        results[target_column] = model.evaluation
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train and evaluate comment classifiers.")
    parser.add_argument("dataset", help="Path to a labeled CSV with the documented columns.")
    parser.add_argument("--output-dir", default="models", help="Directory for model and metric files.")
    parser.add_argument(
        "--task",
        choices=["legacy", "jigsaw-toxicity"],
        default="legacy",
        help="Which classifier family to train: legacy sentiment/bullying or Jigsaw multi-label toxicity.",
    )
    arguments = parser.parse_args(argv)
    try:
        if arguments.task == "jigsaw-toxicity":
            model = train_jigsaw_toxicity_model(
                arguments.dataset,
                arguments.output_dir,
            )
            print(json.dumps(model.evaluation, indent=2))
            return 0
        results = train_dataset(arguments.dataset, arguments.output_dir)
    except (DatasetValidationError, OSError) as error:
        parser.error(str(error))
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())