"""Data-driven text models for sentiment and cyberbullying classification."""

import argparse
import csv
import html
import json
import re
from dataclasses import dataclass
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline

SENTIMENT_LABELS = {"positive", "negative", "neutral"}
CYBERBULLYING_TARGET = "bullying_category"
NON_BULLYING_LABEL = "not_bullying"
REQUIRED_COLUMNS = {"text", "sentiment", CYBERBULLYING_TARGET, "group_id"}


class DatasetValidationError(ValueError):
    """Raised when labeled comments do not meet the documented data contract."""


@dataclass
class TextClassifier:
    pipeline: Pipeline
    target_column: str
    labels: list[str]
    evaluation: dict

    def predict(self, text):
        probabilities = self.pipeline.predict_proba([text])[0]
        class_probabilities = zip(self.pipeline.classes_, probabilities)
        label, confidence = max(class_probabilities, key=lambda item: item[1])
        return {"label": str(label), "confidence": float(confidence)}


def preprocess_comment(text):
    """Normalize surface noise while retaining words and punctuation cues."""
    text = html.unescape(text or "")
    text = re.sub(r"<[^>]*>", " ", text)
    text = re.sub(r"https?://\S+|www\.\S+", " URL ", text, flags=re.IGNORECASE)
    text = re.sub(r"(?<!\w)@[\w.-]+", " USER ", text)
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


def _make_pipeline():
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    preprocessor=preprocess_comment,
                    lowercase=False,
                    ngram_range=(1, 2),
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
    texts = [row["text"] for row in rows]
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
    parser = argparse.ArgumentParser(description="Train and evaluate both comment classifiers.")
    parser.add_argument("dataset", help="Path to a labeled CSV with the documented columns.")
    parser.add_argument("--output-dir", default="models", help="Directory for model and metric files.")
    arguments = parser.parse_args(argv)
    try:
        results = train_dataset(arguments.dataset, arguments.output_dir)
    except (DatasetValidationError, OSError) as error:
        parser.error(str(error))
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())