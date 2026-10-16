import csv

import pytest

from app.services.ml_pipeline import (
    CYBERBULLYING_TARGET,
    DatasetValidationError,
    analyze_comment,
    load_labeled_comments,
    preprocess_comment,
    train_classifier,
    validate_records,
)


@pytest.fixture
def synthetic_records():
    records = []
    for group_number in range(5):
        group_id = f"synthetic-video-{group_number}"
        records.extend(
            [
                {
                    "text": "I really enjoyed this helpful video!",
                    "sentiment": "positive",
                    "bullying_category": "not_bullying",
                    "group_id": group_id,
                },
                {
                    "text": "I disagree with this point; the explanation is wrong.",
                    "sentiment": "negative",
                    "bullying_category": "not_bullying",
                    "group_id": group_id,
                },
                {
                    "text": "You are awful and nobody wants you here.",
                    "sentiment": "negative",
                    "bullying_category": "insult",
                    "group_id": group_id,
                },
                {
                    "text": "The video starts with an introduction.",
                    "sentiment": "neutral",
                    "bullying_category": "not_bullying",
                    "group_id": group_id,
                },
            ]
        )
    return records


def test_preprocessing_normalizes_urls_mentions_and_entities():
    result = preprocess_comment("Thanks @viewer &amp; see https://example.com/page")

    assert result == "thanks user & see url"


def test_preprocessing_translates_emojis_to_semantic_tokens():
    result = preprocess_comment("Sub to my channel 🖤🔥👍")
    assert "emoji_heart" in result
    assert "emoji_fire" in result
    assert "emoji_thumbsup" in result


def test_validation_requires_all_sentiments_and_non_bullying(synthetic_records):
    assert len(validate_records(synthetic_records)) == len(synthetic_records)

    missing_class = [row for row in synthetic_records if row["sentiment"] != "neutral"]
    with pytest.raises(DatasetValidationError, match="missing sentiment labels"):
        validate_records(missing_class)


def test_csv_loader_requires_the_documented_schema(tmp_path, synthetic_records):
    csv_path = tmp_path / "comments.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=synthetic_records[0].keys())
        writer.writeheader()
        writer.writerows(synthetic_records)

    loaded = load_labeled_comments(csv_path)

    assert len(loaded) == len(synthetic_records)
    assert loaded[0]["group_id"] == "synthetic-video-0"


@pytest.mark.parametrize("target", ["sentiment", CYBERBULLYING_TARGET])
def test_training_evaluates_on_disjoint_groups_and_reports_metrics(synthetic_records, target):
    model = train_classifier(synthetic_records, target)
    evaluation = model.evaluation

    assert evaluation["target"] == target
    assert evaluation["split"] == "StratifiedGroupKFold"
    assert evaluation["train_rows"] + evaluation["test_rows"] == len(synthetic_records)
    assert set(evaluation["metrics"]["per_class"]) == set(model.labels)
    assert set(evaluation["metrics"]["macro"]) == {"precision", "recall", "f1-score"}
    assert len(evaluation["confusion_matrix"]["counts"]) == len(model.labels)


def test_comment_predictions_keep_tasks_separate(synthetic_records):
    sentiment_model = train_classifier(synthetic_records, "sentiment")
    bullying_model = train_classifier(synthetic_records, CYBERBULLYING_TARGET)

    result = analyze_comment(
        sentiment_model,
        bullying_model,
        "I disagree with this point; the explanation is wrong.",
    )

    assert set(result) == {"sentiment", "cyberbullying"}
    assert result["sentiment"]["label"] in {"positive", "negative", "neutral"}
    assert 0 <= result["sentiment"]["confidence"] <= 1
    assert result["cyberbullying"]["label"] in {"not_bullying", "insult"}
    assert result["cyberbullying"]["is_bullying"] == (
        result["cyberbullying"]["label"] != "not_bullying"
    )


def test_pipeline_rejects_incomplete_grouped_evaluation(synthetic_records):
    one_group = [
        row for row in synthetic_records if row["group_id"] == "synthetic-video-0"
    ]

    with pytest.raises(DatasetValidationError, match="distinct group_id"):
        train_classifier(one_group, "sentiment")