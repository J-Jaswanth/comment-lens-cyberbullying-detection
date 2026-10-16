# ML training and evaluation plan for Comment Lens

## Verified current project state

The repository contains a training library in `app/services/ml_pipeline.py` and a clearly marked demo fallback for the legacy sentiment and bullying models. The Jigsaw Toxic Comment dataset is kept locally at `data/raw/train.csv`; its trained toxicity artifact is stored locally under `models/` and is not committed to version control.

This means the app was previously using a synthetic fallback rather than a validated production model. The fallback is intentionally marked as illustrative, but it must remain clearly labeled and should never be treated as a production-grade classifier.

## Datasets to use for production-grade training

### Cyberbullying / toxicity

Use the Jigsaw Toxic Comment Classification Challenge dataset from Kaggle.

- Training rows: approximately 159,571 comments
- Labels: `toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, `identity_hate`
- Recommended modeling: multi-label classification
- Important: these are independent labels, not mutually exclusive classes

The app reports these six independent toxicity model scores separately from sentiment and the legacy bullying estimate. They are uncalibrated scores, not a definitive cyberbullying finding. Do not infer that a toxic label establishes context, intent, or repeated targeting.

### Sentiment

Use the Google GoEmotions dataset.

- Approximate size: 58,000 Reddit comments
- Emotion labels: 27 emotion categories plus neutral
- Recommended approach: map emotions to `positive`, `negative`, and `neutral` using an explicit mapping table, or train a dedicated sentiment model if a direct sentiment dataset is available
- Important: not every emotion maps cleanly to sentiment without a policy decision

## Dataset handling requirements

- Keep datasets outside version control
- Store raw files under `data/raw` or `data/private` only locally
- Do not commit secrets, OAuth tokens, or MongoDB credentials
- Validate required columns before training
- Deduplicate rows, drop blank records, and preserve original social-media text where possible
- Keep related comments together across train/validation/test splits to avoid leakage
- Use fixed random seeds and save model metadata

## Training workflow

From a local Windows environment:

```powershell
cd "c:\Users\dwara\OneDrive\Documents\Final Project"
.\.venv\Scripts\python.exe -m app.services.ml_pipeline data\private\labeled_comments.csv --output-dir models
```

The CSV should contain at least:

```csv
text,sentiment,bullying_category,group_id
```

A production-grade pipeline should then save:

- trained sentiment model
- trained bullying model
- evaluation reports with per-class metrics and confusion matrices
- metadata JSON describing dataset version, seeds, and training parameters

### Train Jigsaw toxicity

The Jigsaw CSV has a different schema and must use its dedicated multi-label task. From the project root:

```powershell
.\.venv\Scripts\python.exe -m app.services.ml_pipeline data\raw\train.csv --task jigsaw-toxicity --output-dir models
```

The command evaluates on separate validation and test splits and writes `jigsaw_toxicity_model.joblib`, `jigsaw_toxicity_labels.json`, and `jigsaw_toxicity_metadata.json`. The model scores appear in the comment breakdown only when this artifact is available. The sentiment and legacy bullying outputs remain marked as demo until their own compatible labeled dataset is trained.

## Important policy notes

- Sentiment and cyberbullying are separate outputs
- Negative sentiment alone is not proof of bullying
- Ambiguous short social-media comments should be reviewed rather than forced into a harmful category
- Demo or fallback models must be clearly labeled as illustrative and untrusted
