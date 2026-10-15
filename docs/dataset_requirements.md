# Dataset Requirements

The project does not include or assume a production training dataset. Obtain data only from a source whose license and terms permit this research use, and keep its license and source citation with the dataset documentation. Do not collect private comments or bypass YouTube permissions.

## Labels and annotation

- CSV header: `text,sentiment,bullying_category,group_id`.
- `group_id` should be a stable, non-identifying video or author grouping key. Do not place names, email addresses, or other personal identifiers in it.
- Sentiment: `positive`, `negative`, or `neutral`.
- Cyberbullying: at minimum, `not_bullying` and clearly defined bullying categories such as `insult`, `harassment`, `threat`, and `identity_attack`.
- Label sentiment independently from bullying. A negative opinion or criticism is not automatically bullying.
- Document label definitions, language, annotation process, and how disagreements were resolved. Human review is important for context-sensitive or ambiguous labels.

## Quality and evaluation

- Include enough examples per label to measure performance; report class counts and address class imbalance.
- Remove or protect identifying information and document preprocessing.
- Keep related comments together when splitting data, preferably grouping by video or author, so near-duplicate/context-linked examples cannot leak across training and test sets.
- Report per-class precision, recall, and F1-score, macro-F1, and a confusion matrix. Use a validation set for model selection and reserve the test set for final evaluation.
- Document language coverage, sarcasm, dialect, context limits, false-positive risk, and false-negative risk.

Synthetic examples are appropriate for deterministic unit tests and UI demonstrations only. They cannot support claims about real-world model performance.

## Training

Place an appropriately licensed dataset at `data/private/labeled_comments.csv`; that directory is excluded from Git. From the project root, run:

```powershell
.\.venv\Scripts\python.exe -m app.services.ml_pipeline data/private/labeled_comments.csv --output-dir models
```

The command evaluates each target on held-out groups and prints accuracy, precision, recall, F1-scores, and confusion matrices. It then refits each classifier on all supplied rows and saves model files plus the held-out evaluation JSON in `models/`. The evaluation JSON describes the held-out split, not the final all-data refit. Logistic-regression probability outputs are model scores and are not guaranteed to be calibrated confidence. Do not run this workflow on synthetic test comments and report the resulting metrics as project performance.