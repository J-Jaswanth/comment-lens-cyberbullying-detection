# OAuth-Based Social Media Cyberbullying Detection and Sentiment Analytics

A staged B.Tech project using Flask, HTML/CSS/JavaScript, MongoDB Atlas, scikit-learn, Google OAuth 2.0, and the official YouTube Data API. The workspace already contains account routes, MongoDB stores, a data-driven ML training pipeline, and OAuth/YouTube API code. A read-only Atlas ping and Flask startup have been verified; live Google OAuth and YouTube access have not. Analysis results, dashboard, and per-user history are still to be implemented.

## Requirements

- Python 3.12 (the initial environment was checked with Python 3.12.10)
- A MongoDB Atlas cluster (the local MongoDB server is only used for isolated tests)
- A configured Google Cloud project and OAuth Web client for live YouTube access

## Local setup

Open PowerShell in this project folder and run:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
```

Edit `.env` locally. Set `SECRET_KEY` to a locally generated value. One way to generate it is:

```powershell
.\.venv\Scripts\python.exe -c "import secrets; print(secrets.token_hex(32))"
```

Put that output in your local `.env` only. Also fill in the MongoDB Atlas URI and Google OAuth values from the services you configured. Do not paste secrets into chat or commit `.env`.

Start the current development server with:

```powershell
.\.venv\Scripts\python.exe run.py
```

Open `http://127.0.0.1:5000/auth/register` to create an account. MongoDB Atlas must be reachable for account creation and login; if it is unavailable, the app shows a storage error rather than silently saving accounts in memory. After signing in, the home page confirms the account session. Sign out using the button in the header.

To run the current automated checks:

```powershell
.\.venv\Scripts\python.exe -m pytest
```

The authentication tests use a fake user store so they can run without MongoDB. MongoDB persistence has a separate integration test; opt in explicitly with a disposable local test URI:

```powershell
$env:MONGODB_TEST_URI = "mongodb://localhost:27017/"
$env:RUN_MONGODB_INTEGRATION = "1"
.\.venv\Scripts\python.exe -m pytest -q
```

That test creates a uniquely named temporary database and drops only that database afterward. Do not point it at a production or Atlas database. A `503` message about account storage usually means the Atlas URI, database user, or network access list needs checking. A startup error asking for `SECRET_KEY` means the placeholder in `.env` has not been replaced.

## Configuration and external setup

### MongoDB

For the application, use the Atlas cluster you created. In Atlas, open **Database > Connect > Drivers**, choose **Python**, and copy the connection URI into `MONGODB_URI` in `.env`. Replace the username, password, cluster host, and database placeholders locally. URL-encode special characters in the database password when they appear in a URI. Confirm your current IP remains allowed under **Network Access** and that the database user has appropriate read/write access. Never use the production Atlas URI for the destructive temporary-database integration test; that test should use the local MongoDB URI shown above.

Atlas connectivity was verified with a read-only PyMongo ping using the local `.env`; no collections or documents were changed. Re-run the ping after changing the URI, credentials, or IP allowlist.

### Google OAuth and YouTube Data API

You reported that the project, YouTube Data API v3, External consent screen, test user, Web application client, and client secret are configured. Keep the client ID and secret in `.env` only. The configured redirect URI is `http://127.0.0.1:5000/auth/google/callback`.

Generate a Fernet key for token encryption locally with:

```powershell
.\.venv\Scripts\python.exe -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
```

Put that output in `GOOGLE_TOKEN_ENCRYPTION_KEY` in `.env`. Treat it as a secret and keep a secure backup; losing it makes stored tokens unreadable.

The app's **Connect YouTube with Google** action is available after signing in. It requests the `youtube.readonly` scope, uses state validation and PKCE, and stores token JSON encrypted with your Fernet key. The application never requests your Google password. The canonical callback is `/auth/google/callback`; `/oauth/google/callback` remains as a compatibility alias. The connect page reports safe configuration errors without exposing credential values. No live OAuth or YouTube access has been verified yet. Confirm in Google Cloud that the selected project is `Cyberbullying Detection` and the client is a **Web application** OAuth client.

## Machine-learning data

Sentiment and cyberbullying are separate predictions. Negative sentiment alone is not a bullying label. Model evaluation requires a licensed, documented, labeled dataset with sentiment labels (`positive`, `negative`, `neutral`) and cyberbullying labels that include non-bullying plus well-defined harmful categories. Record language and annotation guidance, check class balance, protect personal information, and split by video or author where possible to reduce leakage. Evaluate precision, recall, F1-score, and a confusion matrix; accuracy alone is insufficient, especially with imbalanced labels. Sarcasm, context, dialect, and ambiguous language can cause false positives and false negatives.

Synthetic comments may be used for software tests only. They are not training evidence and must not be presented as model-quality results. See [docs/dataset_requirements.md](docs/dataset_requirements.md).

When a real, licensed dataset is available, put it at `data/private/labeled_comments.csv` and train/evaluate both independent classifiers with:

```powershell
.\.venv\Scripts\python.exe -m app.services.ml_pipeline data/private/labeled_comments.csv --output-dir models
```

The required CSV columns, grouped split behavior, generated evaluation files, and limitations of probability scores are described in [docs/dataset_requirements.md](docs/dataset_requirements.md). No dataset or trained production model is included yet.

## Planned implementation stages

See [PROJECT_STATUS.md](PROJECT_STATUS.md) for tested features, external setup status, known limitations, and the exact next action. Features are deliberately introduced in small stages rather than presented as complete before they are implemented and tested.