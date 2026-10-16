# Comment Lens: Cyberbullying Detection and Sentiment Analytics

A staged B.Tech project using Flask, HTML/CSS/JavaScript, MongoDB Atlas, scikit-learn, Google OAuth 2.0, and the official YouTube Data API. The workspace contains account routes, MongoDB stores, a data-driven ML training pipeline, OAuth/YouTube API code, and a dashboard for analyzing YouTube comments.

## Requirements

- Python 3.12
- A MongoDB Atlas cluster
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

That test creates a uniquely named temporary database and drops only that database afterward. Do not point it at a production or Atlas database.

## Configuration and external setup

### MongoDB

For the application, use the Atlas cluster you created. In Atlas, open **Database > Connect > Drivers**, choose **Python**, and copy the connection URI into `MONGODB_URI` in `.env`. Replace the username, password, cluster host, and database placeholders locally. URL-encode special characters in the database password when they appear in a URI. Confirm your current IP remains allowed under **Network Access** and that the database user has appropriate read/write access.

### Google OAuth and YouTube Data API

The app requests the `youtube.readonly` scope, uses state validation and PKCE, and stores token JSON encrypted with your Fernet key. The configured redirect URI is `http://127.0.0.1:5000/auth/google/callback`.

Generate a Fernet key for token encryption locally with:

```powershell
.\.venv\Scripts\python.exe -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
```

Put that output in `GOOGLE_TOKEN_ENCRYPTION_KEY` in `.env`.

## Machine-learning data

Sentiment and cyberbullying are separate predictions. Negative sentiment alone is not a bullying label. Model evaluation requires a licensed, documented, labeled dataset with sentiment labels (`positive`, `negative`, `neutral`) and cyberbullying labels that include non-bullying plus well-defined harmful categories.

Jigsaw toxicity uses its own six-label multi-label pipeline and does not create sentiment labels. To train or refresh it using the local dataset at `data/raw/train.csv`, run:

```powershell
.\.venv\Scripts\python.exe -m app.services.ml_pipeline data\raw\train.csv --task jigsaw-toxicity --output-dir models
```

Jigsaw model scores are shown independently in the analysis results and are not calibrated probabilities or definitive bullying verdicts. Until compatible sentiment and bullying artifacts are trained, those legacy outputs remain labeled as illustrative demo predictions.

Synthetic comments may be used for software tests only. They are not training evidence and must not be presented as model-quality results.

## Status

This project is built as a staged B.Tech implementation and is currently validated through automated tests. MongoDB and OAuth flows require the configured external credentials and Atlas network access to work completely in a live environment.
