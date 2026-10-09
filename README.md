# Autonomous Fraud Detection Platform

Real-time fraud detection with **model serving**, **data drift monitoring**, and **automated retraining**.

Built as a full-stack MLOps portfolio project demonstrating production-style observability and a closed-loop retrain pipeline.

**Stack:** FastAPI · LightGBM · Evidently AI · MLflow · React · Docker · GitHub Actions

---

## Highlights

| Capability | Detail |
|------------|--------|
| Fraud model | LightGBM classifier — ~**98.7%** accuracy, ~**0.95** ROC-AUC |
| Drift monitoring | Evidently AI on a **100-event** window; alert at **≥25%** column drift |
| Realistic drift demo | Injected shift typically lands ~**30–40%** drifted columns |
| Retraining | Local webhook or GitHub Actions → train → MLflow log → hot-swap |
| Model versions | Short versions (`v1`, `v2`, …) swapped with **zero downtime** |

---

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  React Dashboard │────▶│  FastAPI Service  │────▶│  LightGBM Model │
│  (Simulator)     │     │  /predict         │     │  (hot-swappable)│
└─────────────────┘     └────────┬─────────┘     └─────────────────┘
                                 │
                                 ▼
                        ┌──────────────────┐
                        │  Evidently AI     │
                        │  Drift Monitor    │
                        └────────┬─────────┘
                                 │ drift ≥ 25%
                                 ▼
                        ┌──────────────────┐     ┌─────────────────┐
                        │  Retrain Trigger  │────▶│  MLflow         │
                        │  (Actions / hook) │     │  Experiment Log │
                        └──────────────────┘     └─────────────────┘
```

### Flow

1. Dashboard streams synthetic credit-card transactions (normal or drift-shifted).
2. FastAPI scores each transaction with LightGBM.
3. Evidently compares the live buffer to a reference distribution (PSI).
4. When drift crosses the threshold, retraining runs, metrics are logged to MLflow, and the new artifact is hot-swapped.

---

## Quick Start

### Option A — Docker (recommended)

**Prerequisites:** Docker Desktop

```bash
git clone https://github.com/AnooshVenkat/autonomous-fraud-detection-platform.git
cd autonomous-fraud-detection-platform
cp .env.example .env   # optional
docker compose up --build
```

| Service | URL |
|---------|-----|
| Dashboard | http://localhost:3000 |
| API docs | http://localhost:8000/docs |
| MLflow UI | http://localhost:5000 |

### Option B — Local development

**Prerequisites:** Python 3.10+, Node 20+

```bash
# Terminal 1 — API
cd backend
python -m venv .venv
# Windows: .\.venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000

# Terminal 2 — Dashboard
cd frontend
npm install
npm run dev
```

Open http://localhost:3000 (Vite proxies `/api` → `http://localhost:8000`).

---

## Demo Walkthrough

1. Open the dashboard.
2. Click **Stream Normal Data** to fill the buffer.
3. Click **Inject Drifted Data** (use batch size **100** to fill the window).
4. Watch the drift gauge rise above **25%** → **Drift Alert**.
5. Click **Trigger Local Retrain** — model version bumps (`v1` → `v2`) and the buffer resets.

Optional: set `GITHUB_REPO` + `GITHUB_TOKEN` in `.env` so drift can dispatch the GitHub Actions retrain workflow.

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Service + model status |
| `POST` | `/predict` | Score one transaction |
| `POST` | `/simulate` | Batch simulate normal or drifted traffic |
| `GET` | `/drift` | Current drift status (cached for polling) |
| `POST` | `/retrain/webhook` | Local retrain + hot-swap |
| `POST` | `/model/reload` | Reload model artifact from disk |

Interactive docs: http://localhost:8000/docs

---

## Project Structure

```
├── backend/
│   ├── app/
│   │   ├── main.py                 # FastAPI routes
│   │   ├── config.py               # Settings / env
│   │   ├── schemas.py              # Pydantic models
│   │   └── services/
│   │       ├── predictor.py        # Serving + training + MLflow
│   │       ├── drift_monitor.py    # Evidently PSI monitoring
│   │       └── retrain_trigger.py  # GitHub Actions dispatch
│   ├── scripts/
│   │   ├── generate_data.py        # Synthetic / drifted batches
│   │   └── train_model.py          # CLI training entrypoint
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   ├── src/App.jsx                 # Drift dashboard UI
│   └── Dockerfile
├── .github/workflows/
│   └── retrain-on-drift.yml        # Automated retrain workflow
├── docker-compose.yml
└── .env.example
```

---

## GitHub Actions Retrain

Workflow: `.github/workflows/retrain-on-drift.yml` (`workflow_dispatch`).

| Env / Secret | Purpose |
|--------------|---------|
| `GITHUB_REPO` | `owner/repo` (API container) |
| `GITHUB_TOKEN` | PAT with `actions:write` |
| `MLFLOW_TRACKING_URI` | Optional remote MLflow |
| `API_RELOAD_URL` | Optional live API for hot-swap |
| `RETRAIN_WEBHOOK_SECRET` | Must match backend secret |

Without GitHub credentials, use **Trigger Local Retrain** on the dashboard (works fully offline with file-based MLflow).

---

## Configuration

See [`.env.example`](.env.example). Important locals defaults:

| Variable | Default (local) |
|----------|-----------------|
| `MLFLOW_TRACKING_URI` | `file:./mlruns` |
| `DRIFT_WINDOW_SIZE` | `100` |
| `DRIFT_THRESHOLD` | `0.25` |
| `RETRAIN_WEBHOOK_SECRET` | `change-me-in-production` |

---

## License

MIT
