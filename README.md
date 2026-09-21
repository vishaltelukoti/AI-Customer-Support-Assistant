# AI Customer Support Assistant

A single-developer AI-powered customer-support POC. **Days 1-3 implement the application foundation, prepared dataset, offline baseline ML classification, and lightweight hyperparameter optimization.** The form submits a customer message to FastAPI and displays a temporary typed acknowledgement. Its category, priority, and confidence remain `null`; tickets are not stored. Trained classifiers are available through separate offline services and CLIs.

Planned later capabilities include deep-learning experiments, API integration, similar historical tickets, suggested resolutions with sources, complex-ticket investigation, security status, and classification explanations. See [the development plan](docs/development-plan.md).

## Stack and structure

Python 3.12+, FastAPI, Uvicorn, Pydantic/pydantic-settings, pytest, httpx, scikit-learn, Optuna, joblib and Matplotlib; React, TypeScript, Vite, Axios and plain CSS. No database or external AI account is required.

```text
backend/
  app/                  # API, schemas, services, preprocessing, baseline ML/evaluation
                        # empty models, rag, agents, security, explainability, monitoring
  tests/                # API, CORS, preprocessing, leakage and real-model inference tests
  requirements.txt
  .env.example
  Dockerfile
frontend/
  src/                  # components, pages, services, types, App and styles
  package.json          # package-lock.json locks frontend dependencies
  .env.example
  Dockerfile
data/
  raw/                  # selected Kaggle CSV; legacy 20-row sample retained
  processed/            # train/validation/test, audit JSON, historical tickets
  knowledge_base/       # four short fictional sample documents
experiments/baseline/   # Day 2 measured results, confusion matrices, ignored model artifacts
experiments/optimization/ # Day 3 search results and ignored optimized model artifacts
airflow/                # placeholder only
mlflow/                 # placeholder only
docs/                   # architecture, dataset, assumptions, development plan
.github/workflows/      # placeholder only; no CI/CD
docker-compose.yml
LICENSE
```

## Run locally

Prerequisites: Python 3.12+ and Node.js 22.12+ with npm. Commands below are PowerShell, starting in the repository root. Use two terminals. Do not run the local servers and Docker stack on the same ports simultaneously.

Backend:

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Frontend (new terminal):

```powershell
cd frontend
npm ci
Copy-Item .env.example .env
npm run dev
```

Open http://localhost:5173. Submit `I was charged twice for the same order.` to see `received`, a UUID, and unavailable AI fields. Do not overwrite an existing customized `.env` when repeating setup. On macOS/Linux use `python3`, `.venv/bin/python`, and `cp` in place of their Windows equivalents.

API documentation: http://localhost:8000/docs. Health: http://localhost:8000/health. Stop either server with Ctrl+C.

## Environment configuration

| Variable | Location | Default / format |
| --- | --- | --- |
| APP_NAME | backend/.env or process environment | AI Customer Support Assistant |
| CORS_ORIGINS | backend/.env or process environment | JSON array containing `http://localhost:5173` and `http://127.0.0.1:5173` |
| VITE_API_BASE_URL | frontend/.env | `http://localhost:8000` |

The backend locates its `.env` relative to its own folder. Process variables take precedence. Restart servers after changing configuration. Frontend `VITE_` values are public and embedded at build time, so never put secrets there. Changing the Docker frontend URL requires rebuilding. Defaults work without `.env` files.

Explicit CORS origins follow [FastAPI's CORS configuration](https://fastapi.tiangolo.com/tutorial/cors/); frontend environment handling follows [Vite's environment documentation](https://vite.dev/guide/env-and-mode).

## API contract

| Method | Endpoint | Behavior |
| --- | --- | --- |
| GET | /health | HTTP 200: `{"status":"healthy"}` |
| POST | /api/v1/tickets | HTTP 200 acknowledgement; HTTP 422 on invalid input |

```powershell
Invoke-RestMethod http://localhost:8000/health
Invoke-RestMethod -Method Post -Uri http://localhost:8000/api/v1/tickets -ContentType 'application/json' -Body '{"ticket_text":"I was charged twice for the same order."}'
```

Example response (the UUID changes per request):

```json
{
  "ticket_id": "d7658bf4-05d0-4f3f-8261-adc5811c1e4d",
  "ticket_text": "I was charged twice for the same order.",
  "category": null,
  "priority": null,
  "confidence": null,
  "status": "received"
}
```

The API accepts 1-10,000 characters after trimming surrounding whitespace. Blank, missing, non-string or oversized text and unexpected fields are rejected. The UI handles empty input, loading, success, validation errors, API failures and a 15-second timeout.

## Prepare the selected ML dataset

The original 20-record `sample_tickets.csv` was used to test the Day 1 foundation. It remains unchanged for historical reference and is no longer an active preprocessing input.

The only selected ML input is `data/raw/aa_dataset-tickets-multi-lang-5-2-50-version.csv`: **Customer IT Support - Ticket Dataset**, by **Tobias Bueck**, from [Kaggle](https://www.kaggle.com/datasets/tobiasbueck/multilingual-customer-support-tickets). The [creator describes it as synthetically generated](https://softoft.de/blog/ticket-dataset/); it is not real customer data. No other dataset variants are used.

From the repository root:

```powershell
.\backend\.venv\Scripts\python.exe -m backend.app.ml.preprocessing
```

Optional explicit configuration:

```powershell
.\backend\.venv\Scripts\python.exe -m backend.app.ml.preprocessing --input data/raw/aa_dataset-tickets-multi-lang-5-2-50-version.csv --output-dir data/processed --seed 42
```

From `backend`, use `.\.venv\Scripts\python.exe -m app.ml.preprocessing`. Defaults resolve relative to the repository. No additional dependencies are required.

```text
Selected raw CSV -> schema/version/language/label/text audit -> English selection
                 -> conservative cleaning + privacy masking -> deduplication
                 -> grouped queue/priority 70/15/15 split -> processed outputs
```

The source has **28,587 rows and 16 columns**. All **16,338 English records**, **10 queue labels** and **3 lowercase priority labels** are retained across all three versions. No invalid or duplicate combined texts were removed. Split sizes are **11,436 train / 2,451 validation / 2,451 test**, seed **42**. Related records stay in one split; priority distributions remain within 0.08 percentage points of the overall distribution. The raw source stays byte-identical.

Outputs in `data/processed/`:

- `train.csv`, `validation.csv`, `test.csv`: exactly `ticket_id,ticket_text,category,priority`.
- `dataset_audit.json`: measured schema, distributions, version overlap, missing values, text statistics, privacy findings, leakage decisions, split checks and hashes.
- `historical_tickets.csv`: masked answers and provenance/tag/type metadata stored separately for future retrieval. Use only training rows as a future evaluation retrieval corpus.

Classification input is **subject + body**. Agent answers, assigned type/tags and other metadata are excluded; category and priority are targets. IDs are generated from the raw checksum and source record because no source ID exists. Repeat runs replace the five named outputs deterministically. Failures before export leave previous outputs in place; inspect the command result before using them.

Important limitations: synthetic label quality, substantial queue imbalance (about 20:1), English-only scope, possible paraphrase/template similarity beyond exact grouping, and lightweight privacy patterns that cannot guarantee PII-free text. One ticket-text phone and 70 answer phones are masked. The API still returns null classification fields and does not load these datasets. See [the full dataset audit](docs/dataset.md).

## Baseline ML classification

Day 2 trains two independent **TF-IDF + Logistic Regression** pipelines for category and priority. Only `train.csv` is fitted; validation and test are evaluation-only. `ticket_text` is the only feature, with answers excluded. The fixed baseline uses unigrams/bigrams, a 50,000-feature cap, seed 42 and `class_weight=None`. No hyperparameter search or resampling is performed.

Run from the repository root after installing `backend/requirements.txt`:

```powershell
.\backend\.venv\Scripts\python.exe -m backend.app.ml.baseline_models
.\backend\.venv\Scripts\python.exe -m backend.app.ml.predict_examples
```

The training command saves full joblib pipelines, metrics JSON, a generated report, and test confusion matrices under `experiments/baseline/`. Model files are ignored by Git. The example command loads those actual models and saves predictions with separate category/priority probabilities. The API endpoint remains unchanged.

| Target | Test accuracy | Test macro-F1 | Test weighted-F1 |
| --- | ---: | ---: | ---: |
| Category | 0.528356 | 0.376067 | 0.495198 |
| Priority | 0.611179 | 0.559960 | 0.593776 |

All seven aggregate metrics, all classes and all three splits are in the [generated results](experiments/baseline/baseline_results.md) and [baseline documentation](docs/baseline-ml.md). Macro-F1 exposes weak minority-class performance that accuracy alone obscures. These synthetic-data results are a reference for later work, not evidence of production readiness.

## Hyperparameter optimization

Day 3 compares **Grid Search**, **Randomized Search**, and **Optuna Bayesian Optimization** for the same category and priority TF-IDF + Logistic Regression pipelines. Searches use `train.csv` only with 3-fold CV and `f1_macro`; validation macro-F1 selects the final configuration. The selected model is retrained on train + validation and evaluated once on test.

Run from the repository root:

```powershell
.\backend\.venv\Scripts\python.exe -m backend.app.ml.optimization
```

The command saves measured search results, optimized model artifacts, and optimized test confusion matrices under `experiments/optimization/`. Model files are ignored by Git.

| Target | Selected method | Validation macro-F1 | Test accuracy | Test macro-F1 | Test weighted-F1 |
| --- | --- | ---: | ---: | ---: | ---: |
| Category | Randomized Search | 0.574854 | 0.652795 | 0.641538 | 0.651417 |
| Priority | Randomized Search | 0.630936 | 0.676867 | 0.662208 | 0.674342 |

Both selected configurations use `C=10.0`, `class_weight=None`, unigrams/bigrams, `min_df=1`, `max_df=0.95`, `max_features=20000`, and `sublinear_tf=False`. Full measured values are in [the generated optimization report](experiments/optimization/optimization_results.md) and [baseline ML documentation](docs/baseline-ml.md). The API endpoint remains unchanged.

## Tests and build

```powershell
cd backend
.\.venv\Scripts\python.exe -m pytest -q
```

```powershell
cd frontend
npm run build
npm run test:api
```

Tests cover health, schema, null AI fields, unique IDs, whitespace normalization, invalid requests, input length boundaries, malformed JSON, and allowed/rejected CORS origins. The frontend build includes strict TypeScript checking. `npm run test:api` requires the backend running and exercises the actual Axios service against it, including API validation errors. With the backend stopped, `npm run test:api -- --unavailable` checks connection-error handling. These service checks are not browser/UI tests. No tests claim future AI functionality works.

Preprocessing tests use small fixtures and cover schema detection, English filtering, version overlap, conservative cleaning, privacy masks, exact deduplication, conflicting labels, grouped queue/priority splits, rare-class handling, cross-split leakage checks, output schemas, repeatability and raw-source preservation. Day 2 tests train lightweight real models, inspect exactly which text/labels are fitted, exclude held-out vocabulary and answers, validate saved artifacts, and check deterministic inference with actual probabilities. Day 3 tests execute all three search methods on a tiny fixture, verify saved optimized models and test metrics, and check that search fitting does not use test rows.

For a manual integration check, submit a valid ticket and inspect the received status; try an empty/whitespace-only ticket for validation; stop the backend and submit again to see the connection error, then restart it and retry.

## Docker

From the repository root, with Docker Desktop running Linux containers:

```powershell
docker compose up --build -d
docker compose ps
docker compose logs
```

Open http://localhost:5173; API docs remain at http://localhost:8000/docs. Compose waits for the backend health check before starting the Nginx frontend. It builds the backend on Python 3.12 and the frontend on Node 22. No ML or database containers are added.

```powershell
docker compose down
```

For another browser-accessible backend address, set `$env:VITE_API_BASE_URL='http://your-host:8000'` before rebuilding and update the backend CORS origins in Compose to match the frontend origin. `backend` is a container hostname, not the browser API address.

## Data and limitations

- [Dataset format and sample-data boundaries](docs/dataset.md)
- [Architecture diagram and extension points](docs/architecture.md)
- [Assumptions and decisions](docs/assumptions.md)
- [Baseline ML experiment and results](docs/baseline-ml.md)
- [Completed Days 1-3 and remaining Days 4-13](docs/development-plan.md)
- [Validation results and browser-check limitation](docs/validation.md)

The selected Kaggle tickets and legacy 20-row sample are synthetic development data; the four knowledge-base documents are fictional sample content, not business policy. The API does not read the raw or processed files or load models. Offline baseline classification and optimization are implemented. API integration, embeddings, RAG, agents, security, explainability and MLOps remain planned. No persistence, authentication, deep learning, retrieval, fairness, monitoring or CI/CD runs now.

Day 2 uses the unchanged selected Kaggle dataset splits. Results on synthetic data do not establish real-world performance. A restrictive placeholder LICENSE is included; the project owner can select an open-source license if needed. See the validation record for verification details.
