# AI Customer Support Assistant

A single-developer AI-powered customer-support POC. **Days 1-12 implement the application foundation, prepared dataset, optimized ML classification, local FAISS similar-ticket retrieval, simple grounded RAG, a lightweight LangGraph multi-agent workflow, deterministic POC security checks, model explainability/subgroup diagnostics, local MLOps, and an integrated FastAPI/UI flow.** The form submits a support-ticket subject and body, then displays classification, similar historical tickets, workflow route, suggested response, source attribution, security status, explanation terms, and timings. Tickets are not stored.

Planned later work is final demo/documentation polish. See [the development plan](docs/development-plan.md).

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
experiments/dl_comparison/ # Day 4 and Day 5 category DL comparisons
experiments/retrieval/ # Day 6 FAISS index, metadata, and retrieval evaluation
experiments/rag/ # Day 7 RAG configuration and evaluation
experiments/agents/ # Day 8 LangGraph workflow evaluation
experiments/security/ # Day 9 deterministic security evaluation
experiments/explainability/ # Day 10 explanations and subgroup diagnostics
experiments/mlops/ # Day 11 metric summaries and MLflow run metadata
experiments/mlflow/ # Day 11 local MLflow file store, internals ignored
experiments/integration/ # Day 12 integrated API/workflow evaluation
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

Open http://localhost:5173. Submit a fictional support-ticket subject and body to see the integrated POC response. Do not overwrite an existing customized `.env` when repeating setup. On macOS/Linux use `python3`, `.venv/bin/python`, and `cp` in place of their Windows equivalents.

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
| POST | /api/v1/tickets | HTTP 200 integrated support-assistant response; HTTP 422 on invalid input |

```powershell
Invoke-RestMethod http://localhost:8000/health
Invoke-RestMethod -Method Post -Uri http://localhost:8000/api/v1/tickets -ContentType 'application/json' -Body '{"subject":"Payment failed","body":"My card payment failed during checkout. Invoice unpaid.","top_k":3}'
```

Example response (the UUID changes per request):

```json
{
  "ticket": {
    "ticket_id": "d7658bf4-05d0-4f3f-8261-adc5811c1e4d",
    "subject": "Payment failed",
    "body": "My card payment failed during checkout. Invoice unpaid.",
    "ticket_text": "Payment failed\n\nMy card payment failed during checkout. Invoice unpaid."
  },
  "classification": {
    "category": "Billing and Payments",
    "category_confidence": 0.9963926576529412,
    "priority": "medium",
    "priority_confidence": 0.8322446406752919
  },
  "similar_tickets": [],
  "workflow": {
    "complexity": "simple",
    "type": "simple_rag",
    "trace": ["retrieval", "resolution"]
  },
  "response": {
    "answer": "Review the similar historical resolutions...",
    "sources": [],
    "retrieval_status": "grounded"
  },
  "explanation": {
    "predicted_category": "Billing and Payments",
    "confidence": 0.9963926576529412,
    "top_features": []
  },
  "security": {
    "allowed": true,
    "status": "safe",
    "reason": "Generated response passed output security check.",
    "category": "safe_output",
    "matched_rule": "none"
  },
  "timings": {
    "total_ms": 198.97,
    "classification_ms": 7.54,
    "retrieval_ms": 39.12,
    "workflow_ms": 151.64
  },
  "status": "completed"
}
```

The API accepts `{subject, body, top_k}` with `top_k` from 1 to 5. The legacy `{ticket_text}` shape is still accepted for compatibility. Blank, missing, non-string or oversized text and unexpected fields are rejected. The UI handles empty input, loading, success, validation errors, API failures and a 15-second timeout.

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

## Deep-learning comparison

Day 4 compares lightweight CNN, vanilla RNN and LSTM text classifiers for the category task only. Day 5 extends the same comparison with a small attention encoder and a pretrained DistilBERT classifier. All models use cleaned `ticket_text`, the unchanged English train/validation/test split, seed 42, and validation Macro-F1 for model selection. Test metrics are final reporting only. No priority DL model, hyperparameter tuning, or API integration is included.

TensorFlow is not compatible with the current Python 3.14 environment, so the CNN/RNN/LSTM/Attention paths use dependency-light NumPy sequence encoders with a trained Logistic Regression head. DistilBERT uses `distilbert-base-uncased`, freezes the base encoder, trains one epoch on a deterministic train-only cap of 60 examples per category, and evaluates validation/test on the full held-out splits.

Run from the repository root:

```powershell
.\backend\.venv\Scripts\python.exe -m backend.app.ml.dl_training
```

The command saves Day 5 artifacts under `experiments/dl_comparison/day5/` because the existing Day 4 artifacts on this machine are locked to SYSTEM/Administrators. The selected Day 5 model is saved as a Hugging Face Transformers artifact in `experiments/dl_comparison/day5/models/distilbert/`.

| Model | Validation macro-F1 | Test accuracy | Test macro-F1 | Test weighted-F1 |
| --- | ---: | ---: | ---: | ---: |
| CNN | 0.045488 | 0.289270 | 0.044873 | 0.129988 |
| RNN | 0.044922 | 0.289678 | 0.044922 | 0.130130 |
| LSTM | 0.044922 | 0.289678 | 0.044922 | 0.130130 |
| Attention | 0.044922 | 0.289678 | 0.044922 | 0.130130 |
| DistilBERT | 0.121775 | 0.194206 | 0.120542 | 0.187129 |

Selected Day 5 DL model: DistilBERT, selected based on validation Macro-F1. Its final test Macro-F1 is 0.120542. This remains far below the Day 3 optimized category TF-IDF result and is not production-ready. Full measured values and limitations are in [the generated Day 5 report](experiments/dl_comparison/day5/dl_comparison_results.md) and [the DL documentation](docs/deep-learning.md).

## Similar-ticket retrieval

Day 6 builds local similar-ticket retrieval for future RAG work. It embeds training-only historical ticket text with `sentence-transformers/all-MiniLM-L6-v2`, stores normalized 384-dimensional vectors in a FAISS `IndexFlatIP`, and saves metadata separately so search results include ticket ID, ticket text, category, priority, answer, source/version/type, tags, and similarity score. No validation or test tickets are indexed, and answer text is not used for embeddings.

Run from the repository root:

```powershell
.\backend\.venv\Scripts\python.exe -m backend.app.rag.retrieval
```

The command saves `tickets.faiss`, `metadata.jsonl`, `retrieval_config.json`, `retrieval_evaluation.json`, and `retrieval_evaluation.md` under `experiments/retrieval/`. Default Top-K is 3. The evaluation uses 250 validation and 250 test tickets as held-out queries; a retrieval is relevant when the returned training ticket has the same category as the query.

| Metric | Value |
| --- | ---: |
| Indexed training tickets | 11,436 |
| Embedding dimension | 384 |
| Recall@1 | 0.656000 |
| Recall@3 | 0.764000 |
| Recall@5 | 0.844000 |
| MRR | 0.722500 |

No API key is required for the local Sentence Transformer + FAISS retrieval implementation. Retrieval is not connected to FastAPI yet, and Day 6 does not implement RAG.

## Suggested-resolution RAG

Day 7 adds a small local RAG pipeline on top of the Day 6 FAISS index. The flow is: incoming `ticket_text` -> retrieve similar training historical tickets -> build bounded context with prior resolutions -> generate a concise suggested resolution with `google/flan-t5-base` -> return supporting source tickets. Retrieved answer text is allowed in the generation context as historical resolution evidence, but answers are still not used for retrieval embeddings.

Run from the repository root:

```powershell
.\backend\.venv\Scripts\python.exe -m backend.app.rag.rag_service --use-local-model-for-eval
```

The command saves `rag_config.json`, `rag_evaluation.json`, and `rag_evaluation.md` under `experiments/rag/`. Default Top-K is 3 and the POC retrieval threshold is 0.55. If no retrieved ticket meets the threshold, the service returns `insufficient_evidence` and does not fabricate a resolution.

| Metric | Value |
| --- | ---: |
| Source attribution rate | 1.000000 |
| Grounded acceptable response rate | 1.000000 |
| Insufficient-information pass rate | 1.000000 |
| Overall acceptance rate | 1.000000 |

No API key is required for the local retrieval plus local generation setup. RAG is not connected to FastAPI yet, and Day 7 does not implement LangGraph, agents, memory, streaming, or production observability.

## LangGraph agent workflow

Day 8 adds a lightweight LangGraph workflow for simple and complex tickets. It has three specialized agents: Retrieval Agent, Investigation Agent, and Resolution Agent. The Retrieval Agent exposes the existing Day 6 FAISS retriever as a callable tool. The Resolution Agent reuses the Day 7 RAG service. Shared state carries ticket text, complexity, retrieved sources, investigation result, final response, trace, status/error fields, and request-local memory.

Run from the repository root:

```powershell
.\backend\.venv\Scripts\python.exe -m backend.app.agents.workflow
```

Simple tickets route Retrieval -> Resolution. Complex tickets route Investigation -> Retrieval -> Resolution. The fixed POC evaluation covers simple payment failure, complex refund/cancellation, insufficient information, and prompt-injection-style input.

| Metric | Value |
| --- | ---: |
| Routing accuracy | 1.000000 |
| Workflow completion rate | 1.000000 |
| Source behavior rate | 1.000000 |
| Trace presence rate | 1.000000 |

Artifacts are saved under `experiments/agents/`. The workflow is not connected to FastAPI yet and does not implement persistent memory.

## Security layer

Day 9 adds deterministic POC security checks around the existing LangGraph workflow. Input checks block obvious prompt injection, jailbreak, and explicit secret-disclosure requests before the AI workflow runs. Explicit email, phone, and card-like values are detected and redacted before allowed workflow execution. Retrieved tickets are treated as untrusted data, and output checks replace obvious generated secrets or PII leakage with a safe fallback.

Run from the repository root:

```powershell
.\backend\.venv\Scripts\python.exe -m backend.app.security.security_service
```

The command saves `security_config.json`, `security_evaluation.json`, and `security_evaluation.md` under `experiments/security/`.

| Metric | Value |
| --- | ---: |
| Security test pass rate | 1.000000 |
| Attack detection rate | 1.000000 |
| Normal-ticket allow rate | 1.000000 |
| Malicious retrieved-content handling rate | 1.000000 |

No API key is required. This is a lightweight rule-based POC layer, not comprehensive security, and it is not connected to FastAPI yet.

## Explainability and subgroup diagnostics

Day 10 explains the saved Day 3 optimized category TF-IDF + Logistic Regression model without retraining. SHAP was attempted, but the current Python 3.14 Windows environment requires Microsoft C++ Build Tools to compile SHAP, so the implementation uses exact linear TF-IDF contribution scores: `tf-idf value * saved Logistic Regression coefficient`.

Run from the repository root:

```powershell
.\backend\.venv\Scripts\python.exe -m backend.app.explainability.shap_explainer
```

The command saves `explainability_config.json`, `explanations.json`, `explanations.md`, `fairness_evaluation.json`, and `fairness_evaluation.md` under `experiments/explainability/`.

| Result | Value |
| --- | ---: |
| Predictions explained | 5 |
| All explanations succeeded | 1 |
| Confidence minimum | 0.174072 |
| Confidence maximum | 0.996855 |

Subgroup diagnostics use only available non-sensitive metadata: `version` and `type`. The classification data is English-only, so it does not support a valid English-vs-German fairness comparison. No sensitive demographic attributes are inferred, and the subgroup metrics are not proof of fairness or bias.

## MLOps

Day 11 adds local MLflow tracking, a compact metrics summary, one Airflow DAG, a backend CI workflow, in-memory monitoring counters, and an expanded health endpoint. It reuses existing Day 3 classification metrics and Day 6 retrieval metrics; no models or indexes are rebuilt.

Run from the repository root:

```powershell
python -m backend.app.monitoring.mlops_metrics
python -m backend.app.monitoring.mlflow_tracking
```

Key tracked metrics:

| Metric | Value |
| --- | ---: |
| Category test Macro F1 | 0.641538 |
| Category test accuracy | 0.652795 |
| Priority test Macro F1 | 0.662208 |
| Priority test accuracy | 0.676867 |
| Retrieval Recall@1 / @3 / @5 | 0.656000 / 0.764000 / 0.844000 |
| Retrieval MRR | 0.722500 |

The single DAG is `ticket_processing_embedding_refresh`: `validate_data -> prepare_ticket_data -> refresh_embeddings -> validate_retrieval_index`. The CI workflow is `.github/workflows/backend-ci.yml`. Docker build validation succeeded with `docker build -t ai-customer-support-backend-day11 ./backend`, though the image is large because existing ML dependencies pull torch/transformers stacks.

## Integrated assistant API

Day 12 wires the prior POC pieces into the `/api/v1/tickets` API and the React UI. The backend flow is input security -> Day 3 optimized category/priority classification -> Day 6 FAISS retrieval -> deterministic complexity routing -> simple Day 7 RAG or complex Day 8 LangGraph workflow -> output security -> Day 10-style explanation -> structured API response. Day 11 monitoring records request, model-prediction and retrieval counters plus latencies.

Run from the repository root:

```powershell
python -m backend.app.services.ticket_service
```

The command saves `integration_config.json`, `integration_evaluation.json`, and `integration_evaluation.md` under `experiments/integration/`.

| Metric | Value |
| --- | ---: |
| Fixed integration cases | 5 |
| Passed | 5 |
| Average latency | 78.41 ms |
| Default Top-K | 3 |

The fixed evaluation covers simple payment failure, complex multiple-charge/refund/cancellation routing, prompt-injection blocking, insufficient evidence, and PII redaction. No API key is required. This is POC integration, not a production support system.

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

Tests cover health, schema, integrated ticket responses, unique IDs, whitespace normalization, invalid requests, input length boundaries, malformed JSON, allowed/rejected CORS origins, security blocks, insufficient evidence, and component-failure fallback. The frontend build includes strict TypeScript checking. `npm run test:api` requires the backend running and exercises the actual Axios service against it, including API validation errors. With the backend stopped, `npm run test:api -- --unavailable` checks connection-error handling. These service checks are not browser/UI tests.

Preprocessing tests use small fixtures and cover schema detection, English filtering, version overlap, conservative cleaning, privacy masks, exact deduplication, conflicting labels, grouped queue/priority splits, rare-class handling, cross-split leakage checks, output schemas, repeatability and raw-source preservation. Day 2 tests train lightweight real models, inspect exactly which text/labels are fitted, exclude held-out vocabulary and answers, validate saved artifacts, and check deterministic inference with actual probabilities. Day 3 tests execute all three search methods on a tiny fixture, verify saved optimized models and test metrics, and check that search fitting does not use test rows. Day 5 DL tests cover CNN/RNN/LSTM/Attention construction and inference, mocked DistilBERT configuration/artifact behavior, validation-based selection, leakage metadata, selected-model loading, and valid category prediction. Day 6 retrieval tests cover embedding shape, FAISS index creation/save/load, Top-K behavior, metadata mapping, search, no self-retrieval, training-only indexing, answer exclusion, and repeatable tiny-embedder behavior. Day 7 RAG tests cover context construction, retrieved answer inclusion, source attribution, configurable Top-K, output schema, insufficient-information behavior, prompt-injection resistance, malicious retrieved content handling, and fabricated-source prevention. Day 8 agent tests cover state, routing, agent nodes, retrieval tool usage, graceful retrieval/generation failure, insufficient evidence, request-local memory, and prompt-injection-as-data behavior. Day 9 security tests cover prompt injection, jailbreaks, secret requests, PII detection/redaction, normal-ticket allow behavior, malicious retrieved content as untrusted data, unsafe output fallback, blocked-workflow behavior, and deterministic checks. Day 10 explainability tests cover saved model loading, explainer initialization, vocabulary-backed feature explanations, confidence bounds, exactly five deterministic examples, allowed subgroup fields, no inferred sensitive attributes, and safe handling of small/empty groups. Day 11 MLOps tests cover metric loading, MLflow tracking config, monitoring counters/latency, health endpoint fields, Airflow DAG structure, and Docker/CI static checks. Day 12 API tests cover normal simple flow, complex flow, classification/similar-ticket/source fields, security blocks, insufficient information, invalid requests and graceful component failure.

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
- [Deep-learning comparisons](docs/deep-learning.md)
- [Similar-ticket retrieval](docs/retrieval.md)
- [Suggested-resolution RAG](docs/rag.md)
- [LangGraph agent workflow](docs/agents.md)
- [POC security layer](docs/security.md)
- [Explainability and subgroup diagnostics](docs/explainability.md)
- [MLOps layer](docs/mlops.md)
- [Integrated assistant API](docs/integration.md)
- [Completed Days 1-12 and remaining Day 13](docs/development-plan.md)
- [Validation results and browser-check limitation](docs/validation.md)

The selected Kaggle tickets and legacy 20-row sample are synthetic development data; the four knowledge-base documents are fictional sample content, not business policy. The API now loads local saved models/artifacts for classification, retrieval, RAG/agents, security and explanations, but it does not persist tickets or authenticate users. The integration remains POC-scoped and should not be treated as production-ready support automation.

Day 2 uses the unchanged selected Kaggle dataset splits. Results on synthetic data do not establish real-world performance. A restrictive placeholder LICENSE is included; the project owner can select an open-source license if needed. See the validation record for verification details.
