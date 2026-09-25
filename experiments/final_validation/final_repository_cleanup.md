# Final Repository Cleanup

Date: 2026-09-25

Status: PASS

This cleanup was limited to repository hygiene. No application logic, model code, dataset splits, generated metrics, API behavior, frontend behavior, or dependencies were intentionally changed.

## Repository Size

| Check | Result |
| --- | ---: |
| Before cleanup, excluding `.git` | 521.44 MB |
| After cleanup, excluding `.git` | 84.02 MB |
| Cleanup status | PASS |

## Removed

- `.pytest_cache`
- `backend/.pytest_cache`
- `backend/.venv`
- Python `__pycache__` directories
- `frontend/node_modules`
- `frontend/dist`
- `experiments/mlflow/mlruns`
- `experiments/dl_comparison/day5/models/distilbert`
- `data/raw/sample_tickets.csv`

The removed Day 5 DistilBERT directory was an unused checkpoint/tokenizer directory. The Day 5 result JSON, Markdown summary, selected-model metadata, and lightweight selected model artifact were retained.

## Retained Evidence And Runtime Artifacts

- Raw assessment dataset: `data/raw/aa_dataset-tickets-multi-lang-5-2-50-version.csv`
- Processed train/validation/test and historical-ticket CSVs under `data/processed/`
- Day 3 optimized classifier artifacts:
  - `experiments/optimization/models/category_optimized.joblib`
  - `experiments/optimization/models/priority_optimized.joblib`
- Day 5 DL comparison result artifacts under `experiments/dl_comparison/`
- Day 6 FAISS index and metadata:
  - `experiments/retrieval/tickets.faiss`
  - `experiments/retrieval/metadata.jsonl`
- Day 7 through Day 12 evaluation artifacts
- Backend, frontend, tests, Docker, CI, Airflow, MLflow helper, and documentation files

## Largest Retained Files

| File | Size | Reason |
| --- | ---: | --- |
| `data/raw/aa_dataset-tickets-multi-lang-5-2-50-version.csv` | 24.79 MB | Source dataset for traceability |
| `experiments/retrieval/tickets.faiss` | 16.75 MB | Required FAISS retrieval index |
| `data/processed/historical_tickets.csv` | 14.03 MB | Retrieval/explainability corpus |
| `experiments/retrieval/metadata.jsonl` | 12.41 MB | FAISS metadata mapping |
| `data/processed/train.csv` | 5.00 MB | Reproducible training split |
| `experiments/baseline/models/category_tfidf_logreg.joblib` | 3.51 MB | Baseline model evidence |
| `experiments/optimization/models/category_optimized.joblib` | 1.68 MB | Required optimized category model |
| `experiments/baseline/models/priority_tfidf_logreg.joblib` | 1.38 MB | Baseline model evidence |
| `data/processed/validation.csv` | 1.09 MB | Reproducible validation split |
| `data/processed/test.csv` | 1.07 MB | Reproducible test split |
| `experiments/optimization/models/priority_optimized.joblib` | 0.66 MB | Required optimized priority model |
| `experiments/dl_comparison/models/selected_model.joblib` | 0.61 MB | Lightweight selected comparison model |

## Required Artifact Check

| Artifact | Status |
| --- | --- |
| `experiments/optimization/optimization_results.json` | PASS |
| `experiments/optimization/models/category_optimized.joblib` | PASS |
| `experiments/optimization/models/priority_optimized.joblib` | PASS |
| `experiments/dl_comparison/day5/dl_comparison_results.json` | PASS |
| `experiments/dl_comparison/day5/selected_model_metadata.json` | PASS |
| `experiments/retrieval/tickets.faiss` | PASS |
| `experiments/retrieval/metadata.jsonl` | PASS |
| `experiments/retrieval/retrieval_evaluation.json` | PASS |
| `experiments/rag/rag_evaluation.json` | PASS |
| `experiments/agents/agent_evaluation.json` | PASS |
| `experiments/security/security_evaluation.json` | PASS |
| `experiments/explainability/explanations.json` | PASS |
| `experiments/explainability/fairness_evaluation.json` | PASS |
| `experiments/mlops/ml_metrics.json` | PASS |
| `experiments/mlops/mlflow_tracking_result.json` | PASS |
| `experiments/integration/integration_evaluation.json` | PASS |

No required assessment artifact was missing.

## Validation Results

| Validation | Result | Status |
| --- | --- | --- |
| Backend tests | `143 passed, 289 warnings in 59.53s` | PASS |
| Frontend build | `npm run build` passed after Windows/sandbox escalation; 87 modules transformed, built in 5.62s | PASS |
| Frontend API smoke | `npm run test:api -- --unavailable` passed | PASS |
| Optimized model load | Category and priority optimized models loaded | PASS |
| Retrieval load | FAISS index and 11,436 metadata rows loaded | PASS |
| API smoke | Health returned 200 healthy; ticket processing returned 200 completed | PASS |
| Docker build | `docker build -t ai-customer-support-backend-final ./backend` passed after Docker permission escalation | PASS |

## Cleanup Residue Check

Checked for `node_modules`, `dist`, `__pycache__`, `.pytest_cache`, `.venv`, and `mlruns`.

Result: `NO_CLEANUP_RESIDUE_FOUND`

## `.gitignore` Review

- Added explicit exceptions so required optimized Day 3 model artifacts are not ignored.
- Removed the obsolete exception for `data/raw/sample_tickets.csv`.
- Kept generated/bulky local paths ignored, including virtualenvs, frontend build outputs, MLflow run stores, and model checkpoint formats.

## Final Top-Level Repository Tree

```text
.
├── .github/
├── airflow/
├── backend/
├── data/
├── docs/
├── experiments/
├── frontend/
├── mlflow/
├── .gitignore
├── docker-compose.yml
├── LICENSE
└── README.md
```

## Warnings

- Optimized model loading emitted scikit-learn `InconsistentVersionWarning` because the artifacts were created with sklearn 1.9.1 and the current environment has sklearn 1.9.0.
- Retrieval/API smoke emitted a Hugging Face cached-model metadata network warning, but cached weights loaded and validation passed.
- Frontend build and Docker build required escalated execution because of local Windows/sandbox permissions, then passed.
- `frontend/node_modules` and `frontend/dist` were removed after validation. Run `npm ci` in `frontend/` only when local frontend development or rebuilding is needed.

## Overall Status

PASS. The repository is cleaned, required assessment artifacts are retained, validation passed, and no manual action is required for assessment review.
