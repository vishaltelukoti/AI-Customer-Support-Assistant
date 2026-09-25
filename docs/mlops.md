# MLOps Layer

Day 11 adds a lightweight local MLOps layer for the POC. It does not retrain models, rebuild retrieval, redesign RAG/agents, add cloud deployment, or introduce production infrastructure.

## MLflow

`backend/app/monitoring/mlflow_tracking.py` logs existing Day 3 classification and Day 6 retrieval metrics to a local file-based MLflow store under `experiments/mlflow/mlruns/`. The compact tracked-run metadata is saved in `experiments/mlops/mlflow_tracking_result.json`, and the local configuration is saved in `experiments/mlops/mlflow_config.json`.

Tracked metrics:

| Metric | Value |
| --- | ---: |
| Category test accuracy | 0.652795 |
| Category test Macro F1 | 0.641538 |
| Priority test accuracy | 0.676867 |
| Priority test Macro F1 | 0.662208 |
| Retrieval Recall@1 | 0.656000 |
| Retrieval Recall@3 | 0.764000 |
| Retrieval Recall@5 | 0.844000 |
| Retrieval MRR | 0.722500 |

Run locally:

```powershell
python -m backend.app.monitoring.mlflow_tracking
```

Inspect with:

```powershell
mlflow ui --backend-store-uri file:///.../experiments/mlflow/mlruns
```

The repository has a placeholder `mlflow/` directory, so the tracking module explicitly imports the installed MLflow package rather than the placeholder.

## Metrics Summary

`backend/app/monitoring/mlops_metrics.py` creates:

- `experiments/mlops/ml_metrics.json`
- `experiments/mlops/ml_metrics.md`

Metrics are loaded from existing artifacts:

- Classification: `experiments/optimization/optimization_results.json`
- Retrieval: `experiments/retrieval/retrieval_evaluation.json`

No training or retrieval evaluation is rerun.

## Airflow

Exactly one DAG was added: `airflow/dags/ticket_processing.py`.

```text
validate_data -> prepare_ticket_data -> refresh_embeddings -> validate_retrieval_index
```

The DAG validates processed data and existing retrieval artifacts. It does not automatically rebuild the FAISS index and does not retrain classifiers. A static validation note is saved in `experiments/mlops/airflow_validation.md`.

## Docker

The existing backend Dockerfile remains a simple FastAPI container:

```powershell
docker build -t ai-customer-support-backend-day11 ./backend
```

The Day 11 build succeeded, but it is heavy because existing ML dependencies include torch/transformers and, on Linux, large CUDA-related torch wheels.

## CI/CD

`.github/workflows/backend-ci.yml` demonstrates a simple backend CI flow:

1. check out repository
2. set up Python 3.12
3. install `backend/requirements.txt`
4. run backend tests
5. build the backend Docker image

It does not deploy to cloud infrastructure.

## Monitoring

`backend/app/monitoring/metrics.py` provides in-memory counters/timers:

- request count
- request error count
- request latency total/average
- model prediction count
- retrieval request count
- retrieval latency total/average

The FastAPI app records request counts and latency through a small middleware. Model and retrieval counters are programmatic utilities for later integration.

## Health

`GET /health` returns:

- status
- service name
- classifier artifact availability
- retrieval index availability

The health check is cheap and does not train models, rebuild indexes, retrieve tickets, or generate RAG output.

## Limitations

This is local-only POC MLOps. MLflow uses a local file store; Airflow is represented by one small DAG; monitoring is in-memory only; CI is a GitHub Actions demonstration; Docker uses the existing heavy ML dependency stack. There is no Kubernetes, Terraform, Prometheus, Grafana, cloud deployment, database-backed monitoring, or production secrets infrastructure.
