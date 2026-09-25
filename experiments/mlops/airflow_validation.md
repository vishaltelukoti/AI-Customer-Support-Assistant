# Day 11 Airflow DAG Validation

The POC DAG lives at `airflow/dags/ticket_processing.py`.

## DAG

- DAG ID: `ticket_processing_embedding_refresh`
- Schedule: manual only (`schedule=None`)
- Retries: 1
- External services: none

## Tasks

```text
validate_data
-> prepare_ticket_data
-> refresh_embeddings
-> validate_retrieval_index
```

The DAG validates existing processed data and existing Day 6 retrieval artifacts. It does not retrain classifiers and does not automatically rebuild the FAISS index.

The current repository has an `airflow/` source directory, so the DAG includes a small fallback class for static import validation when the real Airflow package is not installed or is shadowed by the local folder. In a real Airflow environment, the DAG uses Airflow's `DAG` and `PythonOperator` classes.
