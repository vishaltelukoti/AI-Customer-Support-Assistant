"""Single Day 11 POC Airflow DAG for ticket data and retrieval artifact checks."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

try:
    from airflow import DAG
    from airflow.operators.python import PythonOperator
except ImportError:
    class DAG:
        def __init__(self, dag_id, **kwargs):
            self.dag_id = dag_id
            self.tasks = []

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

    class PythonOperator:
        def __init__(self, task_id, python_callable):
            self.task_id = task_id
            self.python_callable = python_callable
            dag.tasks.append(self)

        def __rshift__(self, other):
            return other

ROOT = Path(__file__).resolve().parents[2]


def validate_data() -> str:
    required = [
        ROOT / "data/processed/train.csv",
        ROOT / "data/processed/validation.csv",
        ROOT / "data/processed/test.csv",
        ROOT / "data/processed/historical_tickets.csv",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing processed data artifacts: {missing}")
    return "processed data artifacts present"


def prepare_ticket_data() -> str:
    historical = ROOT / "data/processed/historical_tickets.csv"
    if historical.stat().st_size <= 0:
        raise ValueError("historical_tickets.csv is empty")
    return "historical ticket metadata available"


def refresh_embeddings() -> str:
    index = ROOT / "experiments/retrieval/tickets.faiss"
    metadata = ROOT / "experiments/retrieval/metadata.jsonl"
    if not index.exists() or not metadata.exists():
        raise FileNotFoundError("retrieval artifacts are missing; run the Day 6 retrieval pipeline manually")
    return "existing retrieval artifacts available; no automatic rebuild performed"


def validate_retrieval_index() -> str:
    config = ROOT / "experiments/retrieval/retrieval_config.json"
    evaluation = ROOT / "experiments/retrieval/retrieval_evaluation.json"
    if not config.exists() or not evaluation.exists():
        raise FileNotFoundError("retrieval configuration/evaluation artifacts are missing")
    return "retrieval index metadata validated"


default_args = {"owner": "poc", "retries": 1, "retry_delay": timedelta(minutes=5)}

with DAG(
    dag_id="ticket_processing_embedding_refresh",
    description="POC validation flow for processed ticket data and existing retrieval artifacts.",
    default_args=default_args,
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    tags=["poc", "customer-support", "mlops"],
) as dag:
    validate_data_task = PythonOperator(task_id="validate_data", python_callable=validate_data)
    prepare_ticket_data_task = PythonOperator(task_id="prepare_ticket_data", python_callable=prepare_ticket_data)
    refresh_embeddings_task = PythonOperator(task_id="refresh_embeddings", python_callable=refresh_embeddings)
    validate_retrieval_index_task = PythonOperator(
        task_id="validate_retrieval_index",
        python_callable=validate_retrieval_index,
    )

    validate_data_task >> prepare_ticket_data_task >> refresh_embeddings_task >> validate_retrieval_index_task
