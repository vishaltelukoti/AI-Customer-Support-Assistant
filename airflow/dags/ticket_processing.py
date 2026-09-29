"""Daily POC refresh of the training-ticket FAISS retrieval index."""

from __future__ import annotations

import json
import sys
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
BACKEND_DIR = ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

PROCESSED_DIR = ROOT / "data/processed"
RETRIEVAL_DIR = ROOT / "experiments/retrieval"


def _retrieval_configuration(config_path: Path):
    from app.rag.retrieval import RetrievalConfig

    if not config_path.exists():
        return RetrievalConfig()
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    configuration = payload.get("configuration", {})
    return RetrievalConfig(**{
        key: configuration[key]
        for key in RetrievalConfig.__dataclass_fields__
        if key in configuration
    })


def validate_data(data_dir: Path = PROCESSED_DIR) -> str:
    required = [data_dir / name for name in ("train.csv", "validation.csv", "test.csv", "historical_tickets.csv")]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing processed data artifacts: {missing}")
    return "processed data artifacts present"


def prepare_ticket_data(data_dir: Path = PROCESSED_DIR) -> str:
    from app.rag.retrieval import load_historical_tickets, training_corpus

    rows = load_historical_tickets(data_dir / "historical_tickets.csv")
    return f"historical tickets validated; training corpus contains {len(training_corpus(rows))} tickets"


def refresh_embeddings(data_dir: Path = PROCESSED_DIR, artifact_dir: Path = RETRIEVAL_DIR, embedder=None) -> str:
    """Rebuild vectors and FAISS artifacts from training tickets only; never append."""
    from app.rag.retrieval import SimilarTicketRetriever, load_historical_tickets

    rows = load_historical_tickets(data_dir / "historical_tickets.csv")
    config = _retrieval_configuration(artifact_dir / "retrieval_config.json")
    retriever = SimilarTicketRetriever.build_index(rows, embedder=embedder, config=config)
    retriever.save(artifact_dir)
    return f"rebuilt {retriever.index.ntotal} training-ticket embeddings and FAISS vectors"


def validate_retrieval_index(data_dir: Path = PROCESSED_DIR, artifact_dir: Path = RETRIEVAL_DIR) -> str:
    from app.rag.retrieval import INDEX_FILENAME, METADATA_FILENAME, TicketMetadata, load_historical_tickets
    import faiss

    metadata_path = artifact_dir / METADATA_FILENAME
    index_path = artifact_dir / INDEX_FILENAME
    if not metadata_path.exists() or not index_path.exists():
        raise FileNotFoundError("Refreshed retrieval index or metadata is missing")
    metadata = [TicketMetadata(**json.loads(line)) for line in metadata_path.read_text(encoding="utf-8").splitlines() if line]
    index = faiss.read_index(str(index_path))
    train_ids = {row["ticket_id"] for row in load_historical_tickets(data_dir / "historical_tickets.csv") if row["split"] == "train"}
    metadata_ids = {item.ticket_id for item in metadata}
    if index.ntotal != len(metadata) or len(metadata_ids) != len(metadata):
        raise ValueError("FAISS index and retrieval metadata counts do not match")
    if metadata_ids - train_ids or any(item.split != "train" for item in metadata):
        raise ValueError("Retrieval index contains validation/test tickets or non-training metadata")
    return f"validated {index.ntotal} FAISS vectors with matching training-only metadata"


default_args = {"owner": "poc", "retries": 1, "retry_delay": timedelta(minutes=5)}

with DAG(
    dag_id="ticket_processing_embedding_refresh",
    description="Daily POC refresh of training-ticket embeddings and the FAISS retrieval index.",
    default_args=default_args,
    start_date=datetime(2026, 1, 1),
    schedule="@daily",
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
