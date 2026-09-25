import importlib.util
import json
from pathlib import Path

from app.monitoring.metrics import MonitoringMetrics
from app.monitoring.mlflow_tracking import run_mlflow_tracking
from app.monitoring.mlops_metrics import load_ml_metrics, write_ml_metrics


ROOT = Path(__file__).resolve().parents[2]


def test_ml_metrics_summary_loads_existing_artifact_metrics():
    metrics = load_ml_metrics()
    assert metrics["classification"]["category"]["macro_f1"] == 0.6415383542430868
    assert metrics["classification"]["priority"]["macro_f1"] == 0.6622079075109244
    assert metrics["retrieval"]["recall@1"] == 0.656
    assert metrics["retrieval"]["mrr"] == 0.7224999999999998


def test_write_ml_metrics_outputs_json_and_markdown(tmp_path):
    metrics = write_ml_metrics(tmp_path)
    assert (tmp_path / "ml_metrics.json").is_file()
    assert (tmp_path / "ml_metrics.md").is_file()
    saved = json.loads((tmp_path / "ml_metrics.json").read_text(encoding="utf-8"))
    assert saved == metrics


def test_mlflow_tracking_configuration_writes_result(tmp_path):
    output = run_mlflow_tracking(output_dir=tmp_path / "mlops", mlflow_dir=tmp_path / "mlflow")
    assert output["config"]["local_only"] is True
    assert "file:///" in output["config"]["tracking_uri"]
    assert output["result"]["tracked_metrics"]["category_test_macro_f1"] == 0.6415383542430868
    assert (tmp_path / "mlops" / "mlflow_config.json").is_file()
    assert (tmp_path / "mlops" / "mlflow_tracking_result.json").is_file()


def test_monitoring_counters_and_latency_tracking():
    metrics = MonitoringMetrics()
    metrics.record_request(0.2)
    metrics.record_request(0.4, error=True)
    metrics.record_model_prediction()
    with metrics.time_retrieval():
        pass
    snapshot = metrics.snapshot()
    assert snapshot["request_count"] == 2
    assert snapshot["request_error_count"] == 1
    assert snapshot["request_latency_seconds_average"] == 0.30000000000000004
    assert snapshot["model_prediction_count"] == 1
    assert snapshot["retrieval_request_count"] == 1
    assert snapshot["retrieval_latency_seconds_total"] >= 0.0


def test_health_endpoint_reports_service_and_artifact_status(client):
    body = client.get("/health").json()
    assert body["status"] == "healthy"
    assert body["service"] == "AI Customer Support Assistant"
    assert set(body) >= {"classifier_artifacts_available", "retrieval_index_available"}


def test_airflow_dag_source_structure():
    dag_path = ROOT / "airflow/dags/ticket_processing.py"
    source = dag_path.read_text(encoding="utf-8")
    assert 'dag_id="ticket_processing_embedding_refresh"' in source
    for task_id in ("validate_data", "prepare_ticket_data", "refresh_embeddings", "validate_retrieval_index"):
        assert f'task_id="{task_id}"' in source
    assert "validate_data_task >> prepare_ticket_data_task >> refresh_embeddings_task >> validate_retrieval_index_task" in source


def test_airflow_import_if_available():
    if importlib.util.find_spec("airflow") is None:
        return
    spec = importlib.util.spec_from_file_location("ticket_processing_dag", ROOT / "airflow/dags/ticket_processing.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.dag.dag_id == "ticket_processing_embedding_refresh"
    assert {task.task_id for task in module.dag.tasks} == {
        "validate_data",
        "prepare_ticket_data",
        "refresh_embeddings",
        "validate_retrieval_index",
    }


def test_docker_and_ci_configuration_static_checks():
    dockerfile = (ROOT / "backend/Dockerfile").read_text(encoding="utf-8")
    workflow = (ROOT / ".github/workflows/backend-ci.yml").read_text(encoding="utf-8")
    assert "FROM python:3.12-slim" in dockerfile
    assert "uvicorn" in dockerfile
    assert "python -m pytest -c backend/pytest.ini -q backend/tests" in workflow
    assert "docker build -t ai-customer-support-backend ./backend" in workflow
