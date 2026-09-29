"""Read-only lightweight monitoring endpoint for the POC."""

from typing import Literal

from fastapi import APIRouter
from pydantic import BaseModel

from app.monitoring.metrics import monitor
from app.monitoring.mlops_metrics import load_ml_metrics

router = APIRouter(tags=["monitoring"])


class QualityMetric(BaseModel):
    metric: Literal["macro_f1", "recall_at_3"]
    value: float


class MonitoringResponse(BaseModel):
    request_count: int
    error_count: int
    average_latency_ms: float
    model_prediction_count: int
    retrieval_request_count: int
    average_retrieval_latency_ms: float
    model_quality: QualityMetric
    retrieval_quality: QualityMetric


@router.get("/monitoring", response_model=MonitoringResponse)
def monitoring() -> MonitoringResponse:
    """Expose process-local counters and existing offline evaluation metrics."""
    snapshot = monitor.snapshot()
    metrics = load_ml_metrics()
    return MonitoringResponse(
        request_count=snapshot["request_count"],
        error_count=snapshot["request_error_count"],
        average_latency_ms=snapshot["request_latency_seconds_average"] * 1000,
        model_prediction_count=snapshot["model_prediction_count"],
        retrieval_request_count=snapshot["retrieval_request_count"],
        average_retrieval_latency_ms=snapshot["retrieval_latency_seconds_average"] * 1000,
        model_quality=QualityMetric(
            metric="macro_f1",
            value=metrics["classification"]["category"]["macro_f1"],
        ),
        retrieval_quality=QualityMetric(
            metric="recall_at_3",
            value=metrics["retrieval"]["recall@3"],
        ),
    )
