from pathlib import Path
from typing import Literal

from fastapi import APIRouter
from pydantic import BaseModel

from app.ml.preprocessing import ROOT

router = APIRouter(tags=["health"])


class HealthResponse(BaseModel):
    status: Literal["healthy"] = "healthy"
    service: str = "AI Customer Support Assistant"
    classifier_artifacts_available: bool = False
    retrieval_index_available: bool = False


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    classifier_dir = ROOT / "experiments/optimization/models"
    retrieval_dir = ROOT / "experiments/retrieval"
    return HealthResponse(
        classifier_artifacts_available=(
            (classifier_dir / "category_optimized.joblib").exists()
            and (classifier_dir / "priority_optimized.joblib").exists()
        ),
        retrieval_index_available=(
            (retrieval_dir / "tickets.faiss").exists()
            and (retrieval_dir / "metadata.jsonl").exists()
        ),
    )
