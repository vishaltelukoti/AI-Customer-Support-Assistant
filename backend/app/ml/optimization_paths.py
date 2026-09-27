"""Artifact locations shared by optimization and runtime inference."""

from .preprocessing import ROOT

OPTIMIZATION_DIR = ROOT / "experiments/optimization"
OPTIMIZED_MODEL_FILES = {
    "category": "category_optimized.joblib",
    "priority": "priority_optimized.joblib",
}
