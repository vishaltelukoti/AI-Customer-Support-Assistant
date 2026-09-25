"""Day 10 explainability and subgroup diagnostics."""

from .fairness import run_fairness_evaluation
from .shap_explainer import LinearTextExplainer, run_explainability

__all__ = ["LinearTextExplainer", "run_explainability", "run_fairness_evaluation"]
