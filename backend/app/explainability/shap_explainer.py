"""Linear-model explainability for the saved optimized TF-IDF classifier.

SHAP was attempted for Day 10, but the current Python environment requires a
local C++ toolchain to build it. This module uses the equivalent local linear
contribution method for TF-IDF + Logistic Regression: tf-idf value multiplied
by the saved class coefficient.
"""

from __future__ import annotations

import argparse
import json
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from ..ml.optimization import OPTIMIZED_MODEL_FILES
from ..ml.preprocessing import ROOT
from .fairness import load_historical_test_records, run_fairness_evaluation

EXPLAINABILITY_DIR = ROOT / "experiments/explainability"
OPTIMIZED_MODELS_DIR = ROOT / "experiments/optimization/models"
DATA_DIR = ROOT / "data/processed"
REPRESENTATIVE_QUANTILES = (0.0, 0.25, 0.5, 0.75, 1.0)


@dataclass(frozen=True)
class PredictionExplanation:
    ticket_id: str
    ticket_text: str
    actual_category: str
    predicted_category: str
    confidence: float
    second_best_category: str
    second_best_probability: float
    top_positive_features: list[dict[str, float | str]]
    top_negative_features: list[dict[str, float | str]]
    explanation_summary: str


class LinearTextExplainer:
    def __init__(self, model_path: Path = OPTIMIZED_MODELS_DIR / OPTIMIZED_MODEL_FILES["category"]):
        self.model_path = model_path
        self.model = joblib.load(model_path)
        self.vectorizer = self.model.named_steps["tfidf"]
        self.classifier = self.model.named_steps["classifier"]
        self.feature_names = np.asarray(self.vectorizer.get_feature_names_out())

    def explain(self, ticket_id: str, ticket_text: str, actual_category: str,
                top_n: int = 8) -> PredictionExplanation:
        probabilities = self.model.predict_proba([ticket_text])[0]
        class_index = int(np.argmax(probabilities))
        predicted = str(self.classifier.classes_[class_index])
        confidence = float(probabilities[class_index])
        second_index = int(np.argsort(probabilities)[-2])
        vector = self.vectorizer.transform([ticket_text])
        contributions = vector.multiply(self.classifier.coef_[class_index]).toarray()[0]
        nonzero = vector.toarray()[0] > 0

        positive = self._top_features(contributions, nonzero, descending=True, top_n=top_n)
        negative = self._top_features(contributions, nonzero, descending=False, top_n=top_n)
        summary = self._summary(predicted, confidence, positive, negative)
        return PredictionExplanation(
            ticket_id=ticket_id,
            ticket_text=ticket_text,
            actual_category=actual_category,
            predicted_category=predicted,
            confidence=confidence,
            second_best_category=str(self.classifier.classes_[second_index]),
            second_best_probability=float(probabilities[second_index]),
            top_positive_features=positive,
            top_negative_features=negative,
            explanation_summary=summary,
        )

    def _top_features(self, contributions: np.ndarray, nonzero: np.ndarray,
                      descending: bool, top_n: int) -> list[dict[str, float | str]]:
        candidates = np.where(nonzero & (contributions > 0 if descending else contributions < 0))[0]
        if candidates.size == 0:
            return []
        order = np.argsort(contributions[candidates])
        if descending:
            order = order[::-1]
        selected = candidates[order[:top_n]]
        return [
            {"feature": str(self.feature_names[index]), "contribution": float(contributions[index])}
            for index in selected
        ]

    @staticmethod
    def _summary(predicted: str, confidence: float, positive: list[dict[str, Any]],
                 negative: list[dict[str, Any]]) -> str:
        positive_terms = ", ".join(item["feature"] for item in positive[:3]) or "no active positive terms"
        negative_terms = ", ".join(item["feature"] for item in negative[:3]) or "no active opposing terms"
        return (
            f"Predicted {predicted} with model probability {confidence:.3f}. "
            f"Strongest supporting terms: {positive_terms}. "
            f"Strongest opposing terms: {negative_terms}."
        )


def select_representative_examples(model, texts: list[str],
                                   count: int = 5) -> list[int]:
    probabilities = model.predict_proba(texts)
    confidences = probabilities.max(axis=1)
    order = np.argsort(confidences)
    targets = [round(q * (len(order) - 1)) for q in REPRESENTATIVE_QUANTILES]
    selected: list[int] = []
    used_predictions: set[str] = set()
    predictions = model.predict(texts)
    for target in targets:
        candidates = sorted(range(len(order)), key=lambda pos: (abs(pos - target), pos))
        chosen = None
        for position in candidates:
            original_index = int(order[position])
            prediction = str(predictions[original_index])
            if original_index not in selected and prediction not in used_predictions:
                chosen = original_index
                break
        if chosen is None:
            for position in candidates:
                original_index = int(order[position])
                if original_index not in selected:
                    chosen = original_index
                    break
        if chosen is not None:
            selected.append(chosen)
            used_predictions.add(str(predictions[chosen]))
    if len(selected) != count:
        raise ValueError(f"Expected {count} representative examples, selected {len(selected)}.")
    return selected


def run_explainability(data_dir: Path = DATA_DIR, model_dir: Path = OPTIMIZED_MODELS_DIR,
                       output_dir: Path = EXPLAINABILITY_DIR) -> dict[str, Any]:
    test_records = load_historical_test_records(data_dir)
    texts = [row["ticket_text"] for row in test_records]
    explainer = LinearTextExplainer(model_dir / OPTIMIZED_MODEL_FILES["category"])
    indices = select_representative_examples(explainer.model, texts, count=5)
    explanations = [
        explainer.explain(
            test_records[index]["ticket_id"],
            test_records[index]["ticket_text"],
            test_records[index]["category"],
        ).__dict__
        for index in indices
    ]
    confidences = [item["confidence"] for item in explanations]
    config = {
        "model_explained": "Day 3 optimized category TF-IDF + Logistic Regression",
        "model_path": str((model_dir / OPTIMIZED_MODEL_FILES["category"]).relative_to(ROOT)),
        "method": "linear_tfidf_logistic_regression_contributions",
        "method_note": (
            "SHAP was attempted but could not be installed because this Python environment requires "
            "Microsoft C++ Build Tools to build SHAP. Contributions are computed exactly as TF-IDF value "
            "times the saved Logistic Regression coefficient for the predicted class."
        ),
        "selection": {
            "split": "test",
            "count": 5,
            "criteria": "Deterministic confidence quantiles [0, 0.25, 0.5, 0.75, 1.0], preferring distinct predicted categories.",
            "test_used_for_model_selection": False,
        },
        "confidence_note": (
            "Confidence is the saved Logistic Regression predict_proba value for the predicted class. "
            "These probabilities are not calibrated correctness probabilities."
        ),
        "environment": {"python": platform.python_version()},
    }
    result = {
        "config": config,
        "summary": {
            "predictions_explained": len(explanations),
            "all_explanations_succeeded": len(explanations) == 5,
            "confidence_min": float(min(confidences)),
            "confidence_max": float(max(confidences)),
        },
        "explanations": explanations,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "explainability_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (output_dir / "explanations.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (output_dir / "explanations.md").write_text(format_explanations_markdown(result), encoding="utf-8")
    return result


def format_explanations_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Day 10 Prediction Explanations",
        "",
        result["config"]["method_note"],
        "",
        f"Predictions explained: {result['summary']['predictions_explained']}",
        f"Confidence range: {result['summary']['confidence_min']:.6f} to {result['summary']['confidence_max']:.6f}",
        "",
    ]
    for item in result["explanations"]:
        lines.extend([
            f"## {item['ticket_id']}",
            "",
            f"- Actual category: {item['actual_category']}",
            f"- Predicted category: {item['predicted_category']}",
            f"- Confidence: {item['confidence']:.6f}",
            f"- Second-best: {item['second_best_category']} ({item['second_best_probability']:.6f})",
            f"- Summary: {item['explanation_summary']}",
            "",
            "Top supporting features:",
        ])
        for feature in item["top_positive_features"]:
            lines.append(f"- {feature['feature']}: {feature['contribution']:.6f}")
        lines.append("")
        lines.append("Top opposing features:")
        for feature in item["top_negative_features"]:
            lines.append(f"- {feature['feature']}: {feature['contribution']:.6f}")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Day 10 explainability and subgroup diagnostics.")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--model-dir", type=Path, default=OPTIMIZED_MODELS_DIR)
    parser.add_argument("--output-dir", type=Path, default=EXPLAINABILITY_DIR)
    args = parser.parse_args()
    explainability = run_explainability(args.data_dir, args.model_dir, args.output_dir)
    fairness = run_fairness_evaluation(args.data_dir, args.model_dir, args.output_dir)
    print(json.dumps({
        "predictions_explained": explainability["summary"]["predictions_explained"],
        "confidence_min": explainability["summary"]["confidence_min"],
        "confidence_max": explainability["summary"]["confidence_max"],
        "fairness_fields": list(fairness["fields"]),
    }, indent=2))


if __name__ == "__main__":
    main()
