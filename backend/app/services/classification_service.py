"""Standalone baseline inference; intentionally not wired into the ticket API."""

from dataclasses import dataclass
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from ..ml.baseline_data import BASELINE_DIR, MODEL_FILES, prepare_ticket_text
from ..ml.dataset_audit import CATEGORIES, PRIORITIES


@dataclass(frozen=True)
class ClassificationPrediction:
    category: str
    category_confidence: float
    priority: str
    priority_confidence: float


class ClassificationService:
    """Load local trusted joblib pipelines once and expose independent probabilities."""

    def __init__(self, model_dir: Path = BASELINE_DIR / "models") -> None:
        self.models: dict[str, Pipeline] = {}
        for target, filename in MODEL_FILES.items():
            path = model_dir / filename
            if not path.is_file():
                raise FileNotFoundError(f"Missing {target} model: {path}. Run python -m backend.app.ml.baseline_models first.")
            try:
                model = joblib.load(path)
                if (not isinstance(model, Pipeline) or list(model.named_steps) != ["tfidf", "classifier"]
                        or not isinstance(model.named_steps['tfidf'], TfidfVectorizer)
                        or not isinstance(model.named_steps['classifier'], LogisticRegression)):
                    raise ValueError("Expected a TF-IDF + Logistic Regression pipeline.")
                check_is_fitted(model.named_steps['tfidf'], 'vocabulary_')
                check_is_fitted(model.named_steps['classifier'], 'classes_')
                expected = CATEGORIES if target == "category" else PRIORITIES
                if set(model.classes_) != set(expected):
                    raise ValueError(f"Unexpected {target} labels in artifact.")
            except Exception as exc:
                raise ValueError(f"Cannot load valid {target} model from {path}: {exc}") from exc
            self.models[target] = model

    def predict(self, ticket_text: str) -> ClassificationPrediction:
        """Predict classes and their actual predict_proba values (not calibrated certainty)."""
        text = prepare_ticket_text(ticket_text)
        prediction = {}
        for target, model in self.models.items():
            probabilities = model.predict_proba([text])[0]
            index = int(probabilities.argmax())
            prediction[target] = str(model.classes_[index])
            prediction[f"{target}_confidence"] = float(probabilities[index])
        return ClassificationPrediction(**prediction)
