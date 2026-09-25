from pathlib import Path

import joblib

from app.explainability.fairness import (
    DATA_DIR,
    NON_SENSITIVE_GROUP_FIELDS,
    SENSITIVE_ATTRIBUTES_NOT_INFERRED,
    load_historical_test_records,
    subgroup_metrics,
)
from app.explainability.shap_explainer import (
    OPTIMIZED_MODELS_DIR,
    LinearTextExplainer,
    run_explainability,
    select_representative_examples,
)
from app.ml.optimization import OPTIMIZED_MODEL_FILES


def test_saved_optimized_category_model_can_be_loaded():
    model = joblib.load(OPTIMIZED_MODELS_DIR / OPTIMIZED_MODEL_FILES["category"])
    assert set(model.named_steps) == {"tfidf", "classifier"}
    assert len(model.named_steps["tfidf"].get_feature_names_out()) > 1000


def test_linear_explainer_initializes():
    explainer = LinearTextExplainer()
    assert explainer.feature_names.size == explainer.classifier.coef_.shape[1]


def test_representative_prediction_can_be_explained_with_vocabulary_features():
    records = load_historical_test_records(DATA_DIR)
    explainer = LinearTextExplainer()
    explanation = explainer.explain(
        records[0]["ticket_id"],
        records[0]["ticket_text"],
        records[0]["category"],
    )
    vocabulary = set(explainer.feature_names)
    assert 0.0 <= explanation.confidence <= 1.0
    assert explanation.top_positive_features
    assert all(item["feature"] in vocabulary for item in explanation.top_positive_features)


def test_exactly_five_representative_examples_are_deterministic():
    records = load_historical_test_records(DATA_DIR)
    texts = [row["ticket_text"] for row in records]
    model = joblib.load(OPTIMIZED_MODELS_DIR / OPTIMIZED_MODEL_FILES["category"])
    first = select_representative_examples(model, texts)
    second = select_representative_examples(model, texts)
    assert first == second
    assert len(first) == 5
    assert len(set(first)) == 5


def test_run_explainability_writes_five_examples(tmp_path):
    result = run_explainability(output_dir=tmp_path)
    assert result["summary"]["predictions_explained"] == 5
    assert result["summary"]["all_explanations_succeeded"] is True
    assert (tmp_path / "explanations.json").is_file()
    assert (tmp_path / "explanations.md").is_file()


def test_fairness_grouping_uses_only_available_non_sensitive_fields():
    assert set(NON_SENSITIVE_GROUP_FIELDS) == {"version", "type"}
    forbidden = {"race", "gender", "age", "religion", "disability"}
    assert forbidden.issubset(set(SENSITIVE_ATTRIBUTES_NOT_INFERRED))


def test_subgroup_metrics_handle_small_and_empty_groups_safely():
    metrics = subgroup_metrics(
        y_true=["A", "A", "B"],
        y_pred=["A", "B", "B"],
        groups=["large", "small", ""],
        field="version",
        min_group_size=2,
    )
    assert metrics == []


def test_subgroup_metrics_calculate_when_threshold_met():
    metrics = subgroup_metrics(
        y_true=["A", "A", "B", "B"],
        y_pred=["A", "A", "A", "B"],
        groups=["v1", "v1", "v1", "v1"],
        field="version",
        min_group_size=2,
    )
    assert len(metrics) == 1
    assert metrics[0].sample_count == 4
    assert 0.0 <= metrics[0].f1_macro <= 1.0
