"""Run representative sanity checks using saved baselines; never assume correctness."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from ..services.classification_service import ClassificationService
from .baseline_data import BASELINE_DIR

EXAMPLES = (
    "My payment failed while placing an order.",
    "I was charged twice for the same order.",
    "I cannot log into my account.",
    "My package has not arrived yet.",
    "Our production network is down and all employees are unable to work. Please restore service urgently.",
    "Where can I find the employee vacation policy and benefits handbook?",
    "Could you send pricing and a product demonstration for your enterprise plan?",
)


def predict_examples(model_dir: Path = BASELINE_DIR / "models") -> list[dict]:
    """Return actual predictions for fixed examples, without ground-truth claims."""
    service = ClassificationService(model_dir)
    return [{"input": text, **asdict(service.predict(text))} for text in EXAMPLES]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=BASELINE_DIR / "models")
    parser.add_argument("--output", type=Path, default=BASELINE_DIR / "example_predictions.json")
    args = parser.parse_args()
    try:
        examples = predict_examples(args.model_dir)
    except (ValueError, OSError) as exc:
        parser.exit(1, f"Inference failed: {exc}\n")
    content = json.dumps(examples, indent=2) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(content, encoding="utf-8")
    print(content)


if __name__ == "__main__":
    main()
