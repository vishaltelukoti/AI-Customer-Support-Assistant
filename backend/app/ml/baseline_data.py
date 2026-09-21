"""Strict loading of the existing classification splits; never reads answers."""

import csv
import hashlib
import io
import json
from dataclasses import dataclass
from pathlib import Path

from .dataset_audit import CATEGORIES, PRIORITIES, clean_text, mask_pii, normalize_text
from .preprocessing import COLUMNS, ROOT, SPLIT_NAMES

BASELINE_DIR = ROOT / "experiments/baseline"
MODEL_FILES = {target: f"{target}_tfidf_logreg.joblib" for target in ("category", "priority")}


def prepare_ticket_text(text: str) -> str:
    """Reuse the stateless dataset cleaner and masks for training and inference."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("ticket_text must be a nonempty string.")
    prepared = mask_pii(clean_text(text))[0]
    if not prepared:
        raise ValueError("ticket_text must contain non-whitespace content.")
    return prepared


@dataclass(frozen=True)
class ClassificationSplit:
    ticket_ids: list[str]
    texts: list[str]
    categories: list[str]
    priorities: list[str]
    sha256: str


def load_classification_split(path: Path) -> ClassificationSplit:
    """Accept only the four-column contract, validating every row and label."""
    try:
        content = path.read_bytes()
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Missing processed split: {path}. Run preprocessing first.") from exc
    reader = csv.DictReader(io.StringIO(content.decode("utf-8-sig")), strict=True)
    if reader.fieldnames is None or len(reader.fieldnames) != len(COLUMNS) or set(reader.fieldnames) != set(COLUMNS):
        raise ValueError(f"{path.name}: expected exactly {', '.join(COLUMNS)}; answers/extra features are forbidden.")
    ids, texts, categories, priorities = [], [], [], []
    for number, row in enumerate(reader, start=1):
        if None in row or any(value is None for value in row.values()):
            raise ValueError(f"{path.name}, record {number}: malformed CSV row.")
        if not row["ticket_id"].strip():
            raise ValueError(f"{path.name}, record {number}: missing ticket_id.")
        if row["category"] not in CATEGORIES or row["priority"] not in PRIORITIES:
            raise ValueError(f"{path.name}, record {number}: unsupported category or priority.")
        ids.append(row["ticket_id"])
        texts.append(prepare_ticket_text(row["ticket_text"]))
        categories.append(row["category"])
        priorities.append(row["priority"])
    if not ids:
        raise ValueError(f"{path.name}: empty split.")
    if set(categories) != set(CATEGORIES) or set(priorities) != set(PRIORITIES):
        raise ValueError(f"{path.name}: all 10 categories and 3 priorities must be represented.")
    return ClassificationSplit(ids, texts, categories, priorities, hashlib.sha256(content).hexdigest())


def load_splits(data_dir: Path) -> tuple[dict[str, ClassificationSplit], bool]:
    """Load fixed splits, reject duplicate IDs/text and check audit hashes if present."""
    splits = {name: load_classification_split(data_dir / f"{name}.csv") for name in SPLIT_NAMES}
    texts, ids = set(), set()
    for name, split in splits.items():
        for ticket_id, text in zip(split.ticket_ids, split.texts):
            key = normalize_text(text)
            if key in texts or ticket_id in ids:
                raise ValueError(f"Duplicate ticket ID or normalized text in/across splits at {name}: {ticket_id}.")
            texts.add(key)
            ids.add(ticket_id)
    audit_path = data_dir / "dataset_audit.json"
    if audit_path.exists():
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        for name, split in splits.items():
            if audit.get("output_sha256", {}).get(f"{name}.csv") != split.sha256:
                raise ValueError(f"{name}.csv differs from the audited split; restore or explicitly regenerate preprocessing.")
    return splits, audit_path.exists()
