"""Read-only dataset audit and lightweight, offline text/privacy helpers."""

import csv
import hashlib
import math
import re
import statistics
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path


CATEGORIES = (
    "Technical Support", "Product Support", "Customer Service", "IT Support",
    "Billing and Payments", "Returns and Exchanges", "Service Outages and Maintenance",
    "Sales and Pre-Sales", "Human Resources", "General Inquiry",
)
PRIORITIES = ("low", "medium", "high")
TAGS = tuple(f"tag_{i}" for i in range(1, 9))
REQUIRED = ("subject", "body", "answer", "type", "queue", "priority", "language", "version")
SOURCE_URL = "https://www.kaggle.com/datasets/tobiasbueck/multilingual-customer-support-tickets"
PROVENANCE_URL = "https://softoft.de/blog/ticket-dataset/"

# Ordered substitutions: specific patterns consume spans before broad numeric rules.
# Counts are non-overlapping occurrences actually masked, not claims of identity.
PII_PATTERNS = (
    ("email", re.compile(r"(?<![\w.+-])[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}")),
    ("url", re.compile(r"\b(?:https?://|www\.)[^\s<>]+", re.I)),
    ("account_number", re.compile(
        r"\b(?:account|acct|iban)\s*(?:number|no\.?|id|#)?\s*[:#-]?\s*"
        r"(?:[A-Z]{2}\d{2}(?:[ -]?[A-Z0-9]){8,30}|[A-Z]{0,3}\d[\dA-Z-]{4,})\b", re.I)),
    ("card_like", re.compile(r"(?<!\w)(?:\d[ -]?){12,18}\d(?!\w)")),
    ("phone", re.compile(
        r"(?<!\w)(?:\+\d{1,3}[ .-]?)?(?:\(\d{2,4}\)[ .-]?)?"
        r"\d{3}[ .-]\d{3}[ .-]\d{4}(?!\w)|(?<!\w)\+\d(?:[ ()-]?\d){7,14}(?!\w)")),
    ("long_numeric_id", re.compile(r"(?<!\w)\d{7,}(?!\w)")),
)


def as_text(value: object) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value)


def clean_text(value: object) -> str:
    """Preserve paragraphs and content; repair observed escaped email formatting.

    Literal escapes are decoded outside tokens with a drive or UNC prefix
    (potential filesystem paths).
    """
    text = as_text(value).replace("\r\n", "\n").replace("\r", "\n")
    # Protect Windows paths; never apply a general unicode_escape decoder.
    paths: list[str] = []

    def protect(match: re.Match) -> str:
        paths.append(match.group())
        return f"\x00PATH{len(paths) - 1}\x00"

    text = re.sub(r"\b[A-Za-z]:\\[^\s]+|\\\\[^\s]+", protect, text)
    text = text.replace(r"\r\n", "\n").replace(r"\n", "\n").replace(r"\t", " ")
    text = "\n".join(re.sub(r"[^\S\n]+", " ", line).strip() for line in text.split("\n"))
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    for index, path in enumerate(paths):
        text = text.replace(f"\x00PATH{index}\x00", path)
    return text


def normalize_text(value: object) -> str:
    """Whitespace-insensitive identity key; output itself retains paragraphs."""
    return " ".join(clean_text(value).split())


def combine_text(subject: object, body: object) -> str:
    return "\n\n".join(part for part in (clean_text(subject), clean_text(body)) if part)


def mask_pii(text: str) -> tuple[str, dict[str, int]]:
    counts = {}
    for name, pattern in PII_PATTERNS:
        text, counts[name] = pattern.subn(f"[{name.upper()}]", text)
    return text, counts


def load_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source, strict=True)
        columns = reader.fieldnames or []
        missing = set(REQUIRED) - set(columns)
        if missing:
            raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")
        if len(set(columns)) != len(columns):
            raise ValueError("Duplicate CSV column names are not supported.")
        rows = []
        for record, row in enumerate(reader, start=1):
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"CSV data record {record} does not match header width.")
            rows.append(row)
    if not rows:
        raise ValueError("The CSV contains no data records.")
    return columns, rows


def select_english(rows: list[dict[str, str]]) -> list[tuple[int, dict[str, str]]]:
    languages = {row.get("language", "") for row in rows}
    if not languages <= {"en", "de"} or "en" not in languages:
        raise ValueError(
            "Unreliable or missing language labels: expected exact en/de values and English rows. "
            "Review the source language field; no language inference is performed."
        )
    return [(index, row) for index, row in enumerate(rows, start=1) if row["language"] == "en"]


def distribution(values: list[str]) -> dict:
    counts = Counter(values)
    total = len(values)
    return {value: {"count": count, "percentage": round(100 * count / total, 6)}
            for value, count in sorted(counts.items())}


def text_statistics(values: list[str]) -> dict:
    lengths = [len(clean_text(value)) for value in values]
    return {
        "missing_count": sum(value == "" for value in values),
        "empty_after_cleaning": lengths.count(0),
        "min_length": min(lengths, default=0), "max_length": max(lengths, default=0),
        "mean_length": round(statistics.mean(lengths), 6) if lengths else 0,
        "median_length": statistics.median(lengths) if lengths else 0,
        "short_nonempty_under_20_chars": sum(0 < n < 20 for n in lengths),
        "long_over_10000_chars": sum(n > 10000 for n in lengths),
    }


def overlap_statistics(rows: list[dict[str, str]], keys: list) -> dict:
    groups = defaultdict(list)
    for index, key in enumerate(keys):
        if key:  # Missing subjects are not evidence of identity.
            groups[key].append(index)
    duplicates = [indices for indices in groups.values() if len(indices) > 1]
    cross = [indices for indices in duplicates if len({rows[i]["version"] for i in indices}) > 1]
    pairs = Counter()
    for indices in cross:
        for left, right in combinations(sorted({rows[i]["version"] for i in indices}), 2):
            pairs[f"{left}|{right}"] += 1
    return {
        "groups": len(duplicates), "rows_in_groups": sum(map(len, duplicates)),
        "excess_rows": sum(len(indices) - 1 for indices in duplicates),
        "cross_version_groups": len(cross), "cross_version_rows": sum(map(len, cross)),
        "version_pair_shared_keys": dict(sorted(pairs.items())),
        "conflicting_label_groups": sum(len({(rows[i]['queue'], rows[i]['priority']) for i in indices}) > 1
                                        for indices in duplicates),
        "source_record_groups": [[i + 1 for i in indices] for indices in duplicates],
    }


def template_key(text: str) -> str:
    """Only obvious long numeric-substitution templates; no semantic similarity."""
    text = normalize_text(mask_pii(text)[0])
    return re.sub(r"\d+", "[NUMBER]", text) if len(text) >= 40 else ""


def duplicate_audit(rows: list[dict[str, str]], columns: list[str]) -> dict:
    combined = [combine_text(row["subject"], row["body"]) for row in rows]
    keys = {
        "exact_rows": [tuple(row[column] for column in columns) for row in rows],
        "subject_body": [(row["subject"], row["body"]) if row["subject"] or row["body"] else "" for row in rows],
        "normalized_ticket_text": [normalize_text(text) for text in combined],
        "exact_subject": [row["subject"] for row in rows],
        "normalized_subject": [normalize_text(row["subject"]) for row in rows],
        "normalized_body": [normalize_text(row["body"]) for row in rows],
        "obvious_template": [template_key(text) for text in combined],
        "obvious_body_template": [template_key(row["body"]) for row in rows],
    }
    result = {name: overlap_statistics(rows, values) for name, values in keys.items()}
    result["ticket_id"] = (overlap_statistics(rows, [row["ticket_id"] for row in rows])
                           if "ticket_id" in columns else {"available": False, "reason": "No source ticket ID column."})
    return result


def leakage_report(columns: list[str]) -> list[dict]:
    reasons = {
        "subject": (True, "Customer email subject available at submission."),
        "body": (True, "Customer email message available at submission."),
        "answer": (False, "Agent response after submission; future historical retrieval only."),
        "queue": (False, "Routing label; category prediction target, not an input."),
        "priority": (False, "Assigned urgency label; priority prediction target, not an input."),
        "type": (False, "Source describes agent-picked ticket type; creation-time availability not established."),
        "language": (False, "Used only for English selection, not a classification feature."),
        "version": (False, "Dataset provenance/version, not customer-provided ticket content."),
        "ticket_id": (False, "Identifier for provenance/grouping only; not an input feature."),
    }
    return [{"field": column, "classification_input": reasons.get(column, (False, "Assigned tag/metadata; creation-time availability not established."))[0],
             "reason": reasons.get(column, (False, "Assigned tag/metadata; creation-time availability not established."))[1]}
            for column in columns]


def audit_dataset(path: Path, columns: list[str], rows: list[dict[str, str]]) -> dict:
    selected = select_english(rows)
    english = [row for _, row in selected]
    english_duplicates = duplicate_audit(english, columns)
    for statistics_ in english_duplicates.values():
        if "source_record_groups" in statistics_:
            statistics_["source_record_groups"] = [
                [selected[i - 1][0] for i in group] for group in statistics_["source_record_groups"]
            ]
    schema = []
    for column in columns:
        values = [row[column] for row in rows]
        nonempty = [value for value in values if value.strip()]
        missing = len(values) - len(nonempty)
        inferred = "integer-like string" if nonempty and all(re.fullmatch(r"[+-]?\d+", v) for v in nonempty) else "string"
        schema.append({"name": column, "dtype": inferred, "storage_type": "str",
                       "missing_count": missing, "missing_percentage": round(100 * missing / len(rows), 6),
                       "unique_nonmissing": len(set(nonempty))})
    return {
        "dataset": {"name": "Customer IT Support - Ticket Dataset", "creator": "Tobias Bueck",
                    "source_url": SOURCE_URL, "synthetic_provenance_url": PROVENANCE_URL,
                    "synthetic": True, "filename": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "file_size_bytes": path.stat().st_size, "rows": len(rows), "columns": len(columns),
                    "text_payload_utf8_bytes": sum(len(value.encode('utf-8')) for row in rows for value in row.values()),
                    "memory_note": "UTF-8 value payload size excludes Python object overhead; not process RAM."},
        "schema": schema,
        "field_mapping": {"ticket_id": "ticket_id" if "ticket_id" in columns else None,
                          "subject": "subject", "body": "body", "answer": "answer", "category": "queue",
                          "priority": "priority", "ticket_type": "type", "language": "language", "version": "version",
                          "tags": [tag for tag in TAGS if tag in columns]},
        "languages": distribution([row["language"] for row in rows]),
        "english_selection": {"total_rows": len(rows), "english_rows": len(english),
                              "non_english_rows": len(rows) - len(english),
                              "retention_percentage": round(100 * len(english) / len(rows), 6),
                              "rule": "language == en; no automatic language inference"},
        "versions": distribution([row["version"] for row in rows]),
        "english_versions": distribution([row["version"] for row in english]),
        "raw_categories": distribution([row["queue"] for row in rows]),
        "english_categories": distribution([row["queue"] for row in english]),
        "raw_priorities": distribution([row["priority"] for row in rows]),
        "english_priorities": distribution([row["priority"] for row in english]),
        "duplicates_raw": duplicate_audit(rows, columns),
        "duplicates_english": english_duplicates,
        "language_reliability": {
            "missing_or_unsupported_labels": 0,
            "limitation": "Labels are trusted source metadata; no automatic language detection. Manual spot checks are documented in docs/dataset.md.",
        },
        "english_text_quality": {
            "subject": text_statistics([row["subject"] for row in english]),
            "body": text_statistics([row["body"] for row in english]),
            "ticket_text": text_statistics([combine_text(row["subject"], row["body"]) for row in english]),
        },
        "leakage_fields": leakage_report(columns),
    }
