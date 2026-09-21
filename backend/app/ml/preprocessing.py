"""Prepare the selected synthetic Kaggle dataset; no training or API integration."""

import argparse
import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

from .dataset_audit import (
    CATEGORIES, PRIORITIES, TAGS, PII_PATTERNS, audit_dataset, clean_text,
    combine_text, distribution, load_csv, mask_pii, normalize_text,
    select_english, template_key,
)

COLUMNS = ("ticket_id", "ticket_text", "category", "priority")
SPLIT_NAMES = ("train", "validation", "test")
SEED = 42
ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT = ROOT / "data/raw/aa_dataset-tickets-multi-lang-5-2-50-version.csv"
Ticket = dict[str, str]


@dataclass
class PreparationResult:
    records: list[Ticket]
    raw_count: int
    duplicates_removed: int
    invalid_records: list[dict]
    audit: dict


def identity_keys(row: Ticket) -> list[tuple[str, str]]:
    """Conservative grouping signals; shared subject/body alone never deletes rows."""
    values = {
        "source_id": row.get("source_ticket_id", ""),
        "text": normalize_text(row["ticket_text"]),
        "subject": normalize_text(row["subject"]),
        "body": normalize_text(row["body"]),
        "template": template_key(row["ticket_text"]),
        "body_template": template_key(row["body"]),
    }
    return [(name, value) for name, value in values.items() if value] + [
        tuple(key) for key in json.loads(row.get("identity_aliases", "[]"))
    ]


def assign_groups(records: list[Ticket]) -> dict:
    parents = list(range(len(records)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    seen = {}
    for index, row in enumerate(records):
        for key in identity_keys(row):
            if key in seen:
                parents[find(index)] = find(seen[key])
            else:
                seen[key] = index
    groups = defaultdict(list)
    for index, row in enumerate(records):
        groups[find(index)].append(row)
    for group in groups.values():
        group_id = min(row["ticket_id"] for row in group)
        for row in group:
            row["group_id"] = group_id
    return {"total_groups": len(groups), "multi_record_groups": sum(len(g) > 1 for g in groups.values()),
            "rows_in_multi_record_groups": sum(len(g) for g in groups.values() if len(g) > 1),
            "largest_group": max(map(len, groups.values()), default=0)}


def prepare_records(rows: list[dict[str, str]], source_hash: str) -> tuple[list[Ticket], dict]:
    records = []
    invalid = []
    duplicates = []
    seen = {}
    pii_counts = {field: Counter({name: 0 for name, _ in PII_PATTERNS}) for field in ("ticket_text", "answer")}
    pii_rows = {field: Counter({name: 0 for name, _ in PII_PATTERNS}) for field in pii_counts}
    for source_record, row in select_english(rows):
        subject, body = clean_text(row["subject"]), clean_text(row["body"])
        text = combine_text(subject, body)
        reasons = []
        if not any(character.isalnum() for character in text):
            reasons.append("no meaningful subject/body text")
        if row["queue"] not in CATEGORIES:
            reasons.append(f"unsupported or missing queue: {row['queue']!r}")
        if row["priority"] not in PRIORITIES:
            reasons.append(f"unsupported or missing priority: {row['priority']!r}")
        if not row["version"].strip():
            reasons.append("missing version")
        if reasons:
            invalid.append({"source_record": source_record, "reasons": reasons})
            continue
        masked_text, text_counts = mask_pii(text)
        answer, answer_counts = mask_pii(clean_text(row["answer"]))
        for field, counts in (("ticket_text", text_counts), ("answer", answer_counts)):
            pii_counts[field].update(counts)
            pii_rows[field].update({key: int(count > 0) for key, count in counts.items()})
        # Separate cleaned fields are internal grouping metadata, never classification features.
        subject, _ = mask_pii(subject)
        body, _ = mask_pii(body)
        key = normalize_text(masked_text)
        if key in seen:
            first = seen[key]
            if (first["category"], first["priority"]) != (row["queue"], row["priority"]):
                raise ValueError(
                    f"Conflicting labels for identical normalized/masked text at data records "
                    f"{first['source_record']} and {source_record}; review labels before splitting."
                )
            duplicates.append({"source_record": source_record, "kept_source_record": int(first["source_record"])})
            # Preserve all grouping evidence before removal, including an alternate source ID.
            aliases = json.loads(first.get("identity_aliases", "[]"))
            aliases.extend(identity_keys({"ticket_text": masked_text, "subject": subject, "body": body,
                                          "source_ticket_id": row.get("ticket_id", "")}))
            first["identity_aliases"] = json.dumps(aliases)
            continue
        ticket = {
            "ticket_id": f"KAGGLE-{source_hash[:12]}-{source_record:06d}",
            "ticket_text": masked_text, "category": row["queue"], "priority": row["priority"],
            "answer": answer, "source_record": str(source_record), "version": row["version"],
            "language": row["language"], "type": row["type"],
            "subject": subject, "body": body, "source_ticket_id": row.get("ticket_id", ""),
        }
        ticket.update({tag: row[tag] for tag in TAGS if tag in row})
        seen[key] = ticket
        records.append(ticket)
    if not records:
        raise ValueError("No valid English tickets remain after cleaning and label validation.")
    if len({row["category"] for row in records}) < 4:
        raise ValueError("At least four meaningful queue classes are required after validation.")
    pii = {field: {name: {"occurrences": pii_counts[field][name], "records": pii_rows[field][name],
                         "action": f"replace matched span with [{name.upper()}]"}
                   for name, _ in PII_PATTERNS} for field in pii_counts}
    return records, {"invalid_records": invalid, "duplicate_records_removed": duplicates, "pii": pii}


def allocate_counts(size: int) -> list[int]:
    targets = [size * ratio for ratio in (.70, .15, .15)]
    counts = [int(value) for value in targets]
    for _ in range(size - sum(counts)):
        counts[max(range(3), key=lambda i: targets[i] - counts[i])] += 1
    if size >= 3:
        for i in range(3):
            if counts[i] == 0:
                donor = max((j for j in range(3) if counts[j] > 1), key=lambda j: counts[j] - targets[j])
                counts[donor] -= 1
                counts[i] += 1
    return counts


def balanced_quotas(totals: Counter) -> dict[tuple[str, str], list[int]]:
    """Balance per-stratum rounding against the global 70/15/15 integer target."""
    quotas = {label: allocate_counts(size) for label, size in totals.items()}
    target = allocate_counts(sum(totals.values()))
    current = [sum(counts[i] for counts in quotas.values()) for i in range(3)]
    while current != target:
        donor = max(range(3), key=lambda i: current[i] - target[i])
        recipient = max(range(3), key=lambda i: target[i] - current[i])
        candidates = [label for label in sorted(totals)
                      if quotas[label][donor] > (1 if totals[label] >= 3 else 0)]
        if not candidates:
            break  # Minimum representation takes precedence for tiny datasets.

        def cost(label: tuple[str, str]) -> float:
            ideal = [totals[label] * ratio for ratio in (.7, .15, .15)]
            old = quotas[label]
            return ((old[donor] - 1 - ideal[donor]) ** 2 - (old[donor] - ideal[donor]) ** 2
                    + (old[recipient] + 1 - ideal[recipient]) ** 2 - (old[recipient] - ideal[recipient]) ** 2)

        label = min(candidates, key=cost)
        quotas[label][donor] -= 1
        quotas[label][recipient] += 1
        current[donor] -= 1
        current[recipient] += 1
    return quotas


def split_by_category(records: list[Ticket], seed: int = SEED) -> tuple[dict[str, list[Ticket]], list[str]]:
    """Group-aware category/priority stratification with explicit rare-group fallback."""
    if not records:
        raise ValueError("No valid tickets to split.")
    groups = defaultdict(list)
    for row in records:
        groups[row["group_id"]].append(row)
    strata = defaultdict(list)
    totals = Counter((row["category"], row["priority"]) for row in records)
    quotas = balanced_quotas(totals)
    assigned = defaultdict(lambda: [0, 0, 0])
    splits = {name: [] for name in SPLIT_NAMES}
    notes = []
    rng = random.Random(seed)
    # Heterogeneous groups are retained in training, never arbitrarily relabelled.
    for group in groups.values():
        labels = {(row["category"], row["priority"]) for row in group}
        if len(labels) > 1:
            splits["train"].extend(group)
            for row in group:
                assigned[(row["category"], row["priority"])][0] += 1
            notes.append(f"Mixed-label identity group {group[0]['group_id']} ({len(group)} rows) reserved for training.")
        else:
            strata[next(iter(labels))].append(group)
    for label, label_groups in sorted(strata.items()):
        desired = quotas[label]
        rng.shuffle(label_groups)
        label_groups.sort(key=len, reverse=True)  # Stable: seed controls equal-size ties.
        if len(label_groups) < 3:
            notes.append(f"{label!r}: only {len(label_groups)} independent groups; all retained in training.")
            for group in label_groups:
                splits["train"].extend(group)
                assigned[label][0] += len(group)
            continue

        for group in label_groups:
            split_index = max(range(3), key=lambda i: desired[i] - assigned[label][i])
            splits[SPLIT_NAMES[split_index]].extend(group)
            assigned[label][split_index] += len(group)
    if any(not split for split in splits.values()):
        raise ValueError("Insufficient independent groups for three nonempty splits; add data or review grouping.")
    for split in splits.values():
        rng.shuffle(split)
    return splits, notes


def verify_splits(splits: dict[str, list[Ticket]]) -> dict:
    owners = {}
    ids = set()
    for name, rows in splits.items():
        for row in rows:
            if row["ticket_id"] in ids:
                raise ValueError("Repeated generated ticket ID in processed data.")
            ids.add(row["ticket_id"])
            for key in [("group", row["group_id"]), *identity_keys(row)]:
                if key in owners and owners[key] != name:
                    raise ValueError(f"Cross-split identity leakage detected for {key[0]}.")
                owners[key] = name
    return {"cross_split_normalized_text_overlap": 0, "cross_split_identity_group_overlap": 0,
            "cross_split_grouping_key_overlap": 0, "generated_ids_unique": True}


def write_csv(path: Path, rows: list[Ticket], columns: tuple[str, ...]) -> None:
    with path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=columns, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def preprocess_dataset(source: Path, output_dir: Path, seed: int = SEED) -> PreparationResult:
    filenames = [f"{name}.csv" for name in SPLIT_NAMES] + ["historical_tickets.csv", "dataset_audit.json"]
    if source.resolve() in {(output_dir / name).resolve() for name in filenames}:
        raise ValueError("Output would overwrite the raw source CSV; choose another output directory.")
    columns, raw = load_csv(source)
    audit = audit_dataset(source, columns, raw)
    records, preparation = prepare_records(raw, audit["dataset"]["sha256"])
    # Include alternate source IDs from removed exact duplicates in grouping.
    grouping = assign_groups(records)
    splits, notes = split_by_category(records, seed)
    verification = verify_splits(splits)
    invalid = preparation["invalid_records"]
    duplicates = preparation["duplicate_records_removed"]
    english_count = audit["english_selection"]["english_rows"]
    audit.update(preparation)
    audit["counts"] = {"raw": len(raw), "english": english_count, "invalid_removed": len(invalid),
                       "after_cleaning": english_count - len(invalid), "duplicates_removed": len(duplicates),
                       "after_deduplication": len(records), "final_english": len(records)}
    audit["grouping"] = grouping
    audit["split"] = {"seed": seed, "target_ratios": [0.70, 0.15, 0.15],
                      "method": "Joint queue/priority stratification with indivisible identity groups; largest-deficit allocation.",
                      "fallbacks": notes, "sets": {
                          name: {"rows": len(rows), "percentage": round(100 * len(rows) / len(records), 6),
                                 "categories": distribution([row['category'] for row in rows]),
                                 "priorities": distribution([row['priority'] for row in rows]),
                                 "versions": distribution([row['version'] for row in rows])}
                          for name, rows in splits.items()}}
    audit["verification"] = verification
    audit["version_decision"] = (
        "Retain every version. Exact combined text is deduplicated after masking; no version-number selection. "
        "Shared nonempty source ID, normalized subject/body/text, or obvious long template links records into "
        "one split. Generic repeated bodies are grouping signals only, not deletion evidence. "
        "No semantic/paraphrase or cross-language equivalence guarantee is made."
    )
    audit["privacy_limitations"] = (
        "Pattern audit only, with ordered non-overlapping masking. Names, addresses, unusual phone formats "
        "and contextual identifiers may remain. URLs and numeric patterns may be nonpersonal; masked conservatively. "
        "No PII-free claim. Counts cover valid English records before deduplication."
    )
    audit["id_strategy"] = "KAGGLE-<first 12 hex of raw SHA256>-<1-based zero-padded data record>; generated, not a source ID."
    audit["history_policy"] = "Answers/metadata are separate from classification. Restrict future retrieval to split=train during evaluation."
    audit["final_categories"] = distribution([row['category'] for row in records])
    audit["final_priorities"] = distribution([row['priority'] for row in records])
    audit["class_balance"] = {
        "majority": max(audit['final_categories'], key=lambda label: audit['final_categories'][label]['count']),
        "minority": min(audit['final_categories'], key=lambda label: audit['final_categories'][label]['count']),
        "majority_minority_ratio": round(max(v['count'] for v in audit['final_categories'].values()) /
                                          min(v['count'] for v in audit['final_categories'].values()), 6),
        "classes_under_30": [label for label, value in audit['final_categories'].items() if value['count'] < 30],
    }
    for field in ("categories", "priorities"):
        overall = audit[f"final_{field}"]
        for name, stats in audit["split"]["sets"].items():
            stats[f"max_{field}_deviation_percentage_points"] = round(max(
                abs(stats[field].get(label, {"percentage": 0})['percentage'] - value['percentage'])
                for label, value in overall.items()), 6)
    audit["output_schemas"] = {f"{name}.csv": list(COLUMNS) for name in SPLIT_NAMES}
    history_columns = COLUMNS + ("answer", "source_record", "version", "language", "split", "type") + tuple(tag for tag in TAGS if tag in columns)
    audit["output_schemas"]["historical_tickets.csv"] = list(history_columns)
    split_for = {row["ticket_id"]: name for name, rows in splits.items() for row in rows}
    history = [{**row, "split": split_for[row["ticket_id"]]} for row in records]
    audit["history_missing_answers"] = sum(not row['answer'] for row in history)
    if hashlib.sha256(source.read_bytes()).hexdigest() != audit["dataset"]["sha256"]:
        raise ValueError("Raw file changed during preprocessing; outputs were not replaced.")
    output_dir.mkdir(parents=True, exist_ok=True)
    # Stage the entire successful run before replacing any existing outputs.
    with TemporaryDirectory(prefix=".preprocessing-", dir=output_dir) as temporary:
        staging = Path(temporary)
        for name, rows in splits.items():
            write_csv(staging / f"{name}.csv", rows, COLUMNS)
        write_csv(staging / "historical_tickets.csv", history, history_columns)
        audit["output_sha256"] = {name: hashlib.sha256((staging / name).read_bytes()).hexdigest()
                                  for name in filenames if name.endswith('.csv')}
        (staging / "dataset_audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
        for name in filenames:
            (staging / name).replace(output_dir / name)
    print("Dataset preprocessing completed (selected Kaggle CSV only).")
    print(json.dumps({"counts": audit["counts"], "splits": {k: len(v) for k, v in splits.items()},
                      "categories": audit["final_categories"], "priorities": audit["final_priorities"],
                      "grouping": grouping, "split_notes": notes, "verification": verification}, indent=2))
    for problem in invalid:
        print(f"Invalid record removed: {problem}")
    print(f"Full audit and processed CSVs: {output_dir.resolve()}")
    return PreparationResult(records, len(raw), len(duplicates), invalid, audit)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/processed")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    try:
        preprocess_dataset(args.input, args.output_dir, args.seed)
    except (ValueError, OSError, csv.Error) as exc:
        parser.exit(1, f"Preprocessing failed: {exc}\n")


if __name__ == "__main__":
    main()

