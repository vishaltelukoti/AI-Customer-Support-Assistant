"""Build and evaluate a local FAISS similar-ticket retrieval index."""

import argparse
import csv
import json
import platform
import time
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Protocol

import faiss
import numpy as np

from ..ml.baseline_data import prepare_ticket_text
from ..ml.dataset_audit import CATEGORIES, PRIORITIES
from ..ml.preprocessing import ROOT, SEED

RETRIEVAL_DIR = ROOT / "experiments/retrieval"
DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_TOP_K = 3
EVALUATION_LIMIT_PER_SPLIT = 250
INDEX_FILENAME = "tickets.faiss"
METADATA_FILENAME = "metadata.jsonl"
CONFIG_FILENAME = "retrieval_config.json"
EVALUATION_JSON = "retrieval_evaluation.json"
EVALUATION_MD = "retrieval_evaluation.md"


class TextEmbedder(Protocol):
    def encode(self, sentences, **kwargs): ...


@dataclass(frozen=True)
class RetrievalConfig:
    model_name: str = DEFAULT_MODEL_NAME
    top_k: int = DEFAULT_TOP_K
    normalize_embeddings: bool = True
    index_type: str = "IndexFlatIP"
    seed: int = SEED
    evaluation_limit_per_split: int = EVALUATION_LIMIT_PER_SPLIT


@dataclass(frozen=True)
class TicketMetadata:
    index_position: int
    ticket_id: str
    ticket_text: str
    category: str
    priority: str
    answer: str
    source_record: str
    version: str
    language: str
    split: str
    type: str
    tags: dict[str, str]


@dataclass(frozen=True)
class RetrievalResult:
    rank: int
    score: float
    ticket_id: str
    ticket_text: str
    category: str
    priority: str
    answer: str
    source_record: str
    version: str
    split: str
    type: str
    tags: dict[str, str]


def _dependency_versions() -> dict[str, str]:
    versions = {"python": platform.python_version()}
    for package in ("sentence-transformers", "faiss-cpu", "numpy"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            if package == "faiss-cpu":
                versions[package] = getattr(faiss, "__version__", "installed")
            else:
                versions[package] = "unknown"
    return versions


def load_sentence_transformer(model_name: str = DEFAULT_MODEL_NAME) -> TextEmbedder:
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def load_historical_tickets(path: Path) -> list[dict[str, str]]:
    try:
        with path.open(encoding="utf-8-sig", newline="") as file:
            rows = list(csv.DictReader(file, strict=True))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Missing historical tickets file: {path}. Run preprocessing first.") from exc
    required = {"ticket_id", "ticket_text", "category", "priority", "answer", "split"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"{path.name}: expected historical ticket columns including {sorted(required)}.")
    seen_ids = set()
    for number, row in enumerate(rows, start=1):
        if row["ticket_id"] in seen_ids:
            raise ValueError(f"{path.name}, record {number}: duplicate ticket_id {row['ticket_id']}.")
        seen_ids.add(row["ticket_id"])
        if row["category"] not in CATEGORIES or row["priority"] not in PRIORITIES:
            raise ValueError(f"{path.name}, record {number}: unsupported category or priority.")
        if row["split"] not in {"train", "validation", "test"}:
            raise ValueError(f"{path.name}, record {number}: unsupported split {row['split']!r}.")
        prepare_ticket_text(row["ticket_text"])
    return rows


def training_corpus(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    corpus = [row for row in rows if row["split"] == "train"]
    if not corpus:
        raise ValueError("No training historical tickets available for retrieval indexing.")
    return corpus


def _to_metadata(rows: list[dict[str, str]]) -> list[TicketMetadata]:
    metadata = []
    for position, row in enumerate(rows):
        metadata.append(TicketMetadata(
            index_position=position,
            ticket_id=row["ticket_id"],
            ticket_text=row["ticket_text"],
            category=row["category"],
            priority=row["priority"],
            answer=row.get("answer", ""),
            source_record=row.get("source_record", ""),
            version=row.get("version", ""),
            language=row.get("language", ""),
            split=row["split"],
            type=row.get("type", ""),
            tags={key: value for key, value in row.items() if key.startswith("tag_") and value},
        ))
    return metadata


def encode_texts(embedder: TextEmbedder, texts: list[str], normalize: bool = True) -> np.ndarray:
    embeddings = embedder.encode(
        texts,
        batch_size=64,
        convert_to_numpy=True,
        normalize_embeddings=normalize,
        show_progress_bar=False,
    )
    embeddings = np.asarray(embeddings, dtype=np.float32)
    if embeddings.ndim != 2 or embeddings.shape[0] != len(texts):
        raise ValueError("Embedding model returned an unexpected shape.")
    if normalize:
        faiss.normalize_L2(embeddings)
    return np.ascontiguousarray(embeddings)


class SimilarTicketRetriever:
    def __init__(self, index, metadata: list[TicketMetadata], embedder: TextEmbedder,
                 config: RetrievalConfig = RetrievalConfig()):
        self.index = index
        self.metadata = metadata
        self.embedder = embedder
        self.config = config
        if index.ntotal != len(metadata):
            raise ValueError("FAISS index size does not match metadata length.")

    @classmethod
    def build_index(cls, historical_rows: list[dict[str, str]], embedder: TextEmbedder | None = None,
                    config: RetrievalConfig = RetrievalConfig()) -> "SimilarTicketRetriever":
        train_rows = training_corpus(historical_rows)
        embedder = embedder or load_sentence_transformer(config.model_name)
        # Answer text is deliberately excluded; only ticket_text is embedded.
        embeddings = encode_texts(embedder, [row["ticket_text"] for row in train_rows], config.normalize_embeddings)
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        return cls(index, _to_metadata(train_rows), embedder, config)

    @classmethod
    def load_index(cls, artifact_dir: Path = RETRIEVAL_DIR, embedder: TextEmbedder | None = None) -> "SimilarTicketRetriever":
        config_data = json.loads((artifact_dir / CONFIG_FILENAME).read_text(encoding="utf-8"))
        config = RetrievalConfig(**{key: config_data["configuration"][key]
                                    for key in RetrievalConfig.__dataclass_fields__
                                    if key in config_data["configuration"]})
        metadata = [TicketMetadata(**json.loads(line)) for line in
                    (artifact_dir / METADATA_FILENAME).read_text(encoding="utf-8").splitlines() if line.strip()]
        index = faiss.read_index(str(artifact_dir / INDEX_FILENAME))
        return cls(index, metadata, embedder or load_sentence_transformer(config.model_name), config)

    def save(self, artifact_dir: Path = RETRIEVAL_DIR) -> None:
        artifact_dir.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix=".retrieval-", dir=artifact_dir) as temporary:
            staging = Path(temporary)
            faiss.write_index(self.index, str(staging / INDEX_FILENAME))
            (staging / METADATA_FILENAME).write_text(
                "\n".join(json.dumps(asdict(item), sort_keys=True) for item in self.metadata) + "\n",
                encoding="utf-8",
            )
            (staging / CONFIG_FILENAME).write_text(json.dumps({
                "configuration": asdict(self.config),
                "embedding_dimension": self.index.d,
                "indexed_tickets": self.index.ntotal,
                "metadata_file": METADATA_FILENAME,
                "index_file": INDEX_FILENAME,
                "embedding_input": "ticket_text",
                "answers_used_for_embeddings": False,
                "corpus_split": "train",
                "environment": _dependency_versions(),
            }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            for path in staging.iterdir():
                path.replace(artifact_dir / path.name)

    def search_similar_tickets(self, query: str, top_k: int = DEFAULT_TOP_K,
                               exclude_ticket_id: str | None = None) -> list[RetrievalResult]:
        if top_k < 1:
            raise ValueError("top_k must be at least 1.")
        query_text = prepare_ticket_text(query)
        candidates = min(self.index.ntotal, top_k + (1 if exclude_ticket_id else 0))
        vector = encode_texts(self.embedder, [query_text], self.config.normalize_embeddings)
        scores, positions = self.index.search(vector, candidates)
        results = []
        for score, position in zip(scores[0], positions[0]):
            if position < 0:
                continue
            item = self.metadata[int(position)]
            if exclude_ticket_id and item.ticket_id == exclude_ticket_id:
                continue
            results.append(RetrievalResult(
                rank=len(results) + 1,
                score=float(score),
                ticket_id=item.ticket_id,
                ticket_text=item.ticket_text,
                category=item.category,
                priority=item.priority,
                answer=item.answer,
                source_record=item.source_record,
                version=item.version,
                split=item.split,
                type=item.type,
                tags=item.tags,
            ))
            if len(results) == top_k:
                break
        return results


def evaluate_retrieval(retriever: SimilarTicketRetriever, historical_rows: list[dict[str, str]],
                       limit_per_split: int = EVALUATION_LIMIT_PER_SPLIT) -> dict:
    queries = []
    for split in ("validation", "test"):
        split_rows = [row for row in historical_rows if row["split"] == split]
        queries.extend(split_rows[:limit_per_split])
    if not queries:
        raise ValueError("No held-out validation/test queries available for retrieval evaluation.")
    metrics = {"recall@1": 0, "recall@3": 0, "recall@5": 0, "mrr": 0.0}
    query_results = []
    indexed_ids = {item.ticket_id for item in retriever.metadata}
    for row in queries:
        results = retriever.search_similar_tickets(row["ticket_text"], top_k=5, exclude_ticket_id=row["ticket_id"])
        relevant_ranks = [result.rank for result in results if result.category == row["category"]]
        for cutoff in (1, 3, 5):
            metrics[f"recall@{cutoff}"] += int(any(rank <= cutoff for rank in relevant_ranks))
        metrics["mrr"] += 1 / relevant_ranks[0] if relevant_ranks else 0
        query_results.append({
            "query_ticket_id": row["ticket_id"],
            "query_split": row["split"],
            "query_category": row["category"],
            "query_id_indexed": row["ticket_id"] in indexed_ids,
            "top_results": [asdict(result) for result in results[:DEFAULT_TOP_K]],
        })
    count = len(queries)
    aggregate = {key: float(value / count) for key, value in metrics.items()}
    return {
        "definition": "A retrieved ticket is relevant when its category matches the held-out query ticket category.",
        "query_splits": ["validation", "test"],
        "queries_evaluated": count,
        "limit_per_split": limit_per_split,
        "metrics": aggregate,
        "leakage_checks": {
            "indexed_split": "train",
            "indexed_tickets": retriever.index.ntotal,
            "validation_or_test_ids_indexed": any(row["ticket_id"] in indexed_ids for row in queries),
            "answers_used_for_embeddings": False,
            "self_retrieval_observed": any(row["query_id_indexed"] for row in query_results),
        },
        "sample_queries": query_results[:10],
    }


def _evaluation_markdown(evaluation: dict, config: RetrievalConfig, embedding_dimension: int) -> str:
    metrics = evaluation["metrics"]
    return "\n".join([
        "# Day 6 retrieval evaluation",
        "",
        "Local Sentence Transformer embeddings with a FAISS IndexFlatIP index over training historical tickets only.",
        "",
        f"Embedding model: `{config.model_name}`",
        f"Embedding dimension: {embedding_dimension}",
        f"Default Top-K: {config.top_k}",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Recall@1 | {metrics['recall@1']:.6f} |",
        f"| Recall@3 | {metrics['recall@3']:.6f} |",
        f"| Recall@5 | {metrics['recall@5']:.6f} |",
        f"| MRR | {metrics['mrr']:.6f} |",
        "",
        "Relevant means the retrieved training ticket has the same category as the held-out query ticket.",
        "No API key is required for the local Sentence Transformer + FAISS retrieval implementation.",
    ]) + "\n"


def run_retrieval_pipeline(data_dir: Path = ROOT / "data/processed", output_dir: Path = RETRIEVAL_DIR,
                           model_name: str = DEFAULT_MODEL_NAME,
                           top_k: int = DEFAULT_TOP_K,
                           evaluation_limit_per_split: int = EVALUATION_LIMIT_PER_SPLIT) -> dict:
    config = RetrievalConfig(model_name=model_name, top_k=top_k,
                             evaluation_limit_per_split=evaluation_limit_per_split)
    rows = load_historical_tickets(data_dir / "historical_tickets.csv")
    started = time.perf_counter()
    retriever = SimilarTicketRetriever.build_index(rows, config=config)
    build_seconds = time.perf_counter() - started
    retriever.save(output_dir)
    evaluation = evaluate_retrieval(retriever, rows, evaluation_limit_per_split)
    evaluation["configuration"] = asdict(config)
    evaluation["embedding_dimension"] = retriever.index.d
    evaluation["index_type"] = config.index_type
    evaluation["build_time_seconds"] = float(build_seconds)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / EVALUATION_JSON).write_text(json.dumps(evaluation, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / EVALUATION_MD).write_text(_evaluation_markdown(evaluation, config, retriever.index.d), encoding="utf-8")
    print(f"Indexed training tickets: {retriever.index.ntotal}")
    print(f"Embedding dimension: {retriever.index.d}")
    print(f"Recall@1: {evaluation['metrics']['recall@1']:.6f}")
    print(f"Recall@3: {evaluation['metrics']['recall@3']:.6f}")
    print(f"Recall@5: {evaluation['metrics']['recall@5']:.6f}")
    print(f"MRR: {evaluation['metrics']['mrr']:.6f}")
    print(f"Saved retrieval artifacts: {output_dir.resolve()}")
    return evaluation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/processed")
    parser.add_argument("--output-dir", type=Path, default=RETRIEVAL_DIR)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--evaluation-limit-per-split", type=int, default=EVALUATION_LIMIT_PER_SPLIT)
    args = parser.parse_args()
    run_retrieval_pipeline(args.data_dir, args.output_dir, args.model_name, args.top_k,
                           args.evaluation_limit_per_split)


if __name__ == "__main__":
    main()
