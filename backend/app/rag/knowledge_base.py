"""Build a local FAISS index for the fictional support knowledge base."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import faiss

from ..ml.baseline_data import prepare_ticket_text
from ..ml.preprocessing import ROOT
from .retrieval import DEFAULT_MODEL_NAME, DEFAULT_TOP_K, TextEmbedder, encode_texts, load_sentence_transformer

KNOWLEDGE_BASE_DIR = ROOT / "data/knowledge_base"
KNOWLEDGE_BASE_ARTIFACT_DIR = ROOT / "experiments/knowledge_base"
INDEX_FILENAME = "kb.faiss"
METADATA_FILENAME = "kb_metadata.jsonl"
CONFIG_FILENAME = "kb_config.json"


@dataclass(frozen=True)
class KnowledgeBaseConfig:
    model_name: str = DEFAULT_MODEL_NAME
    top_k: int = DEFAULT_TOP_K
    normalize_embeddings: bool = True
    index_type: str = "IndexFlatIP"


@dataclass(frozen=True)
class KnowledgeBaseMetadata:
    index_position: int
    ticket_id: str
    title: str
    file_name: str
    chunk_text: str


@dataclass(frozen=True)
class KnowledgeBaseResult:
    rank: int
    score: float
    ticket_id: str
    title: str
    file_name: str
    chunk_text: str
    source_type: str = "knowledge_base"


def _title_from_path(path: Path) -> str:
    return path.stem.replace("_", " ").title().replace("Faq", "FAQ")


def chunk_document(title: str, text: str) -> list[str]:
    """Return paragraph-sized chunks while retaining the document title."""
    sections = [" ".join(part.split()) for part in text.split("\n\n") if part.strip()]
    return [f"{title}: {section}" for section in sections]


def load_knowledge_base(source_dir: Path = KNOWLEDGE_BASE_DIR) -> list[KnowledgeBaseMetadata]:
    paths = sorted(source_dir.glob("*.txt"))
    if not paths:
        raise FileNotFoundError(f"No knowledge-base text files found in {source_dir}.")
    metadata = []
    for path in paths:
        title = _title_from_path(path)
        for chunk_number, chunk_text in enumerate(chunk_document(title, path.read_text(encoding="utf-8")), start=1):
            metadata.append(KnowledgeBaseMetadata(
                index_position=len(metadata),
                ticket_id=f"kb:{path.stem}:{chunk_number}",
                title=title,
                file_name=path.name,
                chunk_text=chunk_text,
            ))
    return metadata


class KnowledgeBaseRetriever:
    def __init__(self, index, metadata: list[KnowledgeBaseMetadata], embedder: TextEmbedder,
                 config: KnowledgeBaseConfig = KnowledgeBaseConfig()):
        if index.ntotal != len(metadata):
            raise ValueError("FAISS knowledge-base index size does not match metadata length.")
        self.index = index
        self.metadata = metadata
        self.embedder = embedder
        self.config = config

    @classmethod
    def build_index(cls, source_dir: Path = KNOWLEDGE_BASE_DIR, embedder: TextEmbedder | None = None,
                    config: KnowledgeBaseConfig = KnowledgeBaseConfig()) -> "KnowledgeBaseRetriever":
        metadata = load_knowledge_base(source_dir)
        embedder = embedder or load_sentence_transformer(config.model_name)
        embeddings = encode_texts(embedder, [item.chunk_text for item in metadata], config.normalize_embeddings)
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        return cls(index, metadata, embedder, config)

    @classmethod
    def load_index(cls, artifact_dir: Path = KNOWLEDGE_BASE_ARTIFACT_DIR,
                   embedder: TextEmbedder | None = None) -> "KnowledgeBaseRetriever":
        config_data = json.loads((artifact_dir / CONFIG_FILENAME).read_text(encoding="utf-8"))
        config = KnowledgeBaseConfig(**config_data["configuration"])
        metadata = [KnowledgeBaseMetadata(**json.loads(line)) for line in
                    (artifact_dir / METADATA_FILENAME).read_text(encoding="utf-8").splitlines() if line.strip()]
        return cls(
            faiss.read_index(str(artifact_dir / INDEX_FILENAME)),
            metadata,
            embedder or load_sentence_transformer(config.model_name),
            config,
        )

    def save(self, artifact_dir: Path = KNOWLEDGE_BASE_ARTIFACT_DIR) -> None:
        artifact_dir.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix=".knowledge-base-", dir=artifact_dir) as temporary:
            staging = Path(temporary)
            faiss.write_index(self.index, str(staging / INDEX_FILENAME))
            (staging / METADATA_FILENAME).write_text(
                "\n".join(json.dumps(asdict(item), sort_keys=True) for item in self.metadata) + "\n",
                encoding="utf-8",
            )
            (staging / CONFIG_FILENAME).write_text(json.dumps({
                "configuration": asdict(self.config),
                "embedding_dimension": self.index.d,
                "indexed_chunks": self.index.ntotal,
                "index_file": INDEX_FILENAME,
                "metadata_file": METADATA_FILENAME,
                "embedding_input": "title_and_chunk_text",
            }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            for path in staging.iterdir():
                path.replace(artifact_dir / path.name)

    def search(self, query: str, top_k: int = DEFAULT_TOP_K) -> list[KnowledgeBaseResult]:
        if top_k < 1:
            raise ValueError("top_k must be at least 1.")
        vector = encode_texts(self.embedder, [prepare_ticket_text(query)], self.config.normalize_embeddings)
        scores, positions = self.index.search(vector, min(self.index.ntotal, top_k))
        results = []
        for score, position in zip(scores[0], positions[0]):
            if position < 0:
                continue
            item = self.metadata[int(position)]
            results.append(KnowledgeBaseResult(
                rank=len(results) + 1,
                score=float(score),
                ticket_id=item.ticket_id,
                title=item.title,
                file_name=item.file_name,
                chunk_text=item.chunk_text,
            ))
        return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=KNOWLEDGE_BASE_DIR)
    parser.add_argument("--output-dir", type=Path, default=KNOWLEDGE_BASE_ARTIFACT_DIR)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    args = parser.parse_args()
    retriever = KnowledgeBaseRetriever.build_index(args.source_dir, config=KnowledgeBaseConfig(model_name=args.model_name))
    retriever.save(args.output_dir)
    print(f"Indexed knowledge-base chunks: {retriever.index.ntotal}")
    print(f"Embedding dimension: {retriever.index.d}")
    print(f"Saved knowledge-base artifacts: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
