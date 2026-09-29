import numpy as np

from app.rag.knowledge_base import KnowledgeBaseConfig, KnowledgeBaseRetriever, chunk_document


class TinyEmbedder:
    def encode(self, sentences, **_kwargs):
        vectors = []
        for text in sentences:
            lower = text.lower()
            vectors.append([
                float("refund" in lower or "charge" in lower),
                float("payment" in lower or "checkout" in lower),
                float("shipping" in lower or "parcel" in lower),
            ])
        array = np.asarray(vectors, dtype=np.float32)
        norms = np.linalg.norm(array, axis=1, keepdims=True)
        return np.divide(array, norms, out=np.zeros_like(array), where=norms > 0)


def test_chunking_keeps_document_title():
    chunks = chunk_document("Refund Policy", "## Refunds\n\nRefunds take five days.\n\n## Charges\n\nDuplicate charges are reviewed.")
    # Each non-empty paragraph section, including headings, is indexed independently.
    assert len(chunks) == 4
    assert all(chunk.startswith("Refund Policy:") for chunk in chunks)


def test_knowledge_base_index_save_load_and_search(tmp_path):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    (source_dir / "refund_policy.txt").write_text("Refund policy\n\nDuplicate charges receive review.", encoding="utf-8")
    (source_dir / "shipping_faq.txt").write_text("Shipping FAQ\n\nMissing parcel investigation.", encoding="utf-8")
    retriever = KnowledgeBaseRetriever.build_index(source_dir, TinyEmbedder(), KnowledgeBaseConfig(model_name="tiny"))
    retriever.save(tmp_path / "artifacts")
    loaded = KnowledgeBaseRetriever.load_index(tmp_path / "artifacts", TinyEmbedder())
    results = loaded.search("I need a refund for a duplicate charge", top_k=1)
    assert loaded.index.ntotal == retriever.index.ntotal
    assert results[0].source_type == "knowledge_base"
    assert results[0].title == "Refund Policy"
    assert (tmp_path / "artifacts" / "kb.faiss").is_file()
    assert (tmp_path / "artifacts" / "kb_metadata.jsonl").is_file()
