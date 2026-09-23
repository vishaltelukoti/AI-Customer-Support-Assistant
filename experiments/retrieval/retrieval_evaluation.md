# Day 6 retrieval evaluation

Local Sentence Transformer embeddings with a FAISS IndexFlatIP index over training historical tickets only.

Embedding model: `sentence-transformers/all-MiniLM-L6-v2`
Embedding dimension: 384
Default Top-K: 3

| Metric | Value |
| --- | ---: |
| Recall@1 | 0.656000 |
| Recall@3 | 0.764000 |
| Recall@5 | 0.844000 |
| MRR | 0.722500 |

Relevant means the retrieved training ticket has the same category as the held-out query ticket.
No API key is required for the local Sentence Transformer + FAISS retrieval implementation.
