# Day 7 RAG evaluation

Simple retrieval-augmented suggested-resolution pipeline using the existing Day 6 FAISS index.

Generation model: `google/flan-t5-base`
Top-K: 3
Retrieval threshold: 0.55

| Metric | Value |
| --- | ---: |
| Source attribution rate | 1.000000 |
| Grounded acceptable response rate | 0.500000 |
| Insufficient-information pass rate | 1.000000 |
| Overall acceptance rate | 0.571429 |

No API key is required for the local retrieval plus local generation setup.
