# Final Evaluation Summary

This page summarizes measured POC results already generated during Days 2-13. These values are historical experiment and validation records, not production accuracy claims or a statement about a current local runtime environment.

## Classical ML

Source: `experiments/optimization/optimization_results.json`.

| Target | Selected method | Test accuracy | Test Macro-F1 |
| --- | --- | ---: | ---: |
| Category | Randomized Search | 0.652795 | 0.641538 |
| Priority | Randomized Search | 0.676867 | 0.662208 |

## Deep Learning

Source: `experiments/dl_comparison/day5/dl_comparison_results.json` and `experiments/dl_comparison/day5/dl_comparison_results.md`.

| Model | Validation Macro-F1 | Test Macro-F1 |
| --- | ---: | ---: |
| CNN | 0.045488 | 0.044873 |
| RNN | 0.044922 | 0.044922 |
| LSTM | 0.044922 | 0.044922 |
| Attention | 0.044922 | 0.044922 |
| DistilBERT | 0.121775 | 0.120542 |

Selected Day 5 DL model: DistilBERT, selected by validation Macro-F1 (`0.121775`); test Macro-F1 (`0.120542`) is reported, not used for selection. This is the DL-comparison selection only. The integrated application uses the Day 3 optimized TF-IDF + Logistic Regression classifiers for both category and priority. Integration of the selected Day 5 model remains an outstanding assessment gap.

## Retrieval

Source: `experiments/retrieval/retrieval_evaluation.json`.

| Metric | Value |
| --- | ---: |
| Indexed training tickets | 11,436 |
| Recall@1 | 0.656000 |
| Recall@3 | 0.764000 |
| Recall@5 | 0.844000 |
| MRR | 0.722500 |

Retrieval relevance is category-level: a retrieved ticket is relevant when it shares the held-out query category.

## RAG

Source: `experiments/rag/rag_evaluation.json`.

| Metric | Value |
| --- | ---: |
| Predefined evaluation cases | 7 |
| Source attribution rate | 1.000000 |
| Grounded acceptable response rate | 0.500000 |
| Insufficient-information pass rate | 1.000000 |
| Overall acceptance rate | 0.571429 |
| Generation model loaded for evaluation | false |

These are predefined POC checks, not universal answer-quality measurements. The recorded run used the deterministic fallback generator rather than loading FLAN-T5.

## Agents

Source: Day 8 generated documentation and validation record.

| Metric | Value |
| --- | ---: |
| Predefined evaluation cases | 4 |
| Routing accuracy | 1.000000 |
| Workflow completion rate | 1.000000 |
| Source behavior rate | 1.000000 |
| Trace presence rate | 1.000000 |

## Security

Source: `experiments/security/security_evaluation.json`.

| Metric | Value |
| --- | ---: |
| Security test cases | 7 |
| Security test pass rate | 1.000000 |
| Attack detection rate | 1.000000 |
| Normal-ticket allow rate | 1.000000 |
| Malicious retrieved-content handling rate | 1.000000 |

This is deterministic POC security, not production-grade security.

## Explainability And Subgroup Diagnostics

Sources: `experiments/explainability/explanations.json` and `experiments/explainability/fairness_evaluation.json`.

| Item | Result |
| --- | --- |
| Representative examples explained | 5 |
| Confidence range | 0.174072 to 0.996855 |
| Explanation method | TF-IDF value times Logistic Regression coefficient |
| Subgroup fields | `version`, `type` |
| Sensitive demographics | Not inferred |

The subgroup analysis uses available metadata only. It is not proof of fairness or bias.

## Integration

Source: `experiments/integration/integration_evaluation.json`.

| Metric | Value |
| --- | ---: |
| Integration cases | 5 |
| Passed cases | 5 |
| Average latency | 78.41 ms |
| Focused ticket/API tests | 20 passed |
| Full backend tests | 143 passed |
| Frontend build | Passed |
| Docker build | Passed |

The recorded integration cases passed; the separate RAG evaluation accepted 4 of 7 predefined cases.
