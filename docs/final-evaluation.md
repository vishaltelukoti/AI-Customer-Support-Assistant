# Final Evaluation Summary

This page summarizes measured POC results already generated during Days 2-12. These values are not production accuracy claims.

## Classical ML

Source: `experiments/optimization/optimization_results.json`.

| Target | Selected method | Test accuracy | Test Macro-F1 |
| --- | --- | ---: | ---: |
| Category | Randomized Search | 0.652795 | 0.641538 |
| Priority | Randomized Search | 0.676867 | 0.662208 |

## Deep Learning

Source: `docs/deep-learning.md` and Day 5 artifacts. The Day 5 JSON files exist, but on this Windows machine their contents are ACL-locked from this session.

| Model | Validation Macro-F1 | Test Macro-F1 |
| --- | ---: | ---: |
| CNN | 0.045488 | 0.044873 |
| RNN | 0.044922 | 0.044922 |
| LSTM | 0.044922 | 0.044922 |
| Attention | 0.044922 | 0.044922 |
| DistilBERT | 0.121775 | 0.120542 |

Selected Day 5 DL model: DistilBERT, selected by validation Macro-F1. The DL comparison stayed below the Day 3 optimized classical category model.

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

Source: Day 7 generated documentation and validation record.

| Metric | Value |
| --- | ---: |
| Predefined evaluation cases | 4 |
| Source attribution rate | 1.000000 |
| Grounded acceptable response rate | 1.000000 |
| Insufficient-information pass rate | 1.000000 |
| Overall acceptance rate | 1.000000 |

These are predefined POC checks, not universal answer-quality measurements.

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

All predefined POC evaluation cases passed.
