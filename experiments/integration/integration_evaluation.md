# Day 12 Integration Evaluation

This fixed POC evaluation exercises the integrated backend flow. It is not an accuracy measurement.

| Case | Expected | Actual | Pass | Latency ms | Workflow | Similar tickets | Sources |
| --- | --- | --- | --- | ---: | --- | ---: | ---: |
| payment_failed | simple_rag | simple_rag | True | 259.77 | simple_rag | 3 | 3 |
| multiple_charges_refund_cancellation | complex_multi_agent | complex_multi_agent | True | 72.57 | complex_multi_agent | 3 | 3 |
| prompt_injection | blocked | blocked | True | 0.80 | blocked | 0 | 0 |
| insufficient_information | insufficient_evidence | insufficient_evidence | True | 30.47 | simple_rag | 3 | 0 |
| pii_ticket | safe_pii_redacted | safe_pii_redacted | True | 28.43 | simple_rag | 3 | 3 |

Passed: 5 / 5
Average latency: 78.41 ms
