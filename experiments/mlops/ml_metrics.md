# Day 11 MLOps Metrics Summary

Metrics are loaded from existing experiment artifacts; no model training or retrieval evaluation is rerun.

## Classification

| Target | Accuracy | Macro F1 | Source |
| --- | ---: | ---: | --- |
| Category | 0.652795 | 0.641538 | experiments\optimization\optimization_results.json |
| Priority | 0.676867 | 0.662208 | experiments\optimization\optimization_results.json |

## Retrieval

| Metric | Value | Source |
| --- | ---: | --- |
| Recall@1 | 0.656000 | experiments\retrieval\retrieval_evaluation.json |
| Recall@3 | 0.764000 | experiments\retrieval\retrieval_evaluation.json |
| Recall@5 | 0.844000 | experiments\retrieval\retrieval_evaluation.json |
| MRR | 0.722500 | experiments\retrieval\retrieval_evaluation.json |
