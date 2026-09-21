# Development plan

Days 1 and 2 are complete. Day 3 has not started; all later phases remain planned.

| Day | Scope / status |
| --- | --- |
| 1 | COMPLETE: architecture, API/UI, configuration, synthetic data/KB, preprocessing and saved splits, tests, Docker and documentation foundation |
| 2 | COMPLETE: independent TF-IDF + Logistic Regression category/priority baselines, train/validation/test evaluation, saved models, tested standalone inference and probability confidence |
| 3 | NOT STARTED: hyperparameter optimization with grid, random and Bayesian search |
| 4 | CNN / RNN / LSTM experiments |
| 5 | Attention and a pre-trained model |
| 6 | Embeddings and FAISS; similar historical tickets |
| 7 | RAG; suggested resolutions with supporting sources |
| 8 | LangGraph multi-agent investigation for complex tickets |
| 9 | Security: prompt injection, jailbreak and PII checks; security status |
| 10 | Explainability and fairness; classification explanations |
| 11 | MLOps: MLflow, Airflow and model monitoring |
| 12 | Integration and testing |
| 13 | Documentation and final demo |

Day 2 uses the unchanged English splits from the selected synthetic Kaggle CSV. Both models are trained only on train.csv; validation and test are evaluation-only. The fixed unweighted baseline has no parameter search, resampling, combined target or API integration. Full measured results, per-class reports, examples and reproducibility details are in [baseline-ml.md](baseline-ml.md). The API still returns null prediction fields.

Day 3 hyperparameter optimization has NOT been implemented. Grid Search: NOT IMPLEMENTED. Random Search: NOT IMPLEMENTED. Bayesian Optimization: NOT IMPLEMENTED. No automatic progression to Day 3 is part of this work.

Synthetic results do not establish real-world performance. Baseline artifacts now populate experiments/baseline; other experiment folders, MLflow, Airflow and future backend modules remain placeholders. Days 4-13 in the table are planned only.
