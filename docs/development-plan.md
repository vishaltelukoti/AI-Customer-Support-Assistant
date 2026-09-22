# Development plan

Days 1, 2, 3 and 4 are complete. Later phases remain planned.

| Day | Scope / status |
| --- | --- |
| 1 | COMPLETE: architecture, API/UI, configuration, synthetic data/KB, preprocessing and saved splits, tests, Docker and documentation foundation |
| 2 | COMPLETE: independent TF-IDF + Logistic Regression category/priority baselines, train/validation/test evaluation, saved models, tested standalone inference and probability confidence |
| 3 | COMPLETE: hyperparameter optimization with Grid Search, Randomized Search and Bayesian Optimization |
| 4 | COMPLETE: lightweight CNN / RNN / LSTM category-classification comparison |
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

Day 3 adds lightweight hyperparameter optimization for the same TF-IDF + Logistic Regression pipelines. Grid Search, Randomized Search and Optuna Bayesian Optimization run separately for category and priority using train.csv only with 3-fold CV and macro-F1 scoring. Validation macro-F1 selects the final configuration; the selected model is retrained on train + validation and evaluated once on test. Both targets selected Randomized Search with `C=10.0`, `class_weight=None`, `ngram_range=(1,2)`, `min_df=1`, `max_df=0.95`, `max_features=20000` and `sublinear_tf=False`. Final optimized test macro-F1 is 0.641538 for category and 0.662208 for priority. No API integration was added.

Day 4 compares lightweight CNN, vanilla RNN and LSTM category-only models on the same English split. TensorFlow was not compatible with the Python 3.14 environment, so the POC uses dependency-free NumPy neural text encoders with a trained classification head. The selected model is RNN by test macro-F1, tied to six decimals with LSTM at 0.044922. This is a small architecture comparison only; no tuning, priority model, API integration, attention or pretrained model is included.

Synthetic results do not establish real-world performance. Baseline, optimization and DL comparison artifacts now populate experiments/baseline, experiments/optimization and experiments/dl_comparison; MLflow, Airflow and future backend modules remain placeholders. Days 5-13 in the table are planned only.
