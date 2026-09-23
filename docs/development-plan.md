# Development plan

Days 1 through 7 are complete. Later phases remain planned.

| Day | Scope / status |
| --- | --- |
| 1 | COMPLETE: architecture, API/UI, configuration, synthetic data/KB, preprocessing and saved splits, tests, Docker and documentation foundation |
| 2 | COMPLETE: independent TF-IDF + Logistic Regression category/priority baselines, train/validation/test evaluation, saved models, tested standalone inference and probability confidence |
| 3 | COMPLETE: hyperparameter optimization with Grid Search, Randomized Search and Bayesian Optimization |
| 4 | COMPLETE: lightweight CNN / RNN / LSTM category-classification comparison |
| 5 | COMPLETE: attention and pretrained DistilBERT category-classification comparison |
| 6 | COMPLETE: local Sentence Transformer embeddings and FAISS similar-ticket retrieval |
| 7 | COMPLETE: simple grounded RAG suggested resolutions with supporting sources |
| 8 | LangGraph multi-agent investigation for complex tickets |
| 9 | Security: prompt injection, jailbreak and PII checks; security status |
| 10 | Explainability and fairness; classification explanations |
| 11 | MLOps: MLflow, Airflow and model monitoring |
| 12 | Integration and testing |
| 13 | Documentation and final demo |

Day 2 uses the unchanged English splits from the selected synthetic Kaggle CSV. Both models are trained only on train.csv; validation and test are evaluation-only. The fixed unweighted baseline has no parameter search, resampling, combined target or API integration. Full measured results, per-class reports, examples and reproducibility details are in [baseline-ml.md](baseline-ml.md). The API still returns null prediction fields.

Day 3 adds lightweight hyperparameter optimization for the same TF-IDF + Logistic Regression pipelines. Grid Search, Randomized Search and Optuna Bayesian Optimization run separately for category and priority using train.csv only with 3-fold CV and macro-F1 scoring. Validation macro-F1 selects the final configuration; the selected model is retrained on train + validation and evaluated once on test. Both targets selected Randomized Search with `C=10.0`, `class_weight=None`, `ngram_range=(1,2)`, `min_df=1`, `max_df=0.95`, `max_features=20000` and `sublinear_tf=False`. Final optimized test macro-F1 is 0.641538 for category and 0.662208 for priority. No API integration was added.

Day 4 compares lightweight CNN, vanilla RNN and LSTM category-only models on the same English split. TensorFlow was not compatible with the Python 3.14 environment, so the POC uses dependency-free NumPy neural text encoders with a trained classification head. Selection uses validation Macro-F1; test metrics are final reporting only. This is a small architecture comparison only; no tuning, priority model, API integration, attention or pretrained model is included in Day 4.

Day 5 extends the category-only DL comparison with an attention encoder and a pretrained DistilBERT classifier. Attention uses the same lightweight tokenizer/embedding sequence representation and computes token attention before the classification head. DistilBERT uses `distilbert-base-uncased`, a frozen base encoder, one epoch, sequence length 48, batch size 64, and a deterministic train-only cap of 60 examples per category to keep CPU runtime within POC bounds; validation and test evaluation use the full held-out splits. DistilBERT completed successfully after caching the public checkpoint locally. The selected DL model is DistilBERT based on validation Macro-F1 0.121775, with final test Macro-F1 0.120542. These results remain well below the Day 3 optimized category classical model and are not production-ready. Day 5 artifacts are in `experiments/dl_comparison/day5/` because the existing Day 4 result files on this machine are locked to SYSTEM/Administrators.

Day 6 adds local similar-ticket retrieval. The index uses `sentence-transformers/all-MiniLM-L6-v2` embeddings over training historical `ticket_text` only, normalized 384-dimensional vectors, and a FAISS `IndexFlatIP` index. Metadata maps FAISS positions back to ticket IDs, text, category, priority, answer, source/version/type and tags. Validation/test tickets are used only as held-out evaluation queries. With 500 held-out queries, Recall@1 is 0.656000, Recall@3 is 0.764000, Recall@5 is 0.844000, and MRR is 0.722500. No API key is required, and no FastAPI or RAG integration was added.

Day 7 adds a simple RAG pipeline over the Day 6 FAISS index. It retrieves top-3 similar training historical tickets, builds bounded context including prior resolutions, and uses local `google/flan-t5-base` generation with grounding rules and source attribution. The retrieval threshold is 0.55; weak evidence returns `insufficient_evidence` instead of a fabricated answer. A four-case POC evaluation checks source attribution, acceptable grounded responses, prompt-injection handling and insufficient-information behavior; all measured rates are 1.000000. No API key is required, and no FastAPI, LangGraph, agent or memory integration was added.

Synthetic results do not establish real-world performance. Baseline, optimization, DL comparison, retrieval and RAG artifacts now populate experiments/baseline, experiments/optimization, experiments/dl_comparison, experiments/retrieval and experiments/rag; MLflow, Airflow and future backend modules remain placeholders. Days 8-13 in the table are planned only.
