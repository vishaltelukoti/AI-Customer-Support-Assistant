# Known POC Limitations

The default API generation mode is the local `google/flan-t5-base` model (`USE_LOCAL_MODEL=true`). If it cannot be loaded, the API logs a warning and falls back to a deterministic template generator so the local demo can still return a clearly limited response. Set `USE_LOCAL_MODEL=false` only when a fast template-only demonstration is intentional.

The recorded seven-case RAG evaluation did not load FLAN-T5 (`generation_model_loaded_for_eval: false`) and therefore measures deterministic fallback output rather than local-model answer quality. When local FLAN-T5 is used, it may reproduce details from retrieved synthetic historical examples; the POC does not independently verify those details as customer facts.

This repository remains a POC: it has no authentication or RBAC, rate limiting, persistent ticket storage, production observability, or production deployment design. Its input/output filter is deterministic pattern matching rather than a comprehensive safety system. A production version would require independently validated controls, operational monitoring, and human review for automated support decisions.

The backend container includes the saved optimized-classifier and FAISS retrieval artifacts required for its demo path. It does not download, retrain, or refresh them at startup; a production image pipeline would version, verify, and promote those artifacts separately.

The tracked retrieval runtime bundle is `tickets.faiss`, `metadata.jsonl`, and `retrieval_config.json`; `python -m backend.app.rag.retrieval` regenerates it. The small retrieval evaluation JSON/Markdown reports remain tracked, while regeneratable experiment confusion-matrix PNGs are not tracked.

The input/output pattern filter catches literal known phrasings only and is trivially bypassed by paraphrase; it demonstrates input/output guardrails, not a production-grade defense.

## Deep-Learning Assessment Gap

Day 5 compares CNN, RNN, LSTM, token-level Attention, and pretrained `distilbert-base-uncased`. DistilBERT was selected using validation Macro-F1 (`0.121775`; recorded test Macro-F1 `0.120542`). The selection applies only within the experiment; the FastAPI classification service still loads the Day 3 optimized classical category and priority classifiers.

| Requirement | Current implementation | Status | Evidence/Notes |
| --- | --- | --- | --- |
| Integrate the selected Day 5 model into the production-like application | API loads Day 3 optimized category and priority TF-IDF + Logistic Regression artifacts; Day 5 DistilBERT is not loaded | NOT COMPLIANT / OUTSTANDING | `backend/app/services/ticket_service.py`; Day 5 selection recorded in `experiments/dl_comparison/day5/dl_comparison_results.json` using validation Macro-F1 |
