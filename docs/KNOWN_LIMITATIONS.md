# Known POC Limitations

The default API generation mode is the local `google/flan-t5-base` model (`USE_LOCAL_MODEL=true`). If it cannot be loaded, the API logs a warning and falls back to a deterministic template generator so the local demo can still return a clearly limited response. Set `USE_LOCAL_MODEL=false` only when a fast template-only demonstration is intentional.

This repository remains a POC: it has no authentication or RBAC, rate limiting, persistent ticket storage, production observability, or production deployment design. Its input/output filter is deterministic pattern matching rather than a comprehensive safety system. A production version would require independently validated controls, operational monitoring, and human review for automated support decisions.

The backend container includes the saved optimized-classifier and FAISS retrieval artifacts required for its demo path. It does not download, retrain, or refresh them at startup; a production image pipeline would version, verify, and promote those artifacts separately.

The tracked retrieval runtime bundle is `tickets.faiss`, `metadata.jsonl`, and `retrieval_config.json`; `python -m backend.app.rag.retrieval` regenerates it. The small retrieval evaluation JSON/Markdown reports remain tracked, while regeneratable experiment confusion-matrix PNGs are not tracked.
