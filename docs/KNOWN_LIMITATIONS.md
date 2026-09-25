# Known POC Limitations

The default API generation mode is the local `google/flan-t5-base` model (`USE_LOCAL_MODEL=true`). If it cannot be loaded, the API logs a warning and falls back to a deterministic template generator so the local demo can still return a clearly limited response. Set `USE_LOCAL_MODEL=false` only when a fast template-only demonstration is intentional.

This repository remains a POC: it has no authentication or RBAC, rate limiting, persistent ticket storage, production observability, or production deployment design. Its input/output filter is deterministic pattern matching rather than a comprehensive safety system. A production version would require independently validated controls, operational monitoring, and human review for automated support decisions.
