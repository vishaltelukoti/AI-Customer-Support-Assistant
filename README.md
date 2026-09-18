# AI Customer Support Assistant

A single-developer customer-support POC. **Day 1 implements ticket intake and the application foundation only.** The form submits a customer message to FastAPI and displays a typed acknowledgement. Category, priority, and confidence are `null`; there are no AI predictions or stored tickets.

Planned later capabilities include classification, similar historical tickets, suggested resolutions with sources, complex-ticket investigation, security status, and classification explanations. See [the development plan](docs/development-plan.md). None of these AI features run yet.

## Stack and structure

Python 3.12+, FastAPI, Uvicorn, Pydantic/pydantic-settings, pytest and httpx; React, TypeScript, Vite, Axios and plain CSS. No database or external AI account is required.

```text
backend/
  app/                  # main, api/routes, core, schemas, services, utils
  tests/                # health, ticket contract, validation and CORS tests
  requirements.txt
  .env.example
  Dockerfile
frontend/
  src/                  # components, pages, services, types, App and styles
  package.json          # package-lock.json locks frontend dependencies
  .env.example
  Dockerfile
data/
  raw/                  # 20 SAMPLE tickets, not approved training data
  processed/            # empty future output directory
  knowledge_base/       # four short fictional sample documents
experiments/            # empty cnn, rnn, lstm, attention, pretrained folders
docs/                   # architecture, dataset, assumptions, development plan
.github/workflows/      # placeholder only; no CI/CD
docker-compose.yml
LICENSE
```

## Run locally

Prerequisites: Python 3.12+ and Node.js 22.12+ with npm. Commands below are PowerShell, starting in the repository root. Use two terminals. Do not run the local servers and Docker stack on the same ports simultaneously.

Backend:

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Frontend (new terminal):

```powershell
cd frontend
npm ci
Copy-Item .env.example .env
npm run dev
```

Open http://localhost:5173. Submit `I was charged twice for the same order.` to see `received`, a UUID, and unavailable AI fields. Do not overwrite an existing customized `.env` when repeating setup. On macOS/Linux use `python3`, `.venv/bin/python`, and `cp` in place of their Windows equivalents.

API documentation: http://localhost:8000/docs. Health: http://localhost:8000/health. Stop either server with Ctrl+C.

## Environment configuration

| Variable | Location | Default / format |
| --- | --- | --- |
| APP_NAME | backend/.env or process environment | AI Customer Support Assistant |
| CORS_ORIGINS | backend/.env or process environment | JSON array containing `http://localhost:5173` and `http://127.0.0.1:5173` |
| VITE_API_BASE_URL | frontend/.env | `http://localhost:8000` |

The backend locates its `.env` relative to its own folder. Process variables take precedence. Restart servers after changing configuration. Frontend `VITE_` values are public and embedded at build time, so never put secrets there. Changing the Docker frontend URL requires rebuilding. Defaults work without `.env` files.

Explicit CORS origins follow [FastAPI's CORS configuration](https://fastapi.tiangolo.com/tutorial/cors/); frontend environment handling follows [Vite's environment documentation](https://vite.dev/guide/env-and-mode).

## API contract

| Method | Endpoint | Behavior |
| --- | --- | --- |
| GET | /health | HTTP 200: `{"status":"healthy"}` |
| POST | /api/v1/tickets | HTTP 200 acknowledgement; HTTP 422 on invalid input |

```powershell
Invoke-RestMethod http://localhost:8000/health
Invoke-RestMethod -Method Post -Uri http://localhost:8000/api/v1/tickets -ContentType 'application/json' -Body '{"ticket_text":"I was charged twice for the same order."}'
```

Example response (the UUID changes per request):

```json
{
  "ticket_id": "d7658bf4-05d0-4f3f-8261-adc5811c1e4d",
  "ticket_text": "I was charged twice for the same order.",
  "category": null,
  "priority": null,
  "confidence": null,
  "status": "received"
}
```

The API accepts 1-10,000 characters after trimming surrounding whitespace. Blank, missing, non-string or oversized text and unexpected fields are rejected. The UI handles empty input, loading, success, validation errors, API failures and a 15-second timeout.

## Tests and build

```powershell
cd backend
.\.venv\Scripts\python.exe -m pytest -q
```

```powershell
cd frontend
npm run build
npm run test:api
```

Tests cover health, schema, null AI fields, unique IDs, whitespace normalization, invalid requests, input length boundaries, malformed JSON, and allowed/rejected CORS origins. The frontend build includes strict TypeScript checking. `npm run test:api` requires the backend running and exercises the actual Axios service against it, including API validation errors. With the backend stopped, `npm run test:api -- --unavailable` checks connection-error handling. These service checks are not browser/UI tests. No tests claim future AI functionality works.

For a manual integration check, submit a valid ticket and inspect the received status; try an empty/whitespace-only ticket for validation; stop the backend and submit again to see the connection error, then restart it and retry.

## Docker

From the repository root, with Docker Desktop running Linux containers:

```powershell
docker compose up --build -d
docker compose ps
docker compose logs
```

Open http://localhost:5173; API docs remain at http://localhost:8000/docs. Compose waits for the backend health check before starting the Nginx frontend. It builds the backend on Python 3.12 and the frontend on Node 22. No ML or database containers are added.

```powershell
docker compose down
```

For another browser-accessible backend address, set `$env:VITE_API_BASE_URL='http://your-host:8000'` before rebuilding and update the backend CORS origins in Compose to match the frontend origin. `backend` is a container hostname, not the browser API address.

## Data and limitations

- [Dataset format and sample-data boundaries](docs/dataset.md)
- [Architecture diagram and extension points](docs/architecture.md)
- [Assumptions and decisions](docs/assumptions.md)
- [Remaining Days 2-13](docs/development-plan.md)
- [Validation results and browser-check limitation](docs/validation.md)

The included 20 sample tickets and four knowledge-base documents are fictional, not an approved dataset or business policy. The API does not read them. No persistence, authentication, actual classification, model training/tuning, deep learning, embeddings, retrieval, RAG, agents, security analysis, explainability, fairness, MLOps, monitoring, or CI/CD is implemented. There is no model performance claim.

Before Day 2, supply the approved assessment dataset and confirm label definitions. A restrictive placeholder LICENSE is included; the project owner can select an open-source license if needed. Git has not been initialized automatically.
