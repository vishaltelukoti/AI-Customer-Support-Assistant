# Day 1 validation record

Validated on 2026-09-17. The workspace began empty. Dependencies were installed in `backend/.venv` and `frontend/node_modules`; frontend dependency versions are recorded in `package-lock.json`.

| Check | Result |
| --- | --- |
| Local backend, Python 3.14.4 | 15 pytest tests passed |
| Backend dependency consistency | `pip check`: no broken requirements |
| Live Uvicorn | Started successfully; GET /health and POST /api/v1/tickets returned expected HTTP 200 bodies |
| Frontend, Node 24.19.0 | Dependency installation and strict TypeScript + Vite production build passed |
| Vite development server | Started successfully on port 5173 |
| Frontend Axios service against live API | Valid ticket returned acknowledgement and null predictions; blank ticket returned 422 and the expected error message |
| Frontend Axios service with backend stopped | Connection error mapped to the expected unavailable-API message |
| Sample CSV | 20 rows, five categories, three priority values |
| Docker build | Both images built successfully; frontend uses Node 22 build / Nginx runtime |
| Docker Compose startup | Backend healthy; frontend running; frontend HTML and JavaScript returned HTTP 200 |
| Container backend, Python 3.12 | All 15 pytest tests passed |
| Container integration | Health, configured CORS preflight, and frontend Axios service checks passed |

Two upstream test-client deprecation warnings occur with the installed Starlette/AnyIO versions: the httpx adapter and BlockingPortal alias are deprecated. Tests pass; httpx is retained as required by the assessment. No future AI capability was tested or claimed.

## Browser verification limitation

Browser UI automation was attempted, but the computer-use tool stopped because it could not determine the active browser URL confidently enough to enforce its policy. No further browser automation was performed. Consequently, visual layout, interactive empty-input validation and loading/error display are implemented but not independently verified through browser automation. The actual frontend Axios service and its error mapping were verified separately, not substituted as a claim of end-to-end browser testing.

To finish the manual UI check, open http://localhost:5173, submit a sample ticket, and confirm the received status and unavailable AI fields. Submit empty/whitespace-only text to verify inline validation. Stop the backend and submit a valid ticket to verify the connection-error display, then restart it. Local and Docker run instructions are in the README.

The Docker Compose stack was left running for review. Stop it from the project root with `docker compose down` before starting local servers on the same ports.
