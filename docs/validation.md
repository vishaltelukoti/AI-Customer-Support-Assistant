# Dataset integration and Day 2 validation record

## Day 2 baseline validation (2026-09-18)

Day 2 is complete. The earlier dataset integration record below is retained as history. The existing raw source, processed splits and Day 1 API/frontend implementation were preserved. Only the backend dependency list, new standalone ML modules/tests, experiment outputs and documentation changed for Day 2.

| Check | Result |
| --- | --- |
| Existing tests | 60 unchanged cases: 15 API and 45 preprocessing |
| New tests | 24 Day 2 cases, using lightweight real fitted classifiers |
| Local full pytest suite | 84 passed, 0 failed on Python 3.14.4 |
| Docker full pytest suite | 84 passed, 0 failed on Python 3.12 with a read-only repository mount |
| Dataset | 11,436 train / 2,451 validation / 2,451 test; all 10 categories and 3 priorities; audited hashes verified |
| Training | Two independent TF-IDF + Logistic Regression pipelines; only training text/labels fitted; 50,000 features each; converged in 112/59 iterations |
| Leakage checks | Exact fit-call inputs checked; held-out tokens absent from fitted vocabulary; answers never loaded; duplicate normalized text/IDs rejected |
| Evaluation | Seven aggregate metrics and per-class reports/confusion arrays for train, validation and test |
| Artifacts | Both complete pipelines, metrics JSON/Markdown, two PNG/CSV test matrices, actual example predictions saved |
| Visual matrix check | Both PNGs inspected; labels and counts readable |
| Full reproducibility | Two complete fixed-configuration runs produced byte-identical models, results JSON, Markdown, PNG and CSV outputs |
| Saved-model inference | Actual example predictions generated; root and backend-directory imports can load saved pipelines |
| Confidence | Per-target winning class probabilities from predict_proba; deterministic and in [0,1] |
| Frontend build | Strict TypeScript and Vite production build passed |
| Docker backend | Image rebuilt with Day 2 dependencies; container started and became healthy |
| Live API | GET /health passed; POST /api/v1/tickets returned a UUID, received status and null category/priority/confidence |
| Frontend integration | Actual Axios service smoke test passed, including API validation-error handling; served frontend returned HTTP 200 |
| API integration | Intentionally absent: no route, schema or ticket-service changes |
| Day 3 | Grid Search, Random Search and Bayesian Optimization NOT IMPLEMENTED |

Commands used from the repository root:

```powershell
.\backend\.venv\Scripts\python.exe -m backend.app.ml.baseline_models
.\backend\.venv\Scripts\python.exe -m backend.app.ml.predict_examples
.\backend\.venv\Scripts\python.exe -m pytest -c backend/pytest.ini -q backend/tests
docker compose build backend
docker compose run --rm --no-deps -v "${PWD}:/repo:ro" -w /repo/backend backend python -m pytest -q -p no:cacheprovider
docker compose up -d
```

From `frontend`: `npm run build` and `npm run test:api`. Full reproducibility was checked with a second `run_baseline` call into a temporary directory, exact parsed-JSON equality, and byte comparison of every generated file against the first run. The temporary outputs were removed automatically. Training configuration was never changed after evaluating held-out data. Existing raw/processed SHA-256 values below remain unchanged.

Both pytest runs emitted 97 upstream deprecation warnings: two existing Starlette/AnyIO test-client warnings and 95 joblib warnings about NumPy 2.5 array shape assignment while loading real test artifacts. They did not cause failures and were not suppressed. Dependency compatibility was also checked with `pip check`. The measured ML dependency versions are recorded in [baseline-ml.md](baseline-ml.md) and the results JSON.

The frontend's source and visual design are unchanged. Browser interaction was not revalidated in Day 2; the prior browser-tool limitation below still distinguishes UI testing from API/Axios/build checks. Docker remains running on ports 5173/8000. No commit or push was performed.

## Earlier dataset integration record

Validated on 2026-09-18. This task extended the existing Day 1 pipeline rather than replacing the application. The initial working tree already contained the preceding Day 1 changes; those were preserved. The project owner's selected raw CSV was inspected in place. No other dataset variants were added or downloaded.

## Checks performed

| Check | Result |
| --- | --- |
| Repository inspection | API, schemas, service, frontend, tests, preprocessing, raw/processed/KB data, experiments, configuration, docs and Docker reviewed |
| Selected CSV | 28,587 rows, 16 columns; field mapping and first/last records inspected |
| Raw integrity | Original and final SHA-256 match; raw CSV unchanged |
| Local complete pytest suite | 60 passed, 0 failed (15 existing API tests and 45 data-pipeline cases) |
| Container pytest, Python 3.12 | 60 passed, 0 failed with the checkout mounted read-only |
| Full preprocessing | 16,338 English rows retained; all 10 queues and 3 priority labels preserved |
| Version/identity handling | All three versions retained; repeated generic-body pair grouped in train; zero cross-split grouping-key or normalized-text overlap |
| Privacy patterns | One phone-like ticket span and 70 phone-like answer spans masked; rescan of exported text/answers found none of the checked patterns |
| Full reproducibility | Two complete runs produced byte-identical train, validation, test, history and audit files |
| Output contract | Classification CSVs have exactly ticket_id,ticket_text,category,priority; no answer fields |
| Historical data | Every row English, aligned to classification ID/split; three missing answers remain blank |
| Frontend production build | Strict TypeScript and Vite build passed |
| Docker build/start | Existing two images built; backend healthy and frontend started |
| Live API | GET /health and valid POST /api/v1/tickets passed; blank/whitespace inputs rejected |
| Placeholder behavior | Category, priority and confidence remain null; status remains received |
| Actual frontend Axios service | npm run test:api passed against the rebuilt backend, including validation-error handling |
| Served frontend | HTML and compiled assets returned HTTP 200 |
| Browser UI | BLOCKED: automation inventory returned no connected apps or browsers |

The frontend/API source, dependencies and Docker configuration were not changed by this dataset task. Two existing test-client deprecation warnings concern Starlette's httpx adapter and AnyIO's BlockingPortal alias; no tests failed. No runtime security or ML feature was added.

## Data result

Raw: 28,587. English: 16,338 (57.151852%). Non-English excluded from working data: 12,249. Invalid removed: 0. Duplicates removed: 0. Final English: 16,338.

Seed 42, grouped queue/priority stratification: train 11,436 (69.996328%); validation 2,451 (15.001836%); test 2,451 (15.001836%). Every set contains all ten queues and all three priorities. Maximum category deviation is 0.070204 percentage points; maximum priority deviation is 0.074317 percentage points. Source/body/template grouping prevents detected overlap without deleting minority classes or changing their labels.

This is not a claim that the dataset is PII-free, free of all semantic overlap, accurately labelled for production, or representative of real customers. See [the detailed dataset audit](dataset.md) for evidence and limitations.

## Reproducibility hashes

Raw SHA-256: `f187c090e59581c2bbf3aa1377c8db4dd647464ecf2ae51bf8966e42e0ed6bc0`.

| Output | SHA-256 |
| --- | --- |
| dataset_audit.json | `ee291a0e6ba97d31c45603fc23db08184fdfb5b8225f973de3f3322aa946214b` |
| historical_tickets.csv | `ab09c8aee33df5affa05053a9f50d0e998241c4d9223afacbc2e2520fcec5bbf` |
| test.csv | `7078fa22d7510476bb2191c027ec474cfea5c62fb30faa48854e3431178a21ee` |
| train.csv | `f1e027454bb4f3d52304bc808a025083e9b821fedc9c6e84dbadbd11bd76a1df` |
| validation.csv | `044ad717a86e9437e8be8592a4b80686d7f4da2a110bdbd13e09556ff46ea734` |

Run from the repository root:

```powershell
.\backend\.venv\Scripts\python.exe -m backend.app.ml.preprocessing
.\backend\.venv\Scripts\python.exe -m pytest -c backend/pytest.ini -q backend/tests
```

Frontend checks from `frontend`: `npm run build`, then `npm run test:api` with the backend running. Container tests were run using a read-only repository bind mount with `python -m pytest -q -p no:cacheprovider` from `/repo/backend` in the backend image.

## Remaining browser verification

The available browser tool returned an empty browser/app inventory. Visual layout and interactive form submission/loading/error states therefore remain unverified through browser automation. API checks and the real Axios service do not substitute for browser interaction tests. Open http://localhost:5173 and submit a fictional ticket; verify received status, echoed text/ID and unavailable classification fields. Check empty and whitespace input, then backend failure/recovery if required.

The Docker application was left running on 5173/8000 for review. Use `docker compose down` before starting local servers on the same ports. No commit or push was performed. At the end of this earlier dataset task, preparation was complete and Day 2 training had not yet begun; the Day 2 record at the top documents its subsequent completion.
