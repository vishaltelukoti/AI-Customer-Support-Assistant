# Assumptions and decisions

- This is a single-developer, local assessment POC. The original workspace was empty and was not a Git repository.
- Python 3.12+ and Node.js 22.12+ are prerequisites. Local validation uses the available Python 3.14 and Node 24; the backend Docker image uses Python 3.12.
- No approved dataset or policies were supplied. All included ticket examples and knowledge-base text are clearly marked fictional samples.
- Ticket category and priority vocabularies are provisional. The final dataset and severity definitions require review before Day 2 training.
- Ticket intake only acknowledges input; it does not persist tickets or read sample files. Reloading the page clears the displayed result.
- The 10,000-character limit is a POC input constraint. Surrounding whitespace is trimmed. HTTP 200 means acknowledged, and HTTP 422 means invalid request.
- No authentication, PII detection, or other Day 9 security feature is implemented. Use fictional tickets for the Day 1 demo.
- Local development uses ports 8000 and 5173. CORS origins can be changed through backend settings. Frontend configuration is bundled at build time.
- No database, router, state-management library, or additional application framework is needed for one form.
- The LICENSE file reserves rights pending the project owner's license choice; no open-source license is assumed.
- No CI/CD workflow is implemented; `.github/workflows/.gitkeep` only preserves the directory.
