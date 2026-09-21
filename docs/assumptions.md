# Assumptions and decisions

- This is a single-developer, local assessment POC built on the existing Git repository. Its working API/UI, raw sample data, tests and Docker configuration are preserved.
- Python 3.12+ and Node.js 22.12+ are prerequisites. Local validation uses the available Python 3.14 and Node 24; the backend Docker image uses Python 3.12.
- Synthetic data is sufficient for this POC. All included ticket examples and knowledge-base text are fictional, not production customer data or business policy.
- Only the selected Kaggle CSV is processed. Its ten queue labels and lowercase low/medium/high priorities are preserved; seed 42 produces grouped queue/priority 70/15/15 splits of English tickets. All source versions are retained and raw bytes stay unchanged. The old 20-row sample is historical only.
- The project owner supplied the selected Kaggle dataset locally. No other variants are used. Synthetic class imbalance, annotation quality and unknown paraphrase relationships limit evaluation claims; review severity definitions before deployment.
- Ticket intake only acknowledges input; it does not persist tickets or read sample files. Reloading the page clears the displayed result.
- The 10,000-character limit is a POC input constraint. Surrounding whitespace is trimmed. HTTP 200 means acknowledged, and HTTP 422 means invalid request.
- No authentication or runtime Day 9 security feature is implemented. The offline dataset preparation masks obvious privacy patterns in ticket text and answers; this is not a PII-free guarantee. Use fictional tickets for the Day 1 demo.
- Local development uses ports 8000 and 5173. CORS origins can be changed through backend settings. Frontend configuration is bundled at build time.
- No database, router, state-management library, or additional application framework is needed for one form.
- The LICENSE file reserves rights pending the project owner's license choice; no open-source license is assumed.
- No CI/CD workflow is implemented; `.github/workflows/.gitkeep` only preserves the directory.
- Day 2 is a fixed unweighted TF-IDF + Logistic Regression baseline for each target. The 50,000-feature cap and single numerical thread keep training practical; no tuning, class balancing or resampling was used. Minority-class limitations remain visible in the reports.
- The standalone ML service returns separate category/priority probabilities. They are uncalibrated model outputs, not guaranteed correctness; the HTTP ticket response still has null AI fields. Load only trusted local joblib artifacts with compatible library versions.
