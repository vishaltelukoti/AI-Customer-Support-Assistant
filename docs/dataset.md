# Dataset foundation

No approved assessment dataset was present in the workspace. `data/raw/sample_tickets.csv` is **SAMPLE DATA ONLY**, containing 20 fictional, manually labelled tickets. Its `SAMPLE-` IDs and accompanying README distinguish it from a future training dataset. It is intended only for local API/UI input examples; it is not loaded by the application, and its labels are never returned as predictions.

## Expected CSV format

Use UTF-8 CSV with a header, quoted fields where commas or line breaks occur, one ticket per row, and these columns:

| Field | Meaning | Expected constraints |
| --- | --- | --- |
| ticket_id | Source ticket identifier | Nonempty, unique string |
| ticket_text | Customer issue and relevant context | Nonblank text; API currently accepts up to 10,000 characters after trimming |
| category | Ground-truth issue category | At least four categories; provisional labels: Payment, Refund, Account, Shipping, Technical |
| priority | Ground-truth priority/severity | Provisional labels: Low, Medium, High |

Sample priority labels are illustrative, not an approved triage policy. High indicates an urgent blocked workflow or significant customer impact; Medium indicates a routine issue requiring investigation; Low indicates a general question or minor request. Resolve actual severity definitions with the dataset owner before training.

Keep the approved source files unchanged in `data/raw/`; document their source, usage approval, label mappings, missing values, duplicates, and class balance when supplied. Place derived files in `data/processed/`. Generated/large data is ignored by Git; the small sample CSV and its README are tracked.

Train/validation/test splitting is future work. Plan a reproducible split (for example 70/15/15 when data volume permits) stratified by category and priority where feasible. Keep duplicates and related conversations together, fit preprocessing on training data only, and reserve the test set for final evaluation. Review a chronological split if the approved data requires time-based evaluation. Do not train or report metrics using these 20 sample rows.

The four short text files in `data/knowledge_base/` are fictional sample content for future retrieval. They are not business policy, are not indexed, and do not support answers in Day 1.
