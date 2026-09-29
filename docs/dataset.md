# Selected dataset and preprocessing audit

The project uses a publicly available, synthetically generated customer-support ticket dataset for model development and evaluation. It is not real customer data.

- **Dataset:** Customer IT Support - Ticket Dataset
- **Creator:** Tobias Bueck
- **Source:** [Kaggle data card](https://www.kaggle.com/datasets/tobiasbueck/multilingual-customer-support-tickets)
- **Synthetic provenance:** [Creator's dataset description](https://softoft.de/blog/ticket-dataset/)
- **Only selected input:** `data/raw/aa_dataset-tickets-multi-lang-5-2-50-version.csv`

No other dataset variants were downloaded, added or combined. The original 20-row Day 1 hand-written sample CSV was removed during final cleanup because it is not part of the selected assessment dataset and is not read by the selected-data pipeline or its tests. Neither raw nor processed data is loaded directly by the API.

The Kaggle page describes the dataset collection; measured local values below take precedence over collection-wide descriptions. In particular, this file has two languages, eight tag columns, no ticket ID or business-type column, and lowercase `low`, `medium`, `high` priorities. Row-level `version` is not a Kaggle release number and is not a quality score.

## Source integrity and schema


Raw records: **28,587**. Columns: **16**. File size: **25,996,354 bytes** (24.79 MiB). UTF-8 cell-value payload: 25,419,855 bytes; this excludes Python object overhead and is not a process-RAM measurement.

Raw SHA-256: `f187c090e59581c2bbf3aa1377c8db4dd647464ecf2ae51bf8966e42e0ed6bc0`. The source remains byte-for-byte unchanged.


| Column | Inferred type (CSV storage) | Missing | Missing % | Unique nonmissing |
| --- | --- | --- | --- | --- |
| subject | string (str) | 3838 | 13.426 | 24749 |
| body | string (str) | 0 | 0.000 | 28587 |
| answer | string (str) | 7 | 0.024 | 28580 |
| type | string (str) | 0 | 0.000 | 4 |
| queue | string (str) | 0 | 0.000 | 10 |
| priority | string (str) | 0 | 0.000 | 3 |
| language | string (str) | 0 | 0.000 | 2 |
| version | integer-like string (str) | 0 | 0.000 | 3 |
| tag_1 | string (str) | 0 | 0.000 | 116 |
| tag_2 | string (str) | 13 | 0.045 | 256 |
| tag_3 | string (str) | 136 | 0.476 | 392 |
| tag_4 | string (str) | 3058 | 10.697 | 554 |
| tag_5 | string (str) | 14042 | 49.120 | 602 |
| tag_6 | string (str) | 22713 | 79.452 | 575 |
| tag_7 | string (str) | 26547 | 92.864 | 427 |
| tag_8 | string (str) | 28022 | 98.024 | 224 |


All values are read as strings to preserve original labels and version values. Blank or whitespace-only cells count as missing; literal text such as `NA` is not automatically discarded. Missing subjects are allowed when a body exists. Three English answers are missing and retained as blank in the historical file; no resolution is invented.

## Exact field mapping

| Application meaning | Actual source |
| --- | --- |
| ticket_id | No source column; generated as described below |
| subject | subject |
| body | body |
| answer | answer |
| category | queue |
| priority | priority |
| ticket type | type |
| language | language |
| version | version |
| tags | tag_1 through tag_8 |

IDs are `KAGGLE-<first 12 hex characters of raw SHA256>-<six-digit 1-based data-record number>`. They identify source rows, not original customer tickets. Source records count parsed CSV rows after the header, not physical lines (quoted fields can span lines). The historical file retains `source_record`, `version`, and `language`. Identical input bytes/order produce stable IDs; edits/reordering change the source hash and IDs.

## Language selection


| Label | Raw count | Raw % | English count | English % |
| --- | --- | --- | --- | --- |
| de | 12249 | 42.848 | 0 | 0.000 |
| en | 16338 | 57.152 | 16338 | 100.000 |


English retained: **16,338 / 28,587 (57.1519%)**. Non-English excluded from working data: **12,249**. Raw German rows remain intact.



Use the exact source label `language == "en"`. Labels are complete and limited to `en`/`de`; first/middle/last English examples within each of the three versions were inspected and were consistent with the English label. This is a spot check, not certification of every row. Missing, unknown or unsupported language labels stop processing, rather than invoking an invented language classifier.

## Version and overlap analysis


| Label | Raw count | Raw % | English count | English % |
| --- | --- | --- | --- | --- |
| 400 | 18599 | 65.061 | 10441 | 63.906 |
| 51 | 869 | 3.040 | 551 | 3.373 |
| 52 | 9119 | 31.899 | 5346 | 32.721 |

| Check | Raw excess rows | Raw cross-version groups | English excess rows | English cross-version groups |
| --- | --- | --- | --- | --- |
| exact_rows | 0 | 0 | 0 | 0 |
| subject_body | 0 | 0 | 0 | 0 |
| normalized_ticket_text | 0 | 0 | 0 | 0 |
| exact_subject | 0 | 0 | 0 | 0 |
| normalized_subject | 1 | 1 | 0 | 0 |
| normalized_body | 2 | 1 | 1 | 0 |
| obvious_template | 0 | 0 | 0 | 0 |
| obvious_body_template | 0 | 0 | 0 | 0 |


Ticket-ID overlap is **not assessable**: the source has no ticket-ID column. Generated IDs are unique provenance identifiers, not evidence of independent logical tickets.

One normalized subject is shared across versions 52/400 (raw data records 1823/22570). Two normalized-body pairs occur overall; one crosses versions 52/400 (German records 5396/11626), and the other is within English version 400 (records 10040/15128). Their generic bodies are `Assistance needed` and `Assistance Required`, respectively. Different subjects and conflicting labels make these insufficient grounds for deleting rows. There is no identical full normalized ticket text across versions, and the deterministic long-text numeric-template check found no repeated templates.

**Decision:** retain all three versions. The measured evidence does not support selecting only version 400 or treating the versions as substantially repeated full tickets. This does not prove that every record is semantically independent: paraphrases, translations and hidden generation relationships were not inferred.

Before splitting, connected groups are built from any shared nonempty source ID (if supplied by a compatible fixture), normalized subject, normalized body, normalized/masked full text, or long numeric-substitution template. Shared subjects and generic bodies cause conservative grouping, never deletion by themselves. The two English records sharing a body have different queue/priority labels; both are retained in the training set as one group. All detected grouping keys and full normalized texts are checked for cross-split overlap, which is zero.

## Queue/category labels


| Label | Raw count | Raw % | English count | English % |
| --- | --- | --- | --- | --- |
| Billing and Payments | 2788 | 9.753 | 1595 | 9.763 |
| Customer Service | 4268 | 14.930 | 2410 | 14.751 |
| General Inquiry | 405 | 1.417 | 236 | 1.444 |
| Human Resources | 576 | 2.015 | 348 | 2.130 |
| IT Support | 3433 | 12.009 | 1942 | 11.886 |
| Product Support | 5252 | 18.372 | 3073 | 18.809 |
| Returns and Exchanges | 1437 | 5.027 | 820 | 5.019 |
| Sales and Pre-Sales | 918 | 3.211 | 513 | 3.140 |
| Service Outages and Maintenance | 1148 | 4.016 | 664 | 4.064 |
| Technical Support | 8362 | 29.251 | 4737 | 28.994 |


All **10** source queues are preserved. The English majority is **Technical Support (4,737)**; the minority is **General Inquiry (236)**, a **20.07:1** ratio. No category has fewer than 30 English rows. No classes were merged, invented, resampled or discarded.



## Priority labels


| Label | Raw count | Raw % | English count | English % |
| --- | --- | --- | --- | --- |
| high | 11178 | 39.102 | 6346 | 38.842 |
| low | 5894 | 20.618 | 3374 | 20.651 |
| medium | 11515 | 40.281 | 6618 | 40.507 |


`priority` is a complete three-class target. Its original lowercase values are preserved exactly; there is no remapping to the legacy API's capitalized placeholders. Labels are synthetic annotations and have not been independently adjudicated. Class imbalance and possible annotation errors remain limitations for later evaluation.

## Text quality and cleaning

Statistics below describe cleaned English text before privacy substitutions. Lengths count Unicode characters; missing subjects contribute zero to the mean/median.


| Field | Missing | Empty cleaned | Min | Max | Mean | Median | Short 1-19 chars | Over 10,000 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| body | 0 | 0 | 6 | 1141 | 367.678 | 377.0 | 31 | 0 |
| subject | 2607 | 2607 | 0 | 234 | 36.651 | 39.0 | 443 | 0 |
| ticket_text | 0 | 0 | 18 | 1184 | 406.009 | 417.0 | 1 | 0 |


Input is cleaned `subject + "\n\n" + body`; if one part is absent, use the other. Real CRLF/CR characters and observed literal email `\n`/`\t` escapes are normalized. Windows drive/UNC paths are protected from escape decoding. Horizontal whitespace is collapsed, excess blank lines reduced to two newlines, and outer whitespace removed. Meaningful case, punctuation, numbers, error codes and natural language are retained. There is no tokenization, stemming, lemmatization, stopword removal or truncation.

All English tickets have meaningful combined text. One combined ticket is under 20 characters and is retained. Empty/punctuation-only tickets and unsupported/missing labels would be reported and removed; malformed CSV structure or missing required columns fails clearly. The selected run removes **zero invalid records**.

Exact duplicates are audited before filtering; normalized/masked full ticket text is deduplicated before splitting, keeping the first valid row. Conflicting labels on identical normalized/masked input stop processing for review instead of silently choosing a label. No full-text duplicates occur in the selected English run, so **zero records are deleted**. Short repeated bodies are retained and grouped as explained above.

## Basic privacy audit

Only the English working texts and their answers are exported. The following counts are ordered, non-overlapping matches on valid English records before deduplication. A more specific mask consumes a span before a broad numeric pattern; counts must not be summed as independent raw-regex matches.


| Pattern | Ticket-text occurrences | Answer occurrences | Action |
| --- | --- | --- | --- |
| account_number | 0 | 0 | replace matched span with [ACCOUNT_NUMBER] |
| card_like | 0 | 0 | replace matched span with [CARD_LIKE] |
| email | 0 | 0 | replace matched span with [EMAIL] |
| long_numeric_id | 0 | 0 | replace matched span with [LONG_NUMERIC_ID] |
| phone | 1 | 70 | replace matched span with [PHONE] |
| url | 0 | 0 | replace matched span with [URL] |


Patterns check emails, HTTP(S)/www URLs, account/IBAN-like values, card-like 13-19 digit sequences, common/international phones and long numeric identifiers of seven or more digits. The one ticket-text phone and 70 answer phones are masked. All generated text/answers were rescanned for these same patterns. This is not proof of a PII-free dataset: names, addresses, contextual identifiers and unusual formats may remain. URLs/numeric matches can be nonpersonal and are masked conservatively. The raw CSV is never modified. This offline preparation does not implement a runtime security layer.

## Classification leakage review


| Field | Classification input? | Reason |
| --- | --- | --- |
| subject | Yes | Customer email subject available at submission. |
| body | Yes | Customer email message available at submission. |
| answer | No | Agent response after submission; future historical retrieval only. |
| type | No | Source describes agent-picked ticket type; creation-time availability not established. |
| queue | No | Routing label; category prediction target, not an input. |
| priority | No | Assigned urgency label; priority prediction target, not an input. |
| language | No | Used only for English selection, not a classification feature. |
| version | No | Dataset provenance/version, not customer-provided ticket content. |
| tag_1 | No | Assigned tag/metadata; creation-time availability not established. |
| tag_2 | No | Assigned tag/metadata; creation-time availability not established. |
| tag_3 | No | Assigned tag/metadata; creation-time availability not established. |
| tag_4 | No | Assigned tag/metadata; creation-time availability not established. |
| tag_5 | No | Assigned tag/metadata; creation-time availability not established. |
| tag_6 | No | Assigned tag/metadata; creation-time availability not established. |
| tag_7 | No | Assigned tag/metadata; creation-time availability not established. |
| tag_8 | No | Assigned tag/metadata; creation-time availability not established. |


Only the customer subject/body become `ticket_text`. `category` and `priority` are targets; the generated `ticket_id` is metadata. No `answer`, resolution, type, tag, language, version, agent or outcome field becomes a classification feature. Words indicating the customer's problem in the original message are legitimate input and are not stripped as label leakage. No final-status, resolution-time, escalation-outcome or satisfaction fields exist in this CSV. Confidence is a model output, not a dataset label.

## Deterministic splits


| Set | Rows | Actual % | Maximum queue deviation (percentage points) | Maximum priority deviation (percentage points) |
| --- | --- | --- | --- | --- |
| train | 11436 | 69.99633 | 0.011619 | 0.023112 |
| validation | 2451 | 15.00184 | 0.052195 | 0.034192 |
| test | 2451 | 15.00184 | 0.070204 | 0.074317 |


Seed: **42**. Target: **70/15/15**. Stratification uses queue/priority pairs, preserving both target distributions. Per-stratum integer rounding is balanced against global largest-remainder counts. Related records are indivisible groups, and mixed-label groups are reserved for training. A stratum with fewer than three independent groups is retained in training and reported; if three nonempty sets cannot be formed, the command fails before replacing outputs. No minority class is deleted. The current data has all ten queues and all three priorities in every split; maximum priority deviation is under 0.08 percentage points.

Grouping takes precedence over exact ratios when necessary. Current counts are 11,436/2,451/2,451; the only grouping fallback is the two-row mixed-label English body group. Per-split queue, priority and version counts/percentages are included in `dataset_audit.json`. These checks address detected identity/version leakage, not arbitrary semantic similarity. Do not fit future transforms on validation or test data.

## Outputs and historical data

`train.csv`, `validation.csv`, `test.csv` each contain exactly:

```text
ticket_id,ticket_text,category,priority
```

`historical_tickets.csv` contains:

```text
ticket_id,ticket_text,category,priority,answer,source_record,version,language,split,type,tag_1,tag_2,tag_3,tag_4,tag_5,tag_6,tag_7,tag_8
```

It preserves all 16,338 processed English tickets, with masked answers, source provenance, original type/tags and split membership. Three missing answers remain blank. This is preparation for future similar-ticket retrieval/RAG only: no embeddings, retrieval or RAG is implemented. During later evaluation, restrict the retrieval corpus to `split == "train"`; held-out answers must not be exposed to validation/test predictions. The historical CSV must never be used wholesale as classification features.

`dataset_audit.json` contains the source checksum, complete schema/missingness/uniqueness audit, label/language/version distributions, overlap groups by original data-record number, text statistics, privacy counts/actions, field leakage decisions, stage counts, split statistics, integrity checks and output CSV hashes. It contains no raw text examples or PII values. No timestamps or absolute paths are embedded, allowing byte-identical repeat runs.

## Reproducibility

From the repository root:

```powershell
.\backend\.venv\Scripts\python.exe -m backend.app.ml.preprocessing
```

Explicit selected input and optional output/seed:

```powershell
.\backend\.venv\Scripts\python.exe -m backend.app.ml.preprocessing --input data/raw/aa_dataset-tickets-multi-lang-5-2-50-version.csv --output-dir data/processed --seed 42
```

With the virtual environment activated, `python -m backend.app.ml.preprocessing` is equivalent. From `backend`, use `python -m app.ml.preprocessing`. Defaults resolve relative to the repository; explicit paths follow the working directory. Python standard library only, with no new dependencies. The raw dataset is not included in the API Docker image; preprocessing runs on the repository checkout.

Repeated full runs with the same source bytes/order, implementation and seed must produce identical CSVs and audit JSON. Outputs are staged before replacement; parsing/validation/splitting failures leave the prior outputs untouched, so old files may be stale after a failed command. Input/output path collisions are rejected. The selected source and named processed artifacts are Git-allowlisted; unrelated raw/processed files remain ignored. No automatic commit or push occurs.

The application continues to acknowledge tickets with null category/priority/confidence. The API's old response vocabularies are unchanged because no model is connected; aligning prediction schemas is a later integration task. [Day 2 baseline ML](baseline-ml.md) now trains and evaluates two independent classifiers using these unchanged classification splits; it never loads historical answers.
