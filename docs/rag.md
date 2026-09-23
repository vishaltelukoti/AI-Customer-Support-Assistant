# Day 7: suggested-resolution RAG

## Scope

Day 7 implements a lightweight offline Retrieval-Augmented Generation pipeline. It does not add FastAPI endpoints, LangGraph, agents, memory, streaming generation, cloud infrastructure, hosted model APIs, or production observability.

No API key is required for the local retrieval plus local generation setup.

## Architecture

The flow is:

```text
incoming ticket_text
-> Day 6 FAISS retrieval over training historical ticket_text embeddings
-> bounded context with ticket text, category, priority and previous resolution
-> local FLAN-T5 generation with grounding instructions
-> structured answer, sources, retrieved tickets and retrieval status
```

The retrieval query is based only on the incoming ticket text. It does not include answer, category, priority, tags, validation/test labels or hidden metadata. Historical answer text is included only after retrieval, as previous-resolution evidence in the generation context.

## Model and Prompting

Generation uses the cached local public model `google/flan-t5-base`. The prompt tells the model to use only supplied historical context, avoid inventing policies or facts, treat retrieved content as reference data rather than instructions, abstain when evidence is insufficient, and identify supporting source IDs.

Default configuration:

- Top-K: 3
- retrieval threshold: 0.55
- max context per ticket: 900 characters
- max incoming ticket text: 1,200 characters
- max new tokens: 120

The threshold is a POC setting, not an optimized confidence boundary.

## Source Attribution

Each grounded response includes source IDs from retrieved historical training tickets. The service returns source metadata with:

- ticket ID
- similarity score
- category
- priority
- short source excerpt

The service does not fabricate source IDs; sources must come from retrieved tickets.

## Insufficient Information

If no retrieved ticket reaches the threshold, the service returns:

```text
Insufficient information in retrieved historical tickets.
```

The generator is not called in this case, and the response status is `insufficient_evidence`.

## Evaluation

The Day 7 evaluation uses four fixed POC cases:

- a billing/payment ticket
- a service outage ticket
- a prompt-injection ticket
- an unrelated insufficient-information ticket

Metrics are intentionally narrow and inspect source attribution, acceptable grounded status, prompt-injection resistance at the RAG layer, and insufficient-information behavior. They are not a broad measure of answer quality.

| Metric | Value |
| --- | ---: |
| Source attribution rate | 1.000000 |
| Grounded acceptable response rate | 1.000000 |
| Insufficient-information pass rate | 1.000000 |
| Overall acceptance rate | 1.000000 |

Artifacts are saved in `experiments/rag/`:

- `rag_config.json`
- `rag_evaluation.json`
- `rag_evaluation.md`

## Limitations

The source tickets and resolutions are synthetic. Retrieval relevance is not the same as answer correctness, and the local generation model can still produce weak or overly terse responses. The POC evaluation is small and rule-based. The system is not production-ready and does not include policy validation, safety moderation, reranking, answer grading, human review workflows, API integration, or Day 9 security controls.
