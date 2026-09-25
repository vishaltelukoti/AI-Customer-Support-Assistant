# Day 12 Integrated Assistant API

Day 12 connects the completed POC components behind the existing FastAPI ticket endpoint and updates the React UI to show the full support-assistant result. It does not retrain classifiers, rebuild the FAISS index, change the RAG design, or add persistent storage.

## Backend Flow

`POST /api/v1/tickets` accepts:

```json
{
  "subject": "Payment failed",
  "body": "My card payment failed during checkout. Invoice unpaid.",
  "top_k": 3
}
```

The legacy `{ "ticket_text": "..." }` payload is still accepted for compatibility.

The integrated flow is:

1. Input security check with the Day 9 deterministic rules.
2. PII redaction when explicit email, phone or card-like values are present.
3. Day 3 optimized category and priority classification.
4. Day 6 FAISS similar-ticket retrieval with default Top-K 3.
5. Day 8 deterministic complexity routing.
6. Simple tickets use Day 7 RAG directly.
7. Complex tickets use the Day 8 LangGraph workflow.
8. Output security check.
9. Day 10-style linear text explanation for the category prediction.
10. Structured API response with timings and Day 11 monitoring counters.

Blocked input does not enter classification, retrieval, RAG or agents. If a local component fails to load or execute, the endpoint returns a structured fallback response with `status: "error"` instead of an unhandled exception.

## Response Shape

The response contains:

- `ticket`: generated ticket ID plus normalized subject/body/text.
- `classification`: category, priority and confidence values from the saved Day 3 optimized classifiers.
- `similar_tickets`: FAISS results with ticket ID, score, category, priority and clipped ticket text.
- `workflow`: simple or complex route plus public trace.
- `response`: grounded support answer, source tickets and retrieval status.
- `explanation`: predicted category confidence and top contributing text features.
- `security`: allowed/status/reason/category/matched rule.
- `timings`: total, classification, retrieval and workflow latency in milliseconds.
- `status`: `completed`, `blocked` or `error`.

## Frontend

The React support page now collects subject and body, posts `top_k: 3`, and renders the integrated response. It shows classification, similar tickets, final response, sources, workflow trace, security status, explanation terms and timings. It keeps the existing validation and connection-error handling.

## Evaluation

The fixed Day 12 evaluation runs from a clean Python process:

```powershell
python -m backend.app.services.ticket_service
```

It writes:

- `experiments/integration/integration_config.json`
- `experiments/integration/integration_evaluation.json`
- `experiments/integration/integration_evaluation.md`

The five cases cover:

- simple payment failure
- complex multiple-charge/refund/cancellation
- prompt-injection blocking
- insufficient information
- PII redaction

Latest measured result:

| Metric | Value |
| --- | ---: |
| Cases | 5 |
| Passed | 5 |
| Average latency | 78.41 ms |
| Default Top-K | 3 |

## Monitoring

The existing Day 11 in-memory monitor records request count/error/latency in middleware. The integrated ticket flow also records model prediction count and retrieval latency. `/health` remains cheap and only checks artifact availability; it does not load models or run inference.

## Limitations

This is still a POC. It uses synthetic historical tickets, local deterministic evaluation, a lightweight deterministic generator for API integration evaluation, simple rule-based complexity routing and POC security rules. It does not provide authentication, persistence, human approval workflows, streaming, production observability, production prompt security, or guaranteed answer correctness.

No API key is required for the integrated local implementation.
