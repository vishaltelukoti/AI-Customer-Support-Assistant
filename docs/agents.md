# Day 8: LangGraph multi-agent workflow

## Scope

Day 8 adds a lightweight LangGraph workflow for complex support-ticket handling. It does not add FastAPI integration, persistent memory, agent memory stores, production orchestration, or Day 9 security controls.

## Architecture

The workflow has three specialized agents plus deterministic routing:

- Retrieval Agent: calls the existing Day 6 FAISS retrieval service through a retrieval tool.
- Investigation Agent: runs only for complex tickets and summarizes issue context.
- Resolution Agent: reuses the Day 7 RAG service to generate a grounded response from retrieved evidence.

Routing is deterministic and POC-scoped:

```text
simple ticket  -> Retrieval Agent -> Resolution Agent
complex ticket -> Investigation Agent -> Retrieval Agent -> Resolution Agent
```

Complexity is based on lightweight keyword/length heuristics. No new classifier is trained.

## Shared State

The LangGraph state keeps:

- ticket text
- complexity
- retrieved sources
- retrieved tickets
- investigation result
- final response
- workflow trace
- status/error fields
- in-execution memory

Memory is request-local only. No database or persistent memory is used.

## Safety And Failure Handling

Retrieved ticket text and answers are treated as data. The workflow does not execute instructions found inside retrieved content. If retrieval fails, the workflow returns a safe insufficient-information fallback. If generation fails, the Resolution Agent also returns a safe fallback instead of fabricating a support answer.

## Evaluation

The fixed POC evaluation contains:

- simple payment failure
- complex refund/cancellation case
- insufficient-information case
- prompt-injection-style ticket

Measured results:

| Metric | Value |
| --- | ---: |
| Routing accuracy | 1.000000 |
| Workflow completion rate | 1.000000 |
| Source behavior rate | 1.000000 |
| Trace presence rate | 1.000000 |

Artifacts are saved in `experiments/agents/`:

- `agent_config.json`
- `agent_evaluation.json`
- `agent_evaluation.md`

## Limitations

The evaluation is small and demonstrates workflow mechanics, not real-world accuracy. Routing is heuristic. The default evaluation can use a deterministic local evaluation generator; the workflow can also load the existing local `google/flan-t5-base` generation path. No production security, human approval flow, monitoring, persistence, or API integration is included.
