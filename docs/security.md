# Security Layer

Day 9 adds a lightweight deterministic security layer around the existing LangGraph workflow. It is a POC control surface, not a comprehensive security framework.

## Design

The flow is:

```text
user ticket
-> input security check
-> existing Day 8 LangGraph workflow when allowed
-> output security check
-> safe response plus security status
```

The implementation lives in `backend/app/security/security_service.py`. It does not rebuild retrieval, RAG, classification, or the agent graph. Normal tickets continue through the existing workflow.

## Checks

Input checks block obvious prompt injection, jailbreak phrasing, and direct requests for secrets such as API keys, passwords, access tokens, credentials, and secret keys. Explicit email, phone, and card-like patterns are detected and redacted before workflow execution.

Retrieved tickets are treated as untrusted data. Instruction-like text in retrieved ticket text or answers is recorded as untrusted retrieved content and is not allowed to override workflow instructions.

Output checks block obvious generated secrets and PII leakage. Unsafe output is replaced with a safe manual-review fallback.

## Evaluation

The fixed Day 9 evaluation set covers normal payment failure, prompt injection, jailbreak, secret request, PII detection/redaction, malicious retrieved content, and insufficient information. Metrics report pass rates for the security test set, attack detection, normal-ticket allow behavior, and malicious retrieved-content handling. They are not called security accuracy.

No API key is required. The implementation uses Python standard-library regex checks.

## Limitations

The layer uses deterministic string and regex rules, so it can miss paraphrased attacks and may flag benign text that resembles a rule. PII detection is limited to obvious explicit patterns. It does not provide OAuth, RBAC, WAF behavior, malware scanning, SIEM integration, persistent audit storage, or formal penetration-test coverage.
