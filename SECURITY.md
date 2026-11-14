# Security threat model

## Scope and assets

This is a synthetic-data capstone demonstration, not a production claims system. Protected assets are submitted claim documents, patient and policy identifiers, adjudication worksheets, reviewer decisions, approval tokens, and persistent workflow/audit databases.

## Trust boundaries and threats

| Boundary | Threat | Current controls |
|---|---|---|
| Hospital or patient form → API | Malformed dates, amounts, identifiers, inconsistent bills, or oversized edits | Pydantic constraints, cross-field checks, browser constraints, and workflow-level edit validation |
| Uploaded claim text → model workflow | Prompt injection or fabricated authority instructions | Document text is treated as data, not instructions; raw documents are excluded from model prompts; deterministic attack screening; settlement remains a separate approval-gated action |
| Retrieval result → reviewer/model | Unsupported policy statements or stale citations | Hybrid retrieval returns passage IDs, source file, unknown source date when unavailable, and ingestion timestamp; model prompts restrict citations to retrieved passages |
| Reviewer → settlement | Forged, stale, or misattributed approval | Named reviewer and role checks; approval token bound to the worksheet hash; edits invalidate the earlier approval; settlement checks caller and token |
| MCP caller → project files | Path traversal or reading dotfiles | Filesystem MCP is read-only, root-confined, rejects hidden paths, and limits file size and listing count |
| MCP caller → audit store | Unauthorized or excessive review-note writes | Audit MCP exposes only typed note read/write tools and stores data in a separate persistent SQLite volume |
| Container processes → Chroma | Concurrent embedded clients deleting each other's collections | API and retrieval MCP use separate Chroma data directories |

## Red-team regression evidence

`python red_team_eval.py` runs 30 synthetic prompt-attack strings and 30 benign clinical notes. The current pattern-based detector flags 30/30 attack fixtures and 0/30 benign fixtures; disabling that defense yields 0/30 detections. These fixtures measure regression behavior only and are not an estimate of real-world recall or false-positive rates.

## Residual risks before real use

- The demo API key defaults to `demo-key`, reviewer levels are self-declared, and the exposed MCP endpoints do not authenticate callers. Do not expose the Compose ports to an untrusted network.
- The Compose stack does not provide production TLS, identity federation/OTP, encrypted database volumes, key rotation, or a tested backup/restore procedure.
- The default Ollama model store may have no weights installed; model-provider access, prompt-injection behavior, and tracing integrations require separate validation.
- Synthetic data, locally authored evaluation fixtures, and the 30-case security set cannot replace an independent security review, real human adjudication labels, or a privacy/legal assessment.
- Tariff values are illustrative and policy source effective dates remain unknown where the source does not supply them.

## Safe operating guidance

Keep the service on a trusted local network, replace demo credentials before any shared deployment, use synthetic or explicitly approved data only, review every settlement decision, and obtain security/privacy approval before connecting real claim records.
