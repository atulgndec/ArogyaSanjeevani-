# Arogya Sanjeevani claims adjudication (Python)

A simple Python port of the AI Studio workstation (React + Express). It keeps the original's module layout, so you can read the two side by side.

- Source code: https://github.com/suhanijain/arogya-adjudicator-1
- Live deployment (Render): https://arogya-adjudicator.onrender.com/ (API docs at `/docs`)

| TypeScript | Python | Notes |
|---|---|---|
| `src/types/adjudication.ts` | `arogya/types/adjudication.py` | pydantic v2, same camelCase JSON |
| `lib/adjudicationCalculator.ts` | `lib/adjudication_calculator.py` | Decimal, whole-rupee rounding like `Math.round` |
| `lib/langgraph.ts` (hand-made) | `lib/claims_graph.py` | real **LangGraph**, `interrupt()` for the adjudicator |
| `lib/vectorDb.ts` (fake embeddings) | `lib/vector_db.py` and `policy_kb.py` | BM25 + vector RRF, semantic query expansion, reranking, and Chroma |
| `/api/gemini/*` in `server.ts` | `lib/llm.py` | LangChain chains on **Google Gemini** (`gemini-3.6-flash`), **LangSmith** `@traceable` with masking |
| `lib/documentExtractor` | `lib/document_extractor.py` | rules first, **DSPy** `Predict` only for lines the rules can't place |
| `lib/claimsMetrics.ts` | `lib/claims_metrics.py` | **Prometheus** metrics, **Grafana** dashboard in `ops/` |
| `lib/fraudDetector.ts`, `reflexionEngine.ts`, `claimsDb.ts`, `tariffDb.ts` | same names in snake_case | |
| `server.ts` | `server.py` (FastAPI) | |

## Run
```
pip install -r requirements.txt
make test            # backend, API, retrieval, MCP, validation, and evaluation tests
make serve           # app at http://localhost:8000 , API docs at /docs
make eval            # DSPy Evaluate on the extractor (80/80)
docker compose up --build -d  # API, three FastMCP servers, Chroma, Prometheus, Grafana
```

### Model: Google Gemini
Model calls use **Google Gemini** (`gemini-3.6-flash` by default) through the Gemini API. LangChain uses `ChatGoogleGenerativeAI` (package `langchain-google-genai`, pinned in `requirements.txt`); DSPy uses the same model through `llm.configure_dspy()` (`dspy.LM("gemini/<model>")`). Get a key from [Google AI Studio](https://aistudio.google.com/app/apikey) and put it in a `.env` file in the project root. Never commit this file or copy it into an image (list `.env` in both `.gitignore` and `.dockerignore`):

```
GEMINI_API_KEY=your-key-here
GEMINI_MODEL=gemini-3.6-flash
ADJUDICATOR_KEY=change-me
CLAIMS_APPROVAL_SECRET=<long random value, for example from: openssl rand -hex 32>
```

| Variable | Default | Purpose |
|---|---|---|
| `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) | none | Gemini API key. Without it, the model steps use deterministic text |
| `GEMINI_MODEL` | `gemini-3.6-flash` | Model for case analysis and settlement notes |
| `LLM_PAUSE_SECONDS` | `60` | After a failed Gemini call, skip Gemini for this long so a quota problem is not made worse |
| `ADJUDICATOR_KEY` | `demo-key` | Adjudicator access key. Set your own |
| `CLAIMS_APPROVAL_SECRET` | none | Signs approval tokens. Use a long random value and keep it out of Git |
| `LANGSMITH_TRACING` | off | Set to `true` with a LangSmith key to trace; health details are masked before they leave |

Running locally, export the variables before starting the server, because Python does not read `.env` by itself (or load it with `python-dotenv`):
```
set -a; source .env; set +a
make serve
```

In Docker Compose the file is passed to the containers with `env_file: [.env]`. Add that line to every service that calls Gemini (`api` and `mcp-retrieval`); `GEMINI_MODEL` in `environment:` alone does **not** supply the key. Compose only reads `.env` at container creation, so after changing it, recreate the services (a plain `restart` does not reload it):
```
docker compose up -d --force-recreate api mcp-retrieval
```

Notes on the Gemini model path:
- **Fallback:** if there is no key, the package is missing, or a call fails (quota, network, bad key), the model steps use deterministic text and everything else works. The result then shows `deterministic-rules`, with a reason such as `(model call failed: quota or rate limit)`; the full error is written to the application log as `Gemini call failed`.
- **Parameters:** `lib/llm.py` sets no `temperature`, `top_p` or `top_k`, because Google has deprecated them for Gemini 3.6 and later. `max_output_tokens` is 2048 because Gemini 3.x models spend part of the limit on internal reasoning.
- **Quota:** free-tier keys have low rate limits. A `429 RESOURCE_EXHAUSTED` error means the quota was used up; see Google's [rate limits](https://aistudio.google.com/docs/rate-limits) page, or enable billing.
- **Data sent to Google:** the structured claim fields and retrieved policy clauses go to the Gemini model, over the internet. Containers therefore need outbound access to Google's API. Check this against your data-protection obligations before using real patient data.
- **Changing the model:** set `GEMINI_MODEL` (for example to `gemini-3.8-flash`); no code change is needed.
- **Verify:** process a claim; the analysis should show `"model": "gemini-3.6-flash"` and `"is_ai": true`. To test the model path on its own:
  ```
  python - <<'EOF'
  from arogya.lib import llm
  r = llm.analyze_case(triggers=["test"], clauses="none", lessons="", claim_data={"x": 1}, fallback="FALLBACK")
  print(r["model"], r["is_ai"])
  EOF
  ```

### Running the tests
The tests are meant to run offline and deterministically, so run them without a Gemini key (unset `GEMINI_API_KEY` and `GOOGLE_API_KEY`, or have `tests/conftest.py` clear them); otherwise they would call Gemini and use quota. Run the integration checks with the pinned container dependencies. The empty `UPLOADED_CLAIMS_DB` override keeps tests from writing fixture claims into the persistent volume, and the empty `GEMINI_API_KEY` override keeps them from calling Gemini when the service loads `.env`:
```
docker compose run --build --no-deps --rm -e UPLOADED_CLAIMS_DB= -e GEMINI_API_KEY= -e GOOGLE_API_KEY= api pytest -q
docker compose exec -T api python eval_retrieval.py
docker compose exec -T api python eval_adjudication.py
docker compose exec -T api python red_team_eval.py
```
Open `http://localhost:8000` for the UI, `http://localhost:8000/docs` for the API, and `http://localhost:3000` for Grafana. Use the example packet in `examples/upload_packet/` to smoke-test file upload; leave the claim at the human-approval step unless an authorized reviewer is making a real decision.

## Three views (open `/`)
- **Patient**: cover left, ongoing treatment (found from history), standing pre-approval and sessions left, request status in plain words, recent treatments and payments.
- **Network hospital**: request approval for a session (answered in seconds when it fits the patient's pattern), see its own requests and a billing scorecard.
- **Adjudicator** (name, level and key in the top bar): **Approval queue**, **Patient 360**, **Network 360**.
  - *Patient 360*: payments to date, regular treatments, a timeline by hospital with out-of-pattern sessions ringed, standing approvals (revocable), and why each past or open session differs.
  - *Network 360*: each hospital's billing vs benchmark, claim flag rate, repudiation rate, share of patients' sessions that break their own pattern, monthly volume.

How it speeds up regular treatment (dialysis, chemotherapy, physiotherapy): the first request for a patient with a steady pattern gets a suggested **standing authorisation** (hospitals, per-session cap, number of sessions, end date). One adjudicator approval covers the coming sessions; each one is still checked, and clears in seconds if it matches. A session that differs is **held** for a person and never rejected automatically. Checks (`lib/patient_360.py`): new contact (someone else may be using the account), new hospital, other city, amount above 1.25x the patient's usual, too soon after the last session, price above tariff ceiling, cover left. Approving a held request needs a senior and a note on how the patient was verified on the registered number. Patients and hospitals see only a status, never the reasons.

Data: `scripts/make_history.py` builds 12 synthetic patients (368 past sessions) and 12 pending requests. Each pattern break is planted on purpose (PA-1002 other city, PA-1003 amount, PA-1004 new phone, PA-1005 same-day repeat) and Sterling Hospital bills dialysis at about 2x benchmark. "Today" is fixed at 2026-10-26 so results are reproducible.

## Claim pre-approval for regular treatment (fast-track)
Hospital: **Pre-approve a regular treatment** → assessed from the hospital's contracted package and room rates in the linked provider database (`data/provider_rates.json`, built by `scripts/make_provider_rates.py`; a live system would sync each provider's rate file) and **guaranteed at once** after the 5% co-payment. After treatment: **Send bill**.
Eligibility (all must pass; each check and its reason is shown to adjudicators, never to patients or hospitals): a regular treatment pattern; the patient's usual network hospital with no pricing or claim-flag review; no past out-of-pattern session; no medium or high fraud flag on the patient's claim record; no open hold or rejected request; session timing, phone and price match the pattern; contracted room rate within the policy cap; cover left at least 1.2x the assessed amount and policy valid. Anyone else gets standard review.
Reconciliation uses only the itemised bill and the discharge summary, and only moves the amount:
- room days limited to those the discharge summary documents (extra documented days move it **up**, padded days **down**);
- room rate limited to the hospital's contracted rate (down);
- non-payable items removed (down, Clause 4.7); a bill below the package moves it down; up to 10% above the package is allowed;
- 5% co-payment, and never more than cover left.
It settles automatically only when nothing else is wrong. These go to an adjudicator with the proposed amount (a senior must approve, with a note): line arithmetic or total errors, duplicate bill number, admission far from the planned date, discharge summary dates not matching its room days, room rate above the policy cap (a proportional cut would be needed), bill more than 10% above the package, or payable more than 25% above the pre-approved amount. Nothing is rejected automatically. A settled session becomes history for the next check, so a repeat on the same day is held.
Automatic approval within these limits is a policy decision (`fast-track-policy-v1` is recorded as the approver): your compliance owner should sign off the limits before real use.

## Vector database and semantic search (`lib/vector_db.py`)
Policy clauses (`data/policy_clauses`) plus a plain-language glossary (`data/glossary.json`: how people phrase things, e.g. "stretcher van" for ambulance) are embedded and stored in a persistent **Chroma** collection (`.vectordb/`, cosine distance, via `langchain-chroma`). The collection name carries a fingerprint of the content and embedder, so changed clauses or a new embedder rebuild automatically (about 8 s at first start; a warm-up thread runs at server start).
- **Embeddings**: Ollama `nomic-embed-text` (`ollama pull nomic-embed-text`) when an Ollama server is reachable; otherwise an offline fallback, LSA (TF-IDF over words and character n-grams, reduced with TruncatedSVD, L2-normalised). The UI sidebar shows which one is active. The Compose stack has no Ollama service, so it uses LSA unless `OLLAMA_BASE_URL` points at a server (see "Embeddings stay on Ollama or LSA" above). This part does not use Gemini.
- **Retrieval** (`hybrid_clauses`, used by the claims graph): BM25 and vector search are merged by weighted reciprocal rank fusion (BM25 weight 1, vector weight 2). A LangChain `BaseRetriever` wraps it.
- **Evaluate**: `python eval_retrieval.py` (a development set used to pick the weights, and a held-out set written afterwards).

The query-side glossary expansion and deterministic reranker are compared with the original BM25+vector RRF baseline. Offline LSA results (Recall@5 / MRR):
| set | baseline RRF | rewritten + reranked |
|---|---:|---:|
| development (24 queries) | 0.96 / 0.86 | 1.00 / 0.86 |
| held-out (12 queries) | 0.83 / 0.69 | 1.00 / 0.75 |

This is a small, locally authored benchmark, not external validation; the held-out set contains 12 questions. The CI gate runs on this held-out set. Neural-embedding behavior should be re-evaluated in an environment that downloads and runs the configured Ollama embedding model.

## Console (Stitch-style UI)
`static/` is a dependency-free SPA in the dark "Clinical Adjudication Engine" style from the supplied Stitch export (tokens, inline icons). Adjudicator tabs: Claims worklist, Adjudication bench (ledger, clauses, pipeline trace, retrieval), Session queue, Patient 360, Network 360, Fast-track, Rules, Audit; plus Patient and Network hospital views; layout works down to 390 px. Fonts load from Google Fonts when online and fall back to system fonts. The bench supports bounded line edits and resubmission through the approval workflow. Checked in Chromium only.

### Uploading a claim packet
From the adjudicator's **Claims worklist**, upload one file in each required field: claim form, discharge summary, itemised bill, and policy schedule. The upload accepts searchable/scanned PDFs, PNG/JPG/TIFF images, and UTF-8 text files. Scanned pages are OCR-processed in English with Tesseract; searchable PDFs and text files are read directly. Each document is limited to 10 MB and PDFs to 30 pages. The extracted claim appears in the existing worklist and adjudication bench, so it can enter the same LangGraph workflow and approval gates as other claims. Review the extracted values and warnings before running adjudication.

An example four-file packet is in `examples/upload_packet/`; it can be uploaded directly as text for a smoke test. Change its `Claim ID` in the claim form before uploading it again. In Docker Compose, extracted claim fields and OCR text are stored in SQLite at `/data/workflow/uploaded_claims.sqlite3` on the persistent `workflow-data` volume, so uploaded claims remain available after an API restart; original uploaded files are not retained. Local runs can set `UPLOADED_CLAIMS_DB` to enable the same persistence. This database contains sensitive claim details and must be protected and retained/deleted according to an approved privacy policy. Docker installs Tesseract and its English language data; local runs need Tesseract installed separately. The OCR model and claim parser are English/template-oriented, so this is a demonstration intake path, not production document recognition; verify all fields and bill lines manually.

## Flow
The LangGraph supervisor emits an explicit plan and dispatches three independent specialist agents in parallel: extraction reconciles bill lines against the claim total, fraud screens the claim, and coverage calculates the deterministic worksheet. A join step applies the fraud-based senior-approval gate before adjudication. The workflow then runs the model note (for complex cases, using Gemini), critic (one bounded coverage replan if needed), human approval (pauses), settlement, and memory. Compose persists graph checkpoints and reviewed Reflexion lessons in the `workflow-data` volume so a paused claim can resume after an API restart.
`POST /api/claims/{id}/run`, then `POST /api/claims/{id}/decision` with `{action: approve|edit|reject, approver, level}`. Server-Sent Events are available at `POST /api/claims/{id}/stream`.

## Docker stack and quality gates
`docker compose up --build` starts FastAPI, the Chroma vector store, three FastMCP services, Prometheus, and Grafana. The language model is the hosted Gemini API, so the `api` and `mcp-retrieval` containers need `GEMINI_API_KEY` (through `env_file`) and outbound internet access. Prometheus loads `ops/prometheus-alerts.yml`; `ArogyaAPIDown` fires after the API target is unavailable for one minute. A step-by-step build, run and troubleshooting guide is in `README-docker.md`; on Render, set the same variables in the service's **Environment** tab, because Render does not read a `.env` file from the repository.

Prompts and the embedding model ID are centralized in `arogya/config.json`; the chat model comes from `GEMINI_MODEL` (default `gemini-3.6-flash`), so the chat entry under `models` in that file is no longer used by `lib/llm.py`. GitHub Actions runs the tests and `python eval_retrieval.py`; held-out hybrid Recall@5 must meet `RETRIEVAL_MIN_RECALL_AT_5=0.80` and MRR must meet `RETRIEVAL_MIN_MRR=0.65` or CI fails. CI must not have a Gemini key set.
CI also runs `python eval_adjudication.py` over the 80 synthetic claims and `python red_team_eval.py` over 30 synthetic attacks plus 30 benign notes. The current detector catches all 30 listed attacks, with 0/30 benign notes flagged and 0/30 attack cases flagged when the security defense is disabled. These are regression fixtures, not a real-world false-positive estimate.

### FastMCP services
- Retrieval (`http://localhost:8100/mcp`): `search_policy` returns passage IDs and source metadata; `get_policy_clause` retrieves a numbered clause.
- Filesystem (`http://localhost:8101/mcp`): `list_project_files` and `read_project_file` are confined to the read-only `/workspace` mount and exclude hidden files (so `.env` is not readable through it).
- Review audit (`http://localhost:8102/mcp`): `record_review_note` and `get_review_notes` write/read notes in the persistent `mcp-audit-data` volume.

The 30 procedure tariff benchmarks each include tier1/tier2/tier3 ranges. They are synthetic illustrative test values, not official provider or insurer prices. Policy source dates are reported as unknown unless the source explicitly supplies one; ingestion timestamps are recorded separately.
See [SECURITY.md](./SECURITY.md) for the threat model, current controls, and deployment risks.

## Deliberate differences from the original
- Waiting-period rules are checked independently (the original skipped one: CLM-004, CLM-044 differ).
- Excluded goods are a low-severity note, not fraud. Tariff outliers are one consolidated rule (27 flags vs the original's 17).
- Hashes are real SHA-256; approval tokens are HMAC-signed and bound to the worksheet hash. An edit voids approval. Senior cases need a senior who is not the junior reviewer.
- Reflexion lessons carry a machine-checked `applies_when` and live fired/edited counters; the original's hard-coded "88.2%" figures are gone.
- Only structured fields and BM25 clauses go to the model, never raw document text.
- Outside those, the calculator matches the TS output exactly (differential test on 80 claims).

## Limits of the new views
Demo access control only: patients are chosen by policy number, hospitals by name, adjudicators by an API key (`ADJUDICATOR_KEY`, default `demo-key`; level is self-declared). Real use needs OTP sign-in for patients and SSO for staff. Thresholds (1.25x, 80% steady rhythm, 4 sessions) are reasonable defaults, not tuned on real data. Tariffs are illustrative, and whether each treatment is payable under Arogya Sanjeevani is assumed, not checked against the policy wording. Hospital claim counts are about 8 each, so claim flag rates are noisy. Fast-track pre-approval requests and the claims ledger are in memory and reset on restart; uploaded claim fields, graph checkpoints, and Reflexion lessons are persisted by Compose. Retrieval and red-team evaluation sets are locally authored and small. No 20+ independent human adjudication labels or judge-agreement result is included; real human labels must be supplied before that metric can be reported. Gemini API quotas and rate limits apply to the model calls; when they are exceeded the app falls back to deterministic text.

## Not ported / not verified
React UI, PDF generator, Java bridge, the TS data generator (replaced by `data/synthetic_claims.json`). Category rules and the English OCR/parser are tuned to the synthetic vocabulary; verify real packets line by line. LangSmith uploads need user credentials. The DSPy call uses LiteLLM's `gemini/` provider, so if it does not recognise the model name, upgrade `dspy` and `litellm`. This is a demo, not production insurance software.

## Verification and remaining capstone evidence
Latest full local verification (2026-11-10, **measured with Ollama before the move to Gemini**): 93 tests passed; the extractor scored 80/80; adjudication invariants passed 80/80; red-team fixtures detected 30/30 attacks with 0/30 benign notes blocked. With `ollama:nomic-embed-text`, held-out hybrid retrieval achieved Recall@5 1.00 and MRR 0.88 on 12 locally authored queries. A live `llama3.1:8b` request succeeded. The rebuilt Compose stack started, and a four-image scanned example packet was OCR'd into nine bill lines with no extraction warnings; its claim reached the expected human-approval pause. A separate upload was confirmed to remain available after an API restart. Grafana is running and provisioned, but you should sign in with your local Grafana credentials to inspect dashboard panels.

Since the switch, a direct call to `llm.analyze_case` returned a live answer from `gemini-3.6-flash` (`is_ai: true`) on a developer machine (2026-11-14). The test suite, the evaluations and the Compose smoke test have not been recorded again against Gemini: re-run them, confirm that a processed claim shows `gemini-3.6-flash` inside the Compose stack, and update this section with the new figures. The Ollama-based numbers above describe the earlier model, not Gemini.

Still needed for a strong capstone submission: a hosted CI run, fresh-machine installation evidence, single-agent and memory-on/off comparisons, Reflexion before/after metrics, and at least 20 independent human adjudication labels with judge-agreement results. These results must be measured and reported; do not infer them from the synthetic regression tests. Production authentication, privacy/security review, and policy-owner approval are needed before any non-demo use.
