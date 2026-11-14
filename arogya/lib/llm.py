"""Model calls. Replaces the three /api/gemini/* handlers in server.ts with LangChain chains.

* Google Gemini (default gemini-3.6-flash; set GEMINI_MODEL and GEMINI_API_KEY). If no key is set, or a call fails
  (quota, network, bad key), every function returns its deterministic fallback, so the pipeline never depends on a model.
* LangSmith tracing is on when LANGSMITH_TRACING=true; inputs and outputs are masked before they leave.
* Claim data goes to the model inside <claim_data> tags with a system rule that it is data, never instructions.
"""
from __future__ import annotations

import json
import logging
import os
import time

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langsmith import traceable

from pathlib import Path

from .claims_metrics import LLM_CALLS
from .privacy import mask

log = logging.getLogger(__name__)

_CONFIG = json.loads((Path(__file__).parent.parent / "config.json").read_text())
SYSTEM = _CONFIG["prompts"]["system"]
ANALYSIS_INSTRUCTIONS = _CONFIG["prompts"]["analysis"]
SETTLEMENT_INSTRUCTIONS = _CONFIG["prompts"]["settlement_report"]

ANALYZE = ChatPromptTemplate.from_messages([
    ("system", SYSTEM),
    ("human", "Case triggers:\n{triggers}\n\nRelevant clauses:\n{clauses}\n\nApproved playbook lessons:\n{lessons}\n\n<claim_data>\n{claim_data}\n</claim_data>\n\n"
              + ANALYSIS_INSTRUCTIONS),
])
REPORT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM + "\n" + SETTLEMENT_INSTRUCTIONS),
    ("human", "<claim_data>\n{claim_data}\n</claim_data>\n\nWorksheet:\n{worksheet}"),
])


GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
PAUSE_SECONDS = int(os.getenv("LLM_PAUSE_SECONDS", "60"))     # after a failed call, skip Gemini for this long (protects the quota)
_down_until = 0.0

# Backward-compatible names, in case other modules still import them from the Ollama version of this file.
OLLAMA_MODEL = GEMINI_MODEL


def _api_key() -> str:
    """Read the key at call time so a key added to the environment later is picked up."""
    return os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or ""


def _gemini_ready() -> bool:
    """True when a key is configured. No network call is made here; a bad key or an outage is handled by the fallback in _run()."""
    return bool(_api_key())


_ollama_up = _gemini_ready          # backward-compatible alias


def get_model():
    if not _gemini_ready():
        return None
    from langchain_google_genai import ChatGoogleGenerativeAI
    return ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        google_api_key=_api_key(),
        # No temperature/top_p/top_k: Google deprecated them for Gemini 3.6 and later.
        # Gemini 3.x models "think" first and thinking tokens count against this limit, so it is well above the 600 used for Ollama.
        max_output_tokens=2048,
        timeout=30,
        max_retries=1,
    )


def configure_dspy() -> bool:
    """Point DSPy at the same Gemini model (used by the line-classifier fallback)."""
    if not _gemini_ready():
        return False
    import dspy
    dspy.configure(lm=dspy.LM(f"gemini/{GEMINI_MODEL}", api_key=_api_key()))
    return True


def _hide(d):          # LangSmith input/output processor: mask health details
    return json.loads(mask(json.dumps(d, default=str)))


def _reason(e: Exception) -> str:
    """Short, readable cause for the result label. The full error goes to the log."""
    m = str(e)
    if "429" in m or "RESOURCE_EXHAUSTED" in m:
        return "quota or rate limit"
    if "503" in m or "UNAVAILABLE" in m:
        return "service busy"
    if "403" in m or "API key" in m or "PERMISSION_DENIED" in m:
        return "key or permission"
    if "404" in m or "not found" in m.lower():
        return "model not found"
    return type(e).__name__


def _run(prompt, variables: dict, fallback: str) -> dict:
    global _down_until
    model = get_model()
    if model is None or time.time() < _down_until:           # no key, or paused after a recent failure
        LLM_CALLS.labels("fallback").inc()
        return {"text": fallback, "model": "deterministic-rules", "is_ai": False}
    try:
        text = (prompt | model | StrOutputParser()).invoke(variables)
        LLM_CALLS.labels("live").inc()
        return {"text": text, "model": GEMINI_MODEL, "is_ai": True}
    except Exception as e:                                   # network, quota, bad key: never block the claim
        _down_until = time.time() + PAUSE_SECONDS
        log.warning("Gemini call failed (%s): %s", type(e).__name__, str(e)[:500])      # full reason goes to the logs
        LLM_CALLS.labels("fallback").inc()
        return {"text": fallback, "model": f"deterministic-rules (model call failed: {_reason(e)})", "is_ai": False}


@traceable(name="llm_analyze_case", process_inputs=_hide, process_outputs=_hide)
def analyze_case(*, triggers: list[str], clauses: str, lessons: str, claim_data: dict, fallback: str) -> dict:
    return _run(ANALYZE, {"triggers": "\n".join(triggers), "clauses": clauses, "lessons": lessons or "none",
                          "claim_data": json.dumps(claim_data, default=str)}, fallback)


@traceable(name="llm_settlement_report", process_inputs=_hide, process_outputs=_hide)
def settlement_report(*, claim_data: dict, worksheet: dict, fallback: str) -> dict:
    return _run(REPORT, {"claim_data": json.dumps(claim_data, default=str), "worksheet": json.dumps(worksheet, default=str)}, fallback)