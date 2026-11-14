"""Thin layer over the graph. Mirrors src/lib/agentOrchestrator.ts: run a claim, resume it, write the letter."""
from __future__ import annotations

import os
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from ..data.synthetic_claims import get_claim
from . import llm
from .claims_graph import build_graph
from .claims_metrics import PENDING


def _create_graph():
    checkpoint_path = os.getenv("LANGGRAPH_CHECKPOINT_DB")
    if not checkpoint_path:
        return build_graph(), None
    database = Path(checkpoint_path)
    database.parent.mkdir(parents=True, exist_ok=True)
    context = SqliteSaver.from_conn_string(str(database))
    return build_graph(checkpointer=context.__enter__()), context


graph, _checkpoint_context = _create_graph()


def _cfg(claim_id: str) -> dict:
    return {"configurable": {"thread_id": claim_id}}


def snapshot(claim_id: str) -> dict:
    st = graph.get_state(_cfg(claim_id))
    v = st.values
    if not v:
        return {"status": "not_started"}
    return {"status": "waiting_for_adjudicator" if st.next else "finished", "next": list(st.next), "outcome": v.get("outcome", "pending"),
            "worksheet": v["ws"].model_dump(mode="json", by_alias=True) if v.get("ws") else None,
            "fraud": v["fraud"].model_dump(mode="json", by_alias=True) if v.get("fraud") else None,
            "agents": v.get("specialist_reviews", {}), "steps": v.get("steps", []), "plan": v.get("plan"),
            "settlement": v.get("settlement")}


def _refresh_pending() -> None:
    PENDING.set(sum(1 for t in _threads if graph.get_state(_cfg(t)).next))


_threads: set[str] = set()


def run_claim(claim_id: str, auto_approve: dict | None = None) -> dict:
    claim = get_claim(claim_id)
    _threads.add(claim_id)
    graph.invoke({"claim": claim, "auto_approve": auto_approve}, _cfg(claim_id))
    _refresh_pending()
    return snapshot(claim_id)


def decide(claim_id: str, decision: dict) -> dict:
    """decision: {action: approve|edit|reject, approver, level, edits?, reason?}"""
    if not graph.get_state(_cfg(claim_id)).next:
        raise ValueError("claim is not waiting for a decision")
    graph.invoke(Command(resume=decision), _cfg(claim_id))
    _refresh_pending()
    return snapshot(claim_id)


def settlement_letter(claim_id: str) -> dict:
    v = graph.get_state(_cfg(claim_id)).values
    if not v or not v.get("ws"):
        raise ValueError("run the claim first")
    ws, c = v["ws"], v["claim"]
    citations = (ws.llm_adjudication or {}).get("citations", [])
    citation_text = "\n\nSources: " + ", ".join(
        f"[{citation['passageId']}] {citation['clause']}" for citation in citations
    ) if citations else ""
    fallback = (f"# Settlement advice: {c.claim_id}\n\nClaimed ₹{c.claimed_total}; deductions ₹{ws.total_deductions}; co-pay ₹{ws.copay_amount}; "
                f"**payable ₹{ws.final_payable_amount}**.\n\n" + "\n".join(f"- {l.raw_description}: ₹{l.deduction_amount} ({l.clause_cited})" for l in ws.lines if l.deduction_amount > 0)
                + "\n\nIf you disagree, write to the insurer's grievance officer, then the Insurance Ombudsman." + citation_text)
    return llm.settlement_report(claim_data={"claimId": c.claim_id, "diagnosis": c.diagnosis, "procedure": c.procedure_name},
                                 worksheet={"payable": str(ws.final_payable_amount),
                                            "deductions": [(l.raw_description, str(l.deduction_amount), l.clause_cited) for l in ws.lines if l.deduction_amount > 0],
                                            "citations": citations}, fallback=fallback)
