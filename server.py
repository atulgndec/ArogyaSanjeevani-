"""FastAPI server. Mirrors server.ts (+ the claim, decision and lesson routes the React app called in-process)."""
import asyncio
import json
import os
from pathlib import Path

from fastapi import Depends, FastAPI, File, HTTPException, Response, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from api_console import adj as console_adj, public as policy_api
from api_360 import adj_api, adjudicator, hospital_api, patient_api
from arogya.data.synthetic_claims import all_claims, get_claim, load_claims
from arogya.lib import agent_orchestrator as orch, llm
from arogya.lib.adjudication_calculator import calculate_worksheet
from arogya.lib.claims_metrics import LESSONS, summarize
from arogya.lib.document_extractor import extract_claim
from arogya.lib.document_ocr import DocumentReadError, read_document
from arogya.lib.fraud_detector import detect_fraud
from arogya.lib.reflexion_engine import reflexion_engine
from arogya.lib.uploaded_claims import register as register_uploaded_claim
from arogya.types.adjudication import RawDocuments

import threading
from decimal import Decimal
from typing import Literal

from arogya.lib import vector_db

app = FastAPI(title="Arogya Sanjeevani claims adjudication")
llm.configure_dspy()   # no-op unless Ollama is running
threading.Thread(target=vector_db.stats, daemon=True).start()      # build or open the vector database in the background


class LineEdit(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    line_id: str = Field(min_length=1, max_length=100)
    deduction_amount: Decimal = Field(ge=0, le=100_000_000)
    comment: str = Field(default="", max_length=2000)


class Decision(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    action: Literal["approve", "edit", "reject"] = "approve"
    approver: str = Field(min_length=1, max_length=120)
    level: Literal["junior", "senior"] = "junior"
    edits: list[LineEdit] = Field(default_factory=list, max_length=100)
    reason: str = Field(default="", max_length=2000)

    @field_validator("approver")
    @classmethod
    def approver_must_not_be_blank(cls, value: str) -> str:
        if not value:
            raise ValueError("approver must not be blank")
        return value

    @model_validator(mode="after")
    def action_payload_is_complete(self):
        if self.action == "edit" and not self.edits:
            raise ValueError("at least one line edit is required")
        if self.action == "reject" and not self.reason:
            raise ValueError("a rejection reason is required")
        return self


class Review(BaseModel):
    reviewer: str
    note: str = ""


def claim_or_404(cid: str):
    c = get_claim(cid)
    if c is None:
        raise HTTPException(404, "unknown claim")
    return c


@app.get("/healthz")
def health():
    return {"ok": True}


@app.get("/api/claims")
def claims():
    return [{"claimId": c.claim_id, "diagnosis": c.diagnosis, "claimedTotal": c.claimed_total} for c in all_claims()]


@app.get("/api/claims/{cid}")
def claim(cid: str):
    return claim_or_404(cid).model_dump(mode="json", by_alias=True, exclude={"raw_documents"})


@app.post("/api/claims/{cid}/run")
def run(cid: str):
    claim_or_404(cid)
    return orch.run_claim(cid)


@app.post("/api/claims/{cid}/stream")
async def stream_claim(cid: str):
    claim_or_404(cid)

    async def events():
        yield "event: status\ndata: {\"status\":\"started\"}\n\n"
        await asyncio.sleep(0)
        snapshot = await asyncio.to_thread(orch.run_claim, cid)
        for step in snapshot.get("steps", []):
            yield "event: step\ndata: " + json.dumps(step, default=str) + "\n\n"
            await asyncio.sleep(0.05)
        yield "event: done\ndata: " + json.dumps(snapshot, default=str) + "\n\n"

    return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/api/claims/{cid}/state")
def state(cid: str):
    return orch.snapshot(cid)


@app.post("/api/claims/{cid}/decision", dependencies=[Depends(adjudicator)])
def decision(cid: str, d: Decision):
    try:
        return orch.decide(cid, d.model_dump())
    except ValueError as e:
        raise HTTPException(409, str(e))


@app.post("/api/llm/settlement-report")
def settlement_report(body: dict):
    try:
        return orch.settlement_letter(body["claimId"])
    except (KeyError, ValueError) as e:
        raise HTTPException(400, str(e))


@app.post("/api/extract-document")
def extract_document(docs: dict):
    claim, warnings = extract_claim(docs)
    return {"claim": claim.model_dump(mode="json", by_alias=True, exclude={"raw_documents"}), "warnings": warnings}


@app.post("/api/claims/upload", dependencies=[Depends(adjudicator)])
async def upload_claim_packet(
    claim_form: UploadFile = File(...),
    discharge_summary: UploadFile = File(...),
    itemised_bill: UploadFile = File(...),
    policy_schedule: UploadFile = File(...),
):
    packet = {
        "claim_form_text": claim_form,
        "discharge_summary_text": discharge_summary,
        "itemised_bill_text": itemised_bill,
        "policy_schedule_text": policy_schedule,
    }
    texts, methods = {}, {}
    try:
        for field_name, upload in packet.items():
            contents = await upload.read(10 * 1024 * 1024 + 1)
            text, method = await asyncio.to_thread(read_document, upload.filename or field_name, contents)
            texts[field_name] = text
            methods[field_name.removesuffix("_text")] = method
        claim, warnings = extract_claim(RawDocuments.model_validate(texts))
    except DocumentReadError as exc:
        raise HTTPException(400, str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(503, str(exc)) from exc
    except (ValueError, TypeError) as exc:
        raise HTTPException(422, f"Could not extract a valid claim from this packet: {exc}") from exc
    if not claim.claim_id.strip() or not claim.line_items:
        raise HTTPException(422, "The packet must contain a claim ID and at least one readable itemised bill line.")
    if any(existing.claim_id == claim.claim_id for existing in all_claims()):
        raise HTTPException(409, f"a claim with ID {claim.claim_id} already exists")
    try:
        register_uploaded_claim(claim)
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
    return {
        "claim": claim.model_dump(mode="json", by_alias=True, exclude={"raw_documents"}),
        "warnings": warnings,
        "documentMethods": methods,
        "retention": (
            "Extracted claim data and OCR text are stored in the configured SQLite database; "
            "original uploaded files are not retained."
            if os.getenv("UPLOADED_CLAIMS_DB")
            else "Extracted claim data and OCR text are held in API memory until the process restarts; "
            "original uploaded files are not retained."
        ),
    }


@app.get("/api/lessons")
def lessons():
    return reflexion_engine.stats()


@app.post("/api/lessons/{lid}/{action}", dependencies=[Depends(adjudicator)])
def review_lesson(lid: str, action: str, r: Review):
    try:
        return reflexion_engine.review(lid, action, r.reviewer, r.note)
    except (KeyError, ValueError) as e:
        raise HTTPException(400, str(e))


@app.get("/api/metrics/summary")
def metrics_summary():
    cs = load_claims()
    wss = [calculate_worksheet(c) for c in cs]
    return summarize(cs, wss, [detect_fraud(c) for c in cs])


@app.get("/metrics")
def metrics():
    for status in ("pending_review", "approved", "rejected", "withdrawn"):
        LESSONS.labels(status).set(sum(l.status == status for l in reflexion_engine.all()))
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


for r in (patient_api, hospital_api, adj_api, console_adj, policy_api):
    app.include_router(r)
STATIC = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC / "index.html")
