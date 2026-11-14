"""Patient, hospital and adjudicator routes. Patients and hospitals see simple statuses; only adjudicators see reasons.
Demo access control: patients are identified by policy number, hospitals by name, adjudicators by an API key
(ADJUDICATOR_KEY, default "demo-key"). A real deployment needs proper sign-in (OTP for patients, SSO for staff)."""
import os
import secrets
from datetime import date
from decimal import Decimal
from typing import Literal

from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from arogya.lib import agent_orchestrator as orch
from arogya.lib import provider_db
from arogya.lib.claim_preapproval import claim_preapprovals as cpa
from arogya.lib.network_360 import network_360, network_overview
from arogya.lib.patient_360 import AS_OF, HOSPITALS, get_patient, patient_360, patients, regular_treatments, remaining_cover
from arogya.lib.preauth import preauth
from arogya.types.adjudication import BillCategory


def adjudicator(x_adjudicator_key: str = Header(default="")):
    if not secrets.compare_digest(x_adjudicator_key, os.getenv("ADJUDICATOR_KEY", "demo-key")):
        raise HTTPException(401, "adjudicator key required")


patient_api, hospital_api = APIRouter(prefix="/api/patient"), APIRouter(prefix="/api/hospital")
adj_api = APIRouter(prefix="/api/adjudicator", dependencies=[Depends(adjudicator)])


def known(policy: str) -> dict:
    p = get_patient(policy)
    if not p:
        raise HTTPException(404, "unknown policy")
    return p


# ---- patient ----------------------------------------------------------------------------------
@patient_api.get("")
def patient_list():
    return [{"policy": p["policy"], "name": p["name"]} for p in patients()]


@patient_api.get("/{policy}")
def patient_home(policy: str):
    p = known(policy)
    eps = sorted(p["episodes"], key=lambda e: e["date"])
    regs = regular_treatments(eps)
    auths = [a for a in preauth.standing.values() if a["policy"] == policy and not a["revoked"]]
    claim = orch.snapshot(p["claimId"])
    return {"name": p["name"], "policy": policy, "validTo": p["policyValidTo"], "registeredContact": p["registeredContactMasked"], "remainingCover": preauth.cover_left(policy),
            "preapprovals": cpa.for_policy(policy),
            "coverTotal": p["sumInsured"] + p["cumulativeBonus"],
            "ongoingTreatments": [{"procedure": r["procedure"], "usualHospital": r["usualHospitals"][0], "everyDays": r["typicalGapDays"], "lastSession": r["lastDate"], "sessions": r["episodes"],
                                   "recentDates": [e["date"] for e in eps if e["procedure"] == r["procedure"]][-16:]} for r in regs],
            "standingApprovals": [{"id": a["id"], "procedure": a["procedure"], "hospitals": a["hospitals"], "sessionsLeft": a["sessionsLeft"], "validTo": a["validTo"]} for a in auths],
            "requests": preauth.for_policy(policy),
            "recent": [{k: e[k] for k in ("date", "hospital", "procedure", "billed", "paid", "status")} for e in eps[-10:][::-1]],
            "claim": {"claimId": p["claimId"], "status": claim["status"], "outcome": claim.get("outcome")}}


# ---- hospital ---------------------------------------------------------------------------------
class SessionRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    policy: str = Field(min_length=1, max_length=80)
    procedure: str = Field(min_length=1, max_length=120)
    serviceDate: str
    amount: int = Field(gt=0, le=10_000_000)
    contactLast4: str = Field(pattern=r"^\d{4}$")

    @field_validator("serviceDate")
    @classmethod
    def valid_service_date(cls, value: str) -> str:
        try:
            date.fromisoformat(value)
        except ValueError as exc:
            raise ValueError("serviceDate must be a real ISO date (YYYY-MM-DD)") from exc
        return value


@hospital_api.get("")
def hospital_list():
    return sorted(HOSPITALS)


@hospital_api.get("/{name}")
def hospital_home(name: str):
    if name not in HOSPITALS:
        raise HTTPException(404, "unknown hospital")
    n = network_360(name)
    return {"hospital": name, "city": n["city"], "requests": preauth.for_hospital(name), "claimPreapprovals": cpa.for_hospital(name), "rates": provider_db.provider(name), "ratesSyncedAt": provider_db.synced_at(),
            "scorecard": {k: n[k] for k in ("treatments", "patients", "billingVsBenchmark", "claims")}}


@hospital_api.post("/{name}/preauth")
def hospital_submit(name: str, r: SessionRequest):
    if name not in HOSPITALS:
        raise HTTPException(404, "unknown hospital")
    p = known(r.policy)
    contact = p["registeredContactId"] if r.contactLast4 == p["registeredContactMasked"][-4:] else f"C-{r.contactLast4}"
    rec = preauth.submit({"policy": r.policy, "hospital": name, "procedure": r.procedure, "serviceDate": r.serviceDate, "amount": r.amount, "contactId": contact, "submittedBy": "hospital"})
    return preauth.public(rec)


class PreApprovalRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    policy: str = Field(min_length=1, max_length=80)
    procedure: str = Field(min_length=1, max_length=120)
    serviceDate: str
    roomCategory: str = Field(default="general_ward", min_length=1, max_length=60)
    roomDays: int = Field(default=0, ge=0, le=7)
    contactLast4: str = Field(pattern=r"^\d{4}$")

    @field_validator("serviceDate")
    @classmethod
    def valid_service_date(cls, value: str) -> str:
        try:
            date.fromisoformat(value)
        except ValueError as exc:
            raise ValueError("serviceDate must be a real ISO date (YYYY-MM-DD)") from exc
        return value


class BillLine(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    description: str = Field(min_length=1, max_length=200)
    category: BillCategory
    qty: Decimal = Field(gt=0, le=100_000)
    rate: Decimal = Field(ge=0, le=100_000_000)
    amount: Decimal = Field(ge=0, le=100_000_000)

    @model_validator(mode="after")
    def amount_matches_quantity_and_rate(self):
        if abs(self.qty * self.rate - self.amount) > Decimal("1"):
            raise ValueError("line amount must equal quantity multiplied by rate (within ₹1 rounding)")
        return self


class Bill(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    billNumber: str = Field(min_length=1, max_length=100)
    totalStated: Decimal = Field(gt=0, le=100_000_000)
    lines: list[BillLine] = Field(min_length=1, max_length=200)

    @model_validator(mode="after")
    def total_matches_lines(self):
        line_total = sum((line.amount for line in self.lines), Decimal(0))
        if abs(line_total - self.totalStated) > Decimal("1"):
            raise ValueError("stated bill total must equal the sum of bill lines (within ₹1 rounding)")
        return self


class Discharge(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    admissionDate: str
    dischargeDate: str
    roomCategory: str = Field(min_length=1, max_length=60)
    roomDays: int = Field(ge=0, le=365)

    @field_validator("admissionDate", "dischargeDate")
    @classmethod
    def valid_discharge_dates(cls, value: str) -> str:
        try:
            date.fromisoformat(value)
        except ValueError as exc:
            raise ValueError("dates must be real ISO dates (YYYY-MM-DD)") from exc
        return value

    @model_validator(mode="after")
    def discharge_not_before_admission(self):
        admission = date.fromisoformat(self.admissionDate)
        discharge = date.fromisoformat(self.dischargeDate)
        if discharge < admission:
            raise ValueError("dischargeDate must not be before admissionDate")
        documented_days = (discharge - admission).days + 1
        if self.roomDays > documented_days:
            raise ValueError("roomDays cannot exceed the inclusive admission-to-discharge period")
        return self


class BillSubmission(BaseModel):
    bill: Bill
    discharge: Discharge


@hospital_api.post("/{name}/claim-preapproval")
def claim_preapproval(name: str, r: PreApprovalRequest):
    if name not in HOSPITALS:
        raise HTTPException(404, "unknown hospital")
    known(r.policy)
    return cpa.public(guarded(cpa.request, name, r.policy, r.procedure, r.serviceDate, r.roomCategory, r.roomDays, r.contactLast4))


@hospital_api.post("/{name}/claim-preapproval/{rid}/bill")
def claim_bill(name: str, rid: str, b: BillSubmission):
    rec = cpa.items.get(rid)
    if not rec or rec["hospital"] != name:
        raise HTTPException(404, "unknown pre-approval for this hospital")
    return cpa.public(guarded(cpa.reconcile, rid, b.bill.model_dump(), b.discharge.model_dump()))


# ---- adjudicator ------------------------------------------------------------------------------
class Act(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    approver: str = Field(min_length=1, max_length=120)
    level: Literal["junior", "senior"] = "junior"
    note: str = Field(default="", max_length=2000)
    terms: dict | None = None

    @field_validator("approver")
    @classmethod
    def approver_must_not_be_blank(cls, value: str) -> str:
        if not value:
            raise ValueError("approver must not be blank")
        return value


def guarded(fn, *a, **k):
    try:
        return fn(*a, **k)
    except KeyError as e:
        raise HTTPException(404, str(e))
    except ValueError as e:
        raise HTTPException(409, str(e))


@adj_api.get("/queue")
def queue():
    out = []
    for r in preauth.queue():
        p = get_patient(r["policy"])
        out.append({**r, "patient": p["name"], "homeCity": p["homeCity"], "hospitalCity": HOSPITALS[r["hospital"]]["city"]})
    return {"asOf": AS_OF.isoformat(), "items": out}


@adj_api.get("/patients/{policy}")
def adj_patient(policy: str):
    known(policy)
    return {**{**patient_360(policy), "remainingCover": preauth.cover_left(policy)}, "standing": [a for a in preauth.standing.values() if a["policy"] == policy],
            "requests": [r for r in preauth.requests.values() if r["policy"] == policy]}


@adj_api.get("/network")
def adj_network():
    return network_overview()


@adj_api.get("/network/{hospital}")
def adj_hospital(hospital: str):
    n = network_360(hospital)
    if not n:
        raise HTTPException(404, "unknown hospital")
    return {**n, "requests": [r for r in preauth.requests.values() if r["hospital"] == hospital]}


@adj_api.post("/preauth/{rid}/approve")
def approve(rid: str, a: Act):
    return guarded(preauth.resolve, rid, "approve", a.approver, a.level, a.note)


@adj_api.post("/preauth/{rid}/reject")
def reject(rid: str, a: Act):
    return guarded(preauth.resolve, rid, "reject", a.approver, a.level, a.note)


@adj_api.post("/preauth/{rid}/standing")
def standing(rid: str, a: Act):
    return guarded(preauth.grant_standing, rid, a.approver, a.level, a.terms)


@adj_api.post("/standing/{aid}/revoke")
def revoke(aid: str, a: Act):
    return guarded(preauth.revoke, aid, a.approver, a.note)


@adj_api.get("/fast-track")
def fast_track():
    order = {"adjudication": 0, "standard_review": 1, "preapproved": 2, "settled": 3, "rejected": 4}
    return sorted(cpa.items.values(), key=lambda x: (order[x["status"]], x["createdAt"]))


@adj_api.post("/fast-track/{rid}/approve")
def ft_approve(rid: str, a: Act):
    return guarded(cpa.resolve, rid, "approve", a.approver, a.level, a.note)


@adj_api.post("/fast-track/{rid}/reject")
def ft_reject(rid: str, a: Act):
    return guarded(cpa.resolve, rid, "reject", a.approver, a.level, a.note)
