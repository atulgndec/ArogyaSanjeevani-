from decimal import Decimal

from fastapi.testclient import TestClient

import server


api = TestClient(server.app, headers={"X-Adjudicator-Key": "demo-key"})


def hospital_and_policy():
    return api.get("/api/hospital").json()[0], api.get("/api/patient").json()[0]["policy"]


def test_session_request_rejects_invalid_date_amount_and_phone():
    hospital, policy = hospital_and_policy()
    response = api.post(
        f"/api/hospital/{hospital}/preauth",
        json={
            "policy": policy,
            "procedure": "Dialysis",
            "serviceDate": "not-a-date",
            "amount": -1,
            "contactLast4": "12x4",
        },
    )
    assert response.status_code == 422


def test_session_request_accepts_a_well_formed_payload():
    hospital, policy = hospital_and_policy()
    response = api.post(
        f"/api/hospital/{hospital}/preauth",
        json={
            "policy": policy,
            "procedure": "Dialysis",
            "serviceDate": "2026-10-28",
            "amount": 2800,
            "contactLast4": "1234",
        },
    )
    assert response.status_code == 200


def test_claim_preapproval_rejects_invalid_room_days_and_date():
    hospital, policy = hospital_and_policy()
    response = api.post(
        f"/api/hospital/{hospital}/claim-preapproval",
        json={
            "policy": policy,
            "procedure": "Dialysis",
            "serviceDate": "2026-02-30",
            "roomCategory": "general_ward",
            "roomDays": 8,
            "contactLast4": "1234",
        },
    )
    assert response.status_code == 422


def test_bill_submission_rejects_arithmetic_and_date_inconsistencies():
    hospital, _ = hospital_and_policy()
    response = api.post(
        f"/api/hospital/{hospital}/claim-preapproval/unknown/bill",
        json={
            "bill": {
                "billNumber": "B-1",
                "totalStated": 100,
                "lines": [
                    {
                        "description": "Treatment",
                        "category": "procedure",
                        "qty": 1,
                        "rate": 50,
                        "amount": 100,
                    }
                ],
            },
            "discharge": {
                "admissionDate": "2026-11-02",
                "dischargeDate": "2026-11-01",
                "roomCategory": "general_ward",
                "roomDays": 1,
            },
        },
    )
    assert response.status_code == 422


def test_decision_rejects_unknown_action_and_blank_approver():
    response = api.post(
        "/api/claims/CLM-001/decision",
        json={"action": "do-anything", "approver": " ", "level": "root"},
    )
    assert response.status_code == 422


def test_decision_requires_reason_for_rejection_and_payload_for_edit():
    api.post("/api/claims/CLM-001/run")
    reject = api.post(
        "/api/claims/CLM-001/decision",
        json={"action": "reject", "approver": "Reviewer"},
    )
    edit = api.post(
        "/api/claims/CLM-001/decision",
        json={"action": "edit", "approver": "Reviewer"},
    )
    assert reject.status_code == 422
    assert edit.status_code == 422


def test_line_edit_cannot_deduct_more_than_claimed_amount():
    snapshot = api.post("/api/claims/CLM-001/run").json()
    line = next(
        row for row in snapshot["worksheet"]["lines"]
        if Decimal(row["claimedAmount"]) > 0
    )
    response = api.post(
        "/api/claims/CLM-001/decision",
        json={
            "action": "edit",
            "approver": "Reviewer",
            "edits": [
                {
                    "line_id": line["lineId"],
                    "deduction_amount": str(Decimal(line["claimedAmount"]) + 1),
                }
            ],
        },
    )
    assert response.status_code == 409


def test_adjudicator_actions_require_named_approver_and_known_level():
    response = api.post(
        "/api/adjudicator/preauth/unknown/approve",
        json={"approver": " ", "level": "root"},
    )
    assert response.status_code == 422
