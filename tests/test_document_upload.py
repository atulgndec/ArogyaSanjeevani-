from pathlib import Path
from uuid import uuid4

from fastapi.testclient import TestClient
from PIL import Image

import server
from arogya.lib.document_ocr import DocumentReadError, read_document

api = TestClient(server.app, headers={"X-Adjudicator-Key": "demo-key"})
PACKET_DIR = Path(__file__).resolve().parents[1] / "examples" / "upload_packet"


def test_text_document_reader_rejects_bad_inputs():
    assert read_document("packet.txt", b"\xef\xbb\xbfClaim ID: A") == ("Claim ID: A", "text")
    try:
        read_document("packet.exe", b"no")
    except DocumentReadError as exc:
        assert "PDF, PNG" in str(exc)
    else:
        raise AssertionError("unsupported file type should fail")


def test_searchable_pdf_and_scanned_image_paths(monkeypatch):
    import fitz
    import arogya.lib.document_ocr as document_ocr

    pdf = fitz.open()
    pdf.new_page().insert_text((72, 72), "A searchable claim document with enough text.")
    content = pdf.tobytes()
    pdf.close()
    text, method = read_document("claim.pdf", content)
    assert "searchable claim document" in text and method == "PDF text"

    monkeypatch.setattr(document_ocr, "_ocr_image", lambda image: "OCR claim text")
    image = Image.new("RGB", (20, 20), "white")
    from io import BytesIO
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    assert read_document("claim.png", buffer.getvalue()) == ("OCR claim text", "OCR")


def test_uploaded_packet_is_extractable_and_joins_existing_worklist():
    files = {}
    for field, filename in (
        ("claim_form", "claim_form.txt"),
        ("discharge_summary", "discharge_summary.txt"),
        ("itemised_bill", "itemised_bill.txt"),
        ("policy_schedule", "policy_schedule.txt"),
    ):
        content = (PACKET_DIR / filename).read_bytes()
        if field == "claim_form":
            content = content.replace(b"UPLOAD-DEMO-001", f"UPLOAD-{uuid4().hex[:12]}".encode())
        files[field] = (filename, content, "text/plain")

    response = api.post("/api/claims/upload", files=files)
    assert response.status_code == 200, response.text
    result = response.json()
    claim_id = result["claim"]["claimId"]
    assert result["claim"]["lineItems"]
    assert result["documentMethods"] == {
        "claim_form": "text",
        "discharge_summary": "text",
        "itemised_bill": "text",
        "policy_schedule": "text",
    }
    assert api.get(f"/api/claims/{claim_id}").status_code == 200
    assert any(row["claimId"] == claim_id for row in api.get("/api/adjudicator/worklist").json())
    run = api.post(f"/api/claims/{claim_id}/run")
    assert run.status_code == 200, run.text
    assert run.json()["status"] == "waiting_for_adjudicator"


def test_claim_upload_requires_adjudicator_authentication():
    unauthenticated = TestClient(server.app)
    response = unauthenticated.post("/api/claims/upload")
    assert response.status_code == 401


def test_upload_rejects_an_id_already_used_by_a_synthetic_claim():
    files = {}
    for field, filename in (
        ("claim_form", "claim_form.txt"),
        ("discharge_summary", "discharge_summary.txt"),
        ("itemised_bill", "itemised_bill.txt"),
        ("policy_schedule", "policy_schedule.txt"),
    ):
        content = (PACKET_DIR / filename).read_bytes()
        if field == "claim_form":
            content = content.replace(b"UPLOAD-DEMO-001", b"CLM-001")
        files[field] = (filename, content, "text/plain")

    response = api.post("/api/claims/upload", files=files)

    assert response.status_code == 409
    assert api.get("/api/claims/CLM-001").json()["claimId"] == "CLM-001"
