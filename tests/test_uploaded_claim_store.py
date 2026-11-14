import pytest

from arogya.data.synthetic_claims import load_claims
from arogya.lib.uploaded_claims import UploadedClaimStore


def test_persistent_store_retains_uploaded_claim_after_recreation(tmp_path):
    database = tmp_path / "workflow" / "uploaded.sqlite3"
    claim = load_claims()[0].model_copy(update={"claim_id": "UPLOAD-PERSIST-001"})

    UploadedClaimStore(database).register(claim)

    restarted_store = UploadedClaimStore(database)
    restored = restarted_store.get(claim.claim_id)
    assert restored == claim
    assert [item.claim_id for item in restarted_store.all()] == [claim.claim_id]


def test_persistent_store_rejects_duplicate_claim_ids(tmp_path):
    store = UploadedClaimStore(tmp_path / "uploaded.sqlite3")
    claim = load_claims()[0].model_copy(update={"claim_id": "UPLOAD-DUPLICATE-001"})
    store.register(claim)

    with pytest.raises(ValueError, match="already exists"):
        UploadedClaimStore(tmp_path / "uploaded.sqlite3").register(claim)
