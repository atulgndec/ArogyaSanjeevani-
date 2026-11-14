"""80 synthetic claims (40 dev, 40 held-out). Mirrors src/data/syntheticClaims.ts.

The TypeScript file generates them; here the generator's output is stored as JSON (same camelCase fields),
so the dataset is fixed and reviewable. Each claim carries its four raw OCR-style documents.
"""
import json
from functools import lru_cache
from pathlib import Path

from ..types.adjudication import ExtractedClaim

DATA = Path(__file__).parent


@lru_cache
def load_claims() -> list[ExtractedClaim]:
    return [ExtractedClaim.model_validate(c) for c in json.loads((DATA / "synthetic_claims.json").read_text(encoding="utf-8"))]


def get_claim(claim_id: str) -> ExtractedClaim | None:
    sample = next((c for c in load_claims() if c.claim_id == claim_id), None)
    if sample is not None:
        return sample
    from ..lib.uploaded_claims import get as get_uploaded_claim
    return get_uploaded_claim(claim_id)


def all_claims() -> list[ExtractedClaim]:
    from ..lib.uploaded_claims import all_claims as uploaded_claims
    return [*load_claims(), *uploaded_claims()]
