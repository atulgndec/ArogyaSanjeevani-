"""Claims created from adjudicator-uploaded document packets."""
from __future__ import annotations

import os
import sqlite3
from contextlib import closing
from pathlib import Path
from threading import RLock

from ..types.adjudication import ExtractedClaim


class UploadedClaimStore:
    def __init__(self, database_path: str | Path | None = None) -> None:
        self.database_path = Path(database_path) if database_path else None
        self._claims: dict[str, str] = {}
        self._lock = RLock()
        if self.database_path:
            with closing(self._connect()) as connection:
                connection.execute(
                    "CREATE TABLE IF NOT EXISTS uploaded_claims "
                    "(claim_id TEXT PRIMARY KEY, payload TEXT NOT NULL)"
                )


    def _connect(self) -> sqlite3.Connection:
        if self.database_path is None:
            raise RuntimeError("A database path is required for persistent uploaded claims.")
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        return sqlite3.connect(self.database_path, timeout=10)

    def register(self, claim: ExtractedClaim) -> None:
        payload = claim.model_dump_json()
        if self.database_path:
            try:
                with closing(self._connect()) as connection:
                    with connection:
                        connection.execute(
                            "INSERT INTO uploaded_claims (claim_id, payload) VALUES (?, ?)",
                            (claim.claim_id, payload),
                        )
            except sqlite3.IntegrityError as exc:
                raise ValueError(f"an uploaded claim with ID {claim.claim_id} already exists") from exc
            return
        with self._lock:
            if claim.claim_id in self._claims:
                raise ValueError(f"an uploaded claim with ID {claim.claim_id} already exists")
            self._claims[claim.claim_id] = payload

    def get(self, claim_id: str) -> ExtractedClaim | None:
        if self.database_path:
            with closing(self._connect()) as connection:
                row = connection.execute(
                    "SELECT payload FROM uploaded_claims WHERE claim_id = ?", (claim_id,)
                ).fetchone()
            return ExtractedClaim.model_validate_json(row[0]) if row else None
        with self._lock:
            payload = self._claims.get(claim_id)
        return ExtractedClaim.model_validate_json(payload) if payload else None

    def all(self) -> list[ExtractedClaim]:
        if self.database_path:
            with closing(self._connect()) as connection:
                rows = connection.execute(
                    "SELECT payload FROM uploaded_claims ORDER BY claim_id"
                ).fetchall()
            return [ExtractedClaim.model_validate_json(row[0]) for row in rows]
        with self._lock:
            payloads = list(self._claims.values())
        return [ExtractedClaim.model_validate_json(payload) for payload in payloads]


_database_path = os.getenv("UPLOADED_CLAIMS_DB")
uploaded_claim_store = UploadedClaimStore(_database_path)


def register(claim: ExtractedClaim) -> None:
    uploaded_claim_store.register(claim)


def get(claim_id: str) -> ExtractedClaim | None:
    return uploaded_claim_store.get(claim_id)


def all_claims() -> list[ExtractedClaim]:
    return uploaded_claim_store.all()
