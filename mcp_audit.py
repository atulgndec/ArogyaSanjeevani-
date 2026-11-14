"""FastMCP tools for durable adjudication review notes."""
import os
import re
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Annotated

from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from pydantic import Field

mcp = FastMCP("Arogya Review Audit")
CLAIM_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,79}$")


def _database() -> sqlite3.Connection:
    path = Path(os.getenv("AROGYA_MCP_AUDIT_DB", "/data/review_notes.sqlite3"))
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    connection.execute(
        """CREATE TABLE IF NOT EXISTS review_notes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            claim_id TEXT NOT NULL,
            reviewer TEXT NOT NULL,
            note TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
        )"""
    )
    return connection


def _validate_claim_id(claim_id: str) -> str:
    claim_id = claim_id.strip()
    if not CLAIM_ID.fullmatch(claim_id):
        raise ToolError("Claim ID must contain 1-80 letters, numbers, underscores, or hyphens.")
    return claim_id


def record_review_note(
    claim_id: Annotated[str, Field(min_length=1, max_length=80)],
    reviewer: Annotated[str, Field(min_length=1, max_length=120)],
    note: Annotated[str, Field(min_length=1, max_length=2000)],
) -> dict:
    """Persist a named review note for a claim in the audit database."""
    claim_id = _validate_claim_id(claim_id)
    reviewer = reviewer.strip()
    note = note.strip()
    if not reviewer or not note:
        raise ToolError("Reviewer and note must contain non-whitespace text.")
    try:
        with closing(_database()) as connection:
            with connection:
                cursor = connection.execute(
                    "INSERT INTO review_notes (claim_id, reviewer, note) VALUES (?, ?, ?)",
                    (claim_id, reviewer, note),
                )
                note_id = cursor.lastrowid
    except sqlite3.Error as error:
        raise ToolError(f"Unable to persist review note: {type(error).__name__}.") from error
    return {"saved": True, "note_id": note_id, "claim_id": claim_id}


def get_review_notes(
    claim_id: Annotated[str, Field(min_length=1, max_length=80)],
) -> list[dict]:
    """Read persisted review notes for a claim."""
    claim_id = _validate_claim_id(claim_id)
    try:
        with closing(_database()) as connection:
            rows = connection.execute(
                "SELECT id, claim_id, reviewer, note, created_at FROM review_notes "
                "WHERE claim_id = ? ORDER BY id",
                (claim_id,),
            ).fetchall()
    except sqlite3.Error as error:
        raise ToolError(f"Unable to read review notes: {type(error).__name__}.") from error
    return [dict(row) for row in rows]


mcp.tool(record_review_note)
mcp.tool(get_review_notes)
app = mcp.http_app(path="/mcp")
