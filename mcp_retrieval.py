"""FastMCP tools for searching and retrieving policy passages."""
from typing import Annotated

from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from pydantic import Field

from arogya.lib.policy_kb import get_clause
from arogya.lib.vector_db import hybrid_clauses

mcp = FastMCP("Arogya Policy Retrieval")


def search_policy(
    query: Annotated[str, Field(min_length=1, max_length=500)],
    top_k: Annotated[int, Field(ge=1, le=10)] = 5,
) -> list[dict]:
    """Search policy passages and return passage IDs with source citations."""
    if not query.strip():
        raise ToolError("Query must contain non-whitespace text.")
    documents = hybrid_clauses(query, k=top_k)
    return [
        {
            "passage_id": document.metadata["passage_id"],
            "clause": document.metadata["clause"],
            "title": document.metadata["title"],
            "source_document": document.metadata["source_document"],
            "source_date": document.metadata["source_date"],
            "ingested_at": document.metadata["ingested_at"],
            "citation": f"[{document.metadata['passage_id']}]",
            "text": document.page_content,
        }
        for document in documents
    ]


def get_policy_clause(
    clause_number: Annotated[str, Field(min_length=1, max_length=80)],
) -> dict:
    """Retrieve a numbered policy clause by its clause number."""
    clause = get_clause(clause_number.strip())
    if clause is None:
        raise ToolError(f"Unknown policy clause: {clause_number}")
    return clause.model_dump(mode="json", by_alias=True) | {
        "passageId": clause.id,
        "sourceDocument": "arogya/data/policy_clauses.json",
        "sourceDate": "unknown",
    }


mcp.tool(search_policy)
mcp.tool(get_policy_clause)
app = mcp.http_app(path="/mcp")
