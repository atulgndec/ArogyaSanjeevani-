import pytest
from fastmcp.exceptions import ToolError

from mcp_audit import get_review_notes, record_review_note
from mcp_filesystem import list_project_files, read_project_file
from mcp_retrieval import get_policy_clause, search_policy


def test_search_policy_returns_passage_citations():
    results = search_policy("room rent proportional cut", top_k=5)
    assert results
    assert results[0]["passage_id"]
    assert results[0]["source_document"]


def test_get_policy_clause_reports_unknown_clause_as_tool_error():
    with pytest.raises(ToolError):
        get_policy_clause("not-a-real-clause")


def test_list_project_files_excludes_hidden_files(tmp_path, monkeypatch):
    (tmp_path / "policy.txt").write_text("policy text", encoding="utf-8")
    (tmp_path / ".env").write_text("secret=value", encoding="utf-8")
    monkeypatch.setenv("AROGYA_MCP_WORKSPACE", str(tmp_path))
    assert "policy.txt" in list_project_files(".")
    assert ".env" not in list_project_files(".")


def test_read_project_file_rejects_traversal_and_hidden_files(tmp_path, monkeypatch):
    (tmp_path / "policy.txt").write_text("policy text", encoding="utf-8")
    (tmp_path / ".env").write_text("secret=value", encoding="utf-8")
    monkeypatch.setenv("AROGYA_MCP_WORKSPACE", str(tmp_path))
    assert read_project_file("policy.txt") == "policy text"
    with pytest.raises(ToolError):
        read_project_file("../outside.txt")
    with pytest.raises(ToolError):
        read_project_file(".env")


def test_record_review_note_persists(tmp_path, monkeypatch):
    monkeypatch.setenv("AROGYA_MCP_AUDIT_DB", str(tmp_path / "audit.sqlite3"))
    result = record_review_note("CLM-0001", "reviewer", "Check duplicate room charges")
    assert result["saved"] is True
    assert result["claim_id"] == "CLM-0001"


def test_get_review_notes_reads_persisted_notes(tmp_path, monkeypatch):
    monkeypatch.setenv("AROGYA_MCP_AUDIT_DB", str(tmp_path / "audit.sqlite3"))
    record_review_note("CLM-0001", "reviewer", "Check duplicate room charges")
    notes = get_review_notes("CLM-0001")
    assert notes[0]["reviewer"] == "reviewer"
    assert notes[0]["note"] == "Check duplicate room charges"
