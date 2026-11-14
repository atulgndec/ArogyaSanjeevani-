import eval_retrieval
from arogya.lib import vector_db


def test_query_rewriting_expands_claim_billing_language():
    rewritten = vector_db.rewrite_query("OT charges")
    assert "Operation Theatre" in rewritten
    assert "Clause 4.1.d" in rewritten


def test_query_rewriting_expands_semantically_related_glossary_terms():
    rewritten = vector_db.rewrite_query("daily fee for a hospital bed beyond what the policy allows")
    assert "Room rent daily limit" in rewritten
    assert "bed charges per night" in rewritten


def test_hybrid_results_retain_passage_and_source_metadata():
    results = vector_db.hybrid_clauses("OT charges", k=5)
    assert results
    assert all(result.metadata["passage_id"] for result in results)
    assert all(result.metadata["source_document"] for result in results)
    assert all(result.metadata["ingested_at"] for result in results)


def test_reranker_returns_ranked_candidates_with_scores():
    candidates = vector_db.hybrid_candidates("room rent proportional cut")
    ranked = vector_db.rerank_clauses("room rent proportional cut", candidates, top=5)
    assert ranked
    assert len(ranked) <= 5
    assert all("rerank_score" in result.metadata for result in ranked)
    assert ranked[0].metadata["clause"] == "Clause 4.1 Note b"


def test_evaluation_reports_recall_at_5_and_before_after_comparison():
    result = eval_retrieval.evaluate("retrieval_heldout")
    assert len(eval_retrieval.SETS["retrieval_eval"]) + len(eval_retrieval.SETS["retrieval_heldout"]) >= 30
    assert "recall@5" in result["hybrid"]
    assert "baseline" in result and "recall@5" in result["baseline"]
