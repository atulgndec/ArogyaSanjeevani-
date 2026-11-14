from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from arogya.data.synthetic_claims import load_claims
from arogya.lib.adjudication_calculator import calculate_worksheet
from arogya.lib.claims_graph import build_graph
from arogya.lib.fraud_detector import detect_fraud
from arogya.lib.reflexion_engine import ReflexionEngine


def _junior_eligible_claim():
    for claim in load_claims():
        worksheet = calculate_worksheet(claim)
        if worksheet.is_admissible and not worksheet.requires_senior_approval and not detect_fraud(claim).has_fraud_flag:
            return claim
    raise AssertionError("Synthetic set must include a junior-eligible claim")


def test_sqlite_checkpoint_can_resume_a_paused_claim_after_graph_recreation(tmp_path):
    claim = _junior_eligible_claim()
    config = {"configurable": {"thread_id": claim.claim_id}}
    database = str(tmp_path / "checkpoints.sqlite3")

    with SqliteSaver.from_conn_string(database) as checkpointer:
        graph = build_graph(checkpointer=checkpointer)
        graph.invoke({"claim": claim}, config)
        assert graph.get_state(config).next

    with SqliteSaver.from_conn_string(database) as checkpointer:
        restarted_graph = build_graph(checkpointer=checkpointer)
        assert restarted_graph.get_state(config).next
        restarted_graph.invoke(Command(resume={"approver": "reviewer", "level": "junior"}), config)
        assert restarted_graph.get_state(config).values["outcome"] == "settled"


def test_reflexion_lessons_survive_engine_recreation(tmp_path):
    database = tmp_path / "lessons.sqlite3"
    engine = ReflexionEngine(database)
    engine.review("LESSON-001", "withdraw", "Named reviewer", "Rejected after policy review")

    restarted_engine = ReflexionEngine(database)
    lesson = next(item for item in restarted_engine.all() if item.id == "LESSON-001")
    assert lesson.status == "withdrawn"
    assert lesson.impact_note == "Rejected after policy review"
