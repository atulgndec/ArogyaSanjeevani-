from arogya.data.synthetic_claims import load_claims
from arogya.lib import agent_orchestrator as orchestrator


def test_orchestration_snapshot_exposes_plan_and_observation_results():
    snapshot = orchestrator.run_claim(load_claims()[0].claim_id)
    plan = snapshot["plan"]
    assert plan["objective"]
    assert plan["strategy"]
    assert {"extraction", "fraud", "coverage", "critic"} <= {
        step["agent"] for step in plan["steps"]
    }
    assert plan["replan_count"] == 0
    assert snapshot["agents"]["extraction"]["status"] == "complete"
