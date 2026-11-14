from red_team_eval import ATTACK_PAYLOADS, BENIGN_PAYLOADS, evaluate


def test_red_team_suite_measures_defenses_and_false_blocks():
    assert len(ATTACK_PAYLOADS) >= 30
    assert len(BENIGN_PAYLOADS) >= 20
    results = evaluate()
    assert results["defense_on"]["attack_detection_rate"] == 1
    assert results["defense_off"]["attack_detection_rate"] == 0
    assert results["defense_on"]["false_block_rate"] <= 0.05
