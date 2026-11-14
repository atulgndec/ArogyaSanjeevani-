from eval_adjudication import evaluate


def test_adjudication_evaluation_covers_at_least_sixty_cases():
    result = evaluate()
    assert result["total_cases"] >= 60
    assert result["invariant_pass_rate"] == 1
    assert result["dev_cases"] >= 1
    assert result["held_out_cases"] >= 1
