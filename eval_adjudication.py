"""Audit exact-money and worksheet invariants over the synthetic adjudication set."""
from __future__ import annotations

import json

from arogya.data.synthetic_claims import load_claims
from arogya.lib.adjudication_calculator import calculate_worksheet


def evaluate() -> dict:
    claims = load_claims()
    failures = []
    passed = 0
    for claim in claims:
        worksheet = calculate_worksheet(claim)
        checks = {
            "money_identity": worksheet.total_deductions + worksheet.admissible_before_copay == claim.claimed_total,
            "payable_within_limit": 0 <= worksheet.final_payable_amount <= claim.total_coverage_limit,
            "line_amounts_bounded": all(0 <= line.deduction_amount <= line.claimed_amount for line in worksheet.lines),
            "deductions_cite_policy": all(
                line.clause_cited and line.deduction_reason
                for line in worksheet.lines if line.deduction_amount > 0
            ),
        }
        if all(checks.values()):
            passed += 1
        else:
            failures.append({"claim_id": claim.claim_id, "failed_checks": [name for name, ok in checks.items() if not ok]})

    return {
        "total_cases": len(claims),
        "dev_cases": sum(claim.split_type == "dev" for claim in claims),
        "held_out_cases": sum(claim.split_type == "held_out" for claim in claims),
        "invariant_pass_rate": round(passed / len(claims), 3) if claims else 0,
        "failed_claims": failures,
    }


if __name__ == "__main__":
    results = evaluate()
    print(json.dumps(results, indent=2))
    if results["total_cases"] < 60 or results["invariant_pass_rate"] != 1:
        raise SystemExit(1)
