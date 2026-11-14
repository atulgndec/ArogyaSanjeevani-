"""Compare retrieval baselines and the rewritten, reranked system on labeled questions."""
import argparse
import json
import os
from pathlib import Path

from arogya.lib import vector_db as v

SETS = {n: json.loads((Path(__file__).parent / f"arogya/data/{n}.json").read_text()) for n in ("retrieval_eval", "retrieval_heldout")}
MODES = {
    "bm25": lambda q: v.bm25_clauses(q, 5),
    "vector": lambda q: v.semantic_clauses(q, 5, min_sim=0.0),
    "baseline": lambda q: v.hybrid_baseline_clauses(q, 5),
    "hybrid": lambda q: v.hybrid_clauses(q, 5),
}


def evaluate(name: str = "retrieval_eval") -> dict:
    CASES, out = SETS[name], {}
    for name, fn in MODES.items():
        hits3, hits5, rr, misses = 0, 0, 0.0, []
        for c in CASES:
            got = [d.metadata["clause"] for d in fn(c["query"])]
            rank = next((i + 1 for i, g in enumerate(got) if g in c["expected"]), None)
            hits3 += rank is not None and rank <= 3
            hits5 += rank is not None and rank <= 5
            rr += 1 / rank if rank else 0
            if rank is None:
                misses.append(c["query"])
        out[name] = {
            "recall@3": round(hits3 / len(CASES), 3),
            "recall@5": round(hits5 / len(CASES), 3),
            "mrr": round(rr / len(CASES), 3),
            "misses": misses,
        }
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-recall-at-5", type=float, default=float(os.getenv(
        "RETRIEVAL_MIN_RECALL_AT_5", os.getenv("RETRIEVAL_MIN_RECALL", "0.80")
    )))
    parser.add_argument("--min-mrr", type=float, default=float(os.getenv("RETRIEVAL_MIN_MRR", "0.65")))
    args = parser.parse_args()
    print("embedder:", v.embedder_name())
    failed = False
    for name in SETS:
        print(f"\n{name} ({len(SETS[name])} queries)" + ("  <- used to choose the weights" if name == "retrieval_eval" else "  <- written afterwards, never tuned on"))
        results = evaluate(name)
        for mode, metrics in results.items():
            print(f"  {mode:8s} Recall@3 {metrics['recall@3']:.2f}  Recall@5 {metrics['recall@5']:.2f}  MRR {metrics['mrr']:.2f}  missed {len(metrics['misses'])}")
            if name == "retrieval_heldout" and mode == "hybrid":
                if metrics["recall@5"] < args.min_recall_at_5 or metrics["mrr"] < args.min_mrr:
                    failed = True
                    print(f"  QUALITY GATE FAILED: hybrid Recall@5 >= {args.min_recall_at_5:.2f}, MRR >= {args.min_mrr:.2f}")
    raise SystemExit(1 if failed else 0)
