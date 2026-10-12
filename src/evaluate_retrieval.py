import argparse
import json
import math
import time
from statistics import mean

from retriever import HybridRetriever
from common import EVAL_DIR, load_jsonl

def precision_at_k(ranked, relevant, k):
    top = ranked[:k]
    return sum(1 for x in top if x in relevant) / k

def recall_at_k(ranked, relevant, k):
    if not relevant:
        return 0.0
    return sum(1 for x in ranked[:k] if x in relevant) / len(relevant)

def rr(ranked, relevant):
    for i, x in enumerate(ranked, 1):
        if x in relevant:
            return 1.0 / i
    return 0.0

def ndcg_at_k(ranked, grades, k):
    dcg = 0.0
    for i, x in enumerate(ranked[:k], 1):
        rel = grades.get(x, 0)
        dcg += (2**rel - 1) / math.log2(i + 1)
    ideal = sorted(grades.values(), reverse=True)[:k]
    idcg = sum((2**rel - 1) / math.log2(i + 1) for i, rel in enumerate(ideal, 1))
    return dcg / idcg if idcg else 0.0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default=str(EVAL_DIR / "retrieval_eval.jsonl"))
    ap.add_argument("--mode", choices=["hybrid", "hybrid_no_rerank"], default="hybrid")
    args = ap.parse_args()

    examples = load_jsonl(args.file)
    use_reranker = args.mode == "hybrid"
    retriever = HybridRetriever(use_reranker=use_reranker)

    rows = []
    for ex in examples:
        t0 = time.perf_counter()
        results = retriever.retrieve(
            ex["query"],
            top_k=10,
            ticker=ex.get("ticker"),
            year=ex.get("year"),
            form=ex.get("form"),
            rerank=use_reranker,
        )
        latency = time.perf_counter() - t0
        ranked = [r["id"] for r in results]
        grades = ex.get("relevance", {})
        relevant = {k for k, v in grades.items() if v > 0}
        rows.append({
            "p@5": precision_at_k(ranked, relevant, 5),
            "r@5": recall_at_k(ranked, relevant, 5),
            "r@10": recall_at_k(ranked, relevant, 10),
            "mrr": rr(ranked, relevant),
            "ndcg@10": ndcg_at_k(ranked, grades, 10),
            "latency_s": latency,
        })

    if not rows:
        print("No evaluation rows found.")
        return

    summary = {k: mean(r[k] for r in rows) for k in rows[0]}
    print(json.dumps({"mode": args.mode, "queries": len(rows), **summary}, indent=2))

if __name__ == "__main__":
    main()
