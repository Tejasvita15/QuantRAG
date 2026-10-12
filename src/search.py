import argparse
import json
import time
from retriever import HybridRetriever

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("--ticker")
    ap.add_argument("--year", type=int)
    ap.add_argument("--form")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--no-rerank", action="store_true")
    args = ap.parse_args()

    t0 = time.perf_counter()
    retriever = HybridRetriever(use_reranker=not args.no_rerank)
    load_s = time.perf_counter() - t0

    t1 = time.perf_counter()
    results = retriever.retrieve(
        args.query,
        top_k=args.top_k,
        ticker=args.ticker,
        year=args.year,
        form=args.form,
        rerank=not args.no_rerank
    )
    query_s = time.perf_counter() - t1

    print(f"Model/index load: {load_s:.3f}s")
    print(f"Query latency:    {query_s:.3f}s")
    print("=" * 90)
    for r in results:
        p = r["passage"]
        print(f"\n#{r['rank']}  {p['ticker']} | {p['form']} | {p['filing_date']} | {p['section']}")
        print(f"id={p['id']} sources={r.get('sources')} rrf={r.get('rrf_score', 0):.5f} rerank={r.get('rerank_score', float('nan')):.4f}")
        print(p["text"][:900].replace("\n", " "))

if __name__ == "__main__":
    main()
