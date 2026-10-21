"""Five required SEC example queries, with complete inspectable passages."""

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

if __package__:
    from .common import EVAL_DIR, load_config
else:
    from common import EVAL_DIR, load_config

QUERIES = [
    ("AAPL", "What risks does Apple identify related to supply chain disruption?"),
    ("MSFT", "What factors affected Microsoft's cloud revenue growth?"),
    ("NVDA", "What risks does NVIDIA report related to export restrictions?"),
    ("JPM", "What does JPMorgan report about credit risk and loan losses?"),
    ("XOM", "What risks does Exxon Mobil report regarding commodity prices?"),
]

def benchmark(retriever, *, top_k=5, warmup=True):
    if top_k < 1:
        raise ValueError("top_k must be positive")
    if warmup:
        retriever.retrieve(QUERIES[0][1], top_k=top_k, ticker=QUERIES[0][0], rerank=True)
    rows = []
    for ticker, query in QUERIES:
        t0 = time.perf_counter()
        res = retriever.retrieve(query, top_k=top_k, ticker=ticker, rerank=True)
        dt = time.perf_counter() - t0
        rows.append({
            "ticker": ticker,
            "query": query,
            "latency_s": dt,
            # Preserve the older flattened keys, but never truncate annotation context.
            "results": [{**r, "filing_date": r["passage"].get("filing_date"),
                         "section": r["passage"].get("section"),
                         "text": r["passage"]["text"]} for r in res],
        })
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=EVAL_DIR / "five_query_results.json")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--no-warmup", action="store_true", help="Include cold first-query effects")
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("Output already exists; choose a new --output path to preserve previous measurements")
    if args.top_k < 1:
        parser.error("--top-k must be positive")
    if __package__:
        from .retriever import HybridRetriever
    else:
        from retriever import HybridRetriever
    started = time.perf_counter()
    retriever = HybridRetriever(use_reranker=True)
    initialization = time.perf_counter() - started
    rows = benchmark(retriever, top_k=args.top_k, warmup=not args.no_warmup)
    payload = {"created_at": datetime.now(timezone.utc).isoformat(),
               "benchmark": "five SEC retrieval queries; not end-to-end LLM latency",
               "warmup": not args.no_warmup, "initialization_s": initialization,
               "config": load_config(), "top_k": args.top_k, "queries": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    for row in rows:
        print(f"{row['ticker']}: {row['latency_s']:.3f}s ({len(row['results'])} passages)")
    print(f"Saved: {args.output}")
    return 0

if __name__ == "__main__":
    main()
