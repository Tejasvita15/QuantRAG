import json
import time
from retriever import HybridRetriever
from common import EVAL_DIR

QUERIES = [
    ("AAPL", "What risks does Apple identify related to supply chain disruption?"),
    ("MSFT", "What factors affected Microsoft's cloud revenue growth?"),
    ("NVDA", "What risks does NVIDIA report related to export restrictions?"),
    ("JPM", "What does JPMorgan report about credit risk and loan losses?"),
    ("XOM", "What risks does Exxon Mobil report regarding commodity prices?"),
]

def main():
    retriever = HybridRetriever(use_reranker=True)
    rows = []
    for ticker, query in QUERIES:
        t0 = time.perf_counter()
        res = retriever.retrieve(query, top_k=5, ticker=ticker)
        dt = time.perf_counter() - t0
        rows.append({
            "ticker": ticker,
            "query": query,
            "latency_s": dt,
            "results": [
                {
                    "rank": r["rank"],
                    "id": r["id"],
                    "filing_date": r["passage"]["filing_date"],
                    "section": r["passage"]["section"],
                    "text": r["passage"]["text"][:600],
                }
                for r in res
            ],
        })
        print(f"{ticker}: {dt:.3f}s")
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    out = EVAL_DIR / "five_query_results.json"
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"Saved: {out}")

if __name__ == "__main__":
    main()
