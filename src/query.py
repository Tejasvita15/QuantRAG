"""Unified query entry point: route a question to one backend, print JSON.

Backends:
- "filing" -> HybridRetriever over the SEC corpus (src/retriever.py)
- "macro"  -> FRED observations (src/live/fred_tool.py)
- "stock"  -> yfinance snapshot (src/live/stock_tool.py)

Routing is automatic via src/query_router.py unless --type is given.

Input:
    positional query string plus optional CLI flags (see argparse help).

Output (stdout, JSON):
    Success: {"query", "route", "latency_s", "result"}
    Error:   {"query", "route", "error"} with exit code 1
"""

import argparse
import json
import time

from query_router import route


def run_filing(query, args):
    """Retrieve ranked passages from the SEC corpus for a filing question.

    Input:
        query (str): natural-language question.
        args (argparse.Namespace): parsed CLI flags (top_k, ticker, year,
            form, no_rerank).

    Output:
        list[dict]: top-ranked passages, each with rank, id, ticker, form,
        filing_date, section, sources and truncated text.
    """
    from retriever import HybridRetriever

    retriever = HybridRetriever(use_reranker=not args.no_rerank)
    results = retriever.retrieve(
        query,
        top_k=args.top_k,
        ticker=args.ticker,
        year=args.year,
        form=args.form,
        rerank=not args.no_rerank,
    )
    return [
        {
            "rank": r["rank"],
            "id": r["id"],
            "ticker": r["passage"]["ticker"],
            "form": r["passage"]["form"],
            "filing_date": r["passage"]["filing_date"],
            "section": r["passage"]["section"],
            "sources": r.get("sources"),
            "text": r["passage"]["text"][:600],
        }
        for r in results
    ]


def run_stock(ticker):
    """Fetch a live market snapshot via yfinance.

    Input:
        ticker (str): uppercase ticker symbol, e.g. "AAPL".

    Output:
        dict: latest trading date, price, previous close, 1-day and 5-day
        returns (percent), volume, market cap and currency.

    Raises:
        ValueError: no ticker supplied.
    """
    from live.stock_tool import get_stock_snapshot

    if not ticker:
        raise ValueError("No ticker detected; pass --ticker AAPL")
    return get_stock_snapshot(ticker)


def run_macro(series):
    """Fetch recent FRED macro observations.

    Input:
        series (str): FRED alias (cpi, unemployment, fed_funds, gdp,
        treasury_10y) or any valid FRED series ID.

    Output:
        dict: {"series_id": str, "observations": [{"date", "value"}, ...]}
        sorted newest first.

    Raises:
        ValueError: no series supplied.
        RuntimeError: FRED_API_KEY missing in .env.
    """
    from live.fred_tool import get_fred_series

    if not series:
        raise ValueError("No macro indicator detected; pass --series cpi")
    return get_fred_series(series)


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Route a query to SEC corpus retrieval, FRED macro data "
            "or live stock data."
        )
    )
    ap.add_argument("query")
    ap.add_argument(
        "--type",
        choices=["auto", "stock", "macro", "filing"],
        default="auto",
        help="Override the automatic route.",
    )
    ap.add_argument("--ticker", help="Force ticker, e.g. AAPL")
    ap.add_argument(
        "--series",
        help=(
            "Force FRED indicator: cpi, unemployment, fed_funds, gdp, "
            "treasury_10y or any FRED series ID"
        ),
    )
    ap.add_argument("--year", type=int)
    ap.add_argument("--form")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--no-rerank", action="store_true")
    args = ap.parse_args()

    kind, hint = route(args.query)
    if args.type != "auto":
        kind = args.type

    t0 = time.perf_counter()
    try:
        if kind == "stock":
            result = run_stock(args.ticker or hint.get("ticker"))
        elif kind == "macro":
            result = run_macro(args.series or hint.get("series"))
        else:
            result = run_filing(args.query, args)
    except Exception as e:
        print(json.dumps(
            {"query": args.query, "route": kind, "error": str(e)},
            indent=2,
        ))
        return 1
    latency = time.perf_counter() - t0

    out = {
        "query": args.query,
        "route": kind,
        "latency_s": round(latency, 3),
        "result": result,
    }
    print(json.dumps(out, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
