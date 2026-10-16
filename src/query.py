"""Unified JSON CLI for routed evidence retrieval and grounded answers.

Examples:
    python src/query.py "Apple supply-chain risks and latest stock price" --answer
    python -m src.query "JPMorgan risks and latest Fed Funds rate" --route-only

Without --answer, the original single-backend result shape is retained. Mixed
queries return results grouped by backend, plus the shared evidence registry.
"""

import argparse
import contextlib
import json
import sys

if __package__:
    from .query_router import plan_route
    from .rag_pipeline import RAGPipeline
else:
    from query_router import plan_route
    from rag_pipeline import RAGPipeline


def _legacy_result(out):
    kind = out["route"]
    if kind == "mixed":
        return out["results"]
    rows = out["results"][kind]
    if kind == "filing":
        return [
            {
                "rank": row.get("rank"), "id": row["passage"]["id"],
                **{key: row["passage"].get(key) for key in (
                    "ticker", "form", "filing_date", "section", "accession", "cik",
                )},
                "sources": row.get("sources"), "text": row["passage"]["text"],
            }
            for row in rows
        ]
    return rows[0] if len(rows) == 1 else rows


def main(argv=None, pipeline=None):
    ap = argparse.ArgumentParser(description="Route financial questions and optionally generate cited answers")
    ap.add_argument("query")
    ap.add_argument("--type", choices=["auto", "stock", "macro", "filing"], default="auto")
    ap.add_argument("--ticker", help="Override inferred company/ticker")
    ap.add_argument("--series", help="Override FRED alias or series ID")
    ap.add_argument("--year", type=int, help="Filing-date year filter (not fiscal year)")
    ap.add_argument("--form", choices=["10-K", "10-Q"])
    ap.add_argument("--top-k", type=int, default=5, help="SEC passages per company")
    ap.add_argument("--no-rerank", action="store_true")
    ap.add_argument("--answer", action="store_true", help="Generate an evidence-grounded LLM answer")
    ap.add_argument("--route-only", action="store_true", help="Inspect routing without data/model/API calls")
    args = ap.parse_args(argv)
    options = dict(ticker=args.ticker, series=args.series, year=args.year,
                   form=args.form, kind=args.type)
    try:
        if args.top_k < 1:
            raise ValueError("--top-k must be positive")
        if args.route_only:
            out = {"query": args.query, "routing": plan_route(args.query, **options).to_dict()}
        else:
            # Provider/model diagnostic prints must not corrupt JSON stdout.
            with contextlib.redirect_stdout(sys.stderr):
                out = (pipeline if pipeline is not None else RAGPipeline()).run(
                    args.query, **options, top_k=args.top_k,
                    rerank=not args.no_rerank, generate=args.answer,
                )
            if not args.answer:
                out["result"] = _legacy_result(out)
    except ValueError as exc:
        print(json.dumps({"query": args.query, "error": str(exc)}, ensure_ascii=False, indent=2))
        return 1
    print(json.dumps(out, ensure_ascii=False, indent=2, default=str))
    if args.route_only:
        return 0
    return 1 if out["status"] == "generation_error" or not out["evidence"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
