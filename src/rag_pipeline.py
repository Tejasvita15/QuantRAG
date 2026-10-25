"""Shared retrieval/live-data/generation orchestration, independent of the UI."""

import json
import time
from datetime import datetime, timezone
from urllib.parse import quote

if __package__:
    from .generator import GenerationError, GroundedGenerator
    from .query_router import plan_route
else:
    from generator import GenerationError, GroundedGenerator
    from query_router import plan_route


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


def _sec_url(passage):
    # The existing passage store has accession/CIK, not the original HTML URL.
    # Link to the filing index without guessing the primary document filename.
    cik = str(passage.get("cik", ""))
    accession = str(passage.get("accession", ""))
    import re
    if cik.isdigit() and re.fullmatch(r"\d{10}-\d{2}-\d{6}", accession):
        return (f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/"
                f"{accession.replace('-', '')}/{accession}-index.html")
    return None


class RAGPipeline:
    """Dependencies are lazy and injectable for reuse and offline tests.

    A failed live tool does not discard SEC evidence. No requested evidence
    produces an explicit insufficient-evidence result without calling an LLM.
    Raw provider exception strings are never exposed (FRED URLs contain keys).
    """

    def __init__(self, retriever=None, stock_tool=None, macro_tool=None, generator=None):
        self.retriever = retriever
        self.stock_tool = stock_tool
        self.macro_tool = macro_tool
        self.generator = generator if generator is not None else GroundedGenerator()

    def _retrieve(self, query, *, top_k, ticker, year, form, rerank):
        if self.retriever is None:
            if __package__:
                from .retriever import HybridRetriever
            else:
                from retriever import HybridRetriever
            # Retrieval loads the reranker lazily when a request asks for it.
            # In particular, the first request's switch is not a lifetime cap.
            self.retriever = HybridRetriever(use_reranker=False)
        return self.retriever.retrieve(
            query, top_k=top_k, ticker=ticker, year=year, form=form, rerank=rerank,
        )

    def _stock(self, ticker):
        if self.stock_tool is None:
            if __package__:
                from .live.stock_tool import get_stock_snapshot
            else:
                from live.stock_tool import get_stock_snapshot
            self.stock_tool = get_stock_snapshot
        return self.stock_tool(ticker)

    def _macro(self, series):
        if self.macro_tool is None:
            if __package__:
                from .live.fred_tool import get_fred_series
            else:
                from live.fred_tool import get_fred_series
            self.macro_tool = get_fred_series
        return self.macro_tool(series)

    def run(self, query, *, ticker=None, series=None, year=None, form=None,
            kind="auto", top_k=5, rerank=True, generate=True):
        if top_k < 1:
            raise ValueError("top_k must be positive")
        started = time.perf_counter()
        plan = plan_route(query, ticker=ticker, series=series, year=year, form=form, kind=kind)
        sources, errors, timings = [], [], {}
        results = {"filing": [], "stock": [], "macro": []}

        def fail(backend, target, message):
            errors.append({"backend": backend, "target": target, "error": message})

        for backend in plan.routes:
            t0 = time.perf_counter()
            if backend == "filing":
                seen = set()
                for symbol in plan.tickers or (None,):
                    try:
                        rows = self._retrieve(
                            query, top_k=top_k, ticker=symbol, year=plan.year,
                            form=plan.form, rerank=rerank,
                        )
                        if not rows:
                            fail("filing", symbol, "No SEC passages matched the query and filters")
                        for row in rows:
                            passage = row["passage"]
                            passage_id = passage["id"]
                            if passage_id in seen or not passage.get("text", "").strip():
                                continue
                            seen.add(passage_id)
                            results["filing"].append(row)
                            metadata = {k: v for k, v in passage.items() if k != "text"}
                            metadata.update({
                                "url": _sec_url(passage), "retrieved_at": _utc_now(),
                                "rank": row.get("rank"),
                                "retrieval_sources": row.get("sources", []),
                            })
                            sources.append({
                                "source_id": f"SEC{len(results['filing'])}",
                                "kind": "sec", "content": passage["text"], "metadata": metadata,
                            })
                    except Exception:
                        fail("filing", symbol, "SEC retrieval unavailable; check corpus, indexes and model dependencies")
            elif backend == "stock":
                if not plan.tickers:
                    fail("stock", None, "No ticker detected; specify --ticker")
                for symbol in plan.tickers:
                    try:
                        snapshot = self._stock(symbol)
                        if snapshot.get("price") is None or not snapshot.get("latest_trading_date"):
                            raise ValueError("Undated or empty stock snapshot")
                        results["stock"].append(snapshot)
                        sources.append({
                            "source_id": f"STOCK{len(results['stock'])}", "kind": "stock",
                            "content": json.dumps(snapshot, ensure_ascii=False),
                            "metadata": {
                                "ticker": symbol, "provider": "yfinance / Yahoo Finance",
                                "url": f"https://finance.yahoo.com/quote/{quote(symbol, safe='')}/",
                                "trading_date": snapshot["latest_trading_date"],
                                "retrieved_at": _utc_now(),
                                "price_basis": "latest available adjusted daily bar close; not a guaranteed real-time quote",
                            },
                        })
                    except Exception:
                        fail("stock", symbol, "Stock data unavailable; check ticker, provider and network")
            else:
                if not plan.series:
                    fail("macro", None, "No macro indicator detected; specify --series")
                for indicator in plan.series:
                    try:
                        data = self._macro(indicator)
                        rows = data.get("observations", [])
                        if not rows or not all(row.get("date") and row.get("value") is not None for row in rows):
                            raise ValueError("Undated or empty macro observations")
                        series_id = data["series_id"]
                        results["macro"].append(data)
                        sources.append({
                            "source_id": f"FRED{len(results['macro'])}", "kind": "macro",
                            "content": json.dumps(data, ensure_ascii=False),
                            "metadata": {
                                "series_id": series_id, "provider": "FRED",
                                "url": f"https://fred.stlouisfed.org/series/{quote(series_id, safe='')}",
                                "latest_observation_date": max(row["date"] for row in rows),
                                "retrieved_at": _utc_now(),
                                "note": "Values are raw series observations; CPI level is not an inflation percentage. Units are not supplied by this tool.",
                            },
                        })
                    except Exception:
                        fail("macro", indicator, "Macro data unavailable; check FRED_API_KEY, indicator and network")
            timings[backend] = round(time.perf_counter() - t0, 4)

        out = {
            "query": query, "route": plan.kind, "routing": plan.to_dict(),
            "results": results, "evidence": sources, "errors": errors,
        }
        if generate:
            t0 = time.perf_counter()
            try:
                out.update(self.generator.generate(
                    query, sources,
                    warnings=[f"{item['backend']} ({item['target'] or 'unspecified'}): {item['error']}" for item in errors],
                ))
            except GenerationError as exc:
                out.update({"status": "generation_error", "answer": None, "claims": [],
                            "citations": [], "sources": sources, "limitations": [],
                            "model": None, "usage": None})
                fail("generation", None, str(exc))
            timings["generation"] = round(time.perf_counter() - t0, 4)
        else:
            out["status"] = "retrieved" if sources else "insufficient_evidence"
        out["timings_s"] = timings
        out["latency_s"] = round(time.perf_counter() - started, 4)
        return out
