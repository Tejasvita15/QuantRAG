"""Lightweight UI validation and diagnostics; never loads models or providers."""

import os
import re
from pathlib import Path
from urllib.parse import urlsplit

from .common import ROOT


def build_options(ticker, series, year, form, kind, top_k, rerank):
    ticker = ticker.strip().upper() or None
    series = series.strip() or None
    year = year.strip()
    if year and not re.fullmatch(r"(?:19|20)\d{2}", year):
        raise ValueError("Filing-date year must be four digits between 1900 and 2099, or blank.")
    if series and not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,79}", series):
        raise ValueError("Use one FRED alias or series ID, e.g. fed_funds or UNRATE.")
    if form not in ("Auto", "10-K", "10-Q"):
        raise ValueError("Filing form must be Auto, 10-K or 10-Q.")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= 20:
        raise ValueError("Passages per company must be between 1 and 20.")
    return {"ticker": ticker, "series": series, "year": int(year) if year else None,
            "form": None if form == "Auto" else form, "kind": kind,
            "top_k": top_k, "rerank": bool(rerank)}


def readiness(root=ROOT, environ=None):
    env = os.environ if environ is None else environ
    root = Path(root)
    paths = ("indexes/bm25/bm25.pkl", "indexes/dense/index.faiss", "indexes/dense/metadata.jsonl")
    return {
        "SEC index files present": all((root / path).is_file() for path in paths),
        "FRED key configured": bool(env.get("FRED_API_KEY", "").strip()),
        "LLM model + key configured": bool(env.get("LLM_MODEL", "").strip()
                                            and (env.get("LLM_API_KEY", "").strip()
                                                 or env.get("OPENAI_API_KEY", "").strip())),
    }


def safe_source_url(value):
    """Only link to the three known evidence providers, never arbitrary schemes."""
    if not isinstance(value, str):
        return None
    try:
        parts = urlsplit(value)
        if (parts.scheme == "https" and parts.hostname in
                {"www.sec.gov", "sec.gov", "finance.yahoo.com", "fred.stlouisfed.org"}
                and not parts.username and not parts.password and parts.port in (None, 443)
                and not parts.query and not parts.fragment):
            return value
    except ValueError:
        pass
    return None
