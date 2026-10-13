"""Rule-based query router.

Classifies a natural-language query into one of the three backends used
by src/query.py:

- "stock"  -> yfinance live market data (src/live/stock_tool.py)
- "macro"  -> FRED macro observations (src/live/fred_tool.py)
- "filing" -> SEC 10-K corpus retrieval (src/retriever.py)

The router is intentionally heuristic: company names and ticker symbols
are matched against the tickers in config.json, and macro indicators are
matched by keyword. It can later be replaced by an LLM function-calling
router without changing src/query.py.
"""

import re

from common import load_config

STOCK_KEYWORDS = (
    "price", "stock", "share", "trading", "quote", "market cap",
    "股价", "股票", "行情", "市值",
)

MACRO_SERIES = {
    "cpi": ("cpi", "inflation", "consumer price"),
    "unemployment": ("unemployment", "jobless", "jobs report"),
    "fed_funds": ("fed funds", "federal funds", "interest rate", "fomc"),
    "gdp": ("gdp", "economic growth", "recession"),
    "treasury_10y": ("treasury", "10-year", "10y", "bond yield"),
}

COMPANY_NAMES = {
    "apple": "AAPL", "microsoft": "MSFT", "nvidia": "NVDA",
    "amazon": "AMZN", "google": "GOOGL", "alphabet": "GOOGL",
    "meta": "META", "facebook": "META", "tesla": "TSLA",
    "jpmorgan": "JPM", "bank of america": "BAC", "walmart": "WMT",
    "exxon": "XOM", "chevron": "CVX", "johnson & johnson": "JNJ",
    "procter & gamble": "PG", "coca-cola": "KO", "coke": "KO",
    "pepsico": "PEP", "costco": "COST", "home depot": "HD",
    "disney": "DIS", "netflix": "NFLX", "ibm": "IBM",
    "oracle": "ORCL", "intel": "INTC", "amd": "AMD",
    "cisco": "CSCO", "qualcomm": "QCOM", "goldman sachs": "GS",
    "morgan stanley": "MS", "visa": "V", "mastercard": "MA",
}


def known_tickers():
    """Return the set of ticker symbols configured in config.json."""
    return set(load_config()["sec"]["tickers"])


def detect_ticker(query):
    """Find a stock ticker mentioned in a query.

    Input:
        query (str): free-form question.

    Output:
        str | None: uppercase ticker symbol (e.g. "AAPL"), or None when no
        configured company name or ticker symbol is mentioned.
    """
    lower = query.lower()
    for name, ticker in COMPANY_NAMES.items():
        if re.search(rf"\b{re.escape(name)}\b", lower):
            return ticker
    tickers = known_tickers()
    for token in re.findall(r"\b[A-Z]{2,5}\b", query):
        if token in tickers:
            return token
    return None


def detect_macro(query):
    """Find a FRED macro indicator mentioned in a query.

    Input:
        query (str): free-form question.

    Output:
        str | None: FRED series alias (cpi, unemployment, fed_funds, gdp,
        treasury_10y), or None when no indicator keyword is detected.
    """
    lower = query.lower()
    for series, keywords in MACRO_SERIES.items():
        if any(k in lower for k in keywords):
            return series
    return None


def has_stock_intent(query):
    """Check whether the query asks about live market data.

    Input:
        query (str): free-form question.

    Output:
        bool: True when a stock-related keyword is present.
    """
    return any(k in query.lower() for k in STOCK_KEYWORDS)


def route(query):
    """Rule-based router.

    Input:
        query (str): free-form question.

    Output:
        tuple[str, dict]: (route, hint) where route is one of:
            ("stock",  {"ticker": t})   live market data via yfinance
            ("macro",  {"series": s})   FRED macro observations
            ("filing", {"ticker": t})   SEC corpus retrieval
    """
    query = query or ""
    ticker = detect_ticker(query)
    if ticker and has_stock_intent(query):
        return "stock", {"ticker": ticker}
    if ticker:
        return "filing", {"ticker": ticker}
    series = detect_macro(query)
    if series:
        return "macro", {"series": series}
    return "filing", {}
