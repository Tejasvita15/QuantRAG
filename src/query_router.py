"""Deterministic routing for SEC, stock, macro and mixed questions.

Routing never calls a model or provider. ``plan_route`` is the shared CLI/UI
interface; ``route`` retains the original tuple interface.
"""

import re
from dataclasses import asdict, dataclass

if __package__:
    from .common import load_config
else:
    from common import load_config


COMPANY_NAMES = {
    "apple": "AAPL", "microsoft": "MSFT", "nvidia": "NVDA",
    "amazon": "AMZN", "google": "GOOGL", "alphabet": "GOOGL",
    "meta": "META", "facebook": "META", "tesla": "TSLA",
    "jpmorgan": "JPM", "jp morgan": "JPM", "bank of america": "BAC",
    "walmart": "WMT", "exxon": "XOM", "chevron": "CVX",
    "johnson & johnson": "JNJ", "procter & gamble": "PG",
    "coca-cola": "KO", "coke": "KO", "pepsico": "PEP", "costco": "COST",
    "home depot": "HD", "disney": "DIS", "netflix": "NFLX", "ibm": "IBM",
    "oracle": "ORCL", "intel": "INTC", "amd": "AMD", "cisco": "CSCO",
    "qualcomm": "QCOM", "goldman sachs": "GS", "morgan stanley": "MS",
    "visa": "V", "mastercard": "MA",
}

MACRO_SERIES = {
    "cpi": ("cpi", "cpiaucsl", "inflation", "consumer price"),
    "unemployment": ("unemployment", "unrate", "jobless", "jobs report"),
    "fed_funds": ("fed funds", "federal funds", "fedfunds", "interest rate",
                  "interest-rate", "fomc"),
    "gdp": ("gdp", "economic growth", "recession"),
    "treasury_10y": ("treasury", "dgs10", "10-year", "10 year", "10y",
                     "bond yield"),
}

STOCK_KEYWORDS = (
    "stock", "stocks", "stock price", "share price", "trading", "quote",
    "market cap", "market capitalization",
)
FILING_KEYWORDS = (
    "filing", "filings", "10-k", "10-q", "10k", "10q", "annual report",
    "report", "reported", "reports", "discuss", "discussed", "discusses",
    "risk", "risks", "revenue", "revenues", "margin", "margins",
    "profitability", "supply chain", "supply-chain", "fundamentals",
    "business strategy", "cash flow", "balance sheet", "earnings",
)
LIVE_KEYWORDS = (
    "current", "currently", "latest", "recent", "today", "now", "live",
)


def _matches(query, term):
    if term.isascii():
        return bool(re.search(rf"(?<!\w){re.escape(term)}(?!\w)", query, re.I))
    return term in query


def _any(query, terms):
    return any(_matches(query, term) for term in terms)


def known_tickers():
    return set(load_config()["sec"]["tickers"])


def detect_tickers(query):
    """Return symbols in mention order, including explicit $SYMBOLs.

    Bare symbols use the configured universe. Short ambiguous symbols
    (V, MA, MS, etc.) must be uppercase or have a dollar prefix.
    """
    mentions = []
    for name, ticker in COMPANY_NAMES.items():
        pattern = rf"(?<!\w){re.escape(name)}(?!\w)" if name.isascii() else re.escape(name)
        mentions.extend((m.start(), ticker) for m in re.finditer(pattern, query, re.I))
    for m in re.finditer(r"\$([A-Za-z][A-Za-z0-9.-]{0,9})\b", query):
        mentions.append((m.start(), m.group(1).upper()))
    tickers = known_tickers()
    for m in re.finditer(r"(?<!\w)[A-Za-z][A-Za-z0-9.-]*(?!\w)", query):
        token = m.group()
        if token.upper() in tickers and (
            token.isupper() or (len(token) >= 4 and token.upper() != "COST")
        ):
            mentions.append((m.start(), token.upper()))
    return tuple(dict.fromkeys(ticker for _, ticker in sorted(mentions)))


def detect_ticker(query):
    tickers = detect_tickers(query)
    return tickers[0] if tickers else None


def detect_macros(query):
    return tuple(series for series, keywords in MACRO_SERIES.items() if _any(query, keywords))


def detect_macro(query):
    series = detect_macros(query)
    return series[0] if series else None


def has_stock_intent(query):
    # Remove filing phrases only, keeping a later 'latest stock price' clause.
    cleaned = re.sub(
        r"\b(?:stock[ -]based|share[ -]based|stock compensation|stock options|"
        r"stock repurchases?|share repurchases?|stock splits?|stock ownership)\b",
        "", query, flags=re.I,
    )
    if _any(cleaned, STOCK_KEYWORDS):
        return True
    if _matches(cleaned, "price") and not _any(cleaned, FILING_KEYWORDS + (
        "product", "products", "commodity", "commodities", "oil", "gas", "consumer price",
    )):
        return True
    return bool(re.search(r"\b(?:current|latest|trading)\s+price\b", cleaned, re.I))


@dataclass(frozen=True)
class RoutePlan:
    routes: tuple
    tickers: tuple = ()
    series: tuple = ()
    year: int | None = None
    form: str | None = None
    missing_inputs: tuple = ()

    @property
    def kind(self):
        return self.routes[0] if len(self.routes) == 1 else "mixed"

    def to_dict(self):
        result = asdict(self)
        for field in ("routes", "tickers", "series", "missing_inputs"):
            result[field] = list(result[field])
        return {"kind": self.kind, **result}


def plan_route(query, *, ticker=None, series=None, year=None, form=None, kind="auto"):
    """Plan only needed tools, retaining SEC evidence for mixed questions.

    Explicit filters override inferred metadata. A single mentioned year is a
    filing-date filter (the retriever's existing semantics); multi-year trends
    are left unfiltered. Historical filing discussion of rates/inflation alone
    must not trigger a latest-observation FRED request.
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Query must be a non-empty string")
    if kind not in ("auto", "filing", "stock", "macro"):
        raise ValueError("Route must be auto, filing, stock or macro")
    if ticker and not re.fullmatch(r"[A-Za-z][A-Za-z0-9.-]{0,9}", ticker):
        raise ValueError("Invalid ticker symbol")
    tickers = (ticker.upper(),) if ticker else detect_tickers(query)
    indicators = (series,) if series else detect_macros(query)
    years = set(re.findall(r"\b(?:19|20)\d{2}\b", query))
    inferred_year = int(next(iter(years))) if len(years) == 1 else None
    forms = {value.upper() for value in re.findall(r"\b10-?([KQ])\b", query, re.I)}
    inferred_form = "10-" + next(iter(forms)).upper() if len(forms) == 1 else None
    filing = _any(query, FILING_KEYWORDS) or bool(inferred_form)
    # 'Latest annual report' describes the filing, not a request for today's
    # macro/market observations. Keep recency scoped away from those phrases.
    live_query = re.sub(
        r"\b(?:current|latest|recent)\s+(?:10-?[KQ]|(?:annual |quarterly )?reports?|filings?)\b",
        "", query, flags=re.I,
    )
    live = _any(live_query, LIVE_KEYWORDS)
    separate_market_request = bool(re.search(
        r"\b(?:and|also)\s+(?:(?:what is|show|get)\s+)?(?:its?\s+)?"
        r"(?:stock price|share price|quote|market cap)",
        query, re.I,
    ))
    stock = has_stock_intent(query) and (not filing or live or separate_market_request)
    macro = bool(indicators) and (bool(series) or not filing or live)
    # A macro-only jobs report is not a company filing.
    explicit_filing = bool(inferred_form) or _any(query, (
        "filing", "filings", "annual report", "annual reports",
    ))
    use_filing = explicit_filing or (filing and bool(tickers)) or not (stock or macro)
    routes = tuple(name for name, use in (
        ("filing", use_filing), ("stock", stock), ("macro", macro),
    ) if use) if kind == "auto" else (kind,)
    missing = []
    if "stock" in routes and not tickers:
        missing.append("ticker")
    if "macro" in routes and not indicators:
        missing.append("series")
    return RoutePlan(
        routes, tickers, indicators, year if year is not None else inferred_year,
        form.upper() if form else inferred_form, tuple(missing),
    )


def route(query):
    """Original tuple API; mixed hints include the full route plan."""
    plan = plan_route(query)
    hints = {}
    if plan.tickers:
        hints["ticker"] = plan.tickers[0]
    if plan.series:
        hints["series"] = plan.series[0]
    if plan.kind == "mixed":
        hints.update(plan.to_dict())
    return plan.kind, hints
