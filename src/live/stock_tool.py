import yfinance as yf

def get_stock_snapshot(ticker):
    ticker = ticker.upper()
    t = yf.Ticker(ticker)
    hist = t.history(period="1mo", interval="1d", auto_adjust=True)
    if hist.empty:
        raise RuntimeError(f"No market data returned for {ticker}")

    latest = hist.iloc[-1]
    prev = hist.iloc[-2] if len(hist) >= 2 else latest
    first5 = hist.iloc[-6] if len(hist) >= 6 else hist.iloc[0]

    price = float(latest["Close"])
    prev_close = float(prev["Close"])
    return_1d = ((price / prev_close) - 1) * 100 if prev_close else None
    return_5d = ((price / float(first5["Close"])) - 1) * 100 if float(first5["Close"]) else None

    fast = t.fast_info
    def safe_get(key):
        try:
            v = fast[key]
            return float(v) if v is not None else None
        except Exception:
            return None

    return {
        "ticker": ticker,
        "latest_trading_date": str(hist.index[-1]),
        "price": price,
        "previous_close": prev_close,
        "return_1d_pct": return_1d,
        "return_5d_pct": return_5d,
        "volume": int(latest["Volume"]) if latest["Volume"] == latest["Volume"] else None,
        "market_cap": safe_get("marketCap"),
        "currency": fast.get("currency") if hasattr(fast, "get") else None,
    }

if __name__ == "__main__":
    import json, sys
    ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    print(json.dumps(get_stock_snapshot(ticker), indent=2, default=str))
