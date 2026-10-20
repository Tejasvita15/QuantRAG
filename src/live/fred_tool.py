import os
import requests
from dotenv import load_dotenv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

SERIES = {
    "cpi": "CPIAUCSL",
    "unemployment": "UNRATE",
    "fed_funds": "FEDFUNDS",
    "gdp": "GDP",
    "treasury_10y": "DGS10",
}

def get_fred_series(name_or_id, limit=12):
    api_key = os.getenv("FRED_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Set FRED_API_KEY in .env")
    series_id = SERIES.get(name_or_id.lower(), name_or_id.upper())
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "sort_order": "desc",
        "limit": int(limit),
    }
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    rows = []
    for x in r.json().get("observations", []):
        if x.get("value") != ".":
            rows.append({"date": x["date"], "value": float(x["value"])})
    return {"series_id": series_id, "observations": rows}

if __name__ == "__main__":
    import json, sys
    key = sys.argv[1] if len(sys.argv) > 1 else "fed_funds"
    print(json.dumps(get_fred_series(key), indent=2))
