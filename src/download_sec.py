import os
import re
import time
import json
from pathlib import Path
from urllib.parse import urljoin

import requests
from dotenv import load_dotenv
from tqdm import tqdm

from common import ROOT, DATA_RAW, load_config, ensure_dirs

TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
ARCHIVES_BASE = "https://www.sec.gov/Archives/edgar/data/"

load_dotenv(ROOT / ".env")

def headers():
    ua = os.getenv("SEC_USER_AGENT", "").strip()
    if not ua or "your_email" in ua:
        raise RuntimeError(
            "Set SEC_USER_AGENT in .env, e.g. "
            'SEC_USER_AGENT="NTU AI6127 QuantRAG name@ntu.edu.sg"'
        )
    return {
        "User-Agent": ua,
        "Accept-Encoding": "gzip, deflate",
        "Accept": "application/json,text/html,*/*",
    }

def get_json(session, url, delay):
    time.sleep(delay)
    r = session.get(url, headers=headers(), timeout=60)
    r.raise_for_status()
    return r.json()

def get_text(session, url, delay):
    time.sleep(delay)
    r = session.get(url, headers=headers(), timeout=60)
    r.raise_for_status()
    return r.text

def ticker_to_cik(session, delay):
    raw = get_json(session, TICKER_MAP_URL, delay)
    return {
        row["ticker"].upper(): str(row["cik_str"]).zfill(10)
        for row in raw.values()
    }

def iter_recent_filings(submission, forms, start_date, end_date):
    recent = submission.get("filings", {}).get("recent", {})
    if not recent:
        return
    n = len(recent.get("accessionNumber", []))
    for i in range(n):
        form = recent["form"][i]
        date = recent["filingDate"][i]
        if form not in forms or not (start_date <= date < end_date):
            continue
        yield {
            "accession": recent["accessionNumber"][i],
            "filing_date": date,
            "report_date": recent.get("reportDate", [""] * n)[i],
            "form": form,
            "primary_document": recent["primaryDocument"][i],
        }

def filing_url(cik10, accession, primary_document):
    cik_nozero = str(int(cik10))
    acc_no_dash = accession.replace("-", "")
    return f"{ARCHIVES_BASE}{cik_nozero}/{acc_no_dash}/{primary_document}"

def safe_name(s):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s)

def main():
    ensure_dirs()
    cfg = load_config()["sec"]
    session = requests.Session()
    mapping = ticker_to_cik(session, cfg["request_delay_seconds"])

    manifest = []
    for ticker in cfg["tickers"]:
        cik = mapping.get(ticker.upper())
        if not cik:
            print(f"[WARN] No CIK found for {ticker}")
            continue

        submission = get_json(
            session,
            SUBMISSIONS_URL.format(cik=cik),
            cfg["request_delay_seconds"]
        )
        company = submission.get("name", ticker)
        filings = list(iter_recent_filings(
            submission,
            set(cfg["forms"]),
            cfg["start_date"],
            cfg["end_date"]
        ))

        for f in tqdm(filings, desc=ticker):
            out_dir = DATA_RAW / ticker / f["form"] / f["filing_date"]
            out_dir.mkdir(parents=True, exist_ok=True)
            out_file = out_dir / safe_name(f["primary_document"])

            if not out_file.exists():
                url = filing_url(cik, f["accession"], f["primary_document"])
                try:
                    html = get_text(session, url, cfg["request_delay_seconds"])
                    out_file.write_text(html, encoding="utf-8", errors="ignore")
                except Exception as e:
                    print(f"[WARN] Failed {ticker} {f['filing_date']}: {e}")
                    continue

            manifest.append({
                "ticker": ticker,
                "company": company,
                "cik": cik,
                "form": f["form"],
                "filing_date": f["filing_date"],
                "report_date": f["report_date"],
                "accession": f["accession"],
                "primary_document": f["primary_document"],
                "path": str(out_file.relative_to(ROOT))
            })

    manifest_path = DATA_RAW / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as fh:
        for row in manifest:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\nDownloaded/indexed filings: {len(manifest)}")
    print(f"Manifest: {manifest_path}")

if __name__ == "__main__":
    main()
