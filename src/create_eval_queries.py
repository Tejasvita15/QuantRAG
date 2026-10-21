import json
from pathlib import Path

OUTPUT = Path("eval/queries.jsonl")

companies = {
    "AAPL": "Apple",
    "MSFT": "Microsoft",
    "NVDA": "NVIDIA",
    "AMZN": "Amazon",
    "GOOGL": "Alphabet",
    "META": "Meta",
    "TSLA": "Tesla",
    "WMT": "Walmart",
    "CVX": "Chevron",
    "JNJ": "Johnson & Johnson",
    "PG": "Procter & Gamble",
    "KO": "Coca-Cola",
    "PEP": "PepsiCo",
    "COST": "Costco",
    "HD": "Home Depot",
    "DIS": "Disney",
    "NFLX": "Netflix",
    "IBM": "IBM",
    "ORCL": "Oracle",
    "INTC": "Intel",
    "AMD": "AMD",
    "CSCO": "Cisco",
    "QCOM": "Qualcomm",
    "V": "Visa",
    "MA": "Mastercard",
}


# Four deliberately different information needs per company.
# 25 companies × 4 queries = 100 queries.
templates = [
    (
        "revenue_growth",
        "What factors affected {company}'s revenue growth?"
    ),
    (
        "risk",
        "What major business risks does {company} identify in its annual filing?"
    ),
    (
        "profitability",
        "What factors affected {company}'s profitability or operating margin?"
    ),
    (
        "operations_strategy",
        "What operational or strategic factors does {company} identify as important to its business performance?"
    ),
]


def main():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    qid = 1

    for ticker, company in companies.items():
        for category, template in templates:
            rows.append({
                "query_id": f"q{qid:03d}",
                "ticker": ticker,
                "category": category,
                "query": template.format(company=company),
            })
            qid += 1

    assert len(rows) == 100

    with open(OUTPUT, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Saved: {OUTPUT}")
    print(f"Queries: {len(rows)}")

    # Verification
    per_category = {}
    per_ticker = {}

    for row in rows:
        per_category[row["category"]] = (
            per_category.get(row["category"], 0) + 1
        )
        per_ticker[row["ticker"]] = (
            per_ticker.get(row["ticker"], 0) + 1
        )

    print("\nQueries per category:")
    for category, count in per_category.items():
        print(f"  {category}: {count}")

    print("\nQueries per company:")
    for ticker, count in per_ticker.items():
        print(f"  {ticker}: {count}")


if __name__ == "__main__":
    main()