import json
import re
from collections import Counter

from common import DATA_PROCESSED, load_jsonl

def main():
    path = DATA_PROCESSED / "passages.jsonl"
    rows = load_jsonl(path)
    vocab = set()
    total_words = 0
    tickers = Counter()
    forms = Counter()

    for r in rows:
        tokens = re.findall(r"[A-Za-z0-9$%.'-]+", r["text"].lower())
        total_words += len(tokens)
        vocab.update(tokens)
        tickers[r["ticker"]] += 1
        forms[r["form"]] += 1

    stats = {
        "passages": len(rows),
        "words": total_words,
        "types_unique_words": len(vocab),
        "companies": len(tickers),
        "passages_by_form": dict(forms),
        "requirement_10000_passages": len(rows) >= 10000,
        "requirement_100000_words": total_words >= 100000,
    }
    print(json.dumps(stats, indent=2))
    (DATA_PROCESSED / "corpus_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

if __name__ == "__main__":
    main()
