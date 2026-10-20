import pickle
import re
from rank_bm25 import BM25Okapi

from common import DATA_PROCESSED, INDEX_BM25, load_jsonl, ensure_dirs

def tokenize(text):
    return re.findall(r"[A-Za-z0-9$%.'-]+", text.lower())

def main():
    ensure_dirs()
    passages = load_jsonl(DATA_PROCESSED / "passages.jsonl")
    corpus_tokens = [tokenize(p["text"]) for p in passages]
    print(f"Building BM25 over {len(passages):,} passages...")
    bm25 = BM25Okapi(corpus_tokens)
    with open(INDEX_BM25 / "bm25.pkl", "wb") as f:
        pickle.dump({"bm25": bm25, "passages": passages}, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved: {INDEX_BM25 / 'bm25.pkl'}")

if __name__ == "__main__":
    main()
