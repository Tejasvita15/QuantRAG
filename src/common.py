import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw" / "sec"
DATA_PROCESSED = ROOT / "data" / "processed"
INDEX_BM25 = ROOT / "indexes" / "bm25"
INDEX_DENSE = ROOT / "indexes" / "dense"
EVAL_DIR = ROOT / "eval"

def load_config():
    with open(ROOT / "config.json", "r", encoding="utf-8") as f:
        return json.load(f)

def ensure_dirs():
    for p in [DATA_RAW, DATA_PROCESSED, INDEX_BM25, INDEX_DENSE, EVAL_DIR]:
        p.mkdir(parents=True, exist_ok=True)

def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def write_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
