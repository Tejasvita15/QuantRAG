import json
import re
from bs4 import BeautifulSoup
from tqdm import tqdm

from common import ROOT, DATA_RAW, DATA_PROCESSED, load_config, ensure_dirs, load_jsonl, write_jsonl

ITEM_RE = re.compile(r"\bITEM\s+(1A|1B|1C|1|2|3|4|5|6|7A|7|8|9A|9B|9C|9|10|11|12|13|14|15|16)\b", re.I)

def clean_text(text):
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n", text)
    return text.strip()

def html_to_text(html):
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "svg"]):
        tag.decompose()

    # Preserve table structure better than plain get_text:
    for table in soup.find_all("table"):
        rows = []
        for tr in table.find_all("tr"):
            cells = [clean_text(c.get_text(" ", strip=True)) for c in tr.find_all(["th", "td"])]
            if cells:
                rows.append(" | ".join(cells))
        if rows:
            table.replace_with("\nTABLE:\n" + "\n".join(rows) + "\n")

    return clean_text(soup.get_text("\n", strip=True))

def section_label(text, pos):
    window = text[max(0, pos - 3500):pos]
    matches = list(ITEM_RE.finditer(window))
    return matches[-1].group(0).upper() if matches else "UNKNOWN"

def chunk_text(text, chunk_words, overlap_words, min_words):
    words = text.split()
    step = max(1, chunk_words - overlap_words)
    out = []
    start = 0
    while start < len(words):
        end = min(len(words), start + chunk_words)
        chunk = " ".join(words[start:end]).strip()
        if len(chunk.split()) >= min_words:
            out.append((start, chunk))
        if end >= len(words):
            break
        start += step
    return out

def main():
    ensure_dirs()
    cfg = load_config()["chunking"]
    manifest_path = DATA_RAW / "manifest.jsonl"
    if not manifest_path.exists():
        raise FileNotFoundError("Run: python src/download_sec.py first")

    manifest = load_jsonl(manifest_path)
    passages = []
    seen = set()

    for filing in tqdm(manifest, desc="Chunking filings"):
        path = ROOT / filing["path"]
        html = path.read_text(encoding="utf-8", errors="ignore")
        text = html_to_text(html)
        chunks = chunk_text(text, cfg["chunk_words"], cfg["overlap_words"], cfg["min_words"])

        for i, (word_start, chunk) in enumerate(chunks):
            norm = re.sub(r"\W+", "", chunk.lower())
            sig = hash(norm)
            if sig in seen:
                continue
            seen.add(sig)

            approx_char = min(len(text)-1, max(0, int((word_start / max(1, len(text.split()))) * len(text))))
            section = section_label(text, approx_char)
            pid = f"{filing['ticker']}_{filing['filing_date']}_{filing['form'].replace('-', '')}_{i:05d}"

            passages.append({
                "id": pid,
                "text": chunk,
                "ticker": filing["ticker"],
                "company": filing["company"],
                "cik": filing["cik"],
                "form": filing["form"],
                "filing_date": filing["filing_date"],
                "report_date": filing["report_date"],
                "section": section,
                "source": "SEC",
                "accession": filing["accession"]
            })

    out = DATA_PROCESSED / "passages.jsonl"
    write_jsonl(out, passages)
    print(f"\nUnique passages: {len(passages):,}")
    print(f"Saved: {out}")

if __name__ == "__main__":
    main()
