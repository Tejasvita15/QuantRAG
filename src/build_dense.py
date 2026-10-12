import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from common import DATA_PROCESSED, INDEX_DENSE, load_jsonl, load_config, ensure_dirs

def main():
    ensure_dirs()
    cfg = load_config()["dense"]
    passages = load_jsonl(DATA_PROCESSED / "passages.jsonl")
    texts = [p["text"] for p in passages]

    print(f"Loading {cfg['model']}...")
    model = SentenceTransformer(cfg["model"])
    embeddings = model.encode(
        texts,
        batch_size=cfg["batch_size"],
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype("float32")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, str(INDEX_DENSE / "index.faiss"))

    with open(INDEX_DENSE / "metadata.jsonl", "w", encoding="utf-8") as f:
        for p in passages:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"Vectors: {index.ntotal:,}; dim={embeddings.shape[1]}")
    print(f"Saved: {INDEX_DENSE / 'index.faiss'}")

if __name__ == "__main__":
    main()
