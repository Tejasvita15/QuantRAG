import pickle
import re
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder

from common import INDEX_BM25, INDEX_DENSE, load_jsonl, load_config

def tokenize(text):
    return re.findall(r"[A-Za-z0-9$%.'-]+", text.lower())

class HybridRetriever:
    def __init__(self, use_reranker=True):
        cfg = load_config()
        self.cfg = cfg
        with open(INDEX_BM25 / "bm25.pkl", "rb") as f:
            data = pickle.load(f)
        self.bm25 = data["bm25"]
        self.passages = data["passages"]

        self.dense_meta = load_jsonl(INDEX_DENSE / "metadata.jsonl")
        self.faiss = faiss.read_index(str(INDEX_DENSE / "index.faiss"))
        self.encoder = SentenceTransformer(cfg["dense"]["model"])
        self.reranker = CrossEncoder(cfg["retrieval"]["reranker"]) if use_reranker else None

    @staticmethod
    def _passes_filters(p, ticker=None, year=None, form=None):
        if ticker and p.get("ticker", "").upper() != ticker.upper():
            return False
        if form and p.get("form", "").upper() != form.upper():
            return False
        if year and not p.get("filing_date", "").startswith(str(year)):
            return False
        return True

    def bm25_search(self, query, top_k=50, ticker=None, year=None, form=None):
        scores = self.bm25.get_scores(tokenize(query))
        # pull extra candidates when filtering
        n = min(len(scores), max(top_k * 20, top_k))
        idxs = np.argpartition(scores, -n)[-n:]
        idxs = idxs[np.argsort(scores[idxs])[::-1]]
        out = []
        for idx in idxs:
            p = self.passages[int(idx)]
            if self._passes_filters(p, ticker, year, form):
                out.append({"id": p["id"], "rank": len(out)+1, "score": float(scores[idx]), "passage": p})
                if len(out) >= top_k:
                    break
        return out

    def dense_search(self, query, top_k=50, ticker=None, year=None, form=None):
        q = self.encoder.encode([query], normalize_embeddings=True, convert_to_numpy=True).astype("float32")
        search_k = min(self.faiss.ntotal, max(top_k * 20, top_k))
        scores, idxs = self.faiss.search(q, search_k)
        out = []
        for idx, score in zip(idxs[0], scores[0]):
            if idx < 0:
                continue
            p = self.dense_meta[int(idx)]
            if self._passes_filters(p, ticker, year, form):
                out.append({"id": p["id"], "rank": len(out)+1, "score": float(score), "passage": p})
                if len(out) >= top_k:
                    break
        return out

    def rrf(self, lists, k=60):
        scores, docs, sources = {}, {}, {}
        for source, rows in lists:
            for r in rows:
                doc_id = r["id"]
                scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + r["rank"])
                docs[doc_id] = r["passage"]
                sources.setdefault(doc_id, []).append(source)
        ranked = sorted(scores, key=scores.get, reverse=True)
        return [
            {
                "id": doc_id,
                "rrf_score": scores[doc_id],
                "passage": docs[doc_id],
                "sources": sources[doc_id],
                "rank": i + 1
            }
            for i, doc_id in enumerate(ranked)
        ]

    def retrieve(self, query, top_k=None, ticker=None, year=None, form=None, rerank=True):
        rcfg = self.cfg["retrieval"]
        candidate_k = rcfg["candidate_k"]
        top_k = top_k or rcfg["final_k"]

        sparse = self.bm25_search(query, candidate_k, ticker, year, form)
        dense = self.dense_search(query, candidate_k, ticker, year, form)
        fused = self.rrf([("bm25", sparse), ("dense", dense)], k=rcfg["rrf_k"])

        if rerank and self.reranker and fused:
            pool = fused[: min(30, len(fused))]
            pairs = [(query, r["passage"]["text"]) for r in pool]
            scores = self.reranker.predict(pairs)
            for r, score in zip(pool, scores):
                r["rerank_score"] = float(score)
            pool.sort(key=lambda x: x["rerank_score"], reverse=True)
            for i, r in enumerate(pool, 1):
                r["rank"] = i
            return pool[:top_k]

        return fused[:top_k]
