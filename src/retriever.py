import pickle
import re
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder

if __package__:
    from .common import INDEX_BM25, INDEX_DENSE, load_jsonl, load_config
else:
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
        # This is a default, not a permanent restriction on future requests.
        # A cached retriever can run an ablation without reranking and then
        # enable it without loading the corpus or dense encoder again.
        self._default_rerank = use_reranker
        self.reranker = None

    def prepare_reranker(self):
        """Load once, optionally before timing a controlled retrieval study.

        This also works when warmup queries return no candidates, so their
        empty results cannot defer model initialization into measured queries.
        It does not change the default reranking preference for requests.
        """
        if self.reranker is None:
            self.reranker = CrossEncoder(self.cfg["retrieval"]["reranker"])
        return self.reranker

    @staticmethod
    def _positive_k(value, name):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
        return value

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
        self._positive_k(top_k, "top_k")
        scores = self.bm25.get_scores(tokenize(query))
        if not len(scores):
            return []
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
        self._positive_k(top_k, "top_k")
        if not self.faiss.ntotal:
            return []
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

    def retrieve(self, query, top_k=None, ticker=None, year=None, form=None, rerank=None,
                 mode="hybrid"):
        """Retrieve with a fixed candidate budget and an independent rerank switch.

        Omitting ``mode`` preserves hybrid retrieval. Omitting ``rerank`` uses
        the constructor's default; an explicit bool overrides it on each call.
        Pure modes expose the same id/rank/passage/sources contract as hybrid
        while retaining their native score instead of inventing an RRF score.
        Model/index failures propagate rather than changing the requested mode.
        """
        if mode not in {"bm25", "dense", "hybrid"}:
            raise ValueError("mode must be one of: bm25, dense, hybrid")
        if rerank is not None and not isinstance(rerank, bool):
            raise ValueError("rerank must be a bool or None")
        use_reranker = self._default_rerank if rerank is None else rerank
        rcfg = self.cfg["retrieval"]
        candidate_k = self._positive_k(rcfg["candidate_k"], "candidate_k")
        top_k = self._positive_k(rcfg["final_k"] if top_k is None else top_k, "top_k")
        if use_reranker:
            pool_k = self._positive_k(rcfg.get("rerank_pool_k", 30), "rerank_pool_k")
            if top_k > pool_k:
                raise ValueError("top_k must not exceed rerank_pool_k when reranking")

        if mode == "hybrid":
            sparse = self.bm25_search(query, candidate_k, ticker, year, form)
            dense = self.dense_search(query, candidate_k, ticker, year, form)
            ranked = self.rrf([("bm25", sparse), ("dense", dense)], k=rcfg["rrf_k"])
        else:
            search = self.bm25_search if mode == "bm25" else self.dense_search
            rows = search(query, candidate_k, ticker, year, form)
            ranked = [dict(row, rank=i, sources=[mode]) for i, row in enumerate(rows, 1)]

        if use_reranker and ranked:
            reranker = self.prepare_reranker()
            pool = ranked[:pool_k]
            pairs = [(query, r["passage"]["text"]) for r in pool]
            scores = reranker.predict(pairs)
            if len(scores) != len(pool):
                raise ValueError("Reranker returned a different number of scores than candidates")
            for r, score in zip(pool, scores):
                r["rerank_score"] = float(score)
            pool.sort(key=lambda x: x["rerank_score"], reverse=True)
            for i, r in enumerate(pool, 1):
                r["rank"] = i
            return pool[:top_k]

        return ranked[:top_k]
