"""Offline retrieval contracts: no corpus, model downloads or ML packages needed."""

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, mock_open, patch


ROOT = Path(__file__).resolve().parents[1]


def offline_retriever_module():
    # Import a private copy so dependency stubs cannot poison a real retriever
    # imported elsewhere in this process (including Streamlit/AppTest).
    module_spec = importlib.util.spec_from_file_location(
        "src._retriever_offline", ROOT / "src" / "retriever.py",
    )
    module = importlib.util.module_from_spec(module_spec)
    numpy = types.ModuleType("numpy")
    numpy.argpartition = Mock()
    faiss = types.ModuleType("faiss")
    faiss.read_index = Mock(return_value=Mock(ntotal=3))
    transformers = types.ModuleType("sentence_transformers")
    transformers.SentenceTransformer = Mock(return_value=Mock())
    transformers.CrossEncoder = Mock()
    with patch.dict(sys.modules, {
        "numpy": numpy, "faiss": faiss, "sentence_transformers": transformers,
    }):
        module_spec.loader.exec_module(module)
    return module


def hit(passage_id, rank, score):
    return {"id": passage_id, "rank": rank, "score": score,
            "passage": {"id": passage_id, "text": f"Evidence {passage_id}"}}


class RetrievalModeTests(unittest.TestCase):
    def setUp(self):
        self.module = offline_retriever_module()
        self.cfg = {
            "dense": {"model": "offline-encoder"},
            "retrieval": {"candidate_k": 50, "final_k": 2, "rrf_k": 60,
                          "reranker": "offline-reranker"},
        }
        with patch.object(self.module, "load_config", return_value=self.cfg), \
                patch.object(self.module, "load_jsonl", return_value=[]), \
                patch("builtins.open", mock_open()), \
                patch.object(self.module.pickle, "load", return_value={"bm25": Mock(), "passages": []}):
            self.retriever = self.module.HybridRetriever(use_reranker=False)
        self.sparse = [hit("sparse", 1, 10.0), hit("shared", 2, 8.0)]
        self.dense = [hit("dense", 1, 0.9), hit("shared", 2, 0.8)]
        self.retriever.bm25_search = Mock(return_value=self.sparse)
        self.retriever.dense_search = Mock(return_value=self.dense)
        self.reranker = Mock()
        self.reranker.predict.side_effect = lambda pairs: list(range(len(pairs)))
        self.module.CrossEncoder.return_value = self.reranker

    def test_pure_modes_use_only_requested_backend_and_preserve_contract(self):
        for mode, expected, source in (("bm25", self.sparse, "bm25"), ("dense", self.dense, "dense")):
            with self.subTest(mode=mode):
                self.retriever.bm25_search.reset_mock()
                self.retriever.dense_search.reset_mock()
                rows = self.retriever.retrieve("query", mode=mode, rerank=False,
                                               ticker="AAPL", year=2024, form="10-K")
                selected = getattr(self.retriever, f"{mode}_search")
                other = self.retriever.dense_search if mode == "bm25" else self.retriever.bm25_search
                selected.assert_called_once_with("query", 50, "AAPL", 2024, "10-K")
                other.assert_not_called()
                self.assertEqual([r["id"] for r in rows], [r["id"] for r in expected])
                self.assertEqual([r["rank"] for r in rows], [1, 2])
                self.assertTrue(all(r["sources"] == [source] for r in rows))
                self.assertTrue(all("passage" in r and "score" in r for r in rows))
                self.assertTrue(all("rrf_score" not in r for r in rows))
        self.module.CrossEncoder.assert_not_called()

    def test_default_mode_is_hybrid_and_uses_same_candidate_budget(self):
        rows = self.retriever.retrieve("query", rerank=False)
        self.retriever.bm25_search.assert_called_once_with("query", 50, None, None, None)
        self.retriever.dense_search.assert_called_once_with("query", 50, None, None, None)
        self.assertEqual(rows[0]["id"], "shared")
        self.assertEqual(rows[0]["sources"], ["bm25", "dense"])
        self.assertIn("rrf_score", rows[0])

    def test_explicit_rerank_can_toggle_false_true_false_on_same_instance(self):
        for rerank, expected in ((False, ["shared", "sparse"]),
                                 (True, ["dense", "sparse"]),
                                 (False, ["shared", "sparse"]),
                                 (True, ["dense", "sparse"])):
            rows = self.retriever.retrieve("query", rerank=rerank)
            self.assertEqual([r["id"] for r in rows], expected)
            self.assertEqual([r["rank"] for r in rows], [1, 2])
            self.assertEqual("rerank_score" in rows[0], rerank)
        self.module.CrossEncoder.assert_called_once_with("offline-reranker")
        self.assertEqual(self.reranker.predict.call_count, 2)

    def test_omitted_switch_uses_constructor_default_after_explicit_override(self):
        self.retriever.retrieve("query")
        self.module.CrossEncoder.assert_not_called()
        self.retriever.retrieve("query", rerank=True)
        self.retriever.retrieve("query")
        self.assertEqual(self.reranker.predict.call_count, 1)
        self.retriever._default_rerank = True
        self.retriever.retrieve("query")
        self.assertEqual(self.reranker.predict.call_count, 2)

    def test_rerank_switch_is_independent_for_both_pure_modes(self):
        for mode, expected in (("bm25", "shared"), ("dense", "shared")):
            rows = self.retriever.retrieve("query", mode=mode, rerank=True)
            self.assertEqual(rows[0]["id"], expected)
            self.assertEqual(rows[0]["sources"], [mode])
            self.assertIn("rerank_score", rows[0])
        self.assertNotIn("rerank_score", self.sparse[0])
        self.assertNotIn("rerank_score", self.dense[0])

    def test_empty_results_do_not_load_reranker(self):
        self.retriever.bm25_search.return_value = []
        self.retriever.dense_search.return_value = []
        for mode in ("bm25", "dense", "hybrid"):
            self.assertEqual(self.retriever.retrieve("query", mode=mode, rerank=True), [])
        self.module.CrossEncoder.assert_not_called()

    def test_explicit_preparation_loads_once_even_when_warmup_ranking_is_empty(self):
        self.retriever.bm25_search.return_value = []
        self.retriever.dense_search.return_value = []
        self.assertIs(self.retriever.prepare_reranker(), self.reranker)
        self.assertEqual(self.retriever.retrieve("empty warmup", rerank=True), [])
        self.reranker.predict.assert_not_called()
        self.retriever.bm25_search.return_value = self.sparse
        self.retriever.retrieve("measured query", mode="bm25", rerank=True)
        self.assertIs(self.retriever.prepare_reranker(), self.reranker)
        self.module.CrossEncoder.assert_called_once_with("offline-reranker")
        self.reranker.predict.assert_called_once()
        self.assertFalse(self.retriever._default_rerank)

    def test_invalid_settings_fail_before_retrieval(self):
        for kwargs in ({"mode": "unknown"}, {"rerank": "false"},
                       {"top_k": 0}, {"top_k": -1}, {"top_k": True}, {"top_k": 1.5},
                       {"top_k": 31, "rerank": True}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.retriever.retrieve("query", **kwargs)
        self.retriever.bm25_search.assert_not_called()
        self.retriever.dense_search.assert_not_called()

    def test_configured_rerank_pool_is_used_and_enforced(self):
        self.cfg["retrieval"]["rerank_pool_k"] = 2
        self.retriever.retrieve("query", rerank=True)
        self.assertEqual(len(self.reranker.predict.call_args.args[0]), 2)
        with self.assertRaisesRegex(ValueError, "rerank_pool_k"):
            self.retriever.retrieve("query", top_k=3, rerank=True)

    def test_model_and_backend_failures_propagate_without_fallback(self):
        self.retriever.dense_search.side_effect = RuntimeError("dense failed")
        with self.assertRaisesRegex(RuntimeError, "dense failed"):
            self.retriever.retrieve("query", rerank=False)
        self.module.CrossEncoder.side_effect = RuntimeError("reranker unavailable")
        with self.assertRaisesRegex(RuntimeError, "reranker unavailable"):
            self.retriever.retrieve("query", mode="bm25", rerank=True)

    def test_reranker_must_return_a_score_for_each_candidate(self):
        self.reranker.predict.side_effect = None
        self.reranker.predict.return_value = []
        with self.assertRaisesRegex(ValueError, "number of scores"):
            self.retriever.retrieve("query", rerank=True)

    def test_empty_bm25_index_skips_argpartition(self):
        self.retriever.bm25.get_scores.return_value = []
        self.assertEqual(self.module.HybridRetriever.bm25_search(self.retriever, "query"), [])
        self.module.np.argpartition.assert_not_called()

    def test_empty_dense_index_skips_encoding_and_faiss_search(self):
        self.retriever.faiss.ntotal = 0
        self.assertEqual(self.module.HybridRetriever.dense_search(self.retriever, "query"), [])
        self.retriever.encoder.encode.assert_not_called()
        self.retriever.faiss.search.assert_not_called()


if __name__ == "__main__":
    unittest.main()
