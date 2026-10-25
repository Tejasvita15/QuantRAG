"""Offline contracts only: synthetic labels/results are never final experiments."""

import contextlib
import io
import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from src.run_ablation import (
    INDEX_FILES, VARIANTS, ValidationError, audit_corpus, load_qrels, main,
    retrieval_metrics, run_experiment,
)


def judgment(query_id="q1", **extra):
    return {"query_id": query_id, "query": f"Apple risk question {query_id}", "ticker": "AAPL",
            "year": 2024, "form": "10-K", "relevance": {"p1": 2, "p2": 1, "p3": 0}, **extra}


class FakeRetriever:
    def __init__(self, use_reranker=False):
        self.cfg = {"retrieval": {"candidate_k": 50, "rerank_pool_k": 30, "rrf_k": 60}}
        self.calls = []
        self.passages = [{"id": f"p{i}", "ticker": "AAPL", "filing_date": "2024-11-01", "form": "10-K",
                          "text": f"Full synthetic passage {i}. " * 100} for i in range(1, 13)]

    def retrieve(self, query, *, top_k, mode, rerank, **filters):
        self.calls.append({"query": query, "mode": mode, "rerank": rerank, "filters": filters,
                           "top_k": top_k, "candidate_k": self.cfg["retrieval"]["candidate_k"]})
        rows = self.passages[:top_k]
        if mode == "dense":
            rows = list(reversed(rows))
        elif mode == "hybrid" and rerank:
            rows = [rows[1], rows[0], *rows[2:]]
        return [{"id": row["id"], "rank": rank, "passage": row,
                 "sources": [mode], "score": float(top_k - rank)} for rank, row in enumerate(rows, 1)]


class QrelsFixture:
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.path = self.root / "human_heldout.jsonl"

    def tearDown(self):
        self.tmp.cleanup()

    def write_rows(self, *rows):
        self.path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
        return self.path


class QrelsTests(QrelsFixture, unittest.TestCase):
    def test_normalizes_filters_and_ignores_unrelated_metadata(self):
        path = self.write_rows(judgment(ticker="aapl", annotator="not copied to results"))
        rows = load_qrels(path)
        self.assertEqual(rows[0]["filters"], {"ticker": "AAPL", "year": 2024, "form": "10-K"})
        self.assertNotIn("annotator", rows[0])

    def test_empty_duplicate_and_invalid_schema(self):
        invalid = [None, {}, judgment(query_id=" "), judgment(query=" "), judgment(relevance={}),
                   judgment(relevance={"p1": 0}), judgment(relevance={"p1": True}),
                   judgment(relevance={"p1": 3}), judgment(relevance={"p1": 1.0}),
                   judgment(relevance={"REPLACE_WITH_PASSAGE_ID": 2}),
                   judgment(relevance={"p1 ": 2}), judgment(ticker="bad/symbol"),
                   judgment(year=True), judgment(year="2024"), judgment(form="10k")]
        for row in invalid:
            with self.subTest(row=row):
                self.write_rows(row)
                with self.assertRaises(ValidationError):
                    load_qrels(self.path)
        self.write_rows(judgment(), judgment())
        with self.assertRaisesRegex(ValidationError, "duplicate query_id"):
            load_qrels(self.path)
        self.path.write_text("\n\n", encoding="utf-8")
        with self.assertRaisesRegex(ValidationError, "no evaluation records"):
            load_qrels(self.path)

    def test_duplicate_json_keys_and_examples_are_rejected(self):
        self.path.write_text('{"query_id":"q1","query_id":"q2"}', encoding="utf-8")
        with self.assertRaises(ValidationError):
            load_qrels(self.path)
        example = self.root / "retrieval_eval.example.jsonl"
        example.write_text(json.dumps(judgment()), encoding="utf-8")
        with self.assertRaisesRegex(ValidationError, "Example/demo"):
            load_qrels(example)

    def test_missing_and_nonobject_input(self):
        with self.assertRaisesRegex(ValidationError, "does not exist"):
            load_qrels(self.path)
        self.path.write_text("not JSON", encoding="utf-8")
        with self.assertRaisesRegex(ValidationError, "invalid JSON"):
            load_qrels(self.path)

    def test_duplicate_normalized_question_rejected_but_different_filters_allowed(self):
        self.write_rows(judgment(query="Apple  supply\tchain risks"),
                        judgment("q2", query="  APPLE supply chain RISKS "))
        with self.assertRaisesRegex(ValidationError, "duplicate normalized question"):
            load_qrels(self.path)
        self.write_rows(judgment(query="Apple risk"),
                        judgment("q2", query="APPLE RISK", year=2023))
        self.assertEqual(len(load_qrels(self.path)), 2)


class MetricAndExperimentTests(QrelsFixture, unittest.TestCase):
    def examples(self, *rows):
        return load_qrels(self.write_rows(*(rows or (judgment(),))))

    def test_hand_calculated_metrics_and_unjudged_coverage(self):
        result = retrieval_metrics(["p3", "p1", "unknown", "p2"], {"p1": 2, "p2": 1, "p3": 0}, 10)
        self.assertEqual(result["precision_at_5"], 2 / 5)
        self.assertEqual(result["precision_at_k"], 2 / 10)
        self.assertEqual(result["recall_at_10"], 1)
        self.assertEqual(result["mrr_at_k"], 1 / 2)
        expected = (3 / math.log2(3) + 1 / math.log2(5)) / (3 + 1 / math.log2(3))
        self.assertAlmostEqual(result["ndcg_at_10"], expected)
        self.assertEqual(result["judged_coverage_at_k"], 3 / 4)
        self.assertEqual(result["unjudged_at_k"], 1)

    def test_empty_ranking_retains_fixed_denominators(self):
        metrics = retrieval_metrics([], {"p1": 2}, 10)
        self.assertTrue(all(value == 0 for value in metrics.values()))

    def test_four_variants_filters_fixed_budget_warmup_and_repeats(self):
        retriever = FakeRetriever()
        ticks = iter(range(1000))
        result = run_experiment(self.examples(judgment(), judgment("q2")), retriever,
                                repeats=2, warmup=1, candidate_k=40, timer=lambda: next(ticks))
        self.assertEqual(result["status"], "completed")
        self.assertEqual(len(result["warmup_runs"]), 4)
        self.assertEqual(len(result["runs"]), 16)
        self.assertEqual(len(retriever.calls), 20)
        self.assertEqual({row["variant"] for row in result["runs"]}, {row[0] for row in VARIANTS})
        self.assertTrue(all(row["latency_s"] == 1 for row in result["runs"]))
        self.assertTrue(all(row["candidate_k"] == 40 and row["top_k"] == 10 for row in retriever.calls))
        self.assertTrue(all(row["filters"] == {"ticker": "AAPL", "year": 2024, "form": "10-K"} for row in retriever.calls))
        self.assertTrue(all(row["measured_calls"] == 4 for row in result["summary"]))
        self.assertGreater(len(result["runs"][0]["ranking"][0]["passage"]["text"]), 600)
        self.assertTrue(any("Unjudged" in warning for warning in result["warnings"]))
        self.assertTrue(any("small judgment pool" in warning for warning in result["warnings"]))

    def test_execution_order_reproducible_and_rankings_frozen(self):
        examples = self.examples(judgment(), judgment("q2"))
        one, two = FakeRetriever(), FakeRetriever()
        result = run_experiment(examples, one, repeats=2, warmup=0, seed=1)
        other = run_experiment(examples, two, repeats=2, warmup=0, seed=1)
        self.assertEqual([(r["query_id"], r["variant"]) for r in result["runs"]],
                         [(r["query_id"], r["variant"]) for r in other["runs"]])
        original_text = result["runs"][0]["ranking"][0]["passage"]["text"]
        for passage in one.passages:
            passage["text"] = "changed after run"
        self.assertEqual(result["runs"][0]["ranking"][0]["passage"]["text"], original_text)

    def test_corpus_id_and_filter_mismatch_fail_before_retrieval(self):
        retriever = FakeRetriever()
        for row in (judgment(relevance={"old_id": 2}), judgment(year=2023), judgment(ticker="MSFT")):
            with self.subTest(row=row), self.assertRaises(ValidationError):
                run_experiment(self.examples(row), retriever)
        self.assertEqual(retriever.calls, [])

    def test_invalid_budget_and_malformed_ranking(self):
        examples = self.examples()
        with self.assertRaises(ValidationError):
            run_experiment([], FakeRetriever())
        for kwargs in ({"top_k": 5}, {"top_k": 31}, {"repeats": 0}, {"warmup": -1}, {"candidate_k": 9}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValidationError):
                run_experiment(examples, FakeRetriever(), **kwargs)
        retriever = FakeRetriever()
        retriever.retrieve = lambda *args, **kwargs: [{"id": "p1", "rank": 2, "passage": retriever.passages[0]}]
        with self.assertRaisesRegex(ValidationError, "IDs/ranks"):
            run_experiment(examples, retriever)

    def test_empty_first_warmup_still_prepares_lazy_reranker_before_measurement(self):
        class LazyFake(FakeRetriever):
            def __init__(self):
                super().__init__()
                self.prepared = False
                self.loads_during_retrieve = 0

            def prepare_reranker(self):
                self.prepared = True

            def retrieve(self, query, **kwargs):
                if query.endswith("q1"):
                    return []
                if kwargs["rerank"] and not self.prepared:
                    self.loads_during_retrieve += 1
                    self.prepared = True
                return super().retrieve(query, **kwargs)

        retriever = LazyFake()
        result = run_experiment(self.examples(judgment(), judgment("q2")), retriever, repeats=1, warmup=1)
        self.assertTrue(result["preparation"]["reranker_explicitly_prepared"])
        self.assertEqual(retriever.loads_during_retrieve, 0)
        self.assertTrue(all(row["retrieved_count"] == 0 for row in result["warmup_runs"]))
        self.assertTrue(any("first-inference" in warning for warning in result["warnings"]))


class CLITests(QrelsFixture, unittest.TestCase):
    def setup_artifacts(self):
        for name in INDEX_FILES:
            target = self.root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"synthetic offline artifact, not a real index")
        (self.root / "config.json").write_text(json.dumps({
            "dense": {"model": "offline-test"},
            "retrieval": {"candidate_k": 50, "rrf_k": 60, "rerank_pool_k": 30, "reranker": "offline-test"},
            "secret_unused_config": "DO_NOT_COPY_THIS_VALUE",
        }), encoding="utf-8")

    def test_check_only_never_builds_retriever_or_writes_output(self):
        self.write_rows(judgment())
        called = []
        with contextlib.redirect_stdout(io.StringIO()) as output:
            status = main(["--file", str(self.path), "--check-only"],
                          retriever_factory=lambda **kwargs: called.append(kwargs), root=self.root)
        self.assertEqual(status, 0)
        self.assertEqual(json.loads(output.getvalue())["status"], "qrels_validated_only")
        self.assertEqual(called, [])
        self.assertEqual(set(self.root.iterdir()), {self.path})

    def test_check_only_and_both_entrypoints_without_site_packages(self):
        self.write_rows(judgment())
        root = Path(__file__).resolve().parents[1]
        for entry in (["-m", "src.run_ablation"], ["src/run_ablation.py"]):
            proc = subprocess.run([sys.executable, "-S", *entry, "--file", str(self.path), "--check-only"],
                                  cwd=root, capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertEqual(json.loads(proc.stdout)["queries"], 1)

    def test_full_offline_output_and_no_overwrite(self):
        self.write_rows(judgment())
        self.setup_artifacts()
        output_dir = self.root / "run01"
        args = ["--file", str(self.path), "--output", str(output_dir), "--repeats", "2"]
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(args, retriever_factory=FakeRetriever, root=self.root), 0)
        self.assertEqual({path.name for path in output_dir.iterdir()},
                         {"results.json", "manifest.json", "per_query.csv", "summary.csv", "report.md"})
        result = json.loads((output_dir / "results.json").read_text())
        self.assertEqual(len(result["runs"]), 8)
        manifest = (output_dir / "manifest.json").read_text()
        self.assertNotIn("DO_NOT_COPY_THIS_VALUE", manifest)
        self.assertIn("sha256", manifest)
        before = (output_dir / "results.json").read_bytes()
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(main(args, retriever_factory=FakeRetriever, root=self.root), 1)
        self.assertEqual(before, (output_dir / "results.json").read_bytes())
        self.assertFalse((output_dir / "failure.json").exists())

    def test_real_run_creates_parent_after_preflight_but_check_only_does_not(self):
        self.write_rows(judgment())
        self.setup_artifacts()
        output_dir = self.root / "new_parent" / "nested" / "run01"
        args = ["--file", str(self.path), "--output", str(output_dir)]
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main([*args, "--check-only"], root=self.root), 0)
        self.assertFalse(output_dir.parent.exists())
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(args, retriever_factory=FakeRetriever, root=self.root), 0)
        self.assertTrue((output_dir / "results.json").is_file())

    def test_missing_indexes_and_failed_provider_do_not_claim_results(self):
        self.write_rows(judgment())
        output_dir = self.root / "failed-run"
        args = ["--file", str(self.path), "--output", str(output_dir)]
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(main(args, retriever_factory=FakeRetriever, root=self.root), 1)
        self.assertFalse(output_dir.exists())
        self.setup_artifacts()

        def broken(**kwargs):
            raise RuntimeError("https://provider.example?token=SECRET_SHOULD_NOT_LEAK")

        with contextlib.redirect_stderr(io.StringIO()) as error:
            self.assertEqual(main(args, retriever_factory=broken, root=self.root), 1)
        self.assertNotIn("SECRET_SHOULD_NOT_LEAK", error.getvalue())
        self.assertEqual({path.name for path in output_dir.iterdir()}, {"failure.json"})
        self.assertNotIn("SECRET_SHOULD_NOT_LEAK", (output_dir / "failure.json").read_text())

    def test_inputs_changed_midrun_are_rejected(self):
        self.write_rows(judgment())
        self.setup_artifacts()
        output_dir = self.root / "changed-run"

        def changing(**kwargs):
            target = self.root / INDEX_FILES[0]
            target.write_bytes(b"new version")
            return FakeRetriever()

        with contextlib.redirect_stderr(io.StringIO()):
            status = main(["--file", str(self.path), "--output", str(output_dir)],
                          retriever_factory=changing, root=self.root)
        self.assertEqual(status, 1)
        self.assertFalse((output_dir / "results.json").exists())
        self.assertIn("changed", (output_dir / "failure.json").read_text())

    def test_cli_prepares_reranker_in_initialization_not_first_measured_call(self):
        self.write_rows(judgment(), judgment("q2"))
        self.setup_artifacts()
        retriever = FakeRetriever()
        events = []
        retriever.prepare_reranker = lambda: events.append("prepare")
        original = retriever.retrieve

        def retrieve(query, **kwargs):
            events.append("retrieve")
            return [] if query.endswith("q1") else original(query, **kwargs)

        retriever.retrieve = retrieve
        with contextlib.redirect_stdout(io.StringIO()):
            status = main(["--file", str(self.path), "--output", str(self.root / "run")],
                          retriever_factory=lambda **kwargs: retriever, root=self.root)
        self.assertEqual(status, 0)
        self.assertEqual(events[0], "prepare")
        self.assertLess(events.index("prepare"), events.index("retrieve"))


if __name__ == "__main__":
    unittest.main()
