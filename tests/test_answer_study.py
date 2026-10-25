import contextlib
import copy
import csv
import hashlib
import io
import json
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from src.answer_study import (
    DIMENSIONS, VARIANTS, aggregate_ratings, load_cases, main, prepare_study,
)
from src.generator import Completion, GroundedGenerator, SYSTEM_PROMPT


def retrieval_fixture():
    return {
        "schema_version": "1.0", "kind": "retrieval_ablation", "status": "completed",
        "queries": [{"query_id": "q1", "query": "What risks does Apple report?",
                     "filters": {"ticker": "AAPL", "year": None, "form": "10-K"}}],
        "runs": [{"query_id": "q1", "variant": variant, "repeat": 1,
                  "top_k": 5, "ranking": [{"id": "p1", "rank": 1,
                    "sources": [variant], "passage": {
                        "id": "p1", "text": "The filing identifies supply-chain risk. " * 30,
                        "ticker": "AAPL", "filing_date": "2024-11-01", "form": "10-K",
                        "cik": "0000320193", "accession": "0000320193-24-000123"}}]}
                 for variant in VARIANTS],
    }


def valid_completion():
    return Completion(json.dumps({"status": "answered", "claims": [
        {"text": "The filing identifies supply-chain risk.", "source_ids": ["SEC1"]}],
        "limitations": []}), "offline-test-model", {"total_tokens": 12})


class AnswerStudyTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.results = self.root / "results.json"
        self.write_json(self.results, retrieval_fixture())

    def write_json(self, path, value):
        path.write_text(json.dumps(value), encoding="utf-8")

    def cases(self, **kwargs):
        return load_cases(self.results, **kwargs)

    def generated_study(self, name="study", cases=None):
        path = self.root / name
        prepare_study(cases or self.cases(), path, generate=True,
                      generator=GroundedGenerator(Mock(side_effect=lambda _: valid_completion())))
        return path, json.loads((path / "answers.json").read_text())

    def write_ratings(self, study, *, mutate=None, name="ratings.csv"):
        rows = []
        for answer in study["answers"]:
            for annotator in ("A", "B"):
                rows.append({"answer_id": answer["answer_id"], "annotator_id": annotator,
                             **{dimension: "5" for dimension in DIMENSIONS}})
        if mutate:
            mutate(rows)
        path = self.root / name
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["answer_id", "annotator_id", *DIMENSIONS])
            writer.writeheader()
            writer.writerows(rows)
        return path

    def test_load_cases_joins_all_variants_and_keeps_full_evidence(self):
        cases = self.cases(max_source_chars=256)
        self.assertEqual([case["variant"] for case in cases], list(VARIANTS))
        for case in cases:
            self.assertEqual(case["filters"]["ticker"], "AAPL")
            self.assertGreater(len(case["evidence"][0]["content"]), 600)
            self.assertEqual(len(case["sources"][0]["content"]), 256)
            self.assertTrue(case["sources"][0]["truncated"])
            self.assertEqual(case["sources"][0]["metadata"]["rank"], 1)
            self.assertIn("-index.html", case["sources"][0]["metadata"]["url"])

    def test_invalid_or_missing_cases_fail_before_writing(self):
        invalids = []
        value = retrieval_fixture()
        value["runs"].pop()
        invalids.append(value)
        value = retrieval_fixture()
        value["runs"][0]["ranking"][0]["passage"].pop("text")
        invalids.append(value)
        value = retrieval_fixture()
        value["runs"][0]["ranking"][0]["rank"] = 2
        invalids.append(value)
        value = retrieval_fixture()
        value["runs"].append(copy.deepcopy(value["runs"][0]))
        invalids.append(value)
        value = retrieval_fixture()
        value["status"] = "failed"
        invalids.append(value)
        for value in invalids:
            with self.subTest(value=value):
                self.write_json(self.results, value)
                with self.assertRaises(ValueError):
                    self.cases()

    def test_repeat_is_explicit_and_requires_complete_four_way_run(self):
        value = retrieval_fixture()
        for run in value["runs"]:
            run["repeat"] = 2
        self.write_json(self.results, value)
        with self.assertRaises(ValueError):
            self.cases()
        self.assertEqual(len(self.cases(repeat=2)), 4)

    def test_identical_frozen_live_snapshots_are_added_to_all_modes(self):
        live_path = self.root / "live.json"
        live = {"q1": [{"source_id": "STOCK1", "kind": "stock",
                         "content": '{"price":100,"date":"2026-10-20"}',
                         "metadata": {"trading_date": "2026-10-20", "units": "USD"}}]}
        self.write_json(live_path, live)
        cases = self.cases(live_sources_path=live_path)
        for case in cases:
            self.assertEqual(case["sources"][0]["source_id"], "STOCK1")
            self.assertEqual(case["sources"][0]["metadata"]["trading_date"], "2026-10-20")
        live["q1"][0]["metadata"] = {"retrieved_at": "2026-10-21"}
        self.write_json(live_path, live)
        with self.assertRaises(ValueError):
            self.cases(live_sources_path=live_path)

    def test_real_pipeline_stock_datetime_and_macro_date_shapes_are_accepted(self):
        live_path = self.root / "pipeline-live.json"
        # Same source structure as RAGPipeline evidence / Streamlit JSON export.
        live = {"q1": [
            {"source_id": "STOCK1", "kind": "stock",
             "content": json.dumps({"price": 100, "latest_trading_date": "2026-10-15T00:00:00-04:00"}),
             "metadata": {"ticker": "AAPL", "provider": "yfinance / Yahoo Finance",
                          "trading_date": "2026-10-15T00:00:00-04:00",
                          "retrieved_at": "2026-10-21T00:00:00+00:00"}},
            {"source_id": "FRED1", "kind": "macro", "content": '{"series_id":"UNRATE"}',
             "metadata": {"series_id": "UNRATE", "latest_observation_date": "2026-09-01"}},
        ]}
        for value in ("2026-10-15T00:00:00-04:00", "2026-10-15T04:00:00Z", "2024-02-29"):
            with self.subTest(valid=value):
                live["q1"][0]["metadata"]["trading_date"] = value
                self.write_json(live_path, live)
                cases = self.cases(live_sources_path=live_path)
                self.assertEqual(cases[0]["sources"][0]["metadata"]["trading_date"], value)
        for value in ("2026-02-30", "2026-13-01", "2026-02-29", "2026-10-15T00:00:00",
                      "2026-02-30T00:00:00+00:00", "2026-10-15T00:00:00+25:00"):
            with self.subTest(invalid=value):
                live["q1"][0]["metadata"]["trading_date"] = value
                self.write_json(live_path, live)
                with self.assertRaises(ValueError):
                    self.cases(live_sources_path=live_path)

    def test_input_hashes_prompt_hash_and_seeded_call_order_are_recorded(self):
        refs = self.root / "refs.jsonl"
        refs.write_text('{"id":"q1","reference_answer":"Human reference.","held_out":true}\n',
                        encoding="utf-8")
        live = self.root / "empty-live.json"
        live.write_text('{"q1":[]}\n', encoding="utf-8")
        cases = self.cases(references_path=refs, live_sources_path=live)
        actual_calls = []

        def generate(query, sources):
            actual_calls.append(sources[0]["metadata"]["retrieval_sources"][0])
            return {"status": "answered", "answer": "Answer [SEC1]", "claims": [],
                    "citations": [sources[0]], "sources": sources,
                    "model": "fake", "usage": {"total_tokens": 12}}

        generator = Mock()
        generator.generate.side_effect = generate
        output = self.root / "seeded"
        manifest = prepare_study(cases, output, generate=True, generator=generator, seed=1234)
        expected = list(VARIANTS)
        random.Random(1234).shuffle(expected)
        self.assertEqual(actual_calls, expected)
        self.assertEqual([row["variant"] for row in manifest["generation_schedule"]], expected)
        self.assertEqual([row["order"] for row in manifest["generation_schedule"]], [1, 2, 3, 4])
        self.assertTrue(all(row["generation_attempted"] for row in manifest["generation_schedule"]))
        self.assertEqual(manifest["schedule_seed"], 1234)
        self.assertEqual(manifest["system_prompt_sha256"], hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest())
        for field, path in (("retrieval_results", self.results), ("reference_answers", refs),
                            ("frozen_live_sources", live)):
            self.assertEqual(manifest["input_sha256"][field], hashlib.sha256(path.read_bytes()).hexdigest())
        second = prepare_study(cases, self.root / "seeded-second", seed=1234)
        self.assertEqual([row["variant"] for row in second["generation_schedule"]], expected)
        self.assertTrue(all(not row["generation_attempted"] for row in second["generation_schedule"]))

    def test_references_are_declarations_not_invented_labels(self):
        refs = self.root / "references.jsonl"
        refs.write_text(json.dumps({"id": "q1", "reference_answer": "Supply-chain risk.",
                                    "held_out": True}) + "\n", encoding="utf-8")
        cases = self.cases(references_path=refs)
        self.assertTrue(all(case["held_out"] for case in cases))
        self.assertEqual(cases[0]["reference_answer"], "Supply-chain risk.")
        refs.write_text(json.dumps({"id": "unknown", "reference_answer": "Reference."}), encoding="utf-8")
        with self.assertRaises(ValueError):
            self.cases(references_path=refs)

    @patch("src.generator.ChatCompletionClient.from_env")
    def test_default_preparation_never_calls_provider_and_blinds_variant(self, provider):
        output = self.root / "prepared"
        manifest = prepare_study(self.cases(), output)
        provider.assert_not_called()
        self.assertEqual(manifest["status"], "prepared_only")
        study = json.loads((output / "answers.json").read_text())
        self.assertTrue(all(answer["status"] == "not_generated" for answer in study["answers"]))
        with (output / "blind_review.csv").open(encoding="utf-8-sig") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 4)
        self.assertNotIn("variant", rows[0])
        for row in rows:
            self.assertRegex(row["answer_id"], r"^[0-9a-f]{32}$")
            self.assertEqual(row["answer"], "")
            self.assertNotIn("retrieval_sources", row["context"])
            for variant in VARIANTS:
                self.assertNotIn(variant, row["context"])
        mapping = json.loads((output / "private_mapping.json").read_text())
        self.assertEqual({item["variant"] for item in mapping.values()}, set(VARIANTS))
        with self.assertRaises(ValueError):
            prepare_study(self.cases(), output)

    def test_explicit_generation_records_real_returned_citations_usage_latency(self):
        path, study = self.generated_study()
        for answer in study["answers"]:
            self.assertEqual(answer["status"], "answered")
            self.assertEqual(answer["model"], "offline-test-model")
            self.assertEqual(answer["usage"], {"total_tokens": 12})
            self.assertGreaterEqual(answer["generation_latency_s"], 0)
            self.assertEqual(answer["citations"][0]["source_id"], "SEC1")
            self.assertIn("[SEC1]", answer["answer"])
        with (path / "blind_review.csv").open(encoding="utf-8-sig") as handle:
            row = next(csv.DictReader(handle))
        self.assertNotIn("retrieval_sources", row["actual_cited_sources"])

    def test_provider_failure_saved_without_exception_body_or_credentials(self):
        output = self.root / "failed"
        generator = Mock()
        generator.generate.side_effect = RuntimeError("secret-api-key provider-body")
        prepare_study(self.cases(), output, generate=True, generator=generator)
        text = (output / "answers.json").read_text()
        self.assertNotIn("secret-api-key", text)
        self.assertNotIn("provider-body", text)
        study = json.loads(text)
        self.assertTrue(all(answer["status"] == "generation_error" for answer in study["answers"]))
        self.assertTrue(all(answer["context_status"] == "planned_context_send_unconfirmed"
                            for answer in study["answers"]))
        self.assertTrue(all(answer["usage"] is None for answer in study["answers"]))
        self.assertEqual(generator.generate.call_count, 4)

    @patch("src.generator.ChatCompletionClient.from_env")
    def test_check_only_has_no_files_or_provider_calls(self, provider):
        proposed = self.root / "not-created"
        with contextlib.redirect_stdout(io.StringIO()) as stdout:
            main(["prepare", "--results", str(self.results), "--output", str(proposed), "--check-only"])
        result = json.loads(stdout.getvalue())
        self.assertEqual(result["api_calls"], 0)
        self.assertEqual(result["files_written"], 0)
        self.assertFalse(proposed.exists())
        provider.assert_not_called()
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            main(["prepare", "--results", str(self.results), "--check-only", "--generate"])

    def test_manual_aggregation_means_exact_agreement_and_unique_question_count(self):
        cases = self.cases()
        for case in cases:
            case.update({"held_out": True, "reference_answer": "Supply-chain risk."})
        study_path, study = self.generated_study(cases=cases)
        hybrid_id = next(answer["answer_id"] for answer in study["answers"] if answer["variant"] == "hybrid")

        def one_disagreement(rows):
            next(row for row in rows if row["answer_id"] == hybrid_id and row["annotator_id"] == "B")["quality"] = "4"

        ratings = self.write_ratings(study, mutate=one_disagreement)
        output = self.root / "metrics"
        result = aggregate_ratings(study_path, ratings, output)
        self.assertEqual(result["rating_rows"], 8)
        self.assertEqual(result["declared_unique_held_out_questions"], 1)
        self.assertEqual(result["held_out_questions_with_all_four_variants_double_rated"], 1)
        self.assertFalse(result["course_checks"]["at_least_1000_declared_held_out_questions"])
        self.assertEqual(result["agreement"]["paired_comparisons"], 16)
        self.assertEqual(result["agreement"]["exact_matches"], 15)
        self.assertEqual(result["agreement"]["overall_exact_agreement"], 0.9375)
        self.assertEqual(result["agreement"]["by_dimension"]["quality"]["exact_agreement"], 0.75)
        hybrid = next(row for row in result["summary"] if row["variant"] == "hybrid")
        self.assertEqual(hybrid["means"]["quality"], 4.5)
        self.assertEqual(hybrid["reported_token_totals"]["total_tokens"], 12)
        self.assertIsNone(hybrid["reported_token_totals"]["prompt_tokens"])
        self.assertTrue((output / "human_metrics.json").exists())
        with self.assertRaises(ValueError):
            aggregate_ratings(study_path, ratings, output)

    def test_duplicate_question_text_does_not_multiply_held_out_record_count(self):
        cases = self.cases()
        second = copy.deepcopy(cases)
        for case in second:
            case.update({"query_id": "q2", "query": "  WHAT risks does Apple report?  "})
        cases.extend(second)
        for case in cases:
            case.update({"held_out": True, "reference_answer": "Reference."})
        path, study = self.generated_study(cases=cases)
        result = aggregate_ratings(path, self.write_ratings(study), self.root / "unique-metrics")
        self.assertEqual(result["declared_unique_held_out_questions"], 1)
        self.assertEqual(result["held_out_questions_with_all_four_variants_double_rated"], 1)

    def test_missing_labels_duplicates_unknown_ids_or_one_annotator_are_rejected(self):
        study_path, study = self.generated_study()
        mutations = [
            lambda rows: rows[0].update({"faithfulness": ""}),
            lambda rows: rows[0].update({"quality": "6"}),
            lambda rows: rows[0].update({"answer_id": "unknown"}),
            lambda rows: rows.append(dict(rows[0])),
            lambda rows: rows.__setitem__(slice(None), [row for row in rows if row["annotator_id"] == "A"]),
        ]
        for index, mutation in enumerate(mutations):
            ratings = self.write_ratings(study, mutate=mutation)
            output = self.root / f"invalid-{index}"
            with self.subTest(index=index), self.assertRaises(ValueError):
                aggregate_ratings(study_path, ratings, output)
            self.assertFalse(output.exists())

    def test_prepared_blank_answers_cannot_be_scored(self):
        study_path = self.root / "prepared"
        prepare_study(self.cases(), study_path)
        study = json.loads((study_path / "answers.json").read_text())
        with self.assertRaises(ValueError):
            aggregate_ratings(study_path, self.write_ratings(study), self.root / "invalid")

    def test_no_evidence_abstention_does_not_call_llm(self):
        value = retrieval_fixture()
        for run in value["runs"]:
            run["ranking"] = []
        self.write_json(self.results, value)
        client = Mock()
        output = self.root / "abstained"
        prepare_study(self.cases(), output, generate=True, generator=GroundedGenerator(client))
        client.assert_not_called()
        study = json.loads((output / "answers.json").read_text())
        self.assertTrue(all(answer["status"] == "insufficient_evidence" for answer in study["answers"]))


if __name__ == "__main__":
    unittest.main()
