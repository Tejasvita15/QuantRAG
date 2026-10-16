import contextlib
import io
import json
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock

from src.generator import Completion, GenerationError, GroundedGenerator
from src.query import main
from src.rag_pipeline import RAGPipeline


ROOT = Path(__file__).resolve().parents[1]


def filing(ticker="AAPL", passage_id="p1"):
    return {"rank": 1, "id": passage_id, "sources": ["bm25", "dense"], "passage": {
        "id": passage_id, "ticker": ticker, "company": "Example company",
        "form": "10-K", "filing_date": "2024-11-01", "report_date": "2024-09-28",
        "section": "ITEM 1A", "cik": "0000320193", "accession": "0000320193-24-000123",
        "text": "Supply-chain risk described in the historical filing. " * 30,
    }}


def stock(ticker):
    return {"ticker": ticker, "latest_trading_date": "2026-10-15T00:00:00-04:00",
            "price": 100.0, "previous_close": 99.0, "currency": "USD"}


def macro(series):
    return {"series_id": "FEDFUNDS", "observations": [{"date": "2026-09-01", "value": 3.0}]}


def answer_client(messages):
    evidence = json.loads(messages[1]["content"])["evidence"]
    claims = [{"text": f"Supported statement from {s['source_id']}.", "source_ids": [s["source_id"]]} for s in evidence]
    return Completion(json.dumps({"status": "answered", "claims": claims, "limitations": []}), "offline-test")


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.retriever = Mock()
        self.retriever.retrieve.return_value = [filing()]
        self.stock = Mock(side_effect=stock)
        self.macro = Mock(side_effect=macro)
        self.client = Mock(side_effect=answer_client)
        self.pipeline = RAGPipeline(self.retriever, self.stock, self.macro, GroundedGenerator(self.client))

    def test_full_mixed_flow_preserves_evidence_and_dates(self):
        out = self.pipeline.run("Apple supply-chain risks, latest stock price and current Fed Funds Rate")
        self.assertEqual(out["route"], "mixed")
        self.assertEqual(out["status"], "answered")
        self.assertEqual({s["source_id"] for s in out["citations"]}, {"SEC1", "STOCK1", "FRED1"})
        sec = next(s for s in out["sources"] if s["kind"] == "sec")
        self.assertGreater(len(sec["content"]), 600)
        self.assertEqual(sec["metadata"]["report_date"], "2024-09-28")
        self.assertEqual(sec["metadata"]["url"], "https://www.sec.gov/Archives/edgar/data/320193/000032019324000123/0000320193-24-000123-index.html")
        live = next(s for s in out["sources"] if s["kind"] == "stock")
        self.assertEqual(live["metadata"]["trading_date"], "2026-10-15T00:00:00-04:00")
        self.assertIn("not a guaranteed real-time quote", live["metadata"]["price_basis"])
        fred = next(s for s in out["sources"] if s["kind"] == "macro")
        self.assertEqual(fred["metadata"]["latest_observation_date"], "2026-09-01")
        self.assertTrue(all("retrieved_at" in s["metadata"] for s in out["sources"]))
        self.assertEqual(set(out["timings_s"]), {"filing", "stock", "macro", "generation"})

    def test_filing_only_uses_metadata_filters_and_no_live_tools(self):
        out = self.pipeline.run("Microsoft cloud revenue in 2024 10-K", rerank=False)
        self.retriever.retrieve.assert_called_once_with(
            "Microsoft cloud revenue in 2024 10-K", top_k=5, ticker="MSFT",
            year=2024, form="10-K", rerank=False,
        )
        self.stock.assert_not_called()
        self.macro.assert_not_called()
        self.assertEqual(out["route"], "filing")

    def test_multiple_companies_are_retrieved_separately(self):
        self.retriever.retrieve.side_effect = lambda query, **kw: [filing(kw["ticker"], kw["ticker"])]
        out = self.pipeline.run("Compare Apple and Microsoft revenue and latest stock prices")
        self.assertEqual(self.retriever.retrieve.call_count, 2)
        self.assertEqual(self.stock.call_count, 2)
        self.assertEqual({s["source_id"] for s in out["sources"]}, {"SEC1", "SEC2", "STOCK1", "STOCK2"})

    def test_live_only_does_not_initialize_retriever(self):
        out = self.pipeline.run("current Apple stock price")
        self.retriever.retrieve.assert_not_called()
        self.macro.assert_not_called()
        self.assertEqual(out["route"], "stock")

    def test_fred_failure_preserves_sec_answer_and_hides_key(self):
        self.macro.side_effect = RuntimeError("api_key=SECRET_IN_URL")
        out = self.pipeline.run("JPMorgan interest rate risks and latest Fed Funds Rate")
        self.assertEqual(out["status"], "partial")
        self.assertEqual([s["source_id"] for s in out["citations"]], ["SEC1"])
        self.assertEqual(out["errors"][0]["backend"], "macro")
        self.assertIn("FRED_API_KEY", out["errors"][0]["error"])
        self.assertNotIn("SECRET_IN_URL", json.dumps(out))

    def test_all_tools_fail_without_llm_call(self):
        self.stock.side_effect = RuntimeError("provider down")
        out = self.pipeline.run("latest AAPL stock price")
        self.client.assert_not_called()
        self.assertEqual(out["status"], "insufficient_evidence")
        self.assertEqual(out["citations"], [])
        self.assertTrue(out["limitations"])

    def test_empty_retrieval_abstains_without_llm_call(self):
        self.retriever.retrieve.return_value = []
        out = self.pipeline.run("Apple risks")
        self.client.assert_not_called()
        self.assertEqual(out["status"], "insufficient_evidence")
        self.assertTrue(out["errors"])

    def test_missing_ticker_is_not_guessed(self):
        out = self.pipeline.run("latest stock price")
        self.stock.assert_not_called()
        self.client.assert_not_called()
        self.assertEqual(out["routing"]["missing_inputs"], ["ticker"])
        self.assertEqual(out["errors"][0]["backend"], "stock")

    def test_undated_live_data_is_not_sent_to_model(self):
        for data in ({"price": 100}, {"latest_trading_date": "2026-10-15"}):
            self.stock.side_effect = None
            self.stock.return_value = data
            out = self.pipeline.run("latest AAPL stock price")
            self.assertEqual(out["evidence"], [])
        self.macro.side_effect = None
        self.macro.return_value = {"series_id": "GDP", "observations": []}
        self.assertEqual(self.pipeline.run("latest GDP")["evidence"], [])
        self.client.assert_not_called()

    def test_unknown_citations_fail_closed(self):
        self.client.side_effect = None
        self.client.return_value = Completion(json.dumps({"status": "answered", "claims": [
            {"text": "Invented fact", "source_ids": ["SEC999"]},
        ], "limitations": []}))
        out = self.pipeline.run("Apple risks")
        self.assertEqual(out["status"], "generation_error")
        self.assertIsNone(out["answer"])
        self.assertEqual(out["citations"], [])
        self.assertTrue(out["evidence"])

    def test_retrieval_only_does_not_call_generator(self):
        out = self.pipeline.run("Apple risks", generate=False)
        self.assertEqual(out["status"], "retrieved")
        self.client.assert_not_called()

    def test_duplicate_passages_are_deduplicated(self):
        self.retriever.retrieve.return_value = [filing(), filing()]
        out = self.pipeline.run("Apple risks")
        self.assertEqual(len(out["sources"]), 1)

    def test_generation_configuration_error_is_reviewable(self):
        self.client.side_effect = GenerationError("Set LLM_MODEL")
        out = self.pipeline.run("Apple risks")
        self.assertEqual(out["status"], "generation_error")
        self.assertEqual(out["errors"][-1]["backend"], "generation")
        self.assertTrue(out["evidence"])


class CLITests(unittest.TestCase):
    setUp = PipelineTests.setUp

    def test_answer_cli_outputs_json(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            status = main(["Apple risks and latest stock price", "--answer"], pipeline=self.pipeline)
        out = json.loads(output.getvalue())
        self.assertEqual(status, 0)
        self.assertEqual(out["route"], "mixed")
        self.assertTrue(out["answer"])
        self.assertEqual(len(out["citations"]), 2)

    def test_original_stock_result_contract(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            status = main(["current price", "--type", "stock", "--ticker", "NVDA"], pipeline=self.pipeline)
        out = json.loads(output.getvalue())
        self.assertEqual(status, 0)
        self.assertEqual(out["result"]["ticker"], "NVDA")
        self.assertIn("latency_s", out)
        self.client.assert_not_called()

    def test_invalid_cli_input(self):
        for args in (["  "], ["Apple risks", "--top-k", "0"]):
            with self.subTest(args=args), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(args, pipeline=self.pipeline), 1)

    def test_route_only_script_and_module_without_optional_dependencies(self):
        for entry in (["src/query.py"], ["-m", "src.query"]):
            proc = subprocess.run([sys.executable, "-S", *entry, "Apple risks and latest stock price", "--route-only"],
                                  cwd=ROOT, text=True, capture_output=True)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertEqual(json.loads(proc.stdout)["routing"]["routes"], ["filing", "stock"])


if __name__ == "__main__":
    unittest.main()
