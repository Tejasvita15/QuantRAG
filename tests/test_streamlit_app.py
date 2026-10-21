"""UI checks use AppTest and injected backends, not real providers or models."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from streamlit.testing.v1 import AppTest

from src.ui_support import build_options, readiness, safe_source_url


ROOT = Path(__file__).resolve().parents[1]


class UISupportTests(unittest.TestCase):
    def test_filters_and_validation(self):
        result = build_options(" aapl ", " UNRATE ", "2024", "10-K", "auto", 5, False)
        self.assertEqual(result["ticker"], "AAPL")
        self.assertEqual(result["year"], 2024)
        self.assertFalse(result["rerank"])
        for year in ("2024abc", "24", "-1"):
            with self.assertRaises(ValueError):
                build_options("", "", year, "Auto", "auto", 5, True)

    def test_readiness_returns_no_secret_values(self):
        with tempfile.TemporaryDirectory() as directory:
            env = {"LLM_MODEL": "test", "LLM_API_KEY": "SECRET", "FRED_API_KEY": "SECRET"}
            checks = readiness(directory, env)
            self.assertFalse(checks["SEC index files present"])
            self.assertTrue(checks["LLM model + key configured"])
            self.assertNotIn("SECRET", json.dumps(checks))

    def test_only_known_https_sources_linked(self):
        good = "https://www.sec.gov/Archives/edgar/data/320193/index.html"
        self.assertEqual(safe_source_url(good), good)
        for value in ("javascript:alert(1)", "https://evil.example", "https://www.sec.gov.evil.example/",
                      "https://user:secret@www.sec.gov/", "https://www.sec.gov/?key=secret", None):
            self.assertIsNone(safe_source_url(value))


class StreamlitTests(unittest.TestCase):
    def make_app(self):
        return AppTest.from_file(str(ROOT / "app.py"), default_timeout=15).run()

    def submit(self, app, query):
        app.text_area(key="query").set_value(query)
        return app.button(key="FormSubmitter:query_form-Submit").click().run()

    def test_initial_page_has_no_backend_calls(self):
        with patch("src.rag_pipeline.RAGPipeline") as pipeline:
            app = self.make_app()
            self.assertFalse(app.exception)
            pipeline.assert_not_called()

    def test_route_only_and_rerun_preserve_output(self):
        with patch("src.rag_pipeline.RAGPipeline") as pipeline:
            app = self.submit(self.make_app(), "Apple risks and latest stock price")
            self.assertFalse(app.exception)
            out = app.session_state["last_response"]
            self.assertEqual(out["routing"]["routes"], ["filing", "stock"])
            self.assertEqual(out["ui"]["execution_mode"], "route")
            app.checkbox(key="rerank").uncheck().run()
            self.assertEqual(app.session_state["last_response"], out)
            pipeline.assert_not_called()

    def test_blank_query_and_bad_year_are_safe(self):
        app = self.submit(self.make_app(), "   ")
        self.assertTrue(app.warning)
        self.assertNotIn("last_response", app.session_state)
        app.text_input(key="year").set_value("not a year").run()
        app = self.submit(app, "Apple risks")
        self.assertFalse(app.exception)
        self.assertTrue(any("four digits" in item.value for item in app.warning))

    def test_retrieve_uses_shared_pipeline_and_renders_full_evidence(self):
        pipeline = Mock()
        pipeline.run.return_value = {
            "query": "Apple risks", "status": "retrieved", "latency_s": .1,
            "routing": {"kind": "filing", "routes": ["filing"], "missing_inputs": []},
            "evidence": [{"source_id": "SEC1", "kind": "sec", "content": "Full passage " * 100,
                          "metadata": {"id": "stable-id", "ticker": "AAPL", "filing_date": "2024-11-01"}}],
            "timings_s": {"filing": .1}, "errors": [],
        }
        app = self.make_app()
        app.session_state["_pipeline"] = pipeline
        app.selectbox(key="execution_mode").select("Retrieve evidence").run()
        app = self.submit(app, "Apple risks")
        self.assertFalse(app.exception)
        self.assertFalse(pipeline.run.call_args.kwargs["generate"])
        self.assertTrue(any("Full passage" in text.value for text in app.text))
        self.assertEqual(app.session_state["last_response"]["status"], "retrieved")

    def test_backend_exception_does_not_expose_credentials(self):
        app = self.make_app()
        pipeline = Mock()
        pipeline.run.side_effect = RuntimeError("api_key=SECRET")
        app.session_state["_pipeline"] = pipeline
        app.selectbox(key="execution_mode").select("Retrieve evidence").run()
        app = self.submit(app, "Apple risks")
        self.assertFalse(app.exception)
        self.assertTrue(app.error)
        self.assertNotIn("SECRET", " ".join(e.value for e in app.error))

    @patch.dict(os.environ, {"LLM_MODEL": "", "LLM_API_KEY": "", "OPENAI_API_KEY": ""})
    def test_missing_llm_config_stops_before_any_backend_call(self):
        with patch("src.rag_pipeline.RAGPipeline") as pipeline:
            app = self.make_app()
            app.selectbox(key="execution_mode").select("Generate answer").run()
            app = self.submit(app, "Apple risks")
            self.assertFalse(app.exception)
            self.assertTrue(any("No provider call was made" in error.value for error in app.error))
            pipeline.assert_not_called()

    def test_reset_clears_saved_result(self):
        app = self.submit(self.make_app(), "Apple risks")
        self.assertIn("last_response", app.session_state)
        app.button(key="reset_session").click().run()
        self.assertNotIn("last_response", app.session_state)

    @patch.dict(os.environ, {"LLM_MODEL": "test", "LLM_API_KEY": "test"})
    def test_partial_answer_and_generation_toggle(self):
        pipeline = Mock()
        pipeline.run.return_value = {
            "query": "Apple risks", "status": "partial", "answer": "Supported statement [SEC1]",
            "routing": {"kind": "filing", "missing_inputs": []}, "evidence": [],
            "limitations": ["Stock provider unavailable"], "errors": [], "latency_s": .2,
        }
        app = self.make_app()
        app.session_state["_pipeline"] = pipeline
        app.selectbox(key="execution_mode").select("Generate answer").run()
        app = self.submit(app, "Apple risks")
        self.assertFalse(app.exception)
        self.assertTrue(pipeline.run.call_args.kwargs["generate"])
        self.assertTrue(any("Supported statement" in t.value for t in app.text))
        self.assertTrue(any("Partial answer" in w.value for w in app.warning))

    @patch.dict(os.environ, {"LLM_MODEL": "test", "LLM_API_KEY": "test"})
    def test_generation_error_does_not_claim_exact_sent_context(self):
        pipeline = Mock()
        source = {"source_id": "SEC1", "kind": "sec", "content": "Raw text " * 1000, "metadata": {}}
        pipeline.run.return_value = {
            "query": "Apple risks", "status": "generation_error", "answer": None,
            "routing": {"kind": "filing", "missing_inputs": []}, "evidence": [source],
            "sources": [source], "errors": [], "latency_s": .2,
        }
        app = self.make_app()
        app.session_state["_pipeline"] = pipeline
        app.selectbox(key="execution_mode").select("Generate answer").run()
        app = self.submit(app, "Apple risks")
        self.assertFalse(app.exception)
        self.assertFalse(any("Exact context supplied" in e.label for e in app.expander))
        self.assertFalse(any("sent to model" in e.label for e in app.expander))
        self.assertTrue(any("exact sent context is not available" in w.value for w in app.warning))


if __name__ == "__main__":
    unittest.main()
