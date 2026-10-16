import json
import os
import unittest
from unittest.mock import Mock, patch

import requests

from src.generator import (
    ChatCompletionClient, Completion, GenerationError, GroundedGenerator,
)


def source(source_id="SEC1", kind="sec", content="Filing evidence"):
    return {"source_id": source_id, "kind": kind, "content": content,
            "metadata": {"filing_date": "2024-11-01", "url": "https://www.sec.gov/"}}


def completion(claims=None, status="answered", limitations=None):
    if claims is None:
        claims = [{"text": "The 2024 filing identifies supply-chain risk.", "source_ids": ["SEC1"]}]
    return Completion(json.dumps({"status": status, "claims": claims,
                                  "limitations": limitations or []}), "test-model", {"total_tokens": 12})


class GeneratorTests(unittest.TestCase):
    def test_citations_are_rendered_from_validated_source_ids(self):
        client = Mock(return_value=completion())
        out = GroundedGenerator(client).generate("Question", [source()])
        self.assertEqual(out["status"], "answered")
        self.assertTrue(out["answer"].endswith("[SEC1]"))
        self.assertEqual(out["citations"][0]["metadata"]["filing_date"], "2024-11-01")
        self.assertEqual(out["usage"]["total_tokens"], 12)
        messages = client.call_args.args[0]
        self.assertEqual([m["role"] for m in messages], ["system", "user"])
        self.assertIn("untrusted data", messages[0]["content"])
        self.assertEqual(json.loads(messages[1]["content"])["evidence"], out["sources"])

    def test_missing_or_invented_citations_are_rejected(self):
        for ids in ([], ["SEC99"], ["SEC1", "FRED99"], "SEC1", [None]):
            with self.subTest(ids=ids), self.assertRaises(GenerationError):
                GroundedGenerator(Mock(return_value=completion([
                    {"text": "An assertion", "source_ids": ids},
                ]))).generate("Question", [source()])

    def test_embedded_fake_links_and_citations_are_rejected(self):
        for text in ("An assertion [SEC99]", "See https://made-up.example", "See [source](link)"):
            with self.subTest(text=text), self.assertRaises(GenerationError):
                GroundedGenerator(Mock(return_value=completion([
                    {"text": text, "source_ids": ["SEC1"]},
                ]))).generate("Question", [source()])

    def test_invalid_json_schema_and_empty_answers_are_rejected(self):
        for content in (
            "not JSON", "[]", "null", "{}",
            '{"status":"answered","claims":[],"limitations":[]}',
            '{"status":"unknown","claims":[],"limitations":[]}',
            '{"status":"insufficient_evidence","claims":[{}],"limitations":[]}',
            '{"status":"answered","claims":[{}],"limitations":[]}',
            '{"status":"answered","claims":[],"limitations":"missing"}',
        ):
            with self.subTest(content=content), self.assertRaises(GenerationError):
                GroundedGenerator(Mock(return_value=Completion(content))).generate("Question", [source()])

    def test_no_evidence_abstains_without_model_or_credentials(self):
        client = Mock()
        out = GroundedGenerator(client).generate("Question", [])
        self.assertEqual(out["status"], "insufficient_evidence")
        self.assertEqual(out["citations"], [])
        client.assert_not_called()

    def test_model_can_reject_irrelevant_evidence(self):
        client = Mock(return_value=completion([], "insufficient_evidence", ["The filing does not cover the question"]))
        out = GroundedGenerator(client).generate("Question", [source()])
        self.assertEqual(out["status"], "insufficient_evidence")
        self.assertEqual(out["claims"], [])

    def test_tool_failures_are_supplied_to_model_and_mark_partial(self):
        client = Mock(return_value=completion())
        out = GroundedGenerator(client).generate("Question", [source()], warnings=["Stock data unavailable"])
        self.assertEqual(out["status"], "partial")
        prompt = json.loads(client.call_args.args[0][1]["content"])
        self.assertEqual(prompt["coverage_limitations"], ["Stock data unavailable"])
        self.assertEqual(out["limitations"], ["Stock data unavailable"])

    def test_context_budget_preserves_live_dates_and_rejects_omitted_ids(self):
        sources = [source(content="x" * 5000), source("STOCK1", "stock", '{"date":"2026-10-15","price":100}')]
        client = Mock(return_value=completion([
            {"text": "A dated price", "source_ids": ["STOCK1"]},
        ]))
        out = GroundedGenerator(client, max_context_chars=1024).generate("Question", sources)
        self.assertEqual([s["source_id"] for s in out["sources"]], ["STOCK1"])
        self.assertIn("2026-10-15", out["sources"][0]["content"])
        self.assertLessEqual(len(json.dumps(out["sources"], ensure_ascii=False)), 1024)
        self.assertEqual(out["status"], "partial")
        with self.assertRaises(GenerationError):
            GroundedGenerator(Mock(return_value=completion()), max_context_chars=1024).generate("Question", sources)

    def test_passage_truncation_is_visible(self):
        out = GroundedGenerator(Mock(return_value=completion()), max_source_chars=256).generate(
            "Question", [source(content="x" * 1000)],
        )
        self.assertTrue(out["sources"][0]["truncated"])
        self.assertEqual(len(out["sources"][0]["content"]), 256)

    def test_duplicate_source_ids_are_rejected(self):
        with self.assertRaises(ValueError):
            GroundedGenerator(Mock()).generate("Question", [source(), source()])


class ClientTests(unittest.TestCase):
    def response(self, reason="stop", content="{}"):
        response = Mock()
        response.json.return_value = {"model": "model-from-provider", "usage": {"total_tokens": 10},
                                      "choices": [{"finish_reason": reason, "message": {"content": content}}]}
        return response

    @patch("requests.post")
    def test_provider_request_contract_and_usage(self, post):
        post.return_value = self.response()
        out = ChatCompletionClient("configured-model", "secret", "https://provider.example/v1/", timeout=10)([])
        args, kwargs = post.call_args
        self.assertEqual(args, ("https://provider.example/v1/chat/completions",))
        self.assertEqual(kwargs["json"]["model"], "configured-model")
        self.assertEqual(kwargs["json"]["response_format"], {"type": "json_object"})
        self.assertEqual(kwargs["timeout"], 10)
        self.assertEqual(out.model, "model-from-provider")
        self.assertEqual(out.usage, {"total_tokens": 10})

    @patch("requests.post")
    def test_plain_json_provider_mode(self, post):
        post.return_value = self.response()
        ChatCompletionClient("model", "secret", json_mode=False)([])
        self.assertNotIn("response_format", post.call_args.kwargs["json"])

    @patch("requests.post", side_effect=requests.HTTPError("URL contains secret-token"))
    def test_transport_errors_do_not_leak_credentials(self, post):
        with self.assertRaises(GenerationError) as ctx:
            ChatCompletionClient("model", "secret")([])
        self.assertNotIn("secret-token", str(ctx.exception))

    @patch("requests.post")
    def test_incomplete_and_refused_responses_are_rejected(self, post):
        for reason in ("length", "content_filter", "tool_calls", None):
            post.return_value = self.response(reason)
            with self.subTest(reason=reason), self.assertRaises(GenerationError):
                ChatCompletionClient("model", "secret")([])
        post.return_value = self.response()
        post.return_value.json.return_value["choices"][0]["message"]["refusal"] = "Cannot answer"
        with self.assertRaises(GenerationError):
            ChatCompletionClient("model", "secret")([])

    @patch("requests.post")
    def test_malformed_provider_response(self, post):
        for body in (None, {}, {"choices": []}, {"choices": [None]}):
            post.return_value = Mock()
            post.return_value.json.return_value = body
            with self.subTest(body=body), self.assertRaises(GenerationError):
                ChatCompletionClient("model", "secret")([])

    def test_model_and_key_must_be_configured(self):
        for model, key in (("", "secret"), ("model", "")):
            with self.assertRaises(GenerationError):
                ChatCompletionClient(model, key)

    @patch.dict(os.environ, {"LLM_MODEL": "test", "OPENAI_API_KEY": "key"}, clear=True)
    @patch("dotenv.load_dotenv")
    def test_env_fallback_and_timeout_validation(self, load):
        client = ChatCompletionClient.from_env()
        self.assertEqual((client.model, client.api_key), ("test", "key"))
        with patch.dict(os.environ, {"LLM_TIMEOUT_SECONDS": "invalid"}), self.assertRaises(GenerationError):
            ChatCompletionClient.from_env()


if __name__ == "__main__":
    unittest.main()
