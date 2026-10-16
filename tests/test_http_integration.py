"""Exercise the real HTTP adapter against a local deterministic test provider.

This verifies the wire contract, not output quality of a real language model.
"""

import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import Mock

from src.generator import ChatCompletionClient, GroundedGenerator
from src.rag_pipeline import RAGPipeline


class HTTPIntegrationTests(unittest.TestCase):
    def test_stock_question_through_http_and_citation_rendering(self):
        seen = {}

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                seen["path"] = self.path
                seen["payload"] = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                prompt = json.loads(seen["payload"]["messages"][1]["content"])
                evidence = prompt["evidence"][0]
                snapshot = json.loads(evidence["content"])
                answer = {
                    "status": "answered", "claims": [{
                        "text": f"The latest available adjusted daily close on {snapshot['latest_trading_date']} is {snapshot['price']} USD.",
                        "source_ids": [evidence["source_id"]],
                    }], "limitations": [],
                }
                body = json.dumps({
                    "model": "local-test-provider", "usage": {"total_tokens": 100},
                    "choices": [{"finish_reason": "stop", "message": {"content": json.dumps(answer)}}],
                }).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            client = ChatCompletionClient("test-model", "test-key",
                                          f"http://127.0.0.1:{server.server_port}/v1", timeout=5)
            retriever = Mock()
            pipeline = RAGPipeline(
                retriever=retriever,
                stock_tool=lambda ticker: {"ticker": ticker, "price": 100.0,
                                           "currency": "USD", "latest_trading_date": "2026-10-15"},
                generator=GroundedGenerator(client),
            )
            out = pipeline.run("What is the latest AAPL stock price?")
            self.assertEqual(seen["path"], "/v1/chat/completions")
            self.assertEqual(seen["payload"]["model"], "test-model")
            self.assertEqual(out["status"], "answered")
            self.assertIn("2026-10-15", out["answer"])
            self.assertIn("[STOCK1]", out["answer"])
            self.assertEqual(out["citations"][0]["metadata"]["url"], "https://finance.yahoo.com/quote/AAPL/")
            self.assertEqual(out["model"], "local-test-provider")
            retriever.retrieve.assert_not_called()
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)


if __name__ == "__main__":
    unittest.main()
