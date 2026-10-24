import unittest
from unittest.mock import Mock

from src.benchmark_queries import QUERIES, benchmark
from src.query_router import plan_route


class BenchmarkTests(unittest.TestCase):
    def test_default_smoke_companies_and_routing(self):
        self.assertEqual([ticker for ticker, _ in QUERIES], ["AAPL", "MSFT", "NVDA", "AMZN", "CVX"])
        for ticker, query in QUERIES:
            plan = plan_route(query)
            self.assertEqual(plan.routes, ("filing",))
            self.assertEqual(plan.tickers, (ticker,))

    def test_five_queries_full_text_and_separate_warmup(self):
        retriever = Mock()
        text = "Full passage. " * 100
        retriever.retrieve.return_value = [{"id": "p1", "rank": 1, "passage": {"id": "p1", "text": text}}]
        rows = benchmark(retriever)
        self.assertEqual(len(rows), 5)
        self.assertEqual(retriever.retrieve.call_count, 6)
        self.assertEqual(rows[0]["results"][0]["text"], text)
        self.assertEqual(rows[0]["results"][0]["passage"]["text"], text)
        self.assertTrue(all(row["latency_s"] >= 0 for row in rows))

    def test_no_warmup(self):
        retriever = Mock()
        retriever.retrieve.return_value = []
        self.assertEqual(len(benchmark(retriever, warmup=False)), 5)
        self.assertEqual(retriever.retrieve.call_count, 5)


if __name__ == "__main__":
    unittest.main()
