import unittest

from src.query_router import detect_ticker, detect_tickers, plan_route, route


class RoutingTests(unittest.TestCase):
    def test_project_query_categories(self):
        cases = [
            ("What risks does NVIDIA report regarding export restrictions?", ("filing",)),
            ("How has Apple discussed supply-chain risk over recent filings?", ("filing",)),
            ("What did Microsoft report about cloud revenue in its 2024 10-K?", ("filing",)),
            ("What does Apple report about margin pressure, and what is its latest available stock price?", ("filing", "stock")),
            ("What interest-rate risks does JPMorgan discuss, and what is the latest Fed Funds Rate?", ("filing", "macro")),
            ("Apple risks, latest stock price and current unemployment", ("filing", "stock", "macro")),
            ("How is Apple's stock doing?", ("stock",)),
            ("What is Apple's price?", ("stock",)),
            ("What is the current inflation rate?", ("macro",)),
            ("Latest GDP and unemployment rate", ("macro",)),
            ("Annual report risks and latest GDP", ("filing", "macro")),
            ("Apple annual report risks and latest stock price and current interest rate", ("filing", "stock", "macro")),
        ]
        for query, expected in cases:
            with self.subTest(query=query):
                self.assertEqual(plan_route(query).routes, expected)

    def test_static_financial_terms_do_not_call_live_tools(self):
        for query in (
            "JPMorgan interest rate risks in its 2024 10-K",
            "Apple stock-based compensation in 2024 10-K",
            "Apple share-based compensation",
            "Apple stock repurchases in 2024",
            "Apple shareholder risks",
            "Apple product price risks",
            "NVIDIA shares outstanding in its annual report",
            "What inflation risks did Apple report in 2024?",
            "Goldman Sachs trading risks in its 2024 10-K",
            "Apple stock price risk in its latest annual report",
            "JPMorgan interest rate risks in its latest 10-K",
            "Interest-rate risks in Apple's latest annual report",
        ):
            with self.subTest(query=query):
                self.assertEqual(plan_route(query).routes, ("filing",))

    def test_word_boundaries(self):
        self.assertEqual(plan_route("Apple enterprise pricing and shareholder risks").routes, ("filing",))
        self.assertEqual(plan_route("industrialgdp consumer prices").series, ())
        self.assertIsNone(detect_ticker("What is the cost of revenue?"))

    def test_multiple_companies_and_indicators(self):
        plan = plan_route("Compare Apple's stock with MSFT and Apple's stock; current CPI and GDP")
        self.assertEqual(plan.tickers, ("AAPL", "MSFT"))
        self.assertEqual(plan.series, ("cpi", "gdp"))
        self.assertEqual(plan.routes, ("stock", "macro"))
        self.assertEqual(detect_tickers("$BRK-B and $V and aapl"), ("BRK-B", "V", "AAPL"))

    def test_metadata_and_explicit_overrides(self):
        plan = plan_route("Microsoft cloud revenue 2024 10-k")
        self.assertEqual((plan.tickers, plan.year, plan.form), (("MSFT",), 2024, "10-K"))
        plan = plan_route("Apple 2024 10-K risks", ticker="nvda", year=2023, form="10-Q")
        self.assertEqual((plan.tickers, plan.year, plan.form), (("NVDA",), 2023, "10-Q"))
        self.assertIsNone(plan_route("Apple revenue trends from 2021 to 2025").year)
        self.assertIsNone(plan_route("Apple 10-K versus 10-Q").form)

    def test_override_is_applied_before_classification(self):
        self.assertEqual(plan_route("latest price", ticker="NVDA").tickers, ("NVDA",))
        plan = plan_route("interest rate risks", ticker="JPM", series="fed_funds")
        self.assertEqual(plan.routes, ("filing", "macro"))
        self.assertEqual(plan_route("Apple latest stock price", kind="filing").routes, ("filing",))

    def test_missing_inputs_and_invalid_queries(self):
        self.assertEqual(plan_route("latest stock price").missing_inputs, ("ticker",))
        self.assertEqual(plan_route("question", kind="macro").missing_inputs, ("series",))
        for query in ("", "  ", None):
            with self.subTest(query=query), self.assertRaises(ValueError):
                plan_route(query)
        with self.assertRaises(ValueError):
            plan_route("price", ticker="invalid/symbol")

    def test_tuple_api(self):
        self.assertEqual(route("Apple stock price"), ("stock", {"ticker": "AAPL"}))
        self.assertEqual(route("current GDP"), ("macro", {"series": "gdp"}))
        self.assertEqual(route("supply-chain risks"), ("filing", {}))
        self.assertEqual(route("Apple risks and latest stock price")[0], "mixed")


if __name__ == "__main__":
    unittest.main()
