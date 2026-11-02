"""Small Streamlit client for the team's existing shared RAG pipeline."""

import json
import time

import streamlit as st
from dotenv import load_dotenv

from src.benchmark_queries import QUERIES
from src.common import ROOT
from src.query_router import plan_route
from src.rag_pipeline import RAGPipeline
from src.ui_support import build_options, readiness, safe_source_url


MODES = {
    "Route only": "route",
    "Retrieve evidence": "retrieve",
    "Generate answer": "answer",
}
EXAMPLES = [query for _, query in QUERIES] + [
    "Apple risks and latest stock price",
    "Microsoft interest-rate risks and latest Fed Funds Rate",
]

SOURCE_LABELS = {"filing": "SEC company filings", "sec": "SEC company filings",
                 "stock": "Recent stock data", "macro": "FRED economic data"}


def show_route(plan):
    routes = plan.get("routes") or ([plan["kind"]] if plan.get("kind") in SOURCE_LABELS else [])
    st.markdown("**Sources selected for this question**")
    for route in routes:
        st.markdown(f"- {SOURCE_LABELS.get(route, route)}")
    if plan.get("tickers"):
        st.caption("Companies: " + ", ".join(plan["tickers"]))
    if plan.get("missing_inputs"):
        st.warning("Add the missing information in Optional settings: " + ", ".join(plan["missing_inputs"]))


def use_example(question):
    st.session_state["query"] = question


def show_source(source, cited_ids):
    metadata = source.get("metadata", {})
    kind = source.get("kind", "")
    label = f"{source['source_id']} · {SOURCE_LABELS.get(kind, kind)}"
    subject = metadata.get("ticker") or metadata.get("series_id")
    if subject:
        label += f" · {subject}"
    if source["source_id"] in cited_ids:
        label += " · cited in answer"
    with st.expander(label):
        date_fields = (("filing_date", "Filed"), ("trading_date", "Trading date"),
                       ("latest_observation_date", "Latest observation"))
        for field, title in date_fields:
            if metadata.get(field):
                st.caption(f"{title}: {metadata[field]}")
        content = source.get("content", "")
        try:
            live = json.loads(content) if kind in {"stock", "macro"} else None
        except (ValueError, TypeError):
            live = None
        if kind == "macro" and isinstance(live, dict) and isinstance(live.get("observations"), list):
            st.caption("Units: " + str(metadata.get("units") or live.get("units") or "not supplied by this source"))
            st.table(live["observations"])
        elif kind == "stock" and isinstance(live, dict):
            names = {"price": "Adjusted daily close", "previous_close": "Previous close",
                     "return_1d_pct": "1-day return (%)", "return_5d_pct": "5-day return (%)",
                     "volume": "Volume", "market_cap": "Market capitalization", "currency": "Currency"}
            st.table([{"Field": title, "Value": str(live[key])} for key, title in names.items()
                      if live.get(key) is not None])
            st.caption("Latest available adjusted daily bar; not a guaranteed real-time quote.")
        else:
            st.text(content)
        url = safe_source_url(metadata.get("url"))
        if url:
            st.link_button("Open original source", url)
        with st.expander("Source details"):
            st.json(metadata)


def show_result(result):
    st.divider()
    status = result.get("status", "route_only")
    st.subheader("Result")
    st.text(result["query"])
    st.caption(f"Last submitted question · {len(result.get('evidence', []))} sources · {result.get('latency_s', 0):.2f} seconds. Changes apply on your next submission.")
    if status == "route_only":
        st.info("Source preview only. No evidence was retrieved and no language model was called.")
        show_route(result["routing"])
        st.info("Next: choose Retrieve evidence to inspect the sources, or Generate answer for a cited explanation. Then submit again.")
    else:
        messages = {
            "partial": (st.warning, "Partial answer: some requested information is unavailable. See the limitations below."),
            "insufficient_evidence": (st.warning, "Insufficient evidence to answer this question."),
            "generation_error": (st.error, "Answer generation failed. You can still inspect the available sources below."),
            "retrieved": (st.success, "Evidence retrieved. Expand a source below to read it. No language model was called."),
        }
        if status in messages:
            display, message = messages[status]
            display(message)
        if result.get("answer"):
            st.markdown("### Answer")
            st.text(result["answer"])
        for limitation in result.get("limitations", []):
            st.warning(limitation)
        st.markdown("### Sources")
        st.caption("Citation labels such as [SEC1] match the sources below. Check the text and date to verify a claim.")
        cited_ids = {s["source_id"] for s in result.get("citations", [])}
        if not result.get("evidence"):
            st.info("No source records are available for this question.")
        for source in result.get("evidence", []):
            show_source(source, cited_ids)
    with st.expander("Technical details"):
        st.caption("Elapsed time may include first-use model loading. Citation validation checks source IDs, not factual support.")
        st.json(result["routing"])
        st.json(result.get("timings_s", {}))
        for error in result.get("errors", []):
            st.warning(f"{error['backend']}: {error['error']}")
        if result.get("usage"):
            st.caption(f"Model: {result.get('model') or 'not reported'} · Token usage, not currency cost")
            st.json(result["usage"])
        context_known = status not in ("generation_error", "retrieved", "route_only")
        if context_known and result.get("sources"):
            with st.expander("Exact context supplied to the model (including truncation flags)"):
                st.json(result["sources"])
        elif status == "generation_error":
            st.warning("The exact sent context is not available from this failed generation. Collected sources are retained for inspection.")
        with st.expander("Full result JSON"):
            st.json(result)
        st.download_button("Download result JSON", json.dumps(result, ensure_ascii=False, indent=2),
                           file_name="quantrag-result.json", mime="application/json", key="download_result")


MODE_DESCRIPTIONS = {
    "Route only": "Preview how the question is routed. No indexes, models or API keys required.",
    "Retrieve evidence": "Search SEC filings, recent stock data and macro indicators, then inspect every source.",
    "Generate answer": "Retrieve evidence and generate a cited answer with your LLM provider.",
}

MODE_ICONS = {"Route only": "🔎", "Retrieve evidence": "📚", "Generate answer": "✨"}


def show_landing():
    st.title("QuantRAG")
    st.write("Ask about company filings, recent stock prices or economic indicators. Read the evidence and follow the sources.")
    st.write("")
    cols = st.columns(3)
    for col, mode_label in zip(cols, list(MODES)):
        with col:
            st.markdown(
                f'<h3 style="font-size:20px;font-weight:700;margin:0 0 10px;">{MODE_ICONS[mode_label]} {mode_label.title().replace(" ", "<br>")}</h3>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div style="min-height:66px;color:#555867;font-size:14px;line-height:1.5;">{MODE_DESCRIPTIONS[mode_label]}</div>',
                unsafe_allow_html=True,
            )
            if st.button("Open →", key=f"open_{mode_label.replace(' ', '_').lower()}", use_container_width=True):
                st.session_state["execution_mode"] = mode_label
                st.session_state["page"] = "query"
                st.rerun()


def show_query_page(checks):
    mode_label = st.session_state.get("execution_mode", "Retrieve evidence")
    mode = MODES[mode_label]

    bar = st.columns([2, 3])
    with bar[0]:
        st.markdown(f"**{MODE_ICONS[mode_label]} {mode_label.title()}**")
    with bar[1]:
        if st.button("← Back to modes", key="back_to_modes"):
            st.session_state["page"] = "landing"
            st.rerun()

    with st.form("query_form"):
        st.markdown("### Your question")
        query = st.text_area("Your question", key="query", height=140, label_visibility="collapsed",
                             placeholder="What supply-chain risks does Apple report?")
        submitted = st.form_submit_button("Submit", type="primary", use_container_width=True)
    st.caption(f"Mode: {mode_label.title()} · Ask in English · press Submit to run")

    with st.expander("Example questions", expanded=False):
        examples = [("Company filings", EXAMPLES[0]),
                    ("Filings + stock", "What supply-chain risks does Apple report, and what is its latest stock price?"),
                    ("Filings + economy", EXAMPLES[-1])]
        for label, question in examples:
            st.button(question, key=f"example_{label}", on_click=use_example, args=(question,))
        st.caption("Examples fill the question box; press Submit to run it.")
        for question in EXAMPLES:
            st.code(question, language=None)

    with st.expander("Optional settings", expanded=False):
        ticker = st.text_input("Company ticker (optional)", key="ticker", help="Leave blank to detect companies in your question.")
        year = st.text_input("Filing-date year (optional)", key="year", help="Filters the filing date, not the fiscal year.")
        form = st.selectbox("Filing type", ["Auto", "10-K", "10-Q"], key="form")
        top_k = st.slider("Sources per company", 1, 20, 5, key="top_k")
        rerank = st.checkbox("Improve passage ranking", True, key="rerank", help="Uses a cross-encoder reranker; may take longer.")
        with st.expander("Advanced source settings"):
            kind = st.selectbox("Source override", ["auto", "filing", "stock", "macro"], key="kind")
            series = st.text_input("FRED series or alias (optional)", key="series", help="For example: fed_funds, unemployment or treasury_10y.")
        with st.expander("Setup status"):
            for label, ready in checks.items():
                st.write(f"{'✓' if ready else '○'} {label}")
            st.caption("Configuration checks only; provider connectivity has not been tested here. Set keys in your local .env.")
        if st.button("Reset session", key="reset_session"):
            st.session_state.pop("_pipeline", None)
            st.session_state.pop("last_response", None)

    if submitted:
        st.session_state.pop("last_response", None)
        try:
            options = build_options(ticker, series, year, form, kind, top_k, rerank)
            clean_query = query.strip()
            plan = plan_route(clean_query, **{k: v for k, v in options.items() if k not in ("top_k", "rerank")})
        except ValueError as exc:
            st.warning(str(exc))
        else:
            if mode == "answer" and not checks["LLM model + key configured"]:
                st.error("Set LLM_MODEL and LLM_API_KEY (or OPENAI_API_KEY) in .env, or choose Retrieve evidence. No provider call was made.")
            else:
                with st.spinner("Finding sources… The first filing search may take longer while models load."):
                    started = time.perf_counter()
                    try:
                        if mode == "route":
                            result = {"query": clean_query, "routing": plan.to_dict(), "status": "route_only",
                                      "latency_s": round(time.perf_counter() - started, 4)}
                        else:
                            if "_pipeline" not in st.session_state:
                                st.session_state["_pipeline"] = RAGPipeline()
                            result = st.session_state["_pipeline"].run(clean_query, **options, generate=mode == "answer")
                        result["ui"] = {"execution_mode": mode, "requested_options": options}
                        st.session_state["last_response"] = result
                    except Exception:
                        st.error("Request failed. Check local dependencies, provider configuration and network access. Raw errors are hidden to protect credentials.")

    if "last_response" in st.session_state:
        show_result(st.session_state["last_response"])

    st.caption("Research use only. Stock prices are adjusted daily closes, not real-time quotes. CPI observations are index levels, not inflation percentages.")
    st.caption("This product uses the FRED® API but is not endorsed or certified by the Federal Reserve Bank of St. Louis.")


def main():
    st.set_page_config(page_title="QuantRAG", page_icon="📄", layout="centered")
    load_dotenv(ROOT / ".env")
    checks = readiness()
    st.session_state.setdefault("page", "landing")
    st.session_state.setdefault("execution_mode", "Retrieve evidence")

    if st.session_state["page"] == "landing":
        show_landing()
        return

    show_query_page(checks)


if __name__ == "__main__":
    main()
