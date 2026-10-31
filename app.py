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

MODE_HELP = {
    "route": "Preview which sources the question needs. No data is fetched and GPT is not called.",
    "retrieve": "Search the filings and fetch requested live data. Inspect the evidence before using GPT.",
    "answer": "Retrieve evidence, then ask your configured language model to write a cited answer. Uses your API quota.",
}
SOURCE_LABELS = {"filing": "SEC company filings", "sec": "SEC company filings",
                 "stock": "Recent stock data", "macro": "FRED economic data"}


def show_route(plan):
    """Explain the actual route plan without constructing a backend."""
    routes = plan.get("routes") or ([plan["kind"]] if plan.get("kind") in SOURCE_LABELS else [])
    st.markdown("**Sources selected for this question**")
    descriptions = {
        "filing": "Search stored company disclosures for business performance, strategy and risks.",
        "stock": "Request the latest available adjusted daily stock close and recent returns.",
        "macro": "Request dated economic observations, such as unemployment or interest rates.",
    }
    if routes:
        for column, route in zip(st.columns(len(routes)), routes):
            with column.container(border=True):
                st.markdown(f"**{SOURCE_LABELS.get(route, route)}**")
                st.caption(descriptions.get(route, ""))
    filters = []
    if plan.get("tickers"):
        filters.append("Companies: " + ", ".join(plan["tickers"]))
    if plan.get("series"):
        filters.append("Economic indicators: " + ", ".join(plan["series"]))
    if plan.get("year"):
        filters.append(f"Filing-date year: {plan['year']}")
    if plan.get("form"):
        filters.append(f"Filing type: {plan['form']}")
    if filters:
        st.caption(" · ".join(filters))
    if plan.get("missing_inputs"):
        st.warning("Add the missing information in Research settings: " + ", ".join(plan["missing_inputs"]))


def show_method():
    st.markdown("### How QuantRAG answers a question")
    steps = [
        ("01 · Select sources", "A rule-based router identifies companies, dates and requests for market or economic data."),
        ("02 · Retrieve evidence", "BM25 matches words; BGE embeddings match meaning. Reciprocal Rank Fusion (RRF) combines their rankings."),
        ("03 · Refine the context", "A cross-encoder reranks candidate passages. Requested stock and FRED observations are added with dates."),
        ("04 · Generate and verify", "GPT receives a bounded evidence context. The app validates citation IDs and shows sources for human review."),
    ]
    for col, (title, description) in zip(st.columns(4), steps):
        with col.container(border=True):
            st.markdown(f"**{title}**")
            st.caption(description)
    st.markdown("**Research question:** Does hybrid retrieval with reranking and live financial context improve evidence retrieval and grounded question answering?")
    st.caption("Retrieval quality is evaluated using Precision@K, Recall@K, MRR and nDCG on manually labelled records. Answer faithfulness requires human evaluation; valid citation IDs alone do not prove support.")


def use_example(question):
    st.session_state["query"] = question


def show_source(source, cited_ids, supplied_ids):
    source_id = source["source_id"]
    metadata = source.get("metadata", {})
    badges = []
    if source_id in cited_ids:
        badges.append("cited")
    if source_id in supplied_ids:
        badges.append("sent to model")
    label = f"{source_id} | {SOURCE_LABELS.get(source.get('kind'), source.get('kind', ''))} | {metadata.get('ticker') or metadata.get('series_id') or ''}"
    if badges:
        label += " | " + ", ".join(badges)
    with st.expander(label):
        date = metadata.get("filing_date") or metadata.get("trading_date") or metadata.get("latest_observation_date")
        if date:
            st.caption(f"Evidence date: {date}")
        st.text(source.get("content", ""))
        with st.expander("Source metadata and stable identifiers"):
            st.json(metadata)
        url = safe_source_url(metadata.get("url"))
        if url:
            st.link_button("Open original source", url)


def show_result(result):
    st.divider()
    st.subheader("Your research result")
    st.text(result["query"])
    st.caption(f"Execution mode: {result['ui']['execution_mode']} · Settings changes take effect on the next submission.")
    if result["ui"]["execution_mode"] == "route":
        st.info("Routing preview only: no retrieval, market-data request or LLM call was made.")
    status = result.get("status", "route_only")
    messages = {
        "answered": (st.success, "Answer generated. Citation IDs are validated, but factual support still needs review."),
        "partial": (st.warning, "Partial answer: some requested information is unavailable."),
        "insufficient_evidence": (st.warning, "Insufficient evidence. No supported complete answer is available."),
        "generation_error": (st.error, "Answer generation failed. Available evidence is preserved below."),
        "retrieved": (st.success, "Evidence retrieved. No LLM call was made."),
    }
    if status in messages:
        display, message = messages[status]
        display(message)
    cols = st.columns(3)
    cols[0].metric("Outcome", {"route_only": "Sources previewed", "retrieved": "Evidence ready", "answered": "Answer ready", "partial": "Partial coverage", "generation_error": "Generation failed", "insufficient_evidence": "Insufficient evidence"}.get(status, status))
    cols[1].metric("Total latency", f"{result.get('latency_s', 0):.3f} s")
    cols[2].metric("Evidence records", len(result.get("evidence", [])))
    st.caption("Latency includes first-use initialization when applicable; it is not a controlled ablation measurement.")

    summary_label = {"route_only": "Source preview", "retrieved": "Retrieval summary"}.get(status, "Answer")
    answer_tab, evidence_tab, details_tab = st.tabs([f"1 · {summary_label}", "2 · Evidence & citations", "3 · Method & diagnostics"])
    with answer_tab:
        if result.get("answer"):
            st.text(result["answer"])
        elif status == "retrieved":
            st.info("Open the Evidence tab to inspect retrieved passages and live observations.")
        elif status == "route_only":
            show_route(result["routing"])
            st.info("Next: choose Retrieve evidence to inspect the sources, or Generate answer for a cited explanation. Then submit again.")
        for limitation in result.get("limitations", []):
            st.warning(limitation)
        if result.get("usage"):
            st.caption(f"Model: {result.get('model') or 'not reported'} · Provider-reported token usage (not a currency cost):")
            st.json(result["usage"])
    with evidence_tab:
        st.markdown("Match each citation in the answer, such as **[SEC1]**, to the corresponding record below. Check the evidence text and date before accepting a claim.")
        evidence = result.get("evidence", [])
        cited_ids = {s["source_id"] for s in result.get("citations", [])}
        # On generation failure the existing pipeline restores all raw evidence
        # in `sources`; it does not guarantee a record of the actual sent context.
        context_known = status not in ("generation_error", "retrieved", "route_only")
        supplied_ids = {s["source_id"] for s in result.get("sources", [])} if context_known else set()
        if not evidence:
            st.info("No evidence collected in this run.")
        for source in evidence:
            show_source(source, cited_ids, supplied_ids)
        if context_known and result.get("sources"):
            with st.expander("Exact context supplied to the model (including truncation flags)"):
                st.json(result["sources"])
        elif status == "generation_error":
            st.warning("Generation failed: the exact sent context is not available from this pipeline result. The records above are collected evidence, not proof of what reached the model.")
        st.caption("SEC1 etc. identify evidence within this answer only. Use metadata.id for stable passage IDs. Filing links open the filing index, not a highlighted passage.")
    with details_tab:
        show_route(result["routing"])
        st.markdown("**Settings used for this submission**")
        st.json(result["ui"].get("requested_options", {}), expanded=False)
        if result["routing"].get("missing_inputs"):
            st.warning("Missing inputs: " + ", ".join(result["routing"]["missing_inputs"]))
        for error in result.get("errors", []):
            st.warning(f"{error['backend']} ({error.get('target') or 'unspecified'}): {error['error']}")
        with st.expander("Source selection and stage timings"):
            st.json(result["routing"])
            st.json(result.get("timings_s", {}))
        with st.expander("Full result JSON"):
            st.json(result)
    st.download_button(
        "Download result JSON",
        json.dumps(result, ensure_ascii=False, indent=2),
        file_name="quantrag-result.json", mime="application/json", key="download_result",
    )


def main():
    st.set_page_config(page_title="QuantRAG", page_icon="📄", layout="wide")
    load_dotenv(ROOT / ".env")
    checks = readiness()
    st.html("""<style>
    .block-container {max-width:1200px;padding-top:2rem;}
    h1,h2,h3 {letter-spacing:-.025em;}
    [data-testid="stSidebar"] {border-right:1px solid #cbd5e133;}
    div[data-testid="stMetric"] {border-bottom:3px solid #149e92;padding:.6rem 0;}
    .q-eyebrow {color:#149e92;font-size:.78rem;font-weight:700;letter-spacing:.16em;}
    .q-intro {font-size:1.13rem;line-height:1.6;max-width:780px;opacity:.85;}
    </style><div class="q-eyebrow">AI6127 · FINANCIAL QUESTION ANSWERING</div>""")
    st.title("QuantRAG Research Assistant")
    st.html('<p class="q-intro">Ask a financial question. Examine the evidence. Trace the answer back to its sources.</p>')
    with st.expander("Start here · What does this project do?", expanded="last_response" not in st.session_state):
        st.write("QuantRAG is a retrieval-augmented generation (RAG) system: it finds relevant financial evidence before asking a language model to explain it. Company filings provide historical disclosures; stock and FRED tools add dated market and economic observations.")
        show_method()
        st.markdown("**First use:** preview the sources → retrieve and inspect evidence → generate an answer. Use the same question at each stage to see what changes.")
    with st.sidebar:
        st.header("Research workspace")
        st.caption("Choose a stage, then submit a question in the main panel.")
        mode_label = st.selectbox("Execution mode", list(MODES), key="execution_mode", help="Route only explains source selection; Retrieve evidence gathers data; Generate answer adds a language model response.")
        mode = MODES[mode_label]
        st.info(MODE_HELP[mode])
        with st.expander("Research settings"):
            top_k = st.slider("SEC passages per company", 1, 20, 5, key="top_k", help="K is the number of filing passages kept per company. More passages provide more context but may increase latency and model input.")
            rerank = st.checkbox("Cross-encoder reranking", True, key="rerank", help="A second model scores each question–passage pair after hybrid retrieval. Disable this to explore the effect of reranking.")
            st.caption("SEC search uses BM25 + BGE dense retrieval, combined with RRF. This switch controls the additional reranking stage.")
            kind = st.selectbox("Route override", ["auto", "filing", "stock", "macro"], key="kind")
            ticker = st.text_input("One ticker override (e.g. AAPL)", key="ticker")
            series = st.text_input("One FRED series/alias override", key="series")
            year = st.text_input("Filing-date year (blank = infer)", key="year")
            form = st.selectbox("Filing form", ["Auto", "10-K", "10-Q"], key="form")
            st.caption("Year is the filing-date year, NOT fiscal year. Leave ticker blank to detect multiple companies from the question. An unavailable form/year may return no passages.")
        with st.expander("Connection configuration", expanded=False):
            for label, ready in checks.items():
                st.write(f"{'✓' if ready else '○'} {label}")
            st.caption("Presence checks only, not provider connectivity or index compatibility tests. Keys are never displayed. Add credentials to the local .env; never commit it.")
        if st.button("Reset session", key="reset_session"):
            st.session_state.pop("_pipeline", None)
            st.session_state.pop("last_response", None)

    st.subheader("1. Choose a research question")
    st.caption("Start with a company disclosure question, then add market or economic context. Example buttons fill the question; they do not run it.")
    examples = [
        ("Company filings", EXAMPLES[0]),
        ("Filings + stock", "What supply-chain risks does Apple report, and what is its latest stock price?"),
        ("Filings + economy", EXAMPLES[-1]),
    ]
    for col, (label, question) in zip(st.columns(3), examples):
        with col.container(border=True):
            st.markdown(f"**{label}**")
            st.caption(question)
            st.button("Use this question", key=f"example_{label}", on_click=use_example, args=(question,), use_container_width=True)
    with st.expander("More example questions"):
        for example in EXAMPLES:
            st.code(example, language=None)
    with st.form("query_form"):
        query = st.text_area("Question", key="query", height=100,
                             placeholder="Apple risks and latest stock price")
        st.caption(f"Selected stage: {mode_label}. {MODE_HELP[mode]}")
        submitted = st.form_submit_button("Submit", type="primary", use_container_width=True)
    if submitted:
        st.session_state.pop("last_response", None)
        try:
            options = build_options(ticker, series, year, form, kind, top_k, rerank)
            clean_query = query.strip()
            plan = plan_route(clean_query, **{k: v for k, v in options.items() if k not in ("top_k", "rerank")})
        except ValueError as exc:
            # Only deterministic local validation messages are shown here.
            st.warning(str(exc))
        else:
            if mode == "answer" and not checks["LLM model + key configured"]:
                st.error("Set LLM_MODEL and LLM_API_KEY (or OPENAI_API_KEY) in .env, or choose Retrieve evidence. No provider call was made.")
            else:
                with st.spinner("Processing… First SEC query may load/download embedding and reranking models."):
                    started = time.perf_counter()
                    try:
                        if mode == "route":
                            result = {"query": clean_query, "routing": plan.to_dict(), "status": "route_only",
                                      "latency_s": round(time.perf_counter() - started, 4)}
                        else:
                            # Per-session reuse, not a global mutable pipeline shared by users.
                            if "_pipeline" not in st.session_state:
                                st.session_state["_pipeline"] = RAGPipeline()
                            result = st.session_state["_pipeline"].run(clean_query, **options, generate=mode == "answer")
                        result["ui"] = {"execution_mode": mode, "requested_options": options}
                        st.session_state["last_response"] = result
                    except Exception:
                        # Never render raw backend tracebacks: they may contain credentials.
                        st.error("Request failed unexpectedly. Check local dependencies/configuration and try again. Raw provider errors are hidden to protect credentials.")
    if "last_response" in st.session_state:
        show_result(st.session_state["last_response"])
    st.divider()
    st.caption("Research and education · Stock observations are adjusted daily closes, not guaranteed real-time quotes. FRED CPI is an index level, not an inflation percentage. Historical filings and latest observations describe different dates.")
    st.caption("This product uses the FRED® API but is not endorsed or certified by the Federal Reserve Bank of St. Louis.")


if __name__ == "__main__":
    main()
