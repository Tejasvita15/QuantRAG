"""Small Streamlit client for the team's existing shared RAG pipeline."""

import json
import time

import streamlit as st
from dotenv import load_dotenv

from src.common import ROOT
from src.query_router import plan_route
from src.rag_pipeline import RAGPipeline
from src.ui_support import build_options, readiness, safe_source_url


MODES = {
    "Route only": "route",
    "Retrieve evidence": "retrieve",
    "Generate answer": "answer",
}
EXAMPLES = [
    "What risks does Apple identify related to supply chain disruption?",
    "What factors affected Microsoft's cloud revenue growth?",
    "What risks does NVIDIA report related to export restrictions?",
    "What does JPMorgan report about credit risk and loan losses?",
    "What risks does Exxon Mobil report regarding commodity prices?",
    "Apple risks and latest stock price",
    "JPMorgan interest-rate risks and latest Fed Funds Rate",
]


def show_source(source, cited_ids, supplied_ids):
    source_id = source["source_id"]
    metadata = source.get("metadata", {})
    badges = []
    if source_id in cited_ids:
        badges.append("cited")
    if source_id in supplied_ids:
        badges.append("sent to model")
    label = f"{source_id} | {source.get('kind', '')} | {metadata.get('ticker') or metadata.get('series_id') or ''}"
    if badges:
        label += " | " + ", ".join(badges)
    with st.expander(label):
        st.text(source.get("content", ""))
        st.json(metadata)
        url = safe_source_url(metadata.get("url"))
        if url:
            st.link_button("Open original source", url)


def show_result(result):
    st.subheader("Last submitted result")
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
    cols[0].metric("Status", status)
    cols[1].metric("Total latency", f"{result.get('latency_s', 0):.3f} s")
    cols[2].metric("Evidence records", len(result.get("evidence", [])))
    st.caption("Latency includes first-use initialization when applicable; it is not a controlled ablation measurement.")

    answer_tab, evidence_tab, details_tab = st.tabs(["Answer", "Evidence", "Routing & diagnostics"])
    with answer_tab:
        if result.get("answer"):
            st.text(result["answer"])
        elif status == "retrieved":
            st.info("Open the Evidence tab to inspect retrieved passages and live observations.")
        elif status == "route_only":
            st.json(result["routing"])
        for limitation in result.get("limitations", []):
            st.warning(limitation)
        if result.get("usage"):
            st.caption(f"Model: {result.get('model') or 'not reported'} · Provider-reported token usage (not a currency cost):")
            st.json(result["usage"])
    with evidence_tab:
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
        st.json(result["routing"])
        if result["routing"].get("missing_inputs"):
            st.warning("Missing inputs: " + ", ".join(result["routing"]["missing_inputs"]))
        for error in result.get("errors", []):
            st.warning(f"{error['backend']} ({error.get('target') or 'unspecified'}): {error['error']}")
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
    st.title("QuantRAG")
    st.caption("Evidence-grounded financial QA · SEC filings + recent stock data + FRED observations")
    st.info("Research/education only, not investment advice. Stock values are latest available adjusted daily closes, not guaranteed real-time quotes. CPI is an index level, not an inflation percentage.")
    with st.sidebar:
        st.header("Run settings")
        mode_label = st.selectbox("Execution mode", list(MODES), key="execution_mode")
        mode = MODES[mode_label]
        top_k = st.slider("SEC passages per company", 1, 20, 5, key="top_k")
        rerank = st.checkbox("Cross-encoder reranking", True, key="rerank")
        with st.expander("Optional overrides"):
            kind = st.selectbox("Route override", ["auto", "filing", "stock", "macro"], key="kind")
            ticker = st.text_input("One ticker override (e.g. AAPL)", key="ticker")
            series = st.text_input("One FRED series/alias override", key="series")
            year = st.text_input("Filing-date year (blank = infer)", key="year")
            form = st.selectbox("Filing form", ["Auto", "10-K", "10-Q"], key="form")
            st.caption("Year is the filing-date year, NOT fiscal year. Leave ticker blank to detect multiple companies from the question. An unavailable form/year may return no passages.")
        with st.expander("Local readiness", expanded=True):
            for label, ready in checks.items():
                st.write(f"{'✓' if ready else '○'} {label}")
            st.caption("Presence checks only, not provider connectivity or index compatibility tests. Keys are never displayed. Add credentials to the local .env; never commit it.")
        if st.button("Reset session", key="reset_session"):
            st.session_state.pop("_pipeline", None)
            st.session_state.pop("last_response", None)

    with st.expander("Example questions"):
        for example in EXAMPLES:
            st.code(example, language=None)
    with st.form("query_form"):
        query = st.text_area("Question", key="query", height=100,
                             placeholder="Apple risks and latest stock price")
        submitted = st.form_submit_button("Submit", type="primary")
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


if __name__ == "__main__":
    main()
