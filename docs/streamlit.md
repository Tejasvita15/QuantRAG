# Task 7: Streamlit and end-to-end integration

## Run locally

Use Python 3.10/3.11 and the project virtual environment:

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py --server.address 127.0.0.1
```

Keep the server local unless your group intentionally configures authenticated hosting. This app does not implement access control. Never publish a `.env`, provider key, or BM25 pickle from an untrusted party. Only load the team's trusted indexes: Python pickle deserialization can execute code.

The default **Route only** mode works without corpus/index files, embedding models, or provider keys. It calls `plan_route`, not a mock answer generator.

The course delivery uses English interface labels, examples, router keywords and documentation. The generation prompt requests English answers.

**Retrieve evidence** calls the shared `RAGPipeline.run(..., generate=False)`. **Generate answer** calls that same pipeline with `generate=True`, and can incur provider charges. Merely editing settings, expanding a passage, changing tabs, or downloading a result does not submit another request. Results remain in the session and are labelled as the *last submitted* result.

## Inputs and outputs

### Guided research interface

The opening **Start here** panel explains RAG, source selection, hybrid
retrieval, reranking and generation, and connects these stages to the research
question. Three example cards fill the question without submitting it.
The sidebar explains each execution mode; passage count, reranking and
metadata overrides are grouped under **Research settings** with help text.

Use the same question in Route only, Retrieve evidence and Generate answer
to follow the pipeline progressively. Results show a readable source preview,
an evidence and citations tab, and a method and diagnostics tab. Source dates
are visible above each record, while raw metadata remains expandable.
Configuration indicators show whether values are present, not whether an API
connection has succeeded. Editing controls does not resubmit a question.

- The question supports detected companies/indicators and mixed SEC + stock + macro requests.
- Optional overrides: one ticker, one FRED series/alias, route, filing-date year, filing form.
- `top_k` is the number of SEC passages **per company**; it is not a global context budget.
- Filing-date year is not the fiscal year. Multiple years mentioned in a question remain unfiltered unless explicitly overridden.
- Reranking is an independent per-request switch. The retriever loads its cross-encoder when first needed; switching off and then on does not permanently disable it.
- Answers display the pipeline's validated citation IDs, availability limitations and explicit failure/partial/abstention states.
- The Evidence tab displays complete collected passages, stable passage IDs and provider metadata. It distinguishes sources cited in the answer from sources actually supplied to the model. The exact bounded/truncated model context is separately inspectable.
- On `generation_error`, the existing pipeline returns raw evidence rather than the exact sent context. The UI explicitly marks that context as unavailable and does not label those records as "sent to model".
- Download JSON preserves routing, answer, evidence, timings, errors and non-secret run options for review. This is not a substitute for controlled experiment logs.

## Local artifacts and operator configuration

The shared trusted bundle and cached retrieval models were verified on this machine on 24 October. See [the validation record](validation_2026-10-24.md). The following setup remains necessary for a fresh clone; pulling Git alone does not install the ignored data/indexes.

For SEC retrieval, copy a **matching, trusted** set into the project:

```text
indexes/bm25/bm25.pkl
indexes/dense/index.faiss
indexes/dense/metadata.jsonl
```

Retain the corpus manifest/statistics and passage collection for reproducibility. The BM25 store, dense metadata and FAISS index must refer to the same passage version; evaluated IDs must match that version. File-existence checks on the page do not prove this compatibility. BGE and cross-encoder models must be cached or downloadable on first use.

Use a local `.env` based on `.env.example`:

```text
FRED_API_KEY="..."
LLM_MODEL="..."
LLM_API_KEY="..."
LLM_BASE_URL="https://your-provider/v1"
```

Only FRED queries need the FRED key. Stock retrieval requires network/provider availability, but no LLM key. Answer generation needs a model and a compatible provider key. Do not commit or send these keys in chat.

## Manual acceptance checks

1. Start the app without any credentials/indexes. Route-only `Apple risks and latest stock price` must show `mixed`, `filing`, `stock`, `AAPL`, without a provider call.
2. Blank/whitespace question and malformed year must show a validation message, not a traceback.
3. With real indexes, retrieve an Apple risk question: inspect the passage text, filing date, ID and source link; download JSON.
4. Test a mixed query with one backend unavailable. Available evidence must remain visible, with errors/partial status. Do not interpret a partial answer as full coverage.
5. With real credentials, generate an answer and inspect every cited claim against the exact supplied context. Valid citation IDs alone do not prove faithfulness.
6. Switch reranking off and on, submit each time, and inspect the evidence. Changing the checkbox alone must not call the pipeline.
7. After submitting, change controls/download/expand evidence. The previous result must stay visible and no extra paid request should occur.

Stock values are adjusted daily closes, not guaranteed real-time quotes. FRED CPI is a raw index level, not an inflation rate; the present tool may not provide enough dated values to calculate year-on-year inflation. The app deliberately discloses these limits instead of inventing values.

## Automated checks

```bash
python -m unittest discover -s tests -v
```

UI tests use Streamlit's [AppTest](https://docs.streamlit.io/develop/api-reference/app-testing/st.testing.v1.apptest) with injected fake backends. They verify interactions, rendering, rerun persistence and safe failures **without** proving real SEC/provider/LLM quality.

The UI uses a [form](https://docs.streamlit.io/develop/api-reference/execution-flow/st.form) to submit one question at a time; downloading is outside that form. Pipelines are reused within each session, not globally shared mutable resources. Restart the server/reset the session after replacing indexes or changing provider configuration. The teaching echo app is preserved at `examples/app_echo.py` for later exercises.
