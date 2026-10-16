# Query routing and grounded generation

This change implements remaining items 5 and 6 of the project work list:
composable query routing and LLM answer generation with source citations. It
uses the existing retriever, FRED tool and stock tool. It does not add the
Streamlit UI, annotation records, ablation results or generation-quality scores.

## Shared application interface

```python
from src.rag_pipeline import RAGPipeline

# Reuse this instance across requests to retain the lazily loaded retriever.
pipeline = RAGPipeline()
result = pipeline.run(
    "What does Apple report about margin pressure, and what is its latest stock price?",
    top_k=5,
    rerank=True,
    generate=True,
)

# Useful fields for a future Streamlit UI:
# result['answer']       rendered answer with [SEC1]/[STOCK1]/[FRED1]
# result['citations']    cited source objects with metadata and URLs
# result['sources']      context actually sent to the model
# result['evidence']     all collected evidence, before context budgeting
# result['limitations']  coverage gaps (including failed/omitted evidence)
# result['errors']       sanitized errors by backend and target
# result['timings_s']    per-backend and generation latency
```

`run` also accepts `ticker`, `series`, `year`, `form`, `kind`, and `generate=False`.
Data providers and the generator can be injected in the constructor. Only the
selected providers are initialized. `top_k` is per detected company. The current
retriever's `year` filter applies to filing dates, not fiscal years. Explicit
filters override detected values, including an explicit ticker overriding all
inferred companies.

Routing can be inspected without provider/model calls:

```python
from src.query_router import plan_route

plan = plan_route("JPMorgan interest-rate risks and latest Fed Funds Rate")
assert plan.routes == ("filing", "macro")
assert plan.tickers == ("JPM",)
assert plan.series == ("fed_funds",)
```

The original `route(query) -> (kind, hints)` API still exists. Its single-backend
route names remain `filing`, `stock`, and `macro`. Mixed queries now return
`mixed`, with the route plan included in hints. Third-party callers that switch
on the original three names should migrate to `plan_route` or `RAGPipeline`.

## Evidence and answer contract

Every source has `source_id`, `kind`, `content`, and `metadata`. Source IDs are
stable within one response, not global database IDs. SEC passage IDs remain in
metadata and provide the connection to the corpus/evaluation records.

| Source | IDs | Traceability |
|---|---|---|
| SEC passage | `SEC1`, `SEC2`, ... | passage ID, company, CIK, accession, section, report/filing date, filing index URL |
| Stock snapshot | `STOCK1`, `STOCK2`, ... | ticker, price/currency fields, trading date, Yahoo Finance URL, adjusted daily bar note |
| FRED observations | `FRED1`, `FRED2`, ... | series ID, dated raw values, FRED URL, missing-units note |

All records retain a UTC `retrieved_at` timestamp. This timestamp is distinct
from the date of the filing, market bar, or macro observation. Undated/empty
live records are rejected before generation.

The model is instructed to use only supplied evidence and return JSON:

```json
{
  "status": "answered",
  "claims": [
    {
      "text": "A statement supported by the supplied historical filing, with its date.",
      "source_ids": ["SEC1"]
    }
  ],
  "limitations": []
}
```

This is a schema illustration, not a recorded model result. The application
rejects malformed JSON/schema, empty answered responses, claims without source
IDs, unknown IDs, embedded links/citation markers, refused outputs and truncated
completions. It renders citation markers itself and returns source URLs from
application-owned metadata, rather than trusting model-generated source URLs.
Citation existence does not prove that a source entails the claim. Semantic
faithfulness, relevance and numerical accuracy require separate evaluation.

Context defaults to at most 40,000 characters of serialized evidence and 6,000
characters per SEC passage. It prioritizes live records, keeps their JSON intact,
marks truncated SEC passages, and reports omitted records as limitations. The
budget is measured in characters, not tokens; choose a compatible budget for the
configured model. `sources` is the exact evidence supplied, so omitted records
cannot become valid citations.

## Errors and prerequisites

Build the SEC corpus/indexes using the existing README pipeline before filing
queries. Install `requirements.txt` in the project's recommended Python 3.10 or
3.11 environment. FRED requests require `FRED_API_KEY`; answer generation requires
`LLM_MODEL` and `LLM_API_KEY` (or `OPENAI_API_KEY`). Both are configured through
`.env`. No keys or models are needed for route inspection.

The LLM adapter uses the [Chat Completions request contract](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create)
and [JSON object mode](https://developers.openai.com/api/docs/guides/structured-outputs).
JSON mode ensures JSON syntax; the application separately validates the answer
shape and citations. Compatible providers without JSON mode can set
`LLM_JSON_MODE=false` and must still produce valid JSON. Requests have a
configurable timeout and no automatic retry that could silently incur extra
model calls.

A provider failure is returned in `errors` without raw exception/HTTP response
text, because FRED exception URLs can contain API keys. Successful evidence from
other sources is retained. No evidence produces a deterministic
`insufficient_evidence` result without a model call. Invalid generation produces
`generation_error` with no answer text while retaining evidence for inspection.

Routing is a lightweight heuristic over known names, configured ticker symbols,
macro indicators and intent phrases. It is not a general entity resolver. The
existing live tools return latest/recent data, not arbitrary historical windows;
questions needing unavailable historical values must remain unanswered. CPI
index levels must not be presented as inflation rates, and a daily stock bar
must not be presented as a guaranteed real-time quote. No new live-data caching
layer is added by this change.

## Verification

```bash
# Minimal dependencies for these tests only; no models, indexes or API keys.
python -m pip install requests python-dotenv
python -m unittest discover -s tests -v
```

The suite covers project query categories, static/live phrase collisions,
multi-company/multi-indicator requests, filter overrides, missing inputs,
full-passage preservation, dated sources, failures with partial evidence,
credential redaction, JSON/citation validation, context bounds, CLI compatibility,
and a full stock-question flow through a real HTTP request to a local test
provider. Both CLI entry styles are checked with site packages disabled.

The local HTTP provider returns deterministic test data; it is not a real LLM.
These tests verify control flow and interface contracts, not financial answer
quality. No live FRED, Yahoo Finance, model API, or built SEC-index quality result
is claimed. GitHub Actions runs this offline suite on Python 3.10 and 3.11.
