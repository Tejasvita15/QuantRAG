# QuantRAG — Retrieval and Grounded Financial QA

This repository implements SEC retrieval, live-data routing and evidence-grounded answer generation for the AI6127 QuantRAG project:

- SEC EDGAR 10-K ingestion
- HTML cleaning and table-preserving text extraction
- fixed-size overlapping chunking
- corpus statistics
- sparse BM25 retrieval
- dense BGE retrieval with FAISS
- hybrid Reciprocal Rank Fusion (RRF)
- cross-encoder reranking
- ticker/year/form metadata filtering
- five-query latency benchmark
- Precision@K, Recall@K, MRR and nDCG evaluation
- FRED live macro tool
- current/recent stock-market tool using yfinance
- composable SEC / stock / macro query routing
- LLM answers with validated source IDs, dates and inspectable citations

## 1. Create environment

Recommended: Python 3.10 or 3.11.

```bash
python -m venv .venv
source .venv/bin/activate      # Linux/macOS
# .venv\Scripts\activate       # Windows PowerShell

pip install --upgrade pip
pip install -r requirements.txt
```

If you have a CUDA PyTorch environment already, keep that PyTorch install and install the remaining requirements.

## 2. Configure credentials

```bash
cp .env.example .env
```

Edit `.env`:

```text
SEC_USER_AGENT="NTU AI6127 QuantRAG your_real_email@ntu.edu.sg"
FRED_API_KEY="your_fred_key"
```

SEC requires automated clients to identify themselves. FRED requires a registered API key. Answer generation also requires `LLM_MODEL` and `LLM_API_KEY` (or `OPENAI_API_KEY`). Set `LLM_BASE_URL` for another OpenAI-compatible Chat Completions provider. Retrieval and `--route-only` do not require LLM credentials.

## 3. Run the entire retrieval pipeline

Linux/macOS:

```bash
bash scripts/run_step1.sh
```

Or run each stage manually:

```bash
python src/download_sec.py
python src/prepare_corpus.py
python src/corpus_stats.py
python src/build_bm25.py
python src/build_dense.py
python src/benchmark_queries.py
```

## 4. Confirm assignment corpus requirements

Open:

```text
data/processed/corpus_stats.json
```

You want both:

```json
"requirement_10000_passages": true,
"requirement_100000_words": true
```

The default config downloads 2021–2025 10-Ks for 30 large companies. If 10,000 passages is not reached, add more tickers to `config.json` or add `"10-Q"` to `forms`, then rerun download + preprocessing + indexing.

## 5. Search

```bash
python src/search.py \
  "What risks does Apple identify related to supply chain disruption?" \
  --ticker AAPL
```

Metadata examples:

```bash
python src/search.py \
  "What affected revenue growth?" \
  --ticker MSFT \
  --year 2024 \
  --form 10-K
```

## 6. Five required example queries + query speed

```bash
python src/benchmark_queries.py
```

Output:

```text
eval/five_query_results.json
```

This records each query, retrieved results and latency.

## 7. Retrieval evaluation

Create:

```text
eval/retrieval_eval.jsonl
```

Each row:

```json
{
  "query_id": "q001",
  "query": "What risks does Apple identify related to supply chain disruption?",
  "ticker": "AAPL",
  "relevance": {
    "AAPL_2024-11-01_10K_00123": 2,
    "AAPL_2024-11-01_10K_00124": 1
  }
}
```

Grades:
- 0 = not relevant
- 1 = partially relevant
- 2 = directly relevant

Then:

```bash
python src/evaluate_retrieval.py --mode hybrid
python src/evaluate_retrieval.py --mode hybrid_no_rerank
```

Metrics:
- Precision@5
- Recall@5
- Recall@10
- MRR
- nDCG@10
- average query latency

For the final assignment, manually label the required held-out records with your team and document inter-annotator agreement. Do not treat an existing benchmark's labels as your own held-out annotation.

## 8. FRED live macro data

Get a FRED API key and set it in `.env`.

```bash
python src/live/fred_tool.py fed_funds
python src/live/fred_tool.py cpi
python src/live/fred_tool.py unemployment
```

Supported aliases:
- `cpi` → CPIAUCSL
- `unemployment` → UNRATE
- `fed_funds` → FEDFUNDS
- `gdp` → GDP
- `treasury_10y` → DGS10

You can also pass any FRED series ID.

Or through the query router:

```bash
python src/query.py "What is the current inflation rate?"
```

## 9. Current/recent stocks

```bash
python src/live/stock_tool.py AAPL
python src/live/stock_tool.py NVDA
```

Returns latest available trading price, previous close, 1-day return, 5-day return, volume and market cap when available.

Or through the query router:

```bash
python src/query.py "How is Apple's stock doing?"
```

## 10. Query routing (corpus + live data)

`src/query_router.py` creates a route plan. Sources can be combined rather than
choosing a single backend for every question.

| Route | Example | Sources |
|-------|---------|---------|
| `filing` | Microsoft cloud revenue in its 2024 10-K | SEC, inferred company/year/form filters |
| `stock` | Latest Apple stock price | Stock snapshot |
| `macro` | Current Fed Funds Rate | FRED observations |
| `mixed` | Apple margin risks and latest stock price | SEC + stock |
| `mixed` | JPMorgan interest-rate risks and latest Fed Funds Rate | SEC + FRED |
| `mixed` | Apple risks, current stock price and current unemployment | SEC + stock + FRED |

Historical filing discussion of interest rates, stock compensation or trading
risks does not by itself request live data. Company names, configured tickers,
explicit `$SYMBOL`s, multiple companies and multiple macro indicators are
supported. Explicit flags override detected values. Missing stock tickers or
macro indicators are returned as missing inputs; no symbol is guessed.

```bash
# Inspect routing without indexes, models, API keys or provider calls.
python src/query.py "Apple risks and latest stock price" --route-only

# Retrieve evidence (the existing single-source result shape is retained).
python src/query.py "current price" --type stock --ticker NVDA
python src/query.py "10-year yield" --type macro --series treasury_10y
python src/query.py "supply chain risks" --type filing --ticker AAPL --top-k 5
```

Options: `--type auto|stock|macro|filing`, `--ticker`, `--series`, `--year`,
`--form`, `--top-k`, `--no-rerank`, `--route-only`, `--answer`.
`--year` means **filing-date year**, matching the existing retriever; it is not
a fiscal-year filter. A single mentioned year/form can be inferred. Queries
mentioning multiple years are left unfiltered for historical comparisons.
`--top-k` applies per company. Routing remains deterministic and heuristic.

## 11. Grounded LLM answers

Set the following in `.env` (see `.env.example`):

```text
LLM_MODEL="your-provider-model"
LLM_API_KEY="your-key"
LLM_BASE_URL="https://api.openai.com/v1"
LLM_TIMEOUT_SECONDS="60"
LLM_JSON_MODE="true"
```

`OPENAI_API_KEY` is accepted as a fallback. No model is hardcoded. The provider
must support Chat Completions; set `LLM_JSON_MODE=false` if JSON object mode is
unavailable. Output must still be valid JSON and pass local validation.

```bash
python src/query.py \
  "What does Apple report about margin pressure, and what is its latest available stock price?" \
  --answer

python -m src.query \
  "What interest-rate risks does JPMorgan discuss, and what is the latest Fed Funds Rate?" \
  --answer
```

`src/rag_pipeline.py` gathers full ranked passages and requested live records.
`src/generator.py` conditions the model on those records and requests structured
claims with source IDs. The application validates those IDs and renders
`[SEC1]`, `[STOCK1]` and `[FRED1]` next to each claim. Sources retain SEC passage
IDs, accession/CIK, report/filing dates, stock trading dates, FRED observation
dates, retrieval timestamps and provider URLs. SEC URLs lead to the filing
index because the corpus does not store a primary-document URL.

JSON output includes `answer`, `status`, `claims`, `citations`, `sources`,
`evidence`, `routing`, `errors`, `timings_s`, `latency_s`, `model` and `usage`.
`citations` contains the cited source records; `sources` contains the bounded
context actually supplied to the model; `evidence` contains all collected
records. Existing `query`, `route`, `latency_s` and `result` keys remain available
without `--answer`. Mixed results are grouped by backend.

- No evidence: return `insufficient_evidence` without a model call.
- Some tools fail: keep available evidence and mark an otherwise answered result
  `partial`, with explicit coverage limitations.
- Invalid JSON, uncited claims, unknown source IDs or incomplete model output:
  return `generation_error` and preserve evidence for inspection.
- CLI exit code is 1 when no evidence is available, generation fails or inputs
  are invalid; a usable partial answer exits 0. A model's valid abstention over
  retrieved but irrelevant evidence also exits 0.

Citation validation checks that references exist; it **does not prove semantic
faithfulness**. The prompt requires supported claims, dated evidence, explicit
calculations and refusal when the context is insufficient. Generation quality
must still be evaluated with the team's manually labelled records. CPIAUCSL is
an index level, not an inflation percentage. The current FRED tool lacks unit
metadata and historical date-range selection. Stock prices come from adjusted
daily bars, not guaranteed real-time quotes. These limits are included in the
model's instructions/evidence; unavailable values must not be invented.

See [module contracts and verification](docs/query_generation.md) for integration
with a future Streamlit UI and the exact offline test scope.

## 12. Implemented components

### Corpus
- real source: SEC EDGAR
- cleaning implemented
- chunking implemented and documented
- JSONL storage
- document/passage count
- total words
- vocabulary/types
- target >=10K passages and >=100K words

### Retrieval
- sparse retrieval: BM25
- dense retrieval: BGE
- vector similarity: FAISS cosine/IP on normalized vectors
- hybrid: Reciprocal Rank Fusion
- reranking: cross-encoder
- metadata/timeline filtering

### Querying/evaluation
- rule-based query routing across corpus / FRED / stock, including mixed questions
- evidence-grounded generation with validated citations and graceful tool failure handling
- five sample queries
- query latency
- P@K
- Recall@K
- MRR
- nDCG@K
- saved retrieved results

### Live extension
- FRED macro observations
- recent/current stock market data

## 13. Remaining project work

These tasks are outside the query-routing/generation implementation:
- RAGAS faithfulness / answer relevance
- final Streamlit UI
- 1,000-record manual held-out annotation completion
- final generation ablations

## Suggested report wording

> We constructed a financial knowledge base from SEC EDGAR 10-K filings. Filing HTML was cleaned while preserving readable table rows and chunked into overlapping 350-word passages with 50-word overlap. We indexed the same passage collection using a lexical BM25 retriever and a dense BGE bi-encoder with FAISS. Candidate rankings were combined using Reciprocal Rank Fusion and optionally reranked with a cross-encoder. Metadata fields including ticker, filing date, filing type and detected filing section support constrained retrieval. We evaluate the retriever using Precision@K, Recall@K, MRR and nDCG, and report end-to-end query latency.
