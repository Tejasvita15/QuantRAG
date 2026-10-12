# QuantRAG — Step 1: Knowledge Base + Retrieval

This repository implements the retrieval portion of the AI6127 QuantRAG project:

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

SEC requires automated clients to identify themselves. FRED requires a registered API key.

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

## 9. Current/recent stocks

```bash
python src/live/stock_tool.py AAPL
python src/live/stock_tool.py NVDA
```

Returns latest available trading price, previous close, 1-day return, 5-day return, volume and market cap when available.

## 10. What is already covered for AI6127 retrieval

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

## 11. What is NOT Step 1

These are downstream tasks and should be added after retrieval works:
- LLM answer generation
- RAGAS faithfulness / answer relevance
- final Streamlit UI
- 1,000-record manual held-out annotation completion
- final generation ablations

## Suggested report wording

> We constructed a financial knowledge base from SEC EDGAR 10-K filings. Filing HTML was cleaned while preserving readable table rows and chunked into overlapping 350-word passages with 50-word overlap. We indexed the same passage collection using a lexical BM25 retriever and a dense BGE bi-encoder with FAISS. Candidate rankings were combined using Reciprocal Rank Fusion and optionally reranked with a cross-encoder. Metadata fields including ticker, filing date, filing type and detected filing section support constrained retrieval. We evaluate the retriever using Precision@K, Recall@K, MRR and nDCG, and report end-to-end query latency.
