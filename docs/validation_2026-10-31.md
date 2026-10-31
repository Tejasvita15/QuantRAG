# Windows startup verification 31 October 2026

The project was started on Windows using Python 3.10.7 in `.venv`.
Dependencies were installed from `requirements.txt`; `pip check` reported no
broken requirements. Use `scripts/start.ps1` from PowerShell to start the
local Streamlit application. Its default address is http://127.0.0.1:8501.
The launcher keeps the Hugging Face cache under `.venv/huggingface` unless
`HF_HOME` is already set.

## Verified

- 109 automated tests passed in 6.900 seconds, including mocked UI and local
  HTTP integration tests. These do not establish real LLM answer quality.
- Streamlit started successfully; its health endpoint returned HTTP 200 `ok`.
- Corpus and dense metadata have the same 25,976 unique passage IDs in order.
- BM25 passage IDs match dense metadata in order; FAISS contains 25,976
  vectors with 768 dimensions.
- Existing corpus statistics report 7,791,693 words across 25 companies.
- BGE and cross-encoder models downloaded successfully into the local cache.
- Five real hybrid retrieval queries with reranking returned five passages
  each. Warmed single-query times were AAPL 4.691 s, MSFT 3.697 s,
  NVDA 3.662 s, AMZN 3.107 s and CVX 3.041 s.
- A real Apple stock query through the shared query CLI returned one evidence
  record with no backend errors.
- Streamlit AppTest submitted an Apple supply-chain question through the real
  pipeline using cached models and actual indexes: `retrieved`, five evidence
  records, no errors, 27.632 s including first-use initialization. This is an
  in-process UI check, not a visual browser inspection.

The five-query output is saved locally at
`eval/runs/startup-2026-10-31/five-queries.json` (ignored by Git).
These are startup checks, not final relevance measurements or full answer
latencies. They do not replace the manually annotated evaluation study.

## Remaining provider setup

At verification time, `LLM_MODEL`, the LLM API key and `FRED_API_KEY` were
unconfigured. No real LLM answer or FRED response was verified.
Set the values in the local `.env`, restart the app and submit a question in
**Generate answer** mode. The existing OpenAI-compatible adapter accepts
`LLM_API_KEY` or `OPENAI_API_KEY` and a provider model name in `LLM_MODEL`.
For OpenAI, the example base URL already points to `https://api.openai.com/v1`.
Keep credentials out of Git and chat.
