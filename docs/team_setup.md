# Team setup

Use Python 3.10 or 3.11. Git provides the application, configuration template,
tests and final evaluation labels. It does not provide secret keys, your Python
environment, model downloads, the SEC corpus or search indexes.

## First setup on Windows PowerShell

After cloning the repository, open a terminal in its root directory:

```powershell
py -3.10 scripts/setup.py
# Edit the newly created .env with your own keys and provider model.
.\.venv\Scripts\python.exe scripts/check_setup.py
.\scripts\start.ps1
```

If PowerShell blocks scripts, use the direct equivalent:

```powershell
$env:HF_HOME = "$PWD\.venv\huggingface"
.\.venv\Scripts\python.exe -m streamlit run app.py --server.address 127.0.0.1
```

Open http://127.0.0.1:8501. Select Route only for a preview without keys or
indexes. The default Retrieve evidence response needs indexes for SEC queries.
Stop the server with Ctrl+C.

## First setup on macOS or Linux

```bash
python3.11 scripts/setup.py
# Edit .env with your own configuration.
.venv/bin/python scripts/check_setup.py
bash scripts/start.sh
```

Python 3.10 is also supported. Setup preserves any existing .env. Never copy
another teammate's virtual environment; create one on your own operating system.
The requirements are intentionally cross-platform; GPU-specific PyTorch setups
may need the appropriate PyTorch install before installing the other packages.

## Enable SEC evidence retrieval

Obtain the trusted data bundle from a teammate and extract it into the repository
root. It must produce these paths, without an extra enclosing folder:

```text
data/processed/passages.jsonl
data/processed/corpus_stats.json
indexes/bm25/bm25.pkl
indexes/dense/index.faiss
indexes/dense/metadata.jsonl
data-bundle-manifest.json
```

Keep the BM25 index, FAISS index and metadata from the same snapshot. The bundle
includes checksums and the originating config.json checksum; do not change the
embedding-model setting independently of the dense index. Python pickle files
must come from the trusted team. The bundle does not include raw SEC HTML or
Hugging Face model weights. BGE and cross-encoder weights download on first use;
internet access, additional disk space and a slower first query are expected.

After extracting, check its file hashes and matching configuration without
loading the pickle:

```powershell
.\.venv\Scripts\python.exe scripts/check_setup.py --verify-bundle
```

On macOS/Linux use `.venv/bin/python scripts/check_setup.py --verify-bundle`.

The teammate holding the matching data can export it with:

```powershell
.\.venv\Scripts\python.exe scripts/export_data_bundle.py
```

Share `dist/quantrag-data.zip` separately through the team's shared drive.
It is ignored by Git. Do not attach API keys to it. Alternatively rebuild from
SEC using README section 3; this requires SEC identification, network access,
time and potentially substantial compute, and creates a different snapshot.

## Enable provider answers

Set these locally in .env:

```dotenv
SEC_USER_AGENT="NTU AI6127 QuantRAG your_real_email@example.com"
FRED_API_KEY="your-own-fred-key"
LLM_MODEL="your-provider-model"
LLM_API_KEY="your-own-llm-key"
LLM_BASE_URL="https://api.openai.com/v1"
```

Only FRED questions need its key. Stock queries require provider availability
but no LLM key. Generate answer mode uses the configured paid model. A readiness
indicator verifies configuration presence, not connectivity or answer quality.
Restart the server after changing keys, model configuration or indexes.

## After pulling an update

Stop Streamlit, then run:

```powershell
git pull
py -3.10 scripts/setup.py
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\scripts\start.ps1
```

On macOS/Linux use python3.11 and .venv/bin/python, then bash scripts/start.sh.
Setup updates dependencies and preserves .env and existing data. Git does not
refresh ignored indexes; coordinate a new bundle if the corpus or model changes.

Generated experiments, recalibration worksheets and local bundles are ignored
and preserved on disk. Share selected research artifacts separately. Never rerun
annotation-generation scripts over completed human labels.
