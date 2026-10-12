#!/usr/bin/env bash
set -euo pipefail

python src/download_sec.py
python src/prepare_corpus.py
python src/corpus_stats.py
python src/build_bm25.py
python src/build_dense.py
python src/benchmark_queries.py

echo
echo "STEP 1 COMPLETE."
echo "Try:"
echo 'python src/search.py "What risks does Apple identify related to supply chain disruption?" --ticker AAPL'
