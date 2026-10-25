# Task 8: reproducible retrieval ablation

This tool runs real retrieval measurements only after the team supplies its human-labelled held-out records and matching corpus/indexes. No experiment scores are bundled. Offline unit-test labels and deterministic fake retrievers test the software contract; they are not assignment results.

## First run: validate the human records

From the repository root, with Python 3.10 or newer:

```bash
python -m src.run_ablation --file eval/retrieval_eval.jsonl --check-only
```

`--file` is mandatory. There is no default fallback to the repository's example. A filename containing an `example` or `demo` component, placeholder passage IDs, empty records, duplicate query IDs or JSON keys, non-integer grades and records without positive judgments are rejected. Different IDs cannot disguise duplicate questions with the same filters: question text is Unicode-normalized, whitespace-normalized and case-folded before checking. The same question with different filters is allowed. Check-only uses only the Python standard library, does not import FAISS/PyTorch/transformers, does not read indexes or `.env`, does not make network/model calls, and does not create output files. Both `python -m src.run_ablation` and `python src/run_ablation.py` are supported.

Each JSONL line uses the existing evaluator's schema:

```json
{"query_id":"human_q001","query":"What supply-chain risks did Apple report?","ticker":"AAPL","year":2024,"form":"10-K","relevance":{"actual_passage_id_from_the_shared_corpus":2,"another_actual_passage_id":0}}
```

This line illustrates the shape only. Replace the IDs and labels with the team's real work. `query_id`, `query` and a nonempty `relevance` object are required; ticker/year/form are optional. Grades are **0 = not relevant, 1 = partially relevant, 2 = directly relevant**. Every query needs at least one positive grade. Year is the **filing-date year**, not the fiscal/report year. A full run also verifies that every judged ID exists in the loaded corpus and every positively judged passage satisfies that query's explicit filters.

Validation cannot prove that labels were made by humans, that the split is held out, that the required annotation record count is complete, or that annotators meet the required agreement. Retain the annotation protocol, annotator records, disagreements/adjudication, and IAA calculation separately. Do not tune candidate budgets, query wording or labels using the final test scores.

## Run all four configurations

Obtain the same corpus and indexes used for annotation from the team. Required local artifacts:

- `indexes/bm25/bm25.pkl`
- `indexes/dense/index.faiss`
- `indexes/dense/metadata.jsonl`

Install the project dependencies and retain the exact model cache or pin its revisions. Only load trusted, team-produced BM25 pickle files. Choose a **new, non-existing** run directory; missing parent directories are created after input/index preflight:

```bash
python -m src.run_ablation \
  --file eval/retrieval_eval.jsonl \
  --output eval/experiments/retrieval_run01 \
  --top-k 10 --candidate-k 50 --repeats 3 --warmup 1 --seed 6127
```

| Saved variant | Retriever mode | Rerank | What the comparison tests |
|---|---|---|---|
| `bm25` | `bm25` | false | Lexical baseline |
| `dense` | `dense` | false | BGE/FAISS baseline |
| `hybrid_no_rerank` | `hybrid` | false | RRF fusion of the two candidate rankings |
| `hybrid` | `hybrid` | true | Fusion plus cross-encoder reranking |

The same instance, corpus, query, ticker/year/form filters, per-branch candidate budget and final K are used. `--candidate-k` defaults to `config.json` when omitted. K must be at least 10, no larger than `rerank_pool_k` (default 30), and no larger than candidate_k. The final K does not silently shrink for a configuration; if fewer passages are available after filtering the tool records that count and warns, while precision retains its fixed K denominator.

`--warmup N` executes N queries per variant, cycling through input order; these calls and model/index initialization are recorded separately and excluded from query-latency averages. When N is positive, the CLI explicitly calls the retriever's `prepare_reranker()` before finishing initialization timing. The experiment API also prepares it before any timed call, so an empty first warmup query cannot move cross-encoder model construction into a later measured query. If all reranking warmups return no passages, a warning still notes that first-inference/kernel effects may enter the first nonempty measured query. Default N is 1. Setting N to 0 skips explicit preparation and produces a cold-initialization warning. Each repeat shuffles query order and each query's variant order using `--seed`. This seed controls the schedule only; it does not claim to seed all model/GPU kernels. Differences in repeated ranking IDs are flagged. Repeats do not create more independent queries or a larger test set.

If a run fails, it exits nonzero and may leave `failure.json` in its reserved directory. No successful result is claimed. Choose another new directory for the next attempt; the tool never replaces an existing directory. Missing indexes are rejected before reserving an output directory. Real runs can initialize/download configured retrieval models if not cached; check-only never does so.

## Files you can inspect or reuse

| File | Contents |
|---|---|
| `results.json` | Completion status, normalized query/filter/qrels records, every repeated ranking with **full passage text**, source/score fields, per-query metrics, latency, actual execution order, warmups, settings and warnings |
| `per_query.csv` | One row per measured query × variant × repeat |
| `summary.csv` | Macro-average quality metrics and mean/median/p95/min latency per variant |
| `manifest.json` | Qrels SHA-256, Git commit/dirty state, Python/platform/package versions, effective configuration, source/config/index/corpus hashes, initialization timing and reproduction limits |
| `report.md` | Filled comparison table, evaluation limits and remaining answer-quality work |

The manifest deliberately excludes environment variables, `.env`, API keys, Git remotes and provider response bodies. Source/configuration/index/qrels fingerprints are checked again after the run; a mid-run change invalidates the run. A model identifier alone is not an immutable revision: retain/pin the exact local model cache and hardware/software setup for strict reproduction. Hashing `config.json` records its identity without copying arbitrary configuration fields into the manifest.

`results.json` has `schema_version="1.0"`, `kind="retrieval_ablation"` and `status="completed"`. Join `runs[].query_id` to `queries[].query_id`. Each run has `variant`, 1-based `repeat`, `mode`, `rerank`, `top_k`, `latency_s`, `retrieved_count`, `metrics` and `ranking`. Each ranking entry retains `id`, `rank`, `passage` (including `text`) and `sources`; additional score fields are preserved. This completion artifact is written after the other reports. The downstream answer study can freeze one repeat and use these exact passages rather than silently re-retrieving.

## Interpret the table correctly

Reported quality metrics are P@5, R@5, R@10, MRR@K, nDCG@10, P@K, R@K and nDCG@K. Grades above zero count as relevant for binary metrics; nDCG uses gain `2^grade - 1`. MRR is truncated at the saved K; it is not an unbounded MRR over the whole corpus. Repeated query measurements receive equal weight; the query set and number of repetitions are identical across variants.

Unjudged passages are treated as nonrelevant for the metrics and counted explicitly. `judged_coverage_at_k` divides judged returned passages by the number actually returned; it does not imply the system returned K results. Recall's denominator is the number of **provided positive judgments**, which may be an incomplete pool rather than all relevant passages in the corpus. Very small pools or low coverage can bias comparisons against systems that find new relevant passages. The tool always states this limit, and warns when a query's judgments are below `max(2*K,candidate_k)`, there are fewer than 20 queries, or returned judged coverage is below 80%. Those are diagnostic heuristics, not new course requirements. Expand and adjudicate a pool collected across all configurations, without changing labels opportunistically after seeing which system wins.

Latency measures sequential wall-clock **retrieval** calls. It is not full RAG latency, concurrent throughput, cold-start cost or generation expense. The shared `HybridRetriever` initializes both indexes and its dense encoder; the BM25 row therefore measures BM25-only query work after shared initialization, not a separately deployed BM25 service's cold-start or memory cost. Keep hardware, competing workloads, thread/device settings and cache conditions comparable. Report initialization and warmup separately; p95 from a small sample is unstable. No significance or confidence-interval claim is generated.

For the final report, use values from `summary.csv` and inspect `per_query.csv` plus full passages to explain failures. Discuss BM25 vs dense, either single retriever vs RRF, and RRF vs RRF+rerank. Do not conclude that reranking improves answer faithfulness merely because nDCG increased. Preserve the input hashes/run folder and state the held-out query count, annotation protocol, label-pool limits, K, candidate budget, rerank pool, model IDs/revisions, hardware and timing conditions.

## Answer quality and what is still blocked locally

Retrieval scores cannot fill the assignment's downstream/RAG quality, numerical correctness, citation support, human agreement or cost results. Those need real questions, answers/reference judgments, frozen live-data snapshots where applicable, and an answer-evaluation protocol. See [answer study](answer_study.md) for the separate downstream workflow. A no-retrieval LLM comparison, if run, is an answer-generation diagnostic and has no meaningful passage-retrieval P@K/Recall.

As of 24 October, the matching shared SEC corpus/indexes and both retrieval models are installed and verified locally. The real SEC Streamlit acceptance checks pass, and five covered-company demonstration queries each return five complete passages. Those warmed, single-pass retrieval timings are smoke-test observations, not a controlled final ablation. See [the current validation record](validation_2026-10-24.md).

Final adjudicated retrieval qrels and the agreement summary are still absent from main. An externally supplied five-query pilot passes structural check-only validation, but its small judgment pool does not establish the final 100-query evaluation set or completion of manual annotation. Do not convert incomplete CSVs into final labels by averaging grades or choosing one annotator. After the evaluation owners collect all independent labels and resolve conflicts, run check-only, verify passage-ID/corpus consistency and perform the real experiment above. No final P@K, nDCG or answer-quality result is claimed. Local FRED and LLM configuration is still absent; real-provider answer acceptance remains a separate outstanding check.

```bash
python -m unittest discover -s tests -p test_ablation.py -v
```

The tests use synthetic temporary data and injected retrievers. They cover hand-calculated metrics, graded/unjudged behavior, invalid/duplicate/placeholder qrels, filter/ID mismatch, all four variants, fixed budgets, warmups/repeats, order reproducibility, full-passage preservation, both lightweight CLI entrypoints, artifact hashes, non-overwrite behavior, immutable inputs and credential-safe failures.
