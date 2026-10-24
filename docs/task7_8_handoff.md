# Tasks 7 and 8: team handoff

## Development does not require merging tasks 1 and 2 first

The interface uses the `plan_route` and `RAGPipeline` APIs already present on `main`; it does not depend on completed evaluation labels. The ablation tool accepts an external JSONL path. Its implementation can be reviewed now, while final quality measurements must wait for the team's independent annotation and adjudication.

This delivery is on `feature/streamlit-ui`. On 2026-10-24, main revision `e039127` was merged into this branch to preserve the latest annotation submissions. This does not merge the UI implementation into main or change any annotator's labels. A team review/merge is still required.

## Status on 24 October

- The implementation passes 109 automated tests and dependency checks. Seven additional Streamlit AppTest acceptance checks pass with the real shared SEC corpus and cached BGE/cross-encoder models; no live-provider or LLM call was made.
- The matching corpus, BM25 index, FAISS index and dense metadata are now installed locally. Each index has 25,976 matching passage records. Do not request the same bundle again for this machine.
- The five default demonstration queries now cover AAPL, MSFT, NVDA, AMZN and CVX, all present in this snapshot. The earlier JPM/XOM examples returned no SEC passages because the snapshot lacks those companies. Unavailable-company handling remains a separate acceptance test. The manually annotated evaluation queries are unchanged.
- Five real demonstration queries each returned five complete passages. These single-pass, warmed retrieval timings are smoke-test observations, not the final ablation or evidence of answer quality.
- Main does not yet contain final adjudicated retrieval qrels or the pre-adjudication agreement calculation. Member CSV submissions do not by themselves provide final qrels. The evaluation owners need to finish collection, calculate overlap agreement and resolve disagreements before the final four-way experiment.
- FRED and LLM model/key configuration are absent locally. Real FRED retrieval, paid-provider generation and answer-quality evaluation remain unverified. Use each operator's own credentials, not a teammate's secret key.
- The shared Google Docs report is view-only for the current account. The English text in `docs/task7_8_report.md` is prepared for the relevant report sections; it has not been inserted into the shared document.

See [the current verification record](validation_2026-10-24.md). The older 21 October record documents the earlier missing-index state and should not be used as the current readiness statement.

## Scope of this delivery

- Task 7: three Streamlit operating modes, shared-pipeline integration, filters, evidence and citation inspection, status/error handling, latency, JSON download, and offline interaction tests.
- Task 8: BM25, dense, RRF and RRF plus reranking retrieval comparisons; controlled conditions, repeated timing, full rankings/passages, judgment-coverage warnings and report exports; separate answer-comparison and human-scoring tools described in their documentation.
- Fixed reranking when a reused retriever first runs with reranking disabled and later enables it.
- Updated the five-query benchmark to preserve full passage text rather than the first 600 characters, and to separate initialization/warmup from measured query latency.
- Retained the original minimal example at `examples/app_echo.py` as a separate reference; the delivered application starts from the repository-root `app.py`.

The FRED integration, stock tool, query routing and grounded generation for tasks 3–6 were already implemented on `main`. The [README](../README.md), [`.env.example`](../.env.example) and [query/generation documentation](query_generation.md) already explain environment setup, FRED configuration, model selection and compatible providers. The remaining dependencies are matching data/evaluation artifacts and each operator's local configuration, not missing tasks 3–6 implementations or missing setup guidance.

## Main-branch verification

Main was pulled and this feature branch fast-forwarded to `ae1d694` on 2026-10-21 before preparing the delivery commit. That revision adds the 100 queries and six annotation sheets, and renames the preparation script to `src/create_annotation_set.py`.

| Task | Implementation already on main |
|---|---|
| 3: FRED | `src/live/fred_tool.py` |
| 4: stocks | `src/live/stock_tool.py` |
| 5: routing | `src/query_router.py`, `src/query.py` |
| 6: grounded generation | `src/generator.py`, `src/rag_pipeline.py` |

The existing README also supplies the SEC download, preprocessing, corpus statistics, BM25 and dense-index build commands. `config.json` names the public BGE and cross-encoder models; those choices are not missing. Building the data locally is possible. To compare against the team's existing labels, however, the corpus snapshot and stable passage IDs must match.

The repository intentionally ignores `data/raw/`, `data/processed/`, `indexes/` and `.env`, so cloning or pulling main does not download them. The trusted shared bundle is installed on this machine; another machine still needs that matching bundle or a reproducible rebuild. Member submissions are tracked, but completed/adjudicated qrels are not yet available.

## Current shared annotation workflow

The team update on 21 October asks each member to complete one numbered CSV independently by the night of 23 October. The sheets contain the complete passage text, so this annotation work does not require retrieval indexes, models or API keys. Confirm which unclaimed Member number you own before editing a sheet; Tejasvita has selected Member 6.

The committed set has 100 unique queries, 1,000 unique query-passage records and 200 independently assigned overlaps, giving 1,200 annotation decisions. Member 1 and 2 have 201 decisions each, Member 3 and 4 have 200 each, and Member 5 and 6 have 199 each. Labels and notes are initially blank. Use 2 for directly relevant, 1 for partially relevant, and 0 for irrelevant; add short notes only for ambiguous/borderline cases. Do not use retrieval ranks/scores or compare overlap labels before completing independent work.

Return the completed CSV after annotation. Only then can the team merge labels, calculate actual agreement, resolve disagreements and produce final qrels for the retrieval experiments. Completed labels or IAA results are therefore not an immediate prerequisite to obtain from another member. Do not auto-fill these human annotation sheets with software-test labels or generated answers.

Do not rerun `src/create_annotation_set.py` over completed annotation sheets: its output uses overwrite mode. That preparation script was not rerun or modified for this delivery. The task 7/8 tools require adjudicated JSONL qrels, not the unlabeled `eval/queries.jsonl`; preserve the original independent labels when exporting final grades.

**Implementation and offline tests do not establish real-data performance, real-provider acceptance or completion of the entire course project.** No final experimental metrics have been fabricated to fill the remaining gaps.

On 21 October, 107 automated tests and the missing-index failure path passed; one real stock-only request also returned a dated record without an LLM call. On 24 October, the expanded suite passes 109 tests and real SEC retrieval/UI acceptance passes with installed indexes. FRED/LLM success remains unverified. See the [current verification record](validation_2026-10-24.md). These checks are not final project-quality measurements.

## Immediate setup and later experiment inputs

1. New operators only: install the trusted shared corpus/index bundle and retain its manifest. It is already installed and verified on this machine; no duplicate handoff is needed here.
2. Evaluation owners: collect the remaining independent annotations, merge by stable record identity, calculate pre-adjudication overlap agreement, resolve conflicts explicitly and export final qrels plus the agreement summary. Preserve all original labels. Do not use the five-query pilot as the final 100-query evaluation set. Keep answer-reference and answer-quality evaluation separate from query-passage relevance labels.
3. The generation/FRED owners and the team: confirm the agreed model and compatible provider for controlled runs, plus any project-specific configuration choices beyond the existing documentation. Each person should configure their own local LLM/FRED credentials when needed, following the README, `.env.example` and query/generation instructions. Do not request or distribute teammates' secret API keys, or include them in the repository or report.
4. The whole team: freeze the question set, model/prompt, context budget, index version and live-data snapshots for answer comparisons. Differences between stock observations collected on different dates must not be attributed to a system improvement.

The assignment requires at least 1,000 manually annotated downstream records and at least 80% inter-annotator agreement; three annotators are recommended and two are permitted. Reusing one question for four system configurations does not create four independent held-out records. A mock-test pass rate cannot substitute for an experimental quality metric.

## Run after the matching artifacts are available

```bash
# Validate final labels without loading retrieval models.
python -m src.run_ablation --file /absolute/path/retrieval_eval.jsonl --check-only

# Run the four real retrieval configurations; choose a new output directory.
python -m src.run_ablation --file /absolute/path/retrieval_eval.jsonl \
  --output eval/runs/final-retrieval --top-k 10 --candidate-k 50 --repeats 3 --warmup 1

# Export the five required demonstration queries and retrieval latency; no LLM call.
python -m src.benchmark_queries --output eval/runs/five-queries.json

# Start the application and follow docs/streamlit.md for real-backend acceptance.
python -m streamlit run app.py --server.address 127.0.0.1
```

See [answer comparison and human scoring](answer_study.md) for the downstream commands, and [retrieval ablation](ablation.md) for metric definitions and exports. After completing the real runs, replace the pending tables in the [report draft](task7_8_report.md) with measured results, then explain failure cases and limitations. Do not submit the pending tables as final experimental findings.
