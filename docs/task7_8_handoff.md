# Tasks 7 and 8: team handoff

## Development does not require merging tasks 1 and 2 first

The interface uses the `plan_route` and `RAGPipeline` APIs already present on `main`; it does not depend on completed evaluation labels. The ablation tool accepts an external JSONL path. Its implementation can be reviewed now, while final quality measurements must wait for the team's independent annotation and adjudication.

This delivery is on `feature/streamlit-ui`, based on main revision `ae1d694`. It does not merge the implementation into main or modify teammates' annotation sheets.

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

The repository intentionally ignores `data/raw/`, `data/processed/`, `indexes/` and `.env`, so cloning or pulling main does not download them. A matching prebuilt corpus/index bundle or shared-folder link is still needed for full SEC retrieval on another machine. The queries and blank annotation sheets are now tracked; completed/adjudicated qrels are not yet available.

## Current shared annotation workflow

The team update on 21 October asks each member to complete one numbered CSV independently by the night of 23 October. The sheets contain the complete passage text, so this annotation work does not require retrieval indexes, models or API keys. Confirm which unclaimed Member number you own before editing a sheet; Tejasvita has selected Member 6.

The committed set has 100 unique queries, 1,000 unique query-passage records and 200 independently assigned overlaps, giving 1,200 annotation decisions. Member 1 and 2 have 201 decisions each, Member 3 and 4 have 200 each, and Member 5 and 6 have 199 each. Labels and notes are initially blank. Use 2 for directly relevant, 1 for partially relevant, and 0 for irrelevant; add short notes only for ambiguous/borderline cases. Do not use retrieval ranks/scores or compare overlap labels before completing independent work.

Return the completed CSV after annotation. Only then can the team merge labels, calculate actual agreement, resolve disagreements and produce final qrels for the retrieval experiments. Completed labels or IAA results are therefore not an immediate prerequisite to obtain from another member. Do not auto-fill these human annotation sheets with software-test labels or generated answers.

Do not rerun `src/create_annotation_set.py` over completed annotation sheets: its output uses overwrite mode. That preparation script was not rerun or modified for this delivery. The task 7/8 tools require adjudicated JSONL qrels, not the unlabeled `eval/queries.jsonl`; preserve the original independent labels when exporting final grades.

**Implementation and offline tests do not establish real-data performance, real-provider acceptance or completion of the entire course project.** No final experimental metrics have been fabricated to fill the remaining gaps.

On the updated revision, 107 automated tests, dependency checks, CLI/routing checks and the missing-index failure path passed. One real stock-only pipeline request returned a dated record with no errors and no LLM call. FRED/LLM success and SEC retrieval remain unverified because their local prerequisites are absent. See the [verification record](validation_2026-10-21.md). These checks are not final project-quality measurements.

## Immediate setup and later experiment inputs

1. Tejasvita: the existing shared-folder/download link for the matching BM25 and FAISS indexes, dense metadata, complete passages and corpus statistics/manifest, with a version identifier. If no shared snapshot exists, agree on a reproducible rebuild before evaluating against existing passage IDs.
2. Each annotator: confirm and complete the assigned Member CSV independently. The query file and sheets are already in main; do not request them again. After the team finishes, the evaluation owners can merge/adjudicate the labels and export final qrels with the agreement calculation. Keep answer-reference and answer-quality evaluation separate from query-passage relevance labels.
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
