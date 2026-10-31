# Retrieval ablation results 1 November 2026

The adjudicated-label retrieval experiment completed 1,200 measured calls:
100 queries, four variants and three repeats. Four warmup calls and 27.7185 s
of shared initialization were excluded from query latency. All 1,000 judged
passage IDs existed in the 25,976-passage corpus, and no positive judgment
contradicted its query filters. Rankings were identical across repeats.

| Variant | P@5 | R@10 | MRR@10 | nDCG@10 | Judged coverage@10 | Mean retrieval seconds |
|---|---:|---:|---:|---:|---:|---:|
| BM25 | 0.5200 | 0.4588 | 0.7728 | 0.4317 | 0.5140 | 0.1153 |
| Dense BGE | 0.6760 | 0.5995 | 0.9439 | 0.6347 | 0.5430 | 0.1320 |
| Hybrid RRF | 0.6320 | 0.5681 | 0.8970 | 0.5738 | 0.5640 | 0.2471 |
| Hybrid RRF + reranker | 0.7420 | 0.6683 | 0.9417 | 0.6629 | 0.6100 | 4.2057 |

Hybrid with reranking had the highest observed P@5, R@10 and nDCG@10.
Dense retrieval had the highest MRR@10 and a lower latency. Hybrid without
reranking underperformed dense retrieval on all four listed quality metrics;
fusion alone did not improve this evaluation. Adding reranking improved
nDCG@10 from 0.5738 to 0.6629 but increased mean retrieval latency from
0.2471 s to 4.2057 s. No statistical significance or answer-quality improvement
is established by this run.

## Configuration and provenance

- Qrels: `eval/retrieval_eval.jsonl`, 100 queries and 1,000 graded pairs.
- 83 disputed records use the user's explicitly confirmed final human decisions;
  917 records retain their undisputed original labels.
- Original independent exact agreement remains 117/200 = 58.5%. Adjudication
  does not establish the project's 80% independent agreement target.
- Final K = 10; candidate K = 50 per branch; rerank pool = 30; RRF k = 60.
- Ticker filters were held fixed across variants. This experiment does not
  isolate metadata filtering or live-data contributions.
- Models: `BAAI/bge-base-en-v1.5` and
  `cross-encoder/ms-marco-MiniLM-L-6-v2`, loaded from the local cache offline.
- Windows AMD64, Python 3.10.7, CPU inference, `OMP_NUM_THREADS=2` and
  `MKL_NUM_THREADS=2`; sequential calls with execution-order seed 6127.
- The monitoring wrapper printed progress after every 25 metric calculations,
  outside each timed retrieval call. Source/config/index/qrels fingerprints
  were verified unchanged by the runner.
- No LLM, FRED or stock request was made. Timing excludes answer generation.

## Interpretation limits

Only 10 passages per query are judged. Average returned judged coverage ranges
from 51.4% to 61.0%; unjudged passages count as nonrelevant, so these metrics
can penalize systems that find useful evidence outside the label pool.
Recall is relative to supplied positive judgments, not every relevant corpus
passage. Expand the cross-system judgment pool before making strong claims
about general retrieval superiority.

BM25 returned six passages for q067 in each repeat; precision still uses a
fixed denominator. Every other measured ranking returned 10 passages. Repeats
measure timing and ranking stability, not 1,200 independent held-out questions.
Answer faithfulness, provider connectivity and the live-context ablation remain
separate evaluations.

## Saved evidence

The completed run is in `eval/runs/final-retrieval-2026-10-31-03/`, named for
its start date. It includes `report.md`, `summary.csv`, `per_query.csv`, full
rankings in `results.json`, `manifest.json` with fingerprints and runtime
versions, and `completion_audit.json`. This directory is ignored by Git;
retain the archive separately when sharing the results. Earlier interrupted
attempts are not completed experiments and are excluded from these results.
