# Retrieval ablation report

Status: completed. These are retrieval measurements, not answer-quality scores.

Queries: 100; repeats: 3; final K: 10.
Warmup calls and initialization are excluded from the measured latency below. Repeats are repeated measurements of the same queries, not extra independent queries.

| Variant | P@5 | R@5 | R@10 | MRR@K | nDCG@10 | Judged coverage@K | Mean s | Median s | p95 s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bm25 | 0.5060 | 0.3104 | 0.4592 | 0.7578 | 0.4176 | 0.5140 | 0.0609 | 0.0573 | 0.0884 |
| dense | 0.6640 | 0.4246 | 0.5928 | 0.9289 | 0.6293 | 0.5430 | 0.0172 | 0.0173 | 0.0189 |
| hybrid_no_rerank | 0.6300 | 0.3936 | 0.5645 | 0.8964 | 0.5638 | 0.5640 | 0.0755 | 0.0730 | 0.1044 |
| hybrid | 0.7440 | 0.4898 | 0.6864 | 0.9417 | 0.6661 | 0.6100 | 0.1590 | 0.1566 | 0.1892 |

## Interpretation and limits

- Unjudged passages count as nonrelevant for reported metrics; recall is relative to supplied positive judgments, not proven complete-corpus recall.
- Schema validation cannot prove manual annotation, held-out provenance, 1,000-record completion or inter-annotator agreement.
- 100/100 queries have a small judgment pool (<max(2*K,candidate_k)); expand pooled judgments across systems before interpreting recall (heuristic warning).
- Some result lists contain fewer than K passages; precision still divides by the fixed K. Inspect filters and candidate coverage.
- At least one ranking has <80% judged coverage; judging additional pooled passages is recommended (heuristic threshold).
- Warmed sequential retrieval timings are not throughput, concurrency, cold-start or full RAG/LLM latency.
- The shared retriever loads both indexes and a dense encoder. The BM25 row measures BM25-only queries after that shared initialization, not a standalone BM25 service's startup or memory cost.
- Comparisons with hybrid_no_rerank isolate reranking; comparing the two single retrievers and their fusion tests complementarity. Keep the same corpus, qrels, filters, candidate budget and K.
- No confidence intervals or significance claims are produced. More repeats reduce timing noise; more independent held-out queries are needed to assess quality uncertainty.

## Reproducibility

See manifest.json for source/index/qrels hashes, configuration, Git commit and package versions.
results.json preserves full passage text and actual execution order; per_query.csv preserves each measured sample.

## Remaining answer evaluation

Human/reference answers, faithfulness, answer relevance, numerical correctness, citation support, annotation agreement and generation cost remain separate work. Do not fill them from these retrieval metrics.
