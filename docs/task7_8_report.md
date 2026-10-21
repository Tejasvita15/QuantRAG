# Task 7/8 report sections - methods ready, real results pending

This is a report draft, not evidence that the experiments have been run. Replace every pending value only with a measured result from the archived run artifacts. Integrate it into the team's report rather than submitting it as a complete course report.

## User interface and integration (Q2)

We implemented a lightweight Streamlit interface over the shared `RAGPipeline`, keeping retrieval and generation independent of presentation. Users can inspect a deterministic route plan, retrieve evidence without generation, or request an evidence-grounded answer. Optional controls include ticker, FRED indicator, filing-date year, filing form, passages per company and cross-encoder reranking. Submitting a form explicitly starts a run; inspecting previous evidence or downloading JSON does not trigger another generation request.

The interface exposes complete collected evidence, stable SEC passage IDs, filing/report dates, observation/trading dates and original-source links. It separately shows the exact bounded context supplied to the model and identifies cited records. Errors, partial coverage, insufficient evidence and generation failure remain visible rather than being converted into apparently complete answers. Prices are labelled as adjusted daily closes, and raw CPI observations are not presented as inflation percentages.

Include a screenshot of one real SEC answer and one mixed/partial result after accepting the app against real backends. Archive each result JSON. The page's latency is total pipeline latency, potentially including first-use model initialization; controlled measurements below use a separate runner.

## Retrieval ablation method (Q3)

We compare BM25, dense BGE/FAISS, BM25+dense RRF without reranking, and RRF followed by a cross-encoder. The corpus/index version, manually judged query set, metadata filters, candidate count, output depth and evaluation definitions are fixed across variants. Warm-up is excluded from repeated measurements; model/index initialization is reported separately. Full rankings, scores, passages, per-query metrics and environment/index manifests are retained for reproducibility.

Precision and recall count relevance grades above zero as relevant. nDCG uses graded gains; MRR is computed over the saved retrieval depth, not the entire corpus. Recall is relative to the known judged relevant set. Unjudged passages are treated as zero for metric computation, so judged coverage and pooling limitations are reported alongside the scores. A pool drawn only from the full system's top five results may systematically favor that system and is not an adequate basis for strong conclusions.

| Variant | P@5 | R@5 | R@10 | MRR@depth | nDCG@10 | Mean / p95 latency | Judged coverage |
|---|---|---|---|---|---|---|---|
| BM25 | pending | pending | pending | pending | pending | pending | pending |
| Dense | pending | pending | pending | pending | pending | pending | pending |
| RRF | pending | pending | pending | pending | pending | pending | pending |
| RRF + rerank | pending | pending | pending | pending | pending | pending | pending |

Report the exact test-set size, number of judgments, positive-grade convention, candidate/reranking pool sizes, number of repeats, hardware and snapshot hashes. Include query-level failures and cases where reranking reduces quality; do not select only favorable examples or claim significance from a five-query pilot.

The current shared retriever constructor loads both index types and the dense encoder even for a BM25 query. The BM25 row therefore measures BM25-only query ranking after shared initialization, not the cold-start time or memory/dependencies of an independently deployed BM25 service.

## Answer-level study (Q4/Q5)

Use the same saved retrieval rankings to generate answers under a fixed model, system prompt and context budget. Mixed questions must reuse identical dated live-data snapshots across variants. Record model identity, provider-reported token usage, per-answer generation latency, output status and supplied/cited evidence. Repeating generation is still stochastic unless the provider supports and honors deterministic settings; state this limitation.

Annotators independently judge faithfulness, answer relevance, context precision and task quality using the documented rubric and blinded answer IDs. Report the number of unique held-out records, number of annotators, score coverage, agreement definition and actual measured agreement. At least 1,000 manually labelled records and at least 80% inter-annotator agreement are required by the assignment. Four variants of the same query do not create four independent held-out queries.

| Variant | Faithfulness | Answer relevance | Context precision | Task quality | Generation latency | Token usage |
|---|---|---|---|---|---|---|---|
| BM25 context | pending | pending | pending | pending | pending | pending |
| Dense context | pending | pending | pending | pending | pending | pending |
| RRF context | pending | pending | pending | pending | pending | pending |
| RRF + rerank context | pending | pending | pending | pending | pending | pending |

The supplied review tool reports manual rubric scores, not RAGAS scores. If a dedicated automatic metric is added, name its exact implementation/model and validate its limitations. For extractive questions with reference answers, add Exact Match/token F1 as appropriate; for open-ended financial synthesis, explain why the chosen reference-based and human metrics fit the task. Do not label citation-ID validity as semantic faithfulness.

The four context variants measure the downstream effect of retrieval composition and reranking. They do not isolate the effect of citation prompting, routing, live-data integration, or other generation innovations. If the report claims contributions from those additional innovations, add matched controls for each claim; otherwise describe them as implemented features without claiming an experimentally isolated gain.

## Performance, cost and limitations

Report sequential query latency and throughput with warm-up/initialization policy, token counts and failure rates. Serial reciprocal-latency throughput does not establish concurrent-user scalability. Report currency cost only using the actual provider/model price and date, not token counts alone. A single-machine local UI is not a load-tested production service.

Known limitations include heuristic routing, filing-date versus fiscal-year ambiguity, incomplete judgment pools, finite reranking depth, live-data availability and date drift, and the inability of citation validation alone to establish claim support. Offline tests verify interfaces and failure handling; they do not replace real-data retrieval/answer-quality evaluation.
