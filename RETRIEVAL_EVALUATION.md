# QuantRAG: Retrieval Evaluation & Manual Annotation Report

This document details the manual annotation dataset construction and quantitative retrieval evaluation (Precision@K, Recall@K, MRR, nDCG) for Step 1 of the QuantRAG project.

---

## 1. Overview & Setup

To evaluate our multi-stage financial retrieval pipeline (Sparse BM25 + Dense BGE Embedding + Reciprocal Rank Fusion + Cross-Encoder Reranking), we performed manual annotation on held-out SEC 10-K financial queries and conducted systematic retrieval evaluations with ablation experiments.

* **Corpus Scale**: 25,976 passages (~7.79M words) across 25 major public companies (exceeding the baseline requirement of $\ge$10,000 passages and $\ge$100,000 words).
* **Dense Embedding**: `BAAI/bge-base-en-v1.5` (768-dim normalized embeddings).
* **Sparse Index**: BM25 with financial-specific tokenization.
* **Cross-Encoder Reranker**: `cross-encoder/ms-marco-MiniLM-L-6-v2`.

---

## 2. Manual Annotation Dataset

### 2.1 Methodology: Pooling Strategy
Given the corpus size (>25,000 passages), manual exhaustive search is infeasible. We adopted the standard Information Retrieval **Pooling Strategy**:
1. Executed 5 core benchmark queries across representative large-cap tech and enterprise companies (`AAPL`, `MSFT`, `NVDA`, `AMZN`, `GOOGL`).
2. Retrieved top candidate passages using hybrid retrieval and exported full context to `eval/five_query_results.json`.
3. Manually reviewed and graded each retrieved passage against standard criteria.

### 2.2 Relevance Grading Scale
We adopted a 3-tier relevance scale aligned with financial research retrieval standards:

| Grade | Meaning | Criterion (SEC 10-K Context) |
| :---: | :--- | :--- |
| **0** | **Not Relevant** | The passage does not address the question, or mentions generic boilerplate (e.g., table of contents, auditor sign-offs, unrelated business lines). |
| **1** | **Partially Relevant** | Mentions related concepts, background, or macroeconomic context without directly answering the specific query. |
| **2** | **Directly Relevant** | Directly, concretely, and accurately answers the query (e.g., specific risk disclosures in Item 1A, revenue breakdown in Item 7). |

### 2.3 Annotated Benchmark Queries

The final evaluation set is saved in [`eval/retrieval_eval.jsonl`](eval/retrieval_eval.jsonl):

1. **`q001` (AAPL)**: *"What risks does Apple identify related to supply chain disruption?"*
   * Key disclosures: Component sole-sourcing, supplier industrial accidents, natural disasters, and geographic concentration.
2. **`q002` (MSFT)**: *"What factors affected Microsoft's cloud revenue growth?"*
   * Key disclosures: Azure consumption-based services, M365 commercial cloud momentum, and AI infrastructure scaling costs.
3. **`q003` (NVDA)**: *"What risks does NVIDIA report related to export restrictions?"*
   * Key disclosures: USG licensing requirements, datacenter/gaming GPU export controls, and exclusion risks from the China and Middle East markets.
4. **`q004` (AMZN)**: *"What factors drove AWS cloud revenue growth?"*
   * Key disclosures: Increased enterprise customer usage, pricing changes under long-term contracts, and macro headwinds.
5. **`q005` (GOOGL)**: *"What risks does Alphabet report regarding advertising revenue?"*
   * Key disclosures: Fluctuations in digital ad spending, fee structures, regulatory antitrust impacts, and foreign currency exchange.

---

## 3. Quantitative Evaluation Results

We evaluated our retrieval system on all required metrics under two configurations to demonstrate the impact of Cross-Encoder reranking:
1. **Hybrid + Cross-Encoder** (Full pipeline: BM25 + Dense FAISS + RRF + Cross-Encoder Reranker)
2. **Hybrid (No Rerank)** (Ablation: BM25 + Dense FAISS + RRF only)

### Evaluation Metrics Summary

| Retrieval Configuration | Precision@5 | Recall@5 | Recall@10 | MRR | nDCG@10 | Latency (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Hybrid + Cross-Encoder** | **0.96** | **1.00** | **1.00** | **1.00** | **0.914** | 5.26s |
| **Hybrid (No Rerank)** | 0.52 | 0.54 | 0.75 | 0.90 | 0.622 | 0.18s |

---

## 4. Key Findings & Discussion

1. **Drastic Precision & Ranking Gain via Reranking**:
   * Applying the Cross-Encoder reranker boosted **Precision@5 from 0.52 to 0.96 (+84.6%)** and **nDCG@10 from 0.622 to 0.914 (+46.9%)**.
   * **MRR reached 1.00**, indicating that the single most directly relevant passage was ranked at position #1 across all evaluation queries.
2. **Trade-off between Retrieval Quality & Latency**:
   * First-stage hybrid retrieval (BM25 + Dense with RRF) is extremely lightweight, completing in **~0.18 seconds**.
   * Cross-Encoder reranking processes the top 30 candidates with deep cross-attention, increasing query latency to **~5.26 seconds** on CPU. This provides a clear latency-accuracy tradeoff suitable for production design discussions.

---

## 5. Reproduction Steps

To reproduce the benchmark queries and evaluation metrics:

```bash
# 1. Activate the environment
conda activate torch_env

# 2. Run the 5-query benchmark (outputs to eval/five_query_results.json)
PYTHONPATH=src python src/benchmark_queries.py

# 3. Run evaluation (Full Hybrid with Reranker)
PYTHONPATH=src python src/evaluate_retrieval.py --mode hybrid

# 4. Run ablation evaluation (Hybrid without Reranker)
PYTHONPATH=src python src/evaluate_retrieval.py --mode hybrid_no_rerank
```

