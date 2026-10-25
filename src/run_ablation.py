"""Reproducible four-way retrieval ablation; imports use the standard library only.

Use --check-only to validate human qrels without loading models or indexes.
Real runs require an explicit, new output directory and never use example labels.
"""

import argparse
import contextlib
import csv
import hashlib
import importlib.metadata
import json
import math
import platform
import random
import re
import statistics
import subprocess
import sys
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VARIANTS = (
    ("bm25", "bm25", False),
    ("dense", "dense", False),
    ("hybrid_no_rerank", "hybrid", False),
    ("hybrid", "hybrid", True),
)
METRICS = (
    "precision_at_5", "recall_at_5", "recall_at_10", "mrr_at_k",
    "ndcg_at_10", "precision_at_k", "recall_at_k", "ndcg_at_k",
    "judged_coverage_at_k", "judged_at_k", "unjudged_at_k",
)
INDEX_FILES = (
    "indexes/bm25/bm25.pkl", "indexes/dense/index.faiss",
    "indexes/dense/metadata.jsonl",
)


class ValidationError(ValueError):
    """A diagnostic safe to show without provider errors or environment values."""


def _unique_object(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise ValidationError("Duplicate JSON object key")
        out[key] = value
    return out


def load_qrels(path):
    """Load explicit, nonempty JSONL with graded, positive human judgments.

    This validates structure, not annotation provenance or agreement. The caller
    must still document human annotation, held-out split and IAA separately.
    """
    path = Path(path)
    if not path.is_file():
        raise ValidationError("Qrels file does not exist or is not a regular file")
    if re.search(r"(?:^|[._-])(?:example|demo)(?:[._-]|$)", path.name, re.I):
        raise ValidationError("Example/demo qrels cannot be used as final evaluation labels")
    examples, seen, seen_questions = [], set(), set()
    with path.open(encoding="utf-8-sig") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line, object_pairs_hook=_unique_object)
            except (ValueError, TypeError):
                raise ValidationError(f"Qrels line {number}: invalid JSON or duplicate object key") from None
            if not isinstance(row, dict):
                raise ValidationError(f"Qrels line {number}: expected an object")
            query_id, query = row.get("query_id"), row.get("query")
            if not isinstance(query_id, str) or not query_id.strip():
                raise ValidationError(f"Qrels line {number}: query_id must be a nonempty string")
            query_id = query_id.strip()
            if query_id in seen:
                raise ValidationError(f"Qrels line {number}: duplicate query_id")
            seen.add(query_id)
            if not isinstance(query, str) or not query.strip():
                raise ValidationError(f"Qrels line {number}: query must be a nonempty string")
            grades = row.get("relevance")
            if not isinstance(grades, dict) or not grades:
                raise ValidationError(f"Qrels line {number}: relevance must be a nonempty object")
            for passage_id, grade in grades.items():
                if not isinstance(passage_id, str) or not passage_id.strip():
                    raise ValidationError(f"Qrels line {number}: passage IDs must be nonempty strings")
                if passage_id != passage_id.strip() or "REPLACE_WITH" in passage_id.upper():
                    raise ValidationError(f"Qrels line {number}: invalid or placeholder passage ID")
                if type(grade) is not int or grade not in (0, 1, 2):
                    raise ValidationError(f"Qrels line {number}: relevance grades must be integers 0, 1 or 2")
            if not any(grade > 0 for grade in grades.values()):
                raise ValidationError(f"Qrels line {number}: at least one positive judgment is required")
            ticker, year, form = (row.get(key) for key in ("ticker", "year", "form"))
            if ticker is not None and (not isinstance(ticker, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9.-]{0,9}", ticker)):
                raise ValidationError(f"Qrels line {number}: invalid ticker filter")
            if year is not None and (type(year) is not int or not 1900 <= year <= 2099):
                raise ValidationError(f"Qrels line {number}: year must be a filing-date year integer")
            if form is not None and form not in ("10-K", "10-Q"):
                raise ValidationError(f"Qrels line {number}: form must be 10-K or 10-Q")
            normalized_query = unicodedata.normalize("NFKC", " ".join(query.split())).casefold()
            question_key = (normalized_query, ticker.upper() if ticker else None, year, form)
            if question_key in seen_questions:
                raise ValidationError(f"Qrels line {number}: duplicate normalized question and filters, even though query_id differs")
            seen_questions.add(question_key)
            examples.append({
                "query_id": query_id, "query": query.strip(),
                "filters": {"ticker": ticker.upper() if ticker else None, "year": year, "form": form},
                "relevance": grades,
            })
    if not examples:
        raise ValidationError("Qrels file contains no evaluation records")
    return examples


def qrels_warnings(examples, top_k, candidate_k):
    warnings = [
        "Unjudged passages count as nonrelevant for reported metrics; recall is relative to supplied positive judgments, not proven complete-corpus recall.",
        "Schema validation cannot prove manual annotation, held-out provenance, 1,000-record completion or inter-annotator agreement.",
    ]
    if len(examples) < 20:
        warnings.append("Fewer than 20 queries: results are a small pilot, not strong evidence of general performance (heuristic warning).")
    small = sum(len(ex["relevance"]) < max(2 * top_k, candidate_k) for ex in examples)
    if small:
        warnings.append(f"{small}/{len(examples)} queries have a small judgment pool (<max(2*K,candidate_k)); expand pooled judgments across systems before interpreting recall (heuristic warning).")
    return warnings


def retrieval_metrics(ranked, grades, top_k):
    relevant = {key for key, grade in grades.items() if grade > 0}

    def precision(k):
        return sum(key in relevant for key in ranked[:k]) / k

    def recall(k):
        return sum(key in relevant for key in ranked[:k]) / len(relevant) if relevant else 0.0

    def ndcg(k):
        dcg = sum((2 ** grades.get(key, 0) - 1) / math.log2(rank + 1)
                  for rank, key in enumerate(ranked[:k], 1))
        ideal = sorted(grades.values(), reverse=True)[:k]
        idcg = sum((2 ** grade - 1) / math.log2(rank + 1) for rank, grade in enumerate(ideal, 1))
        return dcg / idcg if idcg else 0.0

    judged = sum(key in grades for key in ranked[:top_k])
    retrieved = len(ranked[:top_k])
    return {
        "precision_at_5": precision(5), "recall_at_5": recall(5),
        "recall_at_10": recall(10),
        "mrr_at_k": next((1 / rank for rank, key in enumerate(ranked[:top_k], 1) if key in relevant), 0.0),
        "ndcg_at_10": ndcg(10), "precision_at_k": precision(top_k),
        "recall_at_k": recall(top_k), "ndcg_at_k": ndcg(top_k),
        "judged_coverage_at_k": judged / retrieved if retrieved else 0.0,
        "judged_at_k": judged, "unjudged_at_k": retrieved - judged,
    }


def _percentile(values, fraction):
    ordered = sorted(values)
    point = (len(ordered) - 1) * fraction
    low, high = math.floor(point), math.ceil(point)
    return ordered[low] + (ordered[high] - ordered[low]) * (point - low)


def _normalize_ranking(rows, top_k):
    if not isinstance(rows, list) or len(rows) > top_k:
        raise ValidationError("Retriever must return a list of at most top_k ranked passages")
    normalized, seen = [], set()
    for rank, row in enumerate(rows, 1):
        if not isinstance(row, dict) or not isinstance(row.get("passage"), dict):
            raise ValidationError("Retriever result is missing a passage object")
        passage, passage_id = row["passage"], row.get("id")
        if (not isinstance(passage_id, str) or not passage_id or passage_id != passage.get("id")
                or passage_id in seen or row.get("rank") != rank):
            raise ValidationError("Retriever IDs/ranks must be unique, consistent and ordered from 1")
        if not isinstance(passage.get("text"), str) or not passage["text"].strip():
            raise ValidationError("Retriever must preserve nonempty full passage text")
        sources = row.get("sources", [])
        if not isinstance(sources, list) or not all(isinstance(item, str) for item in sources):
            raise ValidationError("Retriever sources must be a list of names")
        seen.add(passage_id)
        normalized.append({**row, "sources": sources})
    # Freeze each result: a retriever may reuse/mutate passage and score objects.
    try:
        return json.loads(json.dumps(normalized, ensure_ascii=False, allow_nan=False))
    except (ValueError, TypeError):
        raise ValidationError("Retriever result contains nonfinite or non-JSON values") from None


def audit_corpus(examples, retriever):
    """Verify judgment IDs/positive filters when the retriever exposes its corpus."""
    passages = getattr(retriever, "passages", None)
    if not isinstance(passages, list) or not passages:
        raise ValidationError("Retriever must expose its nonempty passages list to validate qrels IDs")
    corpus = {row["id"]: row for row in passages if isinstance(row, dict) and "id" in row}
    missing = {key for ex in examples for key in ex["relevance"] if key not in corpus}
    if missing:
        raise ValidationError(f"{len(missing)} judged passage IDs do not exist in the loaded corpus; use the matching corpus/index version")
    conflicts = 0
    for ex in examples:
        filters = ex["filters"]
        for key, grade in ex["relevance"].items():
            if grade <= 0:
                continue
            row = corpus[key]
            if ((filters["ticker"] and str(row.get("ticker", "")).upper() != filters["ticker"])
                    or (filters["form"] and row.get("form") != filters["form"])
                    or (filters["year"] and not str(row.get("filing_date", "")).startswith(str(filters["year"])))):
                conflicts += 1
    if conflicts:
        raise ValidationError(f"{conflicts} positive judgments contradict the query's ticker/year/form filters")
    return {"corpus_passages": len(corpus), "judged_ids_missing": 0, "positive_filter_conflicts": 0}


def _prepare_reranker(retriever, warmup):
    prepare = getattr(retriever, "prepare_reranker", None)
    if warmup and callable(prepare):
        prepare()
        return True
    return False


def run_experiment(examples, retriever, *, top_k=10, repeats=3, warmup=1, seed=6127,
                   candidate_k=50, timer=time.perf_counter):
    """Run four systems in shuffled, reproducible order; no files or API calls."""
    if not examples:
        raise ValidationError("At least one evaluation query is required")
    if type(top_k) is not int or top_k < 10:
        raise ValidationError("top_k must be an integer >=10")
    if type(repeats) is not int or repeats < 1 or type(warmup) is not int or warmup < 0:
        raise ValidationError("repeats must be >=1 and warmup must be >=0")
    if type(candidate_k) is not int or candidate_k < top_k:
        raise ValidationError("candidate_k must be an integer >=top_k")
    cfg = retriever.cfg["retrieval"]
    if top_k > cfg.get("rerank_pool_k", 30):
        raise ValidationError("top_k exceeds rerank_pool_k; all four variants must support the same final K")
    cfg["candidate_k"] = candidate_k
    corpus_audit = audit_corpus(examples, retriever)
    preparation_started = time.perf_counter()
    prepared = _prepare_reranker(retriever, warmup)
    preparation_s = time.perf_counter() - preparation_started
    rng, runs, warmups = random.Random(seed), [], []

    def retrieve(ex, variant):
        name, mode, rerank = variant
        started = timer()
        rows = retriever.retrieve(ex["query"], top_k=top_k, mode=mode, rerank=rerank, **ex["filters"])
        elapsed = timer() - started
        if not math.isfinite(elapsed) or elapsed < 0:
            raise ValidationError("Timer returned an invalid latency")
        return _normalize_ranking(rows, top_k), elapsed

    for variant in VARIANTS:
        for index in range(warmup):
            ex = examples[index % len(examples)]
            ranking, latency = retrieve(ex, variant)
            warmups.append({"variant": variant[0], "query_id": ex["query_id"],
                            "retrieved_count": len(ranking), "latency_s": latency})
    for repeat in range(1, repeats + 1):
        order = list(examples)
        rng.shuffle(order)
        for ex in order:
            variants = list(VARIANTS)
            rng.shuffle(variants)
            for variant in variants:
                ranking, latency = retrieve(ex, variant)
                runs.append({
                    "query_id": ex["query_id"], "variant": variant[0], "repeat": repeat,
                    "mode": variant[1], "rerank": variant[2], "top_k": top_k,
                    "latency_s": latency, "retrieved_count": len(ranking),
                    "metrics": retrieval_metrics([row["id"] for row in ranking], ex["relevance"], top_k),
                    "ranking": ranking,
                })
    warnings = qrels_warnings(examples, top_k, candidate_k)
    if not warmup:
        warnings.append("Warmup is disabled; measured latency may include lazy model initialization and is not a purely warmed comparison.")
    if warmup and not any(row["variant"] == "hybrid" and row["retrieved_count"] for row in warmups):
        warnings.append("All reranking warmup queries returned no passages. Model weights are explicitly prepared when supported, but first-inference/kernel effects may still enter the first nonempty measured query.")
    if any(row["retrieved_count"] < top_k for row in runs):
        warnings.append("Some result lists contain fewer than K passages; precision still divides by the fixed K. Inspect filters and candidate coverage.")
    if any(row["metrics"]["judged_coverage_at_k"] < 0.8 for row in runs):
        warnings.append("At least one ranking has <80% judged coverage; judging additional pooled passages is recommended (heuristic threshold).")
    for ex in examples:
        for variant, _, _ in VARIANTS:
            signatures = {tuple(hit["id"] for hit in row["ranking"]) for row in runs
                          if row["query_id"] == ex["query_id"] and row["variant"] == variant}
            if len(signatures) > 1:
                warnings.append("Rankings changed across repeated runs; inspect nondeterminism before attributing small metric differences to a component.")
                break
    summary = []
    for variant, mode, rerank in VARIANTS:
        rows = [row for row in runs if row["variant"] == variant]
        latency = [row["latency_s"] for row in rows]
        summary.append({
            "variant": variant, "mode": mode, "rerank": rerank,
            "queries": len(examples), "repeats": repeats, "measured_calls": len(rows),
            "metrics": {key: statistics.mean(row["metrics"][key] for row in rows) for key in METRICS},
            "latency_mean_s": statistics.mean(latency), "latency_median_s": statistics.median(latency),
            "latency_p95_s": _percentile(latency, 0.95), "latency_min_s": min(latency),
        })
    return {
        "schema_version": "1.0", "kind": "retrieval_ablation", "status": "completed",
        "queries": examples, "runs": runs, "summary": summary,
        "warmup_runs": warmups, "corpus_audit": corpus_audit,
        "preparation": {"reranker_explicitly_prepared": prepared,
                        "additional_preparation_s_excluded_from_query_latency": preparation_s},
        "settings": {"top_k": top_k, "repeats": repeats, "warmup_queries_per_variant": warmup,
                     "seed": seed, "candidate_k_per_branch": candidate_k,
                     "rerank_pool_k": cfg.get("rerank_pool_k", 30), "rrf_k": cfg.get("rrf_k")},
        "warnings": list(dict.fromkeys(warnings)),
    }


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact_manifest(root):
    paths = [*INDEX_FILES, "data/processed/passages.jsonl", "config.json", "requirements.txt"]
    paths.extend(str(path.relative_to(root)) for path in sorted((root / "src").glob("**/*.py")))
    return {name: {"size_bytes": (root / name).stat().st_size, "sha256": sha256_file(root / name)}
            for name in dict.fromkeys(paths) if (root / name).is_file()}


def _git(root, *args):
    try:
        proc = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, timeout=5)
        return proc.stdout.strip() if proc.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def build_manifest(root, qrels_path, settings, cfg, artifacts, initialization_s):
    packages = {}
    for name in ("numpy", "faiss-cpu", "rank-bm25", "sentence-transformers", "torch", "transformers"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    git_status = _git(root, "status", "--porcelain")
    return {
        "schema_version": "1.0", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "retrieval only; no LLM, live providers, answer quality or financial validity is evaluated",
        "qrels": {"filename": qrels_path.name, "sha256": sha256_file(qrels_path)},
        "git": {"commit": _git(root, "rev-parse", "HEAD"), "branch": _git(root, "branch", "--show-current"),
                "dirty": None if git_status is None else bool(git_status)},
        "runtime": {"python": platform.python_version(), "platform": platform.platform(),
                    "machine": platform.machine(), "processor": platform.processor(), "packages": packages},
        "configuration": {"dense_model": cfg.get("dense", {}).get("model"),
                          "reranker_model": cfg.get("retrieval", {}).get("reranker"), **settings},
        "artifacts": artifacts, "initialization_s_excluded_from_query_latency": initialization_s,
        "model_revision_note": "Model identifiers may resolve to mutable upstream revisions; retain/pin the exact local model cache for a strict rerun.",
        "timing_note": "Sequential wall-clock retrieval calls; model/index initialization and recorded warmup calls excluded. No GPU synchronization is forced; small pilot p95 estimates are unstable.",
        "shared_initialization_note": "The shared retriever loads both indexes and its dense encoder. BM25 measures BM25-only query work after shared initialization; this is not an independently deployed BM25 service's cold-start or memory cost.",
        "secrets_note": "No environment variable values, .env contents, API keys, remotes or provider response bodies are collected.",
    }


def write_report(output, result, manifest):
    def write_json(name, value):
        with (output / name).open("x", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")

    write_json("manifest.json", manifest)
    per_query_fields = ["query_id", "variant", "repeat", "top_k", "retrieved_count", "latency_s", *METRICS]
    with (output / "per_query.csv").open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=per_query_fields)
        writer.writeheader()
        for row in result["runs"]:
            writer.writerow({**{key: row[key] for key in per_query_fields if key in row}, **row["metrics"]})
    summary_fields = ["variant", "queries", "repeats", "measured_calls", *METRICS,
                      "latency_mean_s", "latency_median_s", "latency_p95_s", "latency_min_s"]
    with (output / "summary.csv").open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader()
        for row in result["summary"]:
            writer.writerow({**{key: row[key] for key in summary_fields if key in row}, **row["metrics"]})
    lines = [
        "# Retrieval ablation report", "", "Status: completed. These are retrieval measurements, not answer-quality scores.", "",
        f"Queries: {len(result['queries'])}; repeats: {result['settings']['repeats']}; final K: {result['settings']['top_k']}.",
        "Warmup calls and initialization are excluded from the measured latency below. Repeats are repeated measurements of the same queries, not extra independent queries.", "",
        "| Variant | P@5 | R@5 | R@10 | MRR@K | nDCG@10 | Judged coverage@K | Mean s | Median s | p95 s |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result["summary"]:
        metrics = row["metrics"]
        values = [metrics[key] for key in ("precision_at_5", "recall_at_5", "recall_at_10", "mrr_at_k", "ndcg_at_10", "judged_coverage_at_k")]
        values.extend(row[key] for key in ("latency_mean_s", "latency_median_s", "latency_p95_s"))
        lines.append("| " + row["variant"] + " | " + " | ".join(f"{value:.4f}" for value in values) + " |")
    lines.extend(["", "## Interpretation and limits", ""])
    lines.extend("- " + warning for warning in result["warnings"])
    lines.extend([
        "- Warmed sequential retrieval timings are not throughput, concurrency, cold-start or full RAG/LLM latency.",
        "- The shared retriever loads both indexes and a dense encoder. The BM25 row measures BM25-only queries after that shared initialization, not a standalone BM25 service's startup or memory cost.",
        "- Comparisons with hybrid_no_rerank isolate reranking; comparing the two single retrievers and their fusion tests complementarity. Keep the same corpus, qrels, filters, candidate budget and K.",
        "- No confidence intervals or significance claims are produced. More repeats reduce timing noise; more independent held-out queries are needed to assess quality uncertainty.",
        "", "## Reproducibility", "", "See manifest.json for source/index/qrels hashes, configuration, Git commit and package versions.",
        "results.json preserves full passage text and actual execution order; per_query.csv preserves each measured sample.",
        "", "## Remaining answer evaluation", "", "Human/reference answers, faithfulness, answer relevance, numerical correctness, citation support, annotation agreement and generation cost remain separate work. Do not fill them from these retrieval metrics.",
    ])
    with (output / "report.md").open("x", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    # The downstream machine-readable completion artifact is published last.
    write_json("results.json", result)


def main(argv=None, *, retriever_factory=None, root=ROOT):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", required=True, type=Path, help="Real human-judged held-out JSONL; no implicit example/default")
    parser.add_argument("--check-only", action="store_true", help="Validate qrels only; no models, indexes, network or output writes")
    parser.add_argument("--output", type=Path, help="New, non-existing output directory (required for a real run)")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--candidate-k", type=int, help="Per-branch candidate budget; defaults to config.json")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1, help="Unmeasured queries per variant, cycling through input order")
    parser.add_argument("--seed", type=int, default=6127, help="Only shuffles execution order; does not seed model/GPU libraries")
    args = parser.parse_args(argv)
    output = None
    stage = "input validation"
    try:
        if not args.file.is_file():
            raise ValidationError("Qrels file does not exist or is not a regular file")
        qrels_hash = sha256_file(args.file)
        examples = load_qrels(args.file)
        if qrels_hash != sha256_file(args.file):
            raise ValidationError("Qrels changed while being validated")
        if args.top_k < 10 or args.repeats < 1 or args.warmup < 0:
            raise ValidationError("Require top-k>=10, repeats>=1 and warmup>=0")
        if args.candidate_k is not None and args.candidate_k < args.top_k:
            raise ValidationError("candidate-k must be >=top-k")
        if args.check_only:
            print(json.dumps({"status": "qrels_validated_only", "queries": len(examples),
                              "positive_judgments": sum(sum(g > 0 for g in ex["relevance"].values()) for ex in examples),
                              "warnings": qrels_warnings(examples, args.top_k, args.candidate_k or 50),
                              "not_verified": ["human provenance/IAA", "passage IDs against corpus", "models/indexes", "retrieval or answer quality"]}, indent=2))
            return 0
        if args.output is None:
            raise ValidationError("--output is required for a real run")
        if args.output.exists() or args.output.is_symlink():
            raise ValidationError("Output directory already exists; choose a new directory to preserve existing results")
        for name in INDEX_FILES:
            if not (root / name).is_file():
                raise ValidationError(f"Missing required index artifact: {name}; obtain matching data/indexes before running")
        with (root / "config.json").open(encoding="utf-8") as handle:
            config = json.load(handle)
        candidate_k = args.candidate_k if args.candidate_k is not None else config["retrieval"]["candidate_k"]
        if type(candidate_k) is not int or candidate_k < args.top_k:
            raise ValidationError("Configured candidate_k must be an integer >=top-k")
        if args.top_k > config["retrieval"].get("rerank_pool_k", 30):
            raise ValidationError("top-k must not exceed configured rerank_pool_k")
        stage = "fingerprinting inputs"
        artifacts = artifact_manifest(root)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.mkdir()
        output = args.output
        stage = "loading local retriever/models"
        started = time.perf_counter()
        # Diagnostic prints from providers/libraries must not corrupt JSON stdout.
        with contextlib.redirect_stdout(sys.stderr):
            if retriever_factory is None:
                if __package__:
                    from .retriever import HybridRetriever
                else:
                    from retriever import HybridRetriever
                retriever_factory = HybridRetriever
            retriever = retriever_factory(use_reranker=False)
            audit_corpus(examples, retriever)
            # Empty warmup results would otherwise leave the lazy cross-encoder
            # unloaded and charge its construction to a later measured query.
            _prepare_reranker(retriever, args.warmup)
            initialization_s = time.perf_counter() - started
            stage = "running retrieval ablation"
            result = run_experiment(examples, retriever, top_k=args.top_k, repeats=args.repeats,
                                    warmup=args.warmup, seed=args.seed, candidate_k=candidate_k)
        stage = "verifying immutable inputs"
        if qrels_hash != sha256_file(args.file) or artifacts != artifact_manifest(root):
            raise ValidationError("Qrels, configuration, source or index files changed during the run; no completed results were written")
        stage = "writing reports"
        manifest = build_manifest(root, args.file, result["settings"], config, artifacts, initialization_s)
        write_report(output, result, manifest)
        print(json.dumps({"status": "completed", "output": str(output.resolve()), "queries": len(examples),
                          "variants": [row[0] for row in VARIANTS], "measured_calls": len(result["runs"])}, indent=2))
        return 0
    except (OSError, KeyError, TypeError, ValueError, ImportError, RuntimeError) as exc:
        message = str(exc) if isinstance(exc, ValidationError) else f"{stage} failed ({type(exc).__name__}); check local dependencies, artifacts and configuration"
        if output is not None and output.is_dir():
            with (output / "failure.json").open("x", encoding="utf-8") as handle:
                json.dump({"status": "failed", "stage": stage, "error": message,
                           "note": "No completed experiment is claimed. Choose a new directory for a rerun."}, handle, indent=2)
        print(json.dumps({"status": "failed", "error": message}, indent=2), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
