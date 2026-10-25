"""Frozen-evidence answer ablations and explicitly human-scored evaluation.

Preparation is offline by default. Only ``prepare --generate`` can call the
configured LLM. No live data is fetched: optionally supply a dated snapshot.
Citation ID validation is NOT a measure of semantic faithfulness.
"""

import argparse
import csv
import hashlib
import itertools
import json
import random
import re
import statistics
import time
import uuid
from datetime import date, datetime, timezone
from pathlib import Path

from .generator import GroundedGenerator, SYSTEM_PROMPT, _bounded_sources
from .rag_pipeline import _sec_url


VARIANTS = ("bm25", "dense", "hybrid_no_rerank", "hybrid")
DIMENSIONS = ("faithfulness", "relevance", "context_precision", "quality")
TOKEN_FIELDS = ("prompt_tokens", "completion_tokens", "total_tokens")
REVIEW_FIELDS = (
    "answer_id", "query", "reference_answer", "answer", "context",
    "actual_cited_sources", "annotator_id", *DIMENSIONS, "notes",
)


def _read_json(path, *, with_digest=False):
    try:
        raw = Path(path).read_bytes()
        data = json.loads(raw)
        return (data, hashlib.sha256(raw).hexdigest()) if with_digest else data
    except (OSError, ValueError, UnicodeError):
        raise ValueError("Cannot read a required JSON input; check path and JSON syntax") from None


def _text(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _load_references(path, query_ids):
    references = {}
    if path is None:
        return references, None
    try:
        raw = Path(path).read_bytes()
        lines = raw.decode("utf-8").splitlines()
    except (OSError, UnicodeError):
        raise ValueError("Cannot read reference JSONL input") from None
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError:
            raise ValueError(f"Invalid reference JSON on line {line_number}") from None
        if not isinstance(row, dict):
            raise ValueError("Each reference record must be a JSON object")
        query_id = _text(row.get("id"), "Reference id")
        if query_id not in query_ids or query_id in references:
            raise ValueError("Reference ids must be unique and present in the retrieval run")
        answer = _text(row.get("reference_answer"), "reference_answer")
        held_out = row.get("held_out", False)
        if not isinstance(held_out, bool):
            raise ValueError("held_out must be a JSON boolean")
        references[query_id] = {"reference_answer": answer, "held_out": held_out}
    return references, hashlib.sha256(raw).hexdigest()


def _valid_observation_date(value):
    """Accept ISO calendar dates and timezone-aware ISO datetimes from tools."""
    if not isinstance(value, str):
        return False
    try:
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
            date.fromisoformat(value)
            return True
        if not re.match(r"\d{4}-\d{2}-\d{2}T", value):
            return False
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return parsed.tzinfo is not None and parsed.utcoffset() is not None
    except ValueError:
        return False


def _load_live_sources(path, query_ids):
    if path is None:
        return {}, None
    data, digest = _read_json(path, with_digest=True)
    if not isinstance(data, dict) or set(data) - query_ids:
        raise ValueError("Live snapshot must be an object keyed by known query_id")
    for sources in data.values():
        if not isinstance(sources, list):
            raise ValueError("Each live snapshot value must be a source list")
        seen = set()
        for source in sources:
            if not isinstance(source, dict):
                raise ValueError("Each frozen live source must be an object")
            kind = source.get("kind")
            prefix = {"stock": "STOCK", "macro": "FRED"}.get(kind)
            source_id = source.get("source_id", "")
            if (not prefix or not isinstance(source_id, str)
                    or not re.fullmatch(prefix + r"[1-9]\d*", source_id)
                    or source_id in seen):
                raise ValueError("Frozen live sources need unique STOCKn/FREDn IDs and matching kinds")
            seen.add(source_id)
            _text(source.get("content"), "Frozen source content")
            if not isinstance(source.get("metadata"), dict):
                raise ValueError("Frozen live sources must include metadata with observation dates/units")
            date_field = "trading_date" if kind == "stock" else "latest_observation_date"
            observed_date = source["metadata"].get(date_field)
            if not _valid_observation_date(observed_date):
                raise ValueError("Frozen live sources must record dated observations, not just retrieval time")
    return data, digest


def load_cases(results_path, *, repeat=1, references_path=None, live_sources_path=None,
               max_context_chars=40000, max_source_chars=6000):
    """Validate a completed retrieval study, joining four modes by query ID.

    Never re-run retrieval or live tools. A single selected repeat is reused
    unchanged, so answer comparisons are not confounded by a new search.
    """
    data, results_digest = _read_json(results_path, with_digest=True)
    if (not isinstance(data, dict) or data.get("kind") != "retrieval_ablation"
            or data.get("status") != "completed" or data.get("schema_version") != "1.0"):
        raise ValueError("Expected completed schema 1.0 retrieval_ablation results.json")
    if type(repeat) is not int or repeat < 1:
        raise ValueError("repeat must be a positive integer")
    queries = data.get("queries")
    runs = data.get("runs")
    if not isinstance(queries, list) or not queries or not isinstance(runs, list):
        raise ValueError("Retrieval results need non-empty queries and a runs list")
    registry = {}
    for query in queries:
        if not isinstance(query, dict):
            raise ValueError("Each query must be an object")
        query_id = _text(query.get("query_id"), "query_id")
        _text(query.get("query"), "query")
        if query_id in registry or not isinstance(query.get("filters"), dict):
            raise ValueError("Queries require unique ids and an explicit filters object")
        registry[query_id] = query
    references, references_digest = _load_references(references_path, set(registry))
    live, live_digest = _load_live_sources(live_sources_path, set(registry))
    input_digests = {"retrieval_results": results_digest, "reference_answers": references_digest,
                     "frozen_live_sources": live_digest}
    selected = {}
    for run in runs:
        if not isinstance(run, dict) or run.get("query_id") not in registry:
            raise ValueError("Every retrieval run must reference a known query_id")
        if run.get("variant") not in VARIANTS or type(run.get("repeat")) is not int:
            raise ValueError("Invalid retrieval variant or repeat")
        if run["repeat"] != repeat:
            continue
        key = (run["query_id"], run["variant"])
        if key in selected:
            raise ValueError("Duplicate query/variant in selected retrieval repeat")
        selected[key] = run
    expected = {(query_id, variant) for query_id in registry for variant in VARIANTS}
    if set(selected) != expected:
        raise ValueError("Selected repeat must contain all four variants for every query")
    cases = []
    for query_id, query in registry.items():
        reference = references.get(query_id, {"reference_answer": "", "held_out": False})
        for variant in VARIANTS:
            run = selected[(query_id, variant)]
            ranking = run.get("ranking")
            top_k = run.get("top_k")
            if (not isinstance(ranking, list) or type(top_k) is not int
                    or top_k < 1 or len(ranking) > top_k):
                raise ValueError("Each ranking must be a list no longer than positive top_k")
            sources, seen = [], set()
            for rank, hit in enumerate(ranking, 1):
                passage = hit.get("passage") if isinstance(hit, dict) else None
                if not isinstance(passage, dict) or hit.get("rank") != rank:
                    raise ValueError("Ranking needs ordered 1-based ranks and full passage objects")
                passage_id = _text(passage.get("id"), "passage.id")
                if passage_id in seen or hit.get("id") != passage_id:
                    raise ValueError("Ranking passage ids must be unique and match hit ids")
                seen.add(passage_id)
                content = _text(passage.get("text"), "Full passage text")
                metadata = {key: value for key, value in passage.items() if key != "text"}
                metadata.update({"rank": rank, "url": _sec_url(passage),
                                 "retrieval_sources": hit.get("sources", [])})
                sources.append({"source_id": f"SEC{rank}", "kind": "sec",
                                "content": content, "metadata": metadata})
            sources.extend(live.get(query_id, []))
            # Validate source IDs and budgets even when no generation is requested.
            bounded, omitted = _bounded_sources(sources, max_context_chars, max_source_chars)
            cases.append({"query_id": query_id, "query": query["query"], "variant": variant,
                          "repeat": repeat, "filters": query["filters"], "top_k": top_k,
                          **reference, "evidence": sources, "sources": bounded,
                          "input_sha256": input_digests,
                          "omitted_source_ids": omitted})
    return cases


def _safe_usage(usage):
    if not isinstance(usage, dict):
        return None
    values = {field: usage[field] for field in TOKEN_FIELDS
              if type(usage.get(field)) is int and usage[field] >= 0}
    return values or None


def _write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def _new_output(path):
    output = Path(path)
    try:
        output.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        raise ValueError("Output directory already exists; choose a NEW directory to preserve results") from None
    return output


def _blinded_sources(sources):
    # Retrieval provenance/score fields would directly reveal the variant.
    visible_metadata = {"id", "ticker", "company", "cik", "accession", "filing_date",
                        "form", "year", "url", "provider", "series_id", "trading_date",
                        "latest_observation_date", "price_basis", "note", "units"}
    return [{**{key: value for key, value in source.items() if key != "metadata"},
             "metadata": {key: value for key, value in source.get("metadata", {}).items()
                          if key in visible_metadata}} for source in sources]


def prepare_study(cases, output_path, *, generate=False, generator=None,
                  max_context_chars=40000, max_source_chars=6000, seed=None):
    """Write snapshots and blinded worksheets; generation is an explicit opt-in."""
    if not cases:
        raise ValueError("No cases to prepare")
    input_digests = cases[0].get("input_sha256")
    if any(case.get("input_sha256") != input_digests for case in cases):
        raise ValueError("All answer cases must originate from the same frozen input files")
    if seed is None:
        seed = random.SystemRandom().getrandbits(64)
    if type(seed) is not int:
        raise ValueError("Schedule seed must be an integer")
    scheduled_cases = list(cases)
    random.Random(seed).shuffle(scheduled_cases)
    output = _new_output(output_path)
    if generate and generator is None:
        generator = GroundedGenerator(max_context_chars=max_context_chars,
                                      max_source_chars=max_source_chars)
    answers, mapping, schedule = [], {}, []
    for order, case in enumerate(scheduled_cases, 1):
        answer_id = uuid.uuid4().hex
        record = {key: value for key, value in case.items() if key not in ("evidence", "input_sha256")}
        record.update({"answer_id": answer_id, "answer": None, "status": "not_generated",
                       "claims": [], "citations": [], "model": None, "usage": None,
                       "limitations": [], "generation_latency_s": None, "errors": [],
                       "context_status": "planned_not_generated"})
        if generate:
            started = time.perf_counter()
            try:
                record.update(generator.generate(case["query"], case["evidence"]))
                record["usage"] = _safe_usage(record.get("usage"))
                record["context_status"] = "generator_returned_context"
            except Exception:
                # Do not persist provider exception bodies, URLs, headers or keys.
                record.update({"answer": None, "status": "generation_error", "claims": [],
                               "citations": [], "model": None, "usage": None,
                               "context_status": "planned_context_send_unconfirmed",
                               "errors": ["Generation failed; check local/provider configuration"]})
            record["generation_latency_s"] = round(time.perf_counter() - started, 6)
        answers.append(record)
        mapping[answer_id] = {"query_id": case["query_id"], "variant": case["variant"],
                              "repeat": case["repeat"]}
        schedule.append({"order": order, "answer_id": answer_id, **mapping[answer_id],
                         "generation_attempted": generate})
    # Never give answers.json/private_mapping.json to blinded reviewers.
    random.SystemRandom().shuffle(answers)
    with (output / "blind_review.csv").open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REVIEW_FIELDS)
        writer.writeheader()
        for answer in answers:
            writer.writerow({"answer_id": answer["answer_id"], "query": answer["query"],
                             "reference_answer": answer["reference_answer"],
                             "answer": answer["answer"] or "",
                             "context": json.dumps(_blinded_sources(answer["sources"]), ensure_ascii=False),
                             "actual_cited_sources": json.dumps(_blinded_sources(answer["citations"]), ensure_ascii=False)})
    manifest = {
        "schema_version": "1.0", "kind": "answer_study",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "generated" if generate else "prepared_only", "case_count": len(answers),
        "unique_queries": len({case["query_id"] for case in cases}),
        "variants": list(VARIANTS), "selected_repeat": cases[0]["repeat"],
        "max_context_chars": max_context_chars, "max_source_chars": max_source_chars,
        "api_calls_allowed": generate, "live_tools_called": False,
        "input_sha256": input_digests,
        "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
        "schedule_seed": seed, "generation_schedule": schedule,
        "status_counts": {status: sum(answer["status"] == status for answer in answers)
                          for status in sorted({answer["status"] for answer in answers})},
        "warnings": ["No automatic faithfulness score: citation validity is not semantic entailment.",
                     "held_out is a team-supplied declaration, not independently verified.",
                     "Token counts are provider-reported; no dollar cost is inferred.",
                     "Calling order is seed-shuffled; this does not control provider/LLM stochasticity.",
                     "Failed generation sources are planned context; successful transmission is unconfirmed.",
                     "Prepared-only blank answers are NOT human-evaluable answers."],
    }
    _write_json(output / "answers.json", {"manifest": manifest, "answers": answers})
    _write_json(output / "private_mapping.json", mapping)
    return manifest


def _mean(values):
    return round(statistics.mean(values), 6) if values else None


def aggregate_ratings(study_path, ratings_path, output_path):
    """Aggregate real 1..5 human ratings with an explicit IAA denominator."""
    study = _read_json(Path(study_path) / "answers.json")
    if not isinstance(study, dict) or study.get("manifest", {}).get("kind") != "answer_study":
        raise ValueError("Expected an answer study directory")
    answers = study.get("answers")
    if not isinstance(answers, list) or not answers:
        raise ValueError("Answer study contains no answer records")
    registry = {answer["answer_id"]: answer for answer in answers}
    if len(registry) != len(answers):
        raise ValueError("Study contains duplicate answer ids")
    ratings, annotators = {}, set()
    try:
        handle = Path(ratings_path).open(encoding="utf-8-sig", newline="")
    except OSError:
        raise ValueError("Cannot read ratings CSV") from None
    with handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not {"answer_id", "annotator_id", *DIMENSIONS} <= set(reader.fieldnames):
            raise ValueError("Ratings CSV needs answer_id, annotator_id and four score columns")
        for row in reader:
            if not any((row.get(field) or "").strip() for field in ("annotator_id", *DIMENSIONS)):
                continue  # untouched worksheet row, not a zero score
            answer_id = (row.get("answer_id") or "").strip()
            annotator_id = (row.get("annotator_id") or "").strip()
            if answer_id not in registry or not annotator_id:
                raise ValueError("Every rating needs a known answer_id and non-empty annotator_id")
            answer = registry[answer_id]
            if answer.get("status") in ("not_generated", "generation_error") or not answer.get("answer"):
                raise ValueError("Cannot rate blank, ungenerated or failed-provider answers")
            key = (answer_id, annotator_id)
            if key in ratings:
                raise ValueError("Duplicate answer_id/annotator_id rating")
            values = {}
            for dimension in DIMENSIONS:
                score = (row.get(dimension) or "").strip()
                if score not in ("1", "2", "3", "4", "5"):
                    raise ValueError("All four human scores must be integers 1..5; blanks are not zero")
                values[dimension] = int(score)
            ratings[key] = values
            annotators.add(annotator_id)
    if len(annotators) < 2:
        raise ValueError("Provide real ratings from at least two distinct annotators")
    by_answer = {answer_id: [] for answer_id in registry}
    for (answer_id, annotator_id), values in ratings.items():
        by_answer[answer_id].append({"annotator_id": annotator_id, **values})
    per_answer, agreement = [], {dimension: {"exact_matches": 0, "paired_comparisons": 0}
                                 for dimension in DIMENSIONS}
    for answer_id, rows in by_answer.items():
        answer = registry[answer_id]
        per_answer.append({"answer_id": answer_id, "query_id": answer["query_id"],
                           "variant": answer["variant"], "annotator_count": len(rows),
                           "means": {dimension: _mean([row[dimension] for row in rows])
                                     for dimension in DIMENSIONS}})
        for left, right in itertools.combinations(rows, 2):
            for dimension in DIMENSIONS:
                agreement[dimension]["paired_comparisons"] += 1
                agreement[dimension]["exact_matches"] += int(left[dimension] == right[dimension])
    for item in agreement.values():
        item["exact_agreement"] = (round(item["exact_matches"] / item["paired_comparisons"], 6)
                                   if item["paired_comparisons"] else None)
    total_pairs = sum(item["paired_comparisons"] for item in agreement.values())
    total_matches = sum(item["exact_matches"] for item in agreement.values())
    overall_iaa = round(total_matches / total_pairs, 6) if total_pairs else None
    summaries = []
    for variant in VARIANTS:
        rows = [row for row in per_answer if row["variant"] == variant and row["annotator_count"]]
        generated = [answer for answer in answers if answer["variant"] == variant]
        usage_totals = {}
        for field in TOKEN_FIELDS:
            values = [answer["usage"][field] for answer in generated
                      if isinstance(answer.get("usage"), dict) and field in answer["usage"]]
            usage_totals[field] = sum(values) if values else None
        summaries.append({"variant": variant, "rated_answers": len(rows),
                          "answers_with_two_annotators": sum(row["annotator_count"] >= 2 for row in rows),
                          "status_counts": {status: sum(answer["status"] == status for answer in generated)
                                            for status in sorted({answer["status"] for answer in generated})},
                          "models": sorted({answer["model"] for answer in generated if answer.get("model")}),
                          "means": {dimension: _mean([row["means"][dimension] for row in rows])
                                    for dimension in DIMENSIONS},
                          "mean_generation_latency_s": _mean([answer["generation_latency_s"] for answer in generated
                                                               if answer.get("generation_latency_s") is not None]),
                          "reported_token_totals": usage_totals,
                          "answers_with_reported_usage": sum(bool(answer.get("usage")) for answer in generated)})
    # Count unique QUESTION records, never four variants or repeated ratings.
    declared_held_out, completed_held_out = set(), set()
    for answer in answers:
        if answer.get("held_out") and answer.get("reference_answer", "").strip():
            normalized = " ".join(answer["query"].casefold().split())
            declared_held_out.add(normalized)
            query_cases = [item for item in answers if item["query_id"] == answer["query_id"]]
            if (len(query_cases) == len(VARIANTS)
                    and all(len(by_answer[item["answer_id"]]) >= 2 for item in query_cases)):
                completed_held_out.add(normalized)
    result = {
        "schema_version": "1.0", "kind": "human_answer_evaluation",
        "annotator_count": len(annotators), "rating_rows": len(ratings),
        "declared_unique_held_out_questions": len(declared_held_out),
        "held_out_questions_with_all_four_variants_double_rated": len(completed_held_out),
        "course_checks": {"at_least_1000_declared_held_out_questions": len(declared_held_out) >= 1000,
                          "at_least_1000_all_variants_double_rated": len(completed_held_out) >= 1000,
                          "observed_exact_agreement_at_least_80_percent": overall_iaa is not None and overall_iaa >= 0.8},
        "agreement": {"definition": "Exact equality on 1..5 labels, across every observed annotator pair for the same answer and dimension; not chance-corrected.",
                      "exact_matches": total_matches, "paired_comparisons": total_pairs,
                      "overall_exact_agreement": overall_iaa, "by_dimension": agreement},
        "summary": summaries, "per_answer": per_answer,
        "limitations": ["Human labels are supplied by annotators; the program never invents labels.",
                        "A held-out declaration does not prove independence from training or development.",
                        "Unique counts deduplicate normalized question text, not variants or ratings.",
                        "Means average annotators per answer, then answers per variant; incomplete coverage can bias comparisons.",
                        "Agreement is observed exact label agreement, not evidence of answer correctness or a chance-corrected statistic.",
                        "No dollar cost is inferred; missing token usage is unknown, not zero."],
    }
    output = _new_output(output_path)
    _write_json(output / "human_metrics.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="Freeze contexts and create blind review worksheets")
    prepare.add_argument("--results", required=True, help="Completed run_ablation results.json")
    prepare.add_argument("--output", help="NEW output directory (required unless --check-only)")
    prepare.add_argument("--references", help="Optional JSONL: id, reference_answer, held_out")
    prepare.add_argument("--live-sources", help="Frozen stock/macro source lists keyed by query_id")
    prepare.add_argument("--repeat", type=int, default=1)
    prepare.add_argument("--seed", type=int, help="Reproduce randomized generation-call order; not LLM randomness")
    prepare.add_argument("--max-context-chars", type=int, default=40000)
    prepare.add_argument("--max-source-chars", type=int, default=6000)
    prepare.add_argument("--generate", action="store_true", help="Explicitly allow paid/provider LLM requests")
    prepare.add_argument("--check-only", action="store_true", help="Validate only; no files, APIs or model downloads")
    aggregate = commands.add_parser("aggregate", help="Aggregate actual multi-annotator 1..5 scores")
    aggregate.add_argument("--study", required=True, help="Study directory containing answers.json")
    aggregate.add_argument("--ratings", required=True, help="Combined filled CSV from at least two annotators")
    aggregate.add_argument("--output", required=True, help="NEW metrics output directory")
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            if args.check_only and args.generate:
                raise ValueError("--check-only cannot be combined with --generate")
            if not args.check_only and not args.output:
                raise ValueError("--output NEWDIR is required unless using --check-only")
            cases = load_cases(args.results, repeat=args.repeat, references_path=args.references,
                               live_sources_path=args.live_sources, max_context_chars=args.max_context_chars,
                               max_source_chars=args.max_source_chars)
            if args.check_only:
                print(json.dumps({"status": "validated_only", "cases": len(cases), "api_calls": 0,
                                  "files_written": 0}, ensure_ascii=False))
            else:
                manifest = prepare_study(cases, args.output, generate=args.generate,
                                         max_context_chars=args.max_context_chars,
                                         max_source_chars=args.max_source_chars, seed=args.seed)
                # The full schedule lives in answers.json, not thousands of terminal lines.
                print(json.dumps({key: value for key, value in manifest.items()
                                  if key != "generation_schedule"}, ensure_ascii=False))
        else:
            result = aggregate_ratings(args.study, args.ratings, args.output)
            print(json.dumps({"status": "aggregated", "rating_rows": result["rating_rows"],
                              "agreement": result["agreement"], "course_checks": result["course_checks"]},
                             ensure_ascii=False))
    except (ValueError, OSError):
        # CLI diagnostics never echo input documents/provider payloads/secrets.
        parser.exit(2, "Input/output validation failed; check schema, score completeness, paths and use a new output directory.\n")


if __name__ == "__main__":
    main()
