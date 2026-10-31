from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


DEFAULT_DIR = Path("eval/recalibration")
NUM_MEMBERS = 6
VALID_LABELS = {0, 1, 2}


def read_csv(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows, fieldnames):
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_label(value):
    value = (value or "").strip()

    if value == "":
        return None

    try:
        label = int(value)
    except ValueError as exc:
        raise ValueError(f"Invalid label: {value!r}") from exc

    if label not in VALID_LABELS:
        raise ValueError(f"Label must be 0, 1 or 2. Found: {label}")

    return label


def cohen_kappa(labels_a, labels_b):
    """Standard unweighted Cohen's kappa."""

    if len(labels_a) != len(labels_b):
        raise ValueError("Label arrays have different lengths.")

    n = len(labels_a)

    if n == 0:
        return None

    observed = sum(a == b for a, b in zip(labels_a, labels_b)) / n

    counts_a = Counter(labels_a)
    counts_b = Counter(labels_b)

    expected = 0.0

    for label in VALID_LABELS:
        pa = counts_a[label] / n
        pb = counts_b[label] / n
        expected += pa * pb

    if expected == 1.0:
        return 1.0

    return (observed - expected) / (1.0 - expected)


def fleiss_kappa(items):
    """
    Compute Fleiss' kappa for items with exactly two ratings each.

    Different human annotator pairs may rate different items, but each
    item must receive exactly two ratings.
    """

    if not items:
        return None

    n_raters = 2
    categories = sorted(VALID_LABELS)

    item_agreements = []
    category_totals = Counter()

    for labels in items:
        if len(labels) != n_raters:
            raise ValueError("Each item must have exactly two ratings.")

        counts = Counter(labels)

        numerator = sum(
            count * count
            for count in counts.values()
        ) - n_raters

        denominator = n_raters * (n_raters - 1)

        p_i = numerator / denominator
        item_agreements.append(p_i)

        for label in labels:
            category_totals[label] += 1

    p_bar = sum(item_agreements) / len(item_agreements)

    total_ratings = len(items) * n_raters

    p_e = sum(
        (category_totals[category] / total_ratings) ** 2
        for category in categories
    )

    if p_e == 1.0:
        return 1.0

    return (p_bar - p_e) / (1.0 - p_e)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dir",
        type=Path,
        default=DEFAULT_DIR,
    )

    args = parser.parse_args()

    master_path = args.dir / "recalibration_master.csv"

    if not master_path.exists():
        raise FileNotFoundError(master_path)

    master = read_csv(master_path)

    if not master:
        raise RuntimeError(f"Master file is empty: {master_path}")

    master_by_id = {
        row["record_id"]: row
        for row in master
    }

    ratings = defaultdict(list)
    calibration_ratings = defaultdict(list)

    # ---------------------------------------------------------
    # Load six member files
    # ---------------------------------------------------------

    missing_labels = []

    for member in range(1, NUM_MEMBERS + 1):
        #path = args.dir / f"member_{member}_recalibration.csv"
        path = args.dir / f"member_{member}_recalibration_ai_suggested.csv"

        if not path.exists():
            raise FileNotFoundError(
                f"Missing completed file: {path}"
            )

        rows = read_csv(path)

        for row in rows:
            '''label = parse_label(
                row.get("label")
            )'''
            label = parse_label(
                row.get("ai_suggested_label")
            )

            if label is None:
                missing_labels.append({
                    "member": member,
                    "file": path.name,
                    "record_id": row["record_id"],
                })
                continue

            rid = row["record_id"]

            if rid not in master_by_id:
                raise RuntimeError(
                    f"Unknown record ID: {rid}"
                )

            round_type = (
                row.get("round_type")
                or master_by_id[rid]["round_type"]
            )

            entry = {
                "member": member,
                "label": label,
                "notes": row.get("notes", ""),
            }

            if round_type == "calibration":
                calibration_ratings[rid].append(entry)

            elif round_type == "fresh_overlap":
                ratings[rid].append(entry)

            else:
                raise RuntimeError(
                    f"Unknown round_type {round_type!r} "
                    f"for record {rid}"
                )

    # ---------------------------------------------------------
    # Stop cleanly if annotations are incomplete
    # ---------------------------------------------------------

    if missing_labels:
        print()
        print("=" * 60)
        print("ANNOTATION ROUND NOT COMPLETE")
        print("=" * 60)

        counts = Counter(
            item["member"]
            for item in missing_labels
        )

        for member in range(1, NUM_MEMBERS + 1):
            print(
                f"Member {member}: "
                f"{counts[member]} missing labels"
            )

        print(
            f"\nTotal missing labels: "
            f"{len(missing_labels)}"
        )

        print("\nFirst 20 missing records:")

        for item in missing_labels[:20]:
            print(
                f"  {item['file']} -> "
                f"{item['record_id']}"
            )

        raise SystemExit(
            "\nComplete the missing labels and rerun."
        )

    # ---------------------------------------------------------
    # Validate calibration round
    # ---------------------------------------------------------

    calibration_master = [
        row
        for row in master
        if row["round_type"] == "calibration"
    ]

    if len(calibration_master) != 20:
        raise RuntimeError(
            f"Expected 20 calibration records, "
            f"found {len(calibration_master)}."
        )

    calibration_incomplete = []

    for row in calibration_master:
        rid = row["record_id"]

        if len(calibration_ratings[rid]) != NUM_MEMBERS:
            calibration_incomplete.append(
                (
                    rid,
                    len(calibration_ratings[rid]),
                )
            )

    if calibration_incomplete:
        print(
            "\nCalibration records do not all "
            "have six ratings:"
        )

        for rid, count in calibration_incomplete:
            print(
                f"  {rid}: {count} ratings"
            )

        raise RuntimeError(
            "Calibration round incomplete."
        )

    # ---------------------------------------------------------
    # Validate fresh round
    # ---------------------------------------------------------

    fresh_master = [
        row
        for row in master
        if row["round_type"] == "fresh_overlap"
    ]

    if len(fresh_master) != 200:
        raise RuntimeError(
            f"Expected 200 fresh records, "
            f"found {len(fresh_master)}."
        )

    incomplete = []

    for row in fresh_master:
        rid = row["record_id"]

        if len(ratings[rid]) != 2:
            incomplete.append(
                (
                    rid,
                    len(ratings[rid]),
                )
            )

    if incomplete:
        print(
            "Incomplete fresh records:"
        )

        for rid, count in incomplete:
            print(
                f"  {rid}: {count} labels"
            )

        raise RuntimeError(
            "Fresh agreement round incomplete."
        )

    # ---------------------------------------------------------
    # Agreement
    # ---------------------------------------------------------

    exact_matches = 0
    all_item_labels = []

    pair_records = defaultdict(list)

    disagreements = []
    agreed = []

    for row in fresh_master:
        rid = row["record_id"]

        r1, r2 = sorted(
            ratings[rid],
            key=lambda x: x["member"],
        )

        l1 = r1["label"]
        l2 = r2["label"]

        all_item_labels.append([l1, l2])

        pair = (
            r1["member"],
            r2["member"],
        )

        pair_records[pair].append(
            (l1, l2)
        )

        if l1 == l2:
            exact_matches += 1

            agreed.append({
                "record_id": rid,
                "query_id": row["query_id"],
                "query": row["query"],
                "passage_id": row["passage_id"],
                "member_1": r1["member"],
                "label_1": l1,
                "member_2": r2["member"],
                "label_2": l2,
                "final_label": l1,
            })

        else:
            disagreements.append({
                "record_id": rid,
                "query_id": row["query_id"],
                "query_category": row["query_category"],
                "query": row["query"],
                "ticker": row["ticker"],
                "passage_id": row["passage_id"],
                "passage_text": row["passage_text"],
                "member_1": r1["member"],
                "label_1": l1,
                "member_2": r2["member"],
                "label_2": l2,
                "final_label": "",
                "rationale": "",
            })

    exact_agreement = (
        exact_matches
        / len(fresh_master)
    )

    overall_fleiss = fleiss_kappa(
        all_item_labels
    )

    # ---------------------------------------------------------
    # Pairwise Cohen kappas
    # ---------------------------------------------------------

    pair_results = {}

    for pair, values in sorted(
        pair_records.items()
    ):
        labels_a = [
            a
            for a, _ in values
        ]

        labels_b = [
            b
            for _, b in values
        ]

        exact = sum(
            a == b
            for a, b in values
        ) / len(values)

        kappa = cohen_kappa(
            labels_a,
            labels_b,
        )

        key = (
            f"Member_{pair[0]}"
            f"_vs_"
            f"Member_{pair[1]}"
        )

        pair_results[key] = {
            "n": len(values),
            "exact_agreement": exact,
            "cohen_kappa": kappa,
        }

    # ---------------------------------------------------------
    # Calibration summary
    # ---------------------------------------------------------

    calibration_exact_all_six = 0

    for row in calibration_master:
        labels = [
            rating["label"]
            for rating in calibration_ratings[
                row["record_id"]
            ]
        ]

        if len(set(labels)) == 1:
            calibration_exact_all_six += 1

    calibration_unanimous_rate = (
        calibration_exact_all_six
        / len(calibration_master)
    )

    # ---------------------------------------------------------
    # Save results
    # ---------------------------------------------------------

    results = {
        "calibration_records": len(
            calibration_master
        ),
        "calibration_unanimous_records":
            calibration_exact_all_six,
        "calibration_unanimous_rate":
            calibration_unanimous_rate,
        "fresh_records": len(
            fresh_master
        ),
        "exact_matches": exact_matches,
        "exact_agreement":
            exact_agreement,
        "fleiss_kappa":
            overall_fleiss,
        "disagreements": len(
            disagreements
        ),
        "agreement_target_80_percent_met":
            exact_agreement >= 0.80,
        "pairwise_results":
            pair_results,
    }

    results_path = (
        args.dir
        / "agreement_results.json"
    )

    results_path.write_text(
        json.dumps(
            results,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    disagreements_path = (
        args.dir
        / "fresh_disagreements.csv"
    )

    agreed_path = (
        args.dir
        / "fresh_agreed.csv"
    )

    if disagreements:
        write_csv(
            disagreements_path,
            disagreements,
            list(
                disagreements[0].keys()
            ),
        )
    elif disagreements_path.exists():
        disagreements_path.unlink()

    if agreed:
        write_csv(
            agreed_path,
            agreed,
            list(
                agreed[0].keys()
            ),
        )
    elif agreed_path.exists():
        agreed_path.unlink()

    # ---------------------------------------------------------
    # Print
    # ---------------------------------------------------------

    print()
    print("=" * 60)
    print("CALIBRATION SUMMARY")
    print("=" * 60)

    print(
        f"Calibration records: "
        f"{len(calibration_master)}"
    )

    print(
        f"Unanimous across all 6 members: "
        f"{calibration_exact_all_six}/"
        f"{len(calibration_master)} "
        f"({calibration_unanimous_rate:.1%})"
    )

    print()
    print("=" * 60)
    print("FRESH INDEPENDENT AGREEMENT RESULTS")
    print("=" * 60)

    print(
        f"Fresh records: "
        f"{len(fresh_master)}"
    )

    print(
        f"Exact matches: "
        f"{exact_matches}"
    )

    print(
        f"Exact agreement: "
        f"{exact_agreement:.1%}"
    )

    print(
        f"Fleiss' kappa: "
        f"{overall_fleiss:.3f}"
    )

    print(
        f"Disagreements: "
        f"{len(disagreements)}"
    )

    print()

    if exact_agreement >= 0.80:
        print(
            "[PASS] >=80% exact agreement achieved."
        )
    else:
        print(
            "[NOT YET] <80% exact agreement."
        )

    print()
    print("Per annotator pair:")

    for pair, result in pair_results.items():
        kappa = result["cohen_kappa"]

        kappa_text = (
            f"{kappa:.3f}"
            if kappa is not None
            else "N/A"
        )

        print(
            f"  {pair}: "
            f"N={result['n']}, "
            f"agreement="
            f"{result['exact_agreement']:.1%}, "
            f"kappa={kappa_text}"
        )

    print()
    print(
        f"Saved: {results_path}"
    )

    if disagreements:
        print(
            "Review disagreements in:"
        )
        print(
            disagreements_path
        )


if __name__ == "__main__":
    main()
