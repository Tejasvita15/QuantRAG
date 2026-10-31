from __future__ import annotations

import argparse
import csv
import random
from collections import Counter, defaultdict
from pathlib import Path


DEFAULT_INPUT = Path("eval/annotations/annotation_master.csv")
DEFAULT_OUTPUT_DIR = Path("eval/recalibration")

NUM_MEMBERS = 6
CALIBRATION_SIZE = 20
FRESH_OVERLAP_SIZE = 200
SEED = 42

LABEL_VALUES = {"0", "1", "2"}


def read_csv(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def clean(value):
    return (value or "").strip()


def proportional_sample(rows, n, category_col, rng):
    """
    Approximately preserve the query-category distribution.
    """

    groups = defaultdict(list)

    for row in rows:
        category = clean(row.get(category_col)) or "UNKNOWN"
        groups[category].append(row)

    total = len(rows)

    if n > total:
        raise ValueError(
            f"Cannot sample {n} rows from only {total} available rows."
        )

    # Desired proportional allocations
    allocations = {}
    remainders = []

    allocated = 0

    for category, group in groups.items():
        exact = n * len(group) / total
        base = int(exact)

        allocations[category] = min(base, len(group))
        allocated += allocations[category]

        remainders.append(
            (exact - base, category)
        )

    # Give remaining slots to largest fractional remainders
    remaining = n - allocated

    for _, category in sorted(
        remainders,
        reverse=True,
    ):
        if remaining <= 0:
            break

        if allocations[category] < len(groups[category]):
            allocations[category] += 1
            remaining -= 1

    # If still short due to tiny groups, fill from anywhere
    while remaining > 0:
        candidates = [
            category
            for category, group in groups.items()
            if allocations[category] < len(group)
        ]

        if not candidates:
            break

        rng.shuffle(candidates)

        for category in candidates:
            if remaining <= 0:
                break

            allocations[category] += 1
            remaining -= 1

    result = []

    for category, count in allocations.items():
        group = groups[category][:]
        rng.shuffle(group)
        result.extend(group[:count])

    rng.shuffle(result)

    if len(result) != n:
        raise RuntimeError(
            f"Expected {n} sampled rows, got {len(result)}."
        )

    return result


def blind_row(row, round_type, assigned_pair=""):
    """
    Remove all old labels, retriever scores, ranks,
    and previous annotator information.
    """

    return {
        "record_id": row["record_id"],
        "query_id": row["query_id"],
        "query_category": row.get("query_category", ""),
        "query": row["query"],
        "ticker": row.get("ticker", ""),
        "passage_id": row["passage_id"],
        "passage_text": row["passage_text"],
        "filing_date": row.get("filing_date", ""),
        "form": row.get("form", ""),
        "section": row.get("section", ""),
        "round_type": round_type,
        "assigned_pair": assigned_pair,
        "label": "",
        "notes": "",
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
    )

    args = parser.parse_args()

    rng = random.Random(args.seed)

    if not args.input.exists():
        raise FileNotFoundError(
            f"Could not find annotation master:\n{args.input}"
        )

    rows = read_csv(args.input)

    print(f"Loaded master records: {len(rows)}")

    required = {
        "record_id",
        "query_id",
        "query",
        "passage_id",
        "passage_text",
    }

    missing = required - set(rows[0].keys())

    if missing:
        raise ValueError(
            f"Master CSV missing columns: {sorted(missing)}"
        )

    # ---------------------------------------------------------
    # Identify records that were NOT part of original overlap
    # ---------------------------------------------------------

    singly_annotated = []

    for row in rows:
        secondary = clean(
            row.get("secondary_annotator")
        )

        if not secondary:
            singly_annotated.append(row)

    print(
        f"Previously single-annotated records: "
        f"{len(singly_annotated)}"
    )

    needed = (
        CALIBRATION_SIZE
        + FRESH_OVERLAP_SIZE
    )

    if len(singly_annotated) < needed:
        raise RuntimeError(
            f"Need at least {needed} previously "
            f"single-annotated records, but found "
            f"{len(singly_annotated)}."
        )

    # ---------------------------------------------------------
    # Select 20 calibration examples
    # ---------------------------------------------------------

    calibration = proportional_sample(
        singly_annotated,
        CALIBRATION_SIZE,
        "query_category",
        rng,
    )

    calibration_ids = {
        row["record_id"]
        for row in calibration
    }

    remaining = [
        row
        for row in singly_annotated
        if row["record_id"] not in calibration_ids
    ]

    # ---------------------------------------------------------
    # Select fresh 200 records
    # ---------------------------------------------------------

    fresh = proportional_sample(
        remaining,
        FRESH_OVERLAP_SIZE,
        "query_category",
        rng,
    )

    # ---------------------------------------------------------
    # Assign two annotators to every fresh item
    # ---------------------------------------------------------

    pairs = [
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 5),
        (5, 6),
        (6, 1),
    ]

    rng.shuffle(fresh)

    assignments = {}

    pair_counts = Counter()

    for i, row in enumerate(fresh):
        a, b = pairs[i % len(pairs)]

        assignments[row["record_id"]] = (a, b)

        pair_counts[(a, b)] += 1

    # ---------------------------------------------------------
    # Create master mapping
    # ---------------------------------------------------------

    master_rows = []

    for row in calibration:
        master_rows.append({
            "record_id": row["record_id"],
            "query_id": row["query_id"],
            "query_category": row.get(
                "query_category",
                "",
            ),
            "query": row["query"],
            "ticker": row.get("ticker", ""),
            "passage_id": row["passage_id"],
            "passage_text": row["passage_text"],
            "filing_date": row.get(
                "filing_date",
                "",
            ),
            "form": row.get("form", ""),
            "section": row.get("section", ""),
            "round_type": "calibration",
            "annotator_a": "ALL",
            "annotator_b": "ALL",
        })

    for row in fresh:
        a, b = assignments[row["record_id"]]

        master_rows.append({
            "record_id": row["record_id"],
            "query_id": row["query_id"],
            "query_category": row.get(
                "query_category",
                "",
            ),
            "query": row["query"],
            "ticker": row.get("ticker", ""),
            "passage_id": row["passage_id"],
            "passage_text": row["passage_text"],
            "filing_date": row.get(
                "filing_date",
                "",
            ),
            "form": row.get("form", ""),
            "section": row.get("section", ""),
            "round_type": "fresh_overlap",
            "annotator_a": f"Member_{a}",
            "annotator_b": f"Member_{b}",
        })

    master_fields = list(master_rows[0].keys())

    write_csv(
        args.output_dir / "recalibration_master.csv",
        master_rows,
        master_fields,
    )

    # ---------------------------------------------------------
    # Calibration file
    # ---------------------------------------------------------

    calibration_rows = [
        blind_row(
            row,
            "calibration",
            "ALL",
        )
        for row in calibration
    ]

    annotation_fields = list(
        calibration_rows[0].keys()
    )

    write_csv(
        args.output_dir / "calibration_20.csv",
        calibration_rows,
        annotation_fields,
    )

    # ---------------------------------------------------------
    # Create six blind files
    # ---------------------------------------------------------

    member_counts = Counter()

    for member in range(1, NUM_MEMBERS + 1):

        tasks = []

        # Everyone labels all 20 calibration examples.
        for row in calibration:
            tasks.append(
                blind_row(
                    row,
                    "calibration",
                    "ALL",
                )
            )

        # Fresh agreement records assigned to this member.
        for row in fresh:
            a, b = assignments[row["record_id"]]

            if member in (a, b):
                tasks.append(
                    blind_row(
                        row,
                        "fresh_overlap",
                        f"Member_{a}|Member_{b}",
                    )
                )

        # Randomize order within each file.
        rng.shuffle(tasks)

        member_counts[member] = len(tasks)

        output = (
            args.output_dir
            / f"member_{member}_recalibration.csv"
        )

        write_csv(
            output,
            tasks,
            annotation_fields,
        )

    # ---------------------------------------------------------
    # Annotation guide
    # ---------------------------------------------------------

    guide = """
QUANTRAG RECALIBRATION ANNOTATION GUIDE
=======================================

ANNOTATION UNIT
---------------
One record consists of:

    one financial query
    +
    one SEC filing passage

Judge how useful the passage is for answering
the exact query.


LABEL 2 — DIRECTLY RELEVANT
---------------------------
The passage contains substantive evidence that
directly answers the query or provides a major
answer component.

Examples:
- It explicitly identifies factors that drove
  revenue growth.
- It explicitly lists major business risks.
- It explicitly explains why operating margin
  changed.


LABEL 1 — PARTIALLY RELEVANT
----------------------------
The passage contains useful evidence or context,
but it is incomplete, indirect, prospective, or
only addresses part of the question.

Examples:
- A future risk that could affect profitability,
  when the question asks what actually affected
  profitability.
- A business segment description that gives some
  relevant operational context but does not fully
  answer the query.


LABEL 0 — IRRELEVANT
--------------------
The passage does not materially help answer the
query.

Important:
- Keyword overlap alone is not relevance.
- Merely saying "see Risk Factors" is not evidence.
- General company information is not automatically
  relevant.
- A passage must contribute substantive information
  toward the answer.


DECISION TEST
-------------
Ask:

"Can I point to specific text in this passage that
materially helps answer this exact question?"

No -> 0

Yes, but more evidence is needed to properly answer
the question -> 1

Yes, and this passage itself contains a substantive
answer or major answer component -> 2


INDEPENDENCE
------------
Do not view:
- previous labels
- retrieval rank
- retriever score
- another annotator's label

Do not discuss fresh-overlap labels until everyone
has completed their file.
""".strip()

    guide_path = (
        args.output_dir
        / "ANNOTATION_GUIDE.txt"
    )

    guide_path.write_text(
        guide + "\n",
        encoding="utf-8",
    )

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    print()
    print("=" * 60)
    print("RECALIBRATION ROUND CREATED")
    print("=" * 60)

    print(
        f"Calibration records: {len(calibration)}"
    )

    print(
        f"Fresh independent overlap records: "
        f"{len(fresh)}"
    )

    print()
    print("Fresh pair assignment:")

    for pair, count in sorted(pair_counts.items()):
        print(
            f"  Member_{pair[0]} + "
            f"Member_{pair[1]}: {count}"
        )

    print()
    print("Total work per member:")

    for member in range(1, NUM_MEMBERS + 1):
        fresh_count = (
            member_counts[member]
            - CALIBRATION_SIZE
        )

        print(
            f"  Member {member}: "
            f"{CALIBRATION_SIZE} calibration + "
            f"{fresh_count} fresh = "
            f"{member_counts[member]}"
        )

    print()
    print(
        f"Output directory: "
        f"{args.output_dir.resolve()}"
    )


if __name__ == "__main__":
    main()