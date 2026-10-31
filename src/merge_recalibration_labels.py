import csv
import json
from pathlib import Path
import argparse

DEFAULT_DIR = Path("eval/recalibration")

DEFAULT_QRELS = Path(
    "eval/retrieval_eval.jsonl"
)

DEFAULT_OUTPUT = Path(
    "eval/retrieval_eval_recalibrated.jsonl"
)

VALID_LABELS = {0, 1, 2}


def read_csv(path: Path):
    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        return list(csv.DictReader(f))


def parse_label(value, context):
    value = (value or "").strip()

    if value == "":
        raise RuntimeError(
            f"Missing final label: {context}"
        )

    try:
        label = int(value)

    except ValueError as exc:
        raise RuntimeError(
            f"Invalid label {value!r}: "
            f"{context}"
        ) from exc

    if label not in VALID_LABELS:
        raise RuntimeError(
            f"Label must be 0, 1 or 2: "
            f"{context}"
        )

    return label


def load_jsonl(path: Path):
    rows = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:

        for line_number, line in enumerate(
            f,
            start=1,
        ):

            line = line.strip()

            if not line:
                continue

            try:
                rows.append(
                    json.loads(line)
                )

            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"Invalid JSON on line "
                    f"{line_number} of {path}"
                ) from exc

    return rows


def write_jsonl(path: Path, rows):
    with path.open(
        "w",
        encoding="utf-8",
    ) as f:

        for row in rows:
            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dir",
        type=Path,
        default=DEFAULT_DIR,
        help=(
            "Directory containing recalibration files."
        ),
    )

    parser.add_argument(
        "--input-qrels",
        type=Path,
        default=DEFAULT_QRELS,
        help=(
            "Original retrieval evaluation JSONL."
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=(
            "Output recalibrated evaluation JSONL."
        ),
    )

    args = parser.parse_args()

    # ---------------------------------------------------------
    # Input paths
    # ---------------------------------------------------------

    agreed_path = (
        args.dir
        / "fresh_agreed.csv"
    )

    disagreement_path = (
        args.dir
        / "fresh_disagreements.csv"
    )

    master_path = (
        args.dir
        / "recalibration_master.csv"
    )

    # ---------------------------------------------------------
    # Validate required files
    # ---------------------------------------------------------

    if not agreed_path.exists():
        raise FileNotFoundError(
            f"Missing agreed-record file: "
            f"{agreed_path}"
        )

    if not master_path.exists():
        raise FileNotFoundError(
            f"Missing recalibration master: "
            f"{master_path}"
        )

    if not args.input_qrels.exists():
        raise FileNotFoundError(
            f"Missing input evaluation file: "
            f"{args.input_qrels}"
        )

    # ---------------------------------------------------------
    # Load files
    # ---------------------------------------------------------

    agreed = read_csv(
        agreed_path
    )

    # No disagreement file is expected when agreement is 100%.
    if disagreement_path.exists():
        disagreements = read_csv(
            disagreement_path
        )
    else:
        disagreements = []

    master = read_csv(
        master_path
    )

    print(
        f"Loaded agreed records: "
        f"{len(agreed)}"
    )

    print(
        f"Loaded disagreement records: "
        f"{len(disagreements)}"
    )

    # ---------------------------------------------------------
    # Build final labels for the 200 fresh records
    # ---------------------------------------------------------

    updates = {}

    # Agreed records already have their gold label.
    for row in agreed:

        record_id = row["record_id"]

        label = parse_label(
            row.get("final_label"),
            record_id,
        )

        if record_id in updates:
            raise RuntimeError(
                f"Duplicate record in recalibration "
                f"data: {record_id}"
            )

        updates[record_id] = {
            "query_id": row["query_id"],
            "passage_id": row[
                "passage_id"
            ],
            "label": label,
            "source": (
                "fresh_independent_agreement"
            ),
        }

    # Disagreements require an adjudicated final_label.
    for row in disagreements:

        record_id = row["record_id"]

        label = parse_label(
            row.get("final_label"),
            (
                f"{record_id} "
                f"(adjudication required)"
            ),
        )

        if record_id in updates:
            raise RuntimeError(
                f"Record appears in both agreed and "
                f"disagreement files: {record_id}"
            )

        updates[record_id] = {
            "query_id": row["query_id"],
            "passage_id": row[
                "passage_id"
            ],
            "label": label,
            "source": (
                "fresh_disagreement_adjudicated"
            ),
        }

    # ---------------------------------------------------------
    # Validate fresh-round size
    # ---------------------------------------------------------

    if len(updates) != 200:
        raise RuntimeError(
            f"Expected exactly 200 final "
            f"recalibration labels, but got "
            f"{len(updates)}."
        )

    # ---------------------------------------------------------
    # Validate against master mapping
    # ---------------------------------------------------------

    fresh_master = {
        row["record_id"]: row
        for row in master
        if row.get("round_type")
        == "fresh_overlap"
    }

    if len(fresh_master) != 200:
        raise RuntimeError(
            f"Expected 200 fresh_overlap records "
            f"in recalibration_master.csv, but "
            f"found {len(fresh_master)}."
        )

    update_ids = set(
        updates.keys()
    )

    master_ids = set(
        fresh_master.keys()
    )

    if update_ids != master_ids:

        missing = sorted(
            master_ids
            - update_ids
        )

        extra = sorted(
            update_ids
            - master_ids
        )

        raise RuntimeError(
            "Record mismatch between final labels "
            "and recalibration master.\n"
            f"Missing records: {missing}\n"
            f"Unexpected records: {extra}"
        )

    # Validate query and passage identity.
    for record_id, update in updates.items():

        master_row = (
            fresh_master[
                record_id
            ]
        )

        if (
            update["query_id"]
            != master_row["query_id"]
        ):
            raise RuntimeError(
                f"Query ID mismatch for "
                f"{record_id}: "
                f"{update['query_id']} vs "
                f"{master_row['query_id']}"
            )

        if (
            update["passage_id"]
            != master_row["passage_id"]
        ):
            raise RuntimeError(
                f"Passage ID mismatch for "
                f"{record_id}: "
                f"{update['passage_id']} vs "
                f"{master_row['passage_id']}"
            )

    # ---------------------------------------------------------
    # Load original evaluation JSONL
    # ---------------------------------------------------------

    qrels = load_jsonl(
        args.input_qrels
    )

    if not qrels:
        raise RuntimeError(
            f"No records found in "
            f"{args.input_qrels}"
        )

    qrels_by_query = {}

    for row in qrels:

        query_id = row.get(
            "query_id"
        )

        if not query_id:
            raise RuntimeError(
                "A retrieval_eval.jsonl record "
                "is missing query_id."
            )

        if query_id in qrels_by_query:
            raise RuntimeError(
                f"Duplicate query_id in "
                f"retrieval_eval.jsonl: "
                f"{query_id}"
            )

        qrels_by_query[
            query_id
        ] = row

    audit_rows = []

    # ---------------------------------------------------------
    # Replace the 200 recalibrated labels
    # ---------------------------------------------------------

    for record_id, update in updates.items():

        query_id = update[
            "query_id"
        ]

        passage_id = update[
            "passage_id"
        ]

        new_label = update[
            "label"
        ]

        if query_id not in qrels_by_query:
            raise RuntimeError(
                f"Query not found in qrels: "
                f"{query_id}"
            )

        qrow = qrels_by_query[
            query_id
        ]

        relevance = qrow.setdefault(
            "relevance",
            {},
        )

        old_label = relevance.get(
            passage_id
        )

        if old_label is None:
            raise RuntimeError(
                f"Passage {passage_id} for "
                f"{query_id} was not present "
                f"in the original relevance set."
            )

        relevance[
            passage_id
        ] = new_label

        audit_rows.append({
            "record_id":
                record_id,

            "query_id":
                query_id,

            "passage_id":
                passage_id,

            "old_label":
                old_label,

            "new_label":
                new_label,

            "source":
                update["source"],
        })

    # ---------------------------------------------------------
    # Write recalibrated evaluation file
    # ---------------------------------------------------------

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    write_jsonl(
        args.output,
        qrels,
    )

    # ---------------------------------------------------------
    # Write audit trail
    # ---------------------------------------------------------

    audit_path = (
        args.dir
        / "recalibration_merge_audit.csv"
    )

    with audit_path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "record_id",
                "query_id",
                "passage_id",
                "old_label",
                "new_label",
                "source",
            ],
        )

        writer.writeheader()

        writer.writerows(
            audit_rows
        )

    # ---------------------------------------------------------
    # Summary statistics
    # ---------------------------------------------------------

    changed = sum(
        str(row["old_label"])
        != str(row["new_label"])
        for row in audit_rows
    )

    unchanged = (
        len(audit_rows)
        - changed
    )

    new_label_counts = {
        label: 0
        for label in sorted(
            VALID_LABELS
        )
    }

    for row in audit_rows:
        new_label_counts[
            int(row["new_label"])
        ] += 1

    # ---------------------------------------------------------
    # Output summary
    # ---------------------------------------------------------

    print()
    print("=" * 60)
    print("RECALIBRATION MERGE COMPLETE")
    print("=" * 60)

    print(
        f"Fresh records updated: "
        f"{len(audit_rows)}"
    )

    print(
        f"Labels changed from old "
        f"gold set: {changed}"
    )

    print(
        f"Labels unchanged: "
        f"{unchanged}"
    )

    print()
    print("New-label distribution:")

    for label in sorted(
        new_label_counts
    ):
        print(
            f"  Label {label}: "
            f"{new_label_counts[label]}"
        )

    print()
    print(
        "New evaluation file:"
    )

    print(
        args.output
    )

    print()
    print(
        "Audit trail:"
    )

    print(
        audit_path
    )

    print()
    print(
        "Original retrieval_eval.jsonl "
        "was NOT overwritten."
    )


if __name__ == "__main__":
    main()