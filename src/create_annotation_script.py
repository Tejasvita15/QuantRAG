import csv
import json
import random
from collections import defaultdict
from pathlib import Path

from retriever import HybridRetriever


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

QUERY_FILE = Path("eval/queries.jsonl")

OUTPUT_DIR = Path("eval/annotations")
MASTER_FILE = OUTPUT_DIR / "annotation_master.csv"

NUM_MEMBERS = 6

# Assignment requirement
NUM_QUERIES = 100
CANDIDATES_PER_QUERY = 10

# 200 / 1000 records will receive a second independent label
OVERLAP_RECORDS = 200

# Retrieve more than we finally keep so that duplicates
# across retrieval systems can be removed safely.
RETRIEVAL_POOL_K = 30


# ============================================================
# HELPERS
# ============================================================

def load_queries(path):
    """
    Expected JSONL format:

    {
        "query_id": "q001",
        "ticker": "AAPL",
        "category": "risk",
        "query": "What supply-chain risks does Apple report?"
    }
    """

    queries = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            item = json.loads(line)

            required = [
                "query_id",
                "ticker",
                "category",
                "query",
            ]

            missing = [
                k for k in required
                if k not in item
            ]

            if missing:
                raise ValueError(
                    f"Missing fields {missing} in query:\n{item}"
                )

            queries.append(item)

    if len(queries) != NUM_QUERIES:
        raise ValueError(
            f"Expected exactly {NUM_QUERIES} queries, "
            f"but found {len(queries)}."
        )

    return queries


def add_candidates(
    storage,
    results,
    source,
    max_add=None,
):
    """
    Add results to candidate dictionary while preserving
    information about which retriever(s) returned a passage.
    """

    added = 0

    for result in results:

        passage = result["passage"]
        passage_id = passage["id"]

        if passage_id not in storage:

            storage[passage_id] = {
                "passage": passage,
                "sources": [],
                "source_ranks": {},
                "rrf_score": result.get("rrf_score"),
                "rerank_score": result.get("rerank_score"),
            }

        if source not in storage[passage_id]["sources"]:
            storage[passage_id]["sources"].append(source)

        storage[passage_id]["source_ranks"][source] = result.get(
            "rank"
        )

        # Keep scores if available
        if result.get("rrf_score") is not None:
            storage[passage_id]["rrf_score"] = result.get(
                "rrf_score"
            )

        if result.get("rerank_score") is not None:
            storage[passage_id]["rerank_score"] = result.get(
                "rerank_score"
            )

        added += 1

        if max_add is not None and added >= max_add:
            break


def select_candidates(
    bm25_results,
    dense_results,
    hybrid_results,
    reranked_results,
):
    """
    Build a balanced pool of 10 UNIQUE passages.

    Desired composition:
        2 BM25
        2 Dense
        2 Hybrid RRF
        2 Hybrid + reranker
        2 lower-ranked/hard candidates

    Because results overlap, selection is done with deduplication.
    """

    selected = {}
    selected_order = []

    def take(results, source, number, start=0):

        count = 0

        for result in results[start:]:

            pid = result["id"]

            if pid in selected:
                # Still record that another retriever found it
                if source not in selected[pid]["candidate_sources"]:
                    selected[pid]["candidate_sources"].append(
                        source
                    )

                continue

            passage = result["passage"]

            selected[pid] = {
                "passage": passage,
                "candidate_sources": [source],
                "original_rank": result.get("rank"),
                "rrf_score": result.get("rrf_score"),
                "rerank_score": result.get("rerank_score"),
            }

            selected_order.append(pid)

            count += 1

            if count >= number:
                break

    # Main retrieval systems
    take(
        bm25_results,
        source="bm25",
        number=2,
    )

    take(
        dense_results,
        source="dense",
        number=2,
    )

    take(
        hybrid_results,
        source="hybrid_rrf",
        number=2,
    )

    take(
        reranked_results,
        source="hybrid_reranker",
        number=2,
    )

    # Lower-ranked candidates.
    # IMPORTANT:
    # They are NOT automatically labelled irrelevant.
    # Humans still decide relevance.
    take(
        hybrid_results,
        source="lower_ranked_candidate",
        number=2,
        start=10,
    )

    # Overlap among systems may leave fewer than 10.
    # Fill from all remaining candidates.
    fallback_lists = [
        ("hybrid_reranker", reranked_results),
        ("hybrid_rrf", hybrid_results),
        ("dense", dense_results),
        ("bm25", bm25_results),
    ]

    for source, results in fallback_lists:

        if len(selected_order) >= CANDIDATES_PER_QUERY:
            break

        for result in results:

            if len(selected_order) >= CANDIDATES_PER_QUERY:
                break

            pid = result["id"]

            if pid in selected:
                continue

            selected[pid] = {
                "passage": result["passage"],
                "candidate_sources": [source],
                "original_rank": result.get("rank"),
                "rrf_score": result.get("rrf_score"),
                "rerank_score": result.get(
                    "rerank_score"
                ),
            }

            selected_order.append(pid)

    if len(selected_order) < CANDIDATES_PER_QUERY:
        raise RuntimeError(
            f"Could only obtain {len(selected_order)} "
            f"unique candidates."
        )

    return [
        selected[pid]
        for pid in selected_order[
            :CANDIDATES_PER_QUERY
        ]
    ]


# ============================================================
# CREATE MASTER DATASET
# ============================================================

def build_annotation_dataset(
    retriever,
    queries,
):
    rows = []

    record_number = 1

    for i, item in enumerate(queries, start=1):

        query_id = item["query_id"]
        query = item["query"]
        ticker = item["ticker"]
        category = item["category"]

        print(
            f"[{i:03d}/{len(queries)}] "
            f"{query_id} | {ticker} | {query}"
        )

        # ----------------------------------------------------
        # BM25
        # ----------------------------------------------------

        bm25_results = retriever.bm25_search(
            query=query,
            top_k=RETRIEVAL_POOL_K,
            ticker=ticker,
        )

        # ----------------------------------------------------
        # Dense
        # ----------------------------------------------------

        dense_results = retriever.dense_search(
            query=query,
            top_k=RETRIEVAL_POOL_K,
            ticker=ticker,
        )

        # ----------------------------------------------------
        # Hybrid RRF
        # ----------------------------------------------------

        hybrid_results = retriever.rrf(
            [
                ("bm25", bm25_results),
                ("dense", dense_results),
            ],
            k=retriever.cfg["retrieval"]["rrf_k"],
        )

        # Need enough candidates for the lower-ranked pool
        hybrid_results = hybrid_results[
            :RETRIEVAL_POOL_K
        ]

        # ----------------------------------------------------
        # Hybrid + reranker
        # ----------------------------------------------------

        reranked_results = retriever.retrieve(
            query=query,
            top_k=RETRIEVAL_POOL_K,
            ticker=ticker,
            rerank=True,
        )

        # ----------------------------------------------------
        # Select 10 unique candidates
        # ----------------------------------------------------

        candidates = select_candidates(
            bm25_results=bm25_results,
            dense_results=dense_results,
            hybrid_results=hybrid_results,
            reranked_results=reranked_results,
        )

        for candidate_index, candidate in enumerate(
            candidates,
            start=1,
        ):

            passage = candidate["passage"]

            rows.append({
                "record_id": f"r{record_number:04d}",
                "query_id": query_id,
                "query_category": category,
                "query": query,
                "ticker": ticker,

                "passage_id": passage["id"],
                "passage_text": passage["text"],

                "filing_date": passage.get(
                    "filing_date",
                    "",
                ),

                "form": passage.get(
                    "form",
                    "",
                ),

                "section": passage.get(
                    "section",
                    "",
                ),

                # Internal provenance
                "candidate_sources": "|".join(
                    candidate["candidate_sources"]
                ),

                "candidate_number": candidate_index,

                "rrf_score": (
                    candidate["rrf_score"]
                    if candidate["rrf_score"] is not None
                    else ""
                ),

                "rerank_score": (
                    candidate["rerank_score"]
                    if candidate["rerank_score"]
                    is not None
                    else ""
                ),

                # Annotation fields
                "primary_annotator": "",
                "primary_label": "",

                "secondary_annotator": "",
                "secondary_label": "",

                "final_label": "",

                "notes": "",
            })

            record_number += 1

    expected = (
        NUM_QUERIES
        * CANDIDATES_PER_QUERY
    )

    if len(rows) != expected:
        raise RuntimeError(
            f"Expected {expected} records, "
            f"generated {len(rows)}."
        )

    return rows


# ============================================================
# ASSIGN ANNOTATORS
# ============================================================

def assign_annotators(rows):
    """
    1,000 primary annotations:
      ~166-167 per member

    200 secondary annotations:
      ~33-34 per member

    Total:
      ~200 decisions/member
    """

    random.seed(SEED)

    indices = list(range(len(rows)))
    random.shuffle(indices)

    # --------------------------------------------------------
    # Primary assignment
    # --------------------------------------------------------

    for position, row_index in enumerate(indices):

        member = (
            position % NUM_MEMBERS
        ) + 1

        rows[row_index][
            "primary_annotator"
        ] = f"Member_{member}"

    # --------------------------------------------------------
    # Select 200 overlap records
    # --------------------------------------------------------

    overlap_indices = random.sample(
        indices,
        OVERLAP_RECORDS,
    )

    secondary_counts = defaultdict(int)

    for row_index in overlap_indices:

        primary = rows[row_index][
            "primary_annotator"
        ]

        primary_num = int(
            primary.split("_")[1]
        )

        # Rotate to a different member.
        # This keeps overlap assignments reasonably balanced.
        possible_members = [
            x
            for x in range(
                1,
                NUM_MEMBERS + 1,
            )
            if x != primary_num
        ]

        # Pick member with lowest current secondary workload
        secondary = min(
            possible_members,
            key=lambda x: secondary_counts[x],
        )

        rows[row_index][
            "secondary_annotator"
        ] = f"Member_{secondary}"

        secondary_counts[secondary] += 1

    return rows


# ============================================================
# WRITE OUTPUT FILES
# ============================================================

MASTER_FIELDS = [
    "record_id",
    "query_id",
    "query_category",
    "query",
    "ticker",
    "passage_id",
    "passage_text",
    "filing_date",
    "form",
    "section",
    "candidate_sources",
    "candidate_number",
    "rrf_score",
    "rerank_score",
    "primary_annotator",
    "primary_label",
    "secondary_annotator",
    "secondary_label",
    "final_label",
    "notes",
]


ANNOTATOR_FIELDS = [
    "record_id",
    "query_id",
    "query_category",
    "query",
    "ticker",
    "passage_id",
    "passage_text",
    "filing_date",
    "form",
    "section",
    "assignment_type",
    "label",
    "notes",
]


def write_master(rows):
    with open(
        MASTER_FILE,
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=MASTER_FIELDS,
        )

        writer.writeheader()
        writer.writerows(rows)


def write_member_files(rows):
    """
    Member files deliberately DO NOT contain retrieval scores.

    This reduces annotation bias.
    """

    for member_number in range(
        1,
        NUM_MEMBERS + 1,
    ):

        member_name = (
            f"Member_{member_number}"
        )

        tasks = []

        for row in rows:

            # Primary task
            if (
                row["primary_annotator"]
                == member_name
            ):

                tasks.append({
                    "record_id":
                        row["record_id"],

                    "query_id":
                        row["query_id"],

                    "query_category":
                        row[
                            "query_category"
                        ],

                    "query":
                        row["query"],

                    "ticker":
                        row["ticker"],

                    "passage_id":
                        row["passage_id"],

                    "passage_text":
                        row["passage_text"],

                    "filing_date":
                        row["filing_date"],

                    "form":
                        row["form"],

                    "section":
                        row["section"],

                    "assignment_type":
                        "primary",

                    "label": "",

                    "notes": "",
                })

            # Secondary / agreement task
            if (
                row["secondary_annotator"]
                == member_name
            ):

                tasks.append({
                    "record_id":
                        row["record_id"],

                    "query_id":
                        row["query_id"],

                    "query_category":
                        row[
                            "query_category"
                        ],

                    "query":
                        row["query"],

                    "ticker":
                        row["ticker"],

                    "passage_id":
                        row["passage_id"],

                    "passage_text":
                        row["passage_text"],

                    "filing_date":
                        row["filing_date"],

                    "form":
                        row["form"],

                    "section":
                        row["section"],

                    "assignment_type":
                        "secondary",

                    "label": "",

                    "notes": "",
                })

        # Randomize annotation order
        random.shuffle(tasks)

        output = (
            OUTPUT_DIR
            / f"member_{member_number}_annotation.csv"
        )

        with open(
            output,
            "w",
            newline="",
            encoding="utf-8-sig",
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=ANNOTATOR_FIELDS,
            )

            writer.writeheader()
            writer.writerows(tasks)

        primary_count = sum(
            x["assignment_type"]
            == "primary"
            for x in tasks
        )

        secondary_count = sum(
            x["assignment_type"]
            == "secondary"
            for x in tasks
        )

        print(
            f"Member {member_number}: "
            f"{primary_count} primary + "
            f"{secondary_count} secondary = "
            f"{len(tasks)} decisions"
        )


# ============================================================
# ANNOTATION GUIDE
# ============================================================

def write_annotation_guide():

    guide = """
QUANTRAG MANUAL RETRIEVAL ANNOTATION GUIDE
===========================================

Each record consists of:

    Query
    +
    One SEC filing passage


LABEL 2 — DIRECTLY RELEVANT
----------------------------

The passage contains the evidence needed to answer
the query.

Examples:

Query:
What risks does NVIDIA report regarding export restrictions?

Passage:
Discusses export licensing restrictions and their
impact on NVIDIA sales.

Label:
2


LABEL 1 — PARTIALLY RELEVANT
-----------------------------

The passage is clearly useful for answering the
query but does not provide the complete evidence.

Example:

Query:
What factors affected Microsoft's cloud revenue growth?

Passage:
Discusses Azure demand but does not give the main
revenue drivers requested by the question.

Label:
1


LABEL 0 — IRRELEVANT
--------------------

The passage does not materially help answer the
query.

Keyword overlap alone does NOT make a passage
relevant.

Example:

Query:
What factors affected Microsoft's cloud revenue growth?

Passage:
Discusses Microsoft gaming revenue.

Label:
0


NUMERICAL QUESTIONS
-------------------

Label 2:
Passage contains the values/evidence necessary to
perform the requested comparison/calculation.

Label 1:
Passage contains some but not all necessary values.

Label 0:
Passage does not provide useful quantitative
evidence.


IMPORTANT
---------

Do NOT use the retrieval rank or model score to
decide the label.

Judge only:

    QUERY
      vs
    PASSAGE


For overlap records, annotators must work
independently.

Do NOT discuss overlap labels before both people
finish.


ANNOTATION CALIBRATION
----------------------

Before the full annotation:

1. All team members independently label the same
   20 calibration records.

2. Compare labels.

3. Discuss disagreements.

4. Finalize interpretation of labels 0 / 1 / 2.

5. Start the main annotation only after calibration.


TARGET
------

Final dataset:
    1,000 unique query-passage records

Agreement subset:
    200 records labelled by a second annotator

Total annotation decisions:
    1,200

Inter-annotator agreement target:
    >= 80%
"""

    path = (
        OUTPUT_DIR
        / "ANNOTATION_GUIDE.txt"
    )

    path.write_text(
        guide.strip() + "\n",
        encoding="utf-8",
    )


# ============================================================
# MAIN
# ============================================================

def main():

    random.seed(SEED)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "Loading evaluation queries..."
    )

    queries = load_queries(
        QUERY_FILE
    )

    print(
        f"Loaded {len(queries)} queries."
    )

    print(
        "Loading retrieval models/indexes..."
    )

    retriever = HybridRetriever(
        use_reranker=True
    )

    print()
    print(
        "Creating pooled annotation set..."
    )

    rows = build_annotation_dataset(
        retriever,
        queries,
    )

    print()
    print(
        f"Unique annotation records: "
        f"{len(rows)}"
    )

    rows = assign_annotators(
        rows
    )

    write_master(rows)
    write_member_files(rows)
    write_annotation_guide()

    print()
    print("=" * 60)
    print(
        "ANNOTATION SET CREATED"
    )
    print("=" * 60)

    print(
        f"Unique records: "
        f"{len(rows)}"
    )

    overlap_count = sum(
        bool(
            r["secondary_annotator"]
        )
        for r in rows
    )

    print(
        f"Overlap records: "
        f"{overlap_count}"
    )

    print(
        f"Total annotation decisions: "
        f"{len(rows) + overlap_count}"
    )

    print()
    print(
        f"Master file:\n"
        f"{MASTER_FILE}"
    )

    print()
    print(
        "Annotator files:"
    )

    for i in range(
        1,
        NUM_MEMBERS + 1,
    ):

        print(
            OUTPUT_DIR
            / f"member_{i}_annotation.csv"
        )

    print()
    print(
        "IMPORTANT:"
    )
    print(
        "Run a 20-record calibration round "
        "before full annotation."
    )


if __name__ == "__main__":
    main()