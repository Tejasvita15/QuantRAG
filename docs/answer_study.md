# Task 8: frozen-evidence answer study

This tool completes the **answer-experiment workflow**, not the human evaluation
itself. It cannot manufacture 1,000 held-out records, reference answers, annotator
agreement or final report results. No real LLM, live-data or retrieval request is
made by default. Unit-test fixtures are synthetic software tests, not project
experimental evidence.

## What is compared

`src.answer_study` reads the completed `results.json` from `src.run_ablation`.
It compares the same four retrieval variants: `bm25`, `dense`,
`hybrid_no_rerank`, and `hybrid` (hybrid plus cross-encoder reranking). It uses
the saved full passages, ranks and per-query filters; it does **not** run search
again. `--repeat 1` selects the first measured repeat by default. All four
variants must be present for every query in that repeat, otherwise validation
fails. Keep the query set, filters, top-k, LLM model/provider and context budgets
fixed across variants. Selecting another repeat is an explicit experiment change.
Generation-call order is shuffled rather than always running BM25 first. Use
`--seed 42` to reproduce that schedule; if omitted, a random seed is generated
and saved. This mitigates systematic first-call/cold-start timing bias but does
not control provider load, network variance or stochastic LLM output. It is not
an LLM sampling seed.

Only `--generate` enables the existing `GroundedGenerator`. It sends frozen
evidence to the provider configured by `LLM_MODEL`, `LLM_API_KEY` (or
`OPENAI_API_KEY`) and optional `LLM_BASE_URL` in the local `.env`. Never commit
`.env`, API keys, provider response bodies or private financial data. Generating
four variants of N questions can require up to 4N provider calls; confirm the
budget first. This tool never fetches Yahoo/FRED data itself.

## 1. Validate without costs or writes

Run from the repository root in the activated Python environment. Replace the
example paths with your actual run directory:

```bash
python -m src.answer_study prepare --results experiments/run-001/results.json --check-only
```

This checks schemas, complete four-way runs, full text, IDs, ranks and context
budgets. It does not contact a provider, download models or create files.

## 2. Optional reference answers and frozen live observations

Reference answers are a **separate answer-level gold set**, not retrieval qrels.
One UTF-8 JSONL line per query, using the original query ID:

```json
{"id":"q1","reference_answer":"Human-written, evidence-checked reference answer.","held_out":true}
```

Only assert `held_out: true` after the team has verified that the question was
not used to choose prompts, tune parameters or develop the system. The script
records that declaration; it cannot independently establish it. Unknown or
duplicate IDs are rejected. Missing references remain empty and do not count
toward held-out coverage. Do not fill empty references with generated answers
and call them human gold labels.

For questions combining SEC evidence with market/macro data, save one live
snapshot **once per query**, then reuse it unchanged for all four variants.
The optional JSON file has this form:

```json
{
  "q1": [
    {
      "source_id": "STOCK1",
      "kind": "stock",
      "content": "{\"price\":100,\"latest_trading_date\":\"2026-10-20\"}",
      "metadata": {
        "ticker": "AAPL",
        "trading_date": "2026-10-20",
        "units": "USD",
        "price_basis": "latest available adjusted daily bar close, not real time"
      }
    }
  ]
}
```

The example numbers above illustrate the input format and are **not observed
market values**. Replace them with genuine saved tool results before any real
experiment. Macro sources use `kind: "macro"`, `FRED1` etc., and metadata field
`latest_observation_date`. Stock sources require `trading_date`. Valid ISO
calendar dates and timezone-aware ISO datetimes are accepted, including the
actual pipeline shape `2026-10-15T00:00:00-04:00` and UTC `...Z`. Invalid calendar
dates and datetimes without a timezone are rejected. Preserve the original
observation timestamp; do not convert a market date into the current date. Both
use dated observations, not merely a retrieval timestamp. Retain the provider, true units,
series/ticker and price basis where available. Missing snapshot keys mean no live
evidence for those questions; the script never silently fills gaps with a new
network request. SEC-only evaluation is valid but must not be described as a
full mixed-route/live-data evaluation.

## 3. Prepare, then explicitly generate

To inspect frozen contexts and review-sheet layout **without generating answers**:

```bash
python -m src.answer_study prepare --results experiments/run-001/results.json --references eval/answer_references.jsonl --output experiments/answer-prepared-001
```

For real answer generation, explicitly opt in and choose a **new** directory:

```bash
python -m src.answer_study prepare --results experiments/run-001/results.json --references eval/answer_references.jsonl --output experiments/answer-generated-001 --generate
```

Add `--live-sources eval/frozen_live_sources.json` to either command if needed.
Use exactly the same snapshot file across comparisons. Optional flags
`--max-context-chars 40000 --max-source-chars 6000` control evidence budgets.
The existing generator prioritizes compact live observations, truncates SEC
passages at the per-source limit, and marks truncated/omitted evidence explicitly.
On successful generation, `sources` records the bounded context returned by the
generator, which may be smaller than the original retrieved ranking. With no
usable sources, abstention occurs without a provider request. For prepared-only
or failed-generation cases, it records only **planned candidate context**;
successful transmission to the provider is not established. The per-case
`context_status` makes that distinction explicit. Hold both budgets fixed and
report them; do not describe a 6,000-character excerpt as an entire filing.

Existing output directories are always rejected. Original results are never
overwritten. To retry after a failed provider run, use a new output directory;
this small tool does not yet resume successful cases or avoid repeat charges.

Generated output:

- `answers.json`: private case records with query ID, variant, selected repeat,
  filters, bounded sources, answer/status, claims, actual citations, limitations,
  provider-reported model/token usage, and generation latency.
- `private_mapping.json`: opaque `answer_id` to query ID/variant mapping.
- `blind_review.csv`: shuffled answers with opaque IDs, query, reference answer,
  bounded context, **actual cited sources**, and blank annotation columns.

Give only `blind_review.csv` to annotators. The private files reveal variants.
The public worksheet removes retrieval provenance/score metadata that would
directly reveal the method; passage order/content can still offer indirect clues,
so blinding is practical rather than a guarantee of perfect concealment.

The `answers.json` manifest freezes SHA-256 digests of the exact input bytes for
retrieval results, optional references and optional live snapshots (absent inputs
are `null`), without saving their paths or credentials. It also records the
`GroundedGenerator` system-prompt SHA-256, context budgets, shuffle seed and full
scheduled order keyed by answer/query/variant/repeat. Each order entry marks
whether generation was attempted. This is an execution schedule, **not** an
exact provider-call count: an evidence-free abstention or configuration failure
can finish without an HTTP request. Keep the original input files to verify
digests later. The prompt hash identifies prompt content; it does not freeze an
external model version, provider behavior or the generator's entire source code.

Provider failures produce `generation_error` with a generic diagnostic, not raw
exception bodies. Their context is marked `planned_context_send_unconfirmed`,
and usage remains unknown rather than assuming zero or inventing billed tokens.
`insufficient_evidence` is a genuine abstention, not a transport
failure. `not_generated` and `generation_error` have no answer to rate and are
rejected by aggregation. Inspect these statuses and rerun/fix the underlying
problem before claiming complete evaluation. Prepared-only sheets have blank
answers and cannot be scored.

## 4. Real, independent human annotation

Ask at least two distinct annotators to fill their own copy independently
(three are preferable if the team can manage it). Enter a stable `annotator_id`
on every filled row. Combine their rows into one CSV with the same columns;
there must be one row per `(answer_id, annotator_id)`, without duplicates.
Untouched blank template rows are ignored; a partially filled scoring row is an
error, not a zero score. Every completed row needs all four integer scores 1–5:

| Column | What the annotator judges |
| --- | --- |
| `faithfulness` | Do the answer's factual claims follow from the supplied evidence, including dates, units and calculations? Read the evidence; a valid source ID is not proof. |
| `relevance` | Does the answer directly address all parts of this question, without unnecessary material? |
| `context_precision` | How much of the supplied bounded context is useful/relevant for answering this question? Judge **all supplied context**, not only cited snippets. |
| `quality` | Overall correctness, completeness and clarity, checked against a human reference when available. |

Agree on a written rubric first: 1 = serious failure, 2 = substantial issues,
3 = mixed/partially satisfactory, 4 = minor issues, 5 = fully satisfactory.
These are **human Likert ratings**, including a context-precision proxy; they are
not canonical automatic precision ratios, RAGAS scores, probabilities, or
percentages. Report their scale and rubric explicitly. In particular, do not
divide a 1–5 score by 5 and claim factual accuracy or a precision percentage.

For legitimate abstentions, establish the rubric before annotation (e.g. judge
whether the refusal is justified and addresses the missing evidence). State this
policy and report abstention coverage separately in the report; averaging these
ratings with substantive answers without disclosure can be misleading. The tool
does not silently assign an abstention a perfect faithfulness score. If a
dimension cannot be scored under the agreed rubric, revise the protocol or
analyze that subset separately; do not invent a numeric label just to fill CSV.

Use the `notes` column to flag unsupported claims, wrong dates/units, irrelevant
context, justified refusals and missing evidence. Keep independently entered
scores before any consensus discussion so agreement is measured honestly.

```bash
python -m src.answer_study aggregate --study experiments/answer-generated-001 --ratings eval/combined_answer_ratings.csv --output experiments/answer-metrics-001
```

`human_metrics.json` contains:

- Means per answer across its annotators, then per variant across answers, so a
  heavily annotated answer does not automatically receive greater weight.
- **Observed exact agreement**: for each answer and dimension, compare every
  unordered annotator pair with both labels present. The denominator counts
  those actual pair–answer–dimension comparisons; the numerator counts exactly
  equal scores. It reports per-dimension and pooled overall values. For two
  annotators, four answers and four dimensions, the maximum denominator is 16.
  This is not Cohen/Fleiss kappa, Krippendorff alpha or a chance-corrected score.
- Unique self-declared held-out question counts, deduplicating normalized
  question text. Four variants and several annotators do **not** multiply one
  question into several independent held-out records.
- A stricter coverage check counting questions with **all four** variants rated
  by at least two annotators. This is this tool's completeness diagnostic, not
  an additional requirement quoted from the assignment.
- Checks for at least 1,000 self-declared held-out questions and observed exact
  agreement of at least 80%. Passing software checks alone does not certify the
  data's independence or compliance with the course's annotation protocol.
- Generation latency and available provider-reported token totals. Missing
  usage is unknown (`null`), not zero. No dollar cost is inferred without an
  independently verified provider/model/date pricing table.

The script requires at least two distinct annotators overall but preserves and
reports incomplete per-answer coverage; do not present a partially double-rated
study as if every answer had two ratings. For a fair final comparison, use the
same fully rated questions for all four variants and report failures/abstentions.

## Report method and limitations (adapt after real experiments)

> We compared BM25, dense retrieval, reciprocal-rank-fusion hybrid retrieval,
> and hybrid retrieval followed by cross-encoder reranking. The query set,
> filters, selected top-k, evidence budgets and generation model were held fixed.
> Answer generation reused saved, ranked passage snapshots rather than executing
> retrieval again, and calling order was randomized with a recorded schedule
> seed. Input-byte hashes and the system-prompt hash identify the frozen inputs
> and prompt. Where market/macro observations were required, the same dated
> observation snapshot was reused across all four variants. Independent human
> annotators, blinded to explicit variant labels, scored faithfulness, relevance,
> a context-precision proxy and overall quality on a predefined 1–5 scale. Scores
> were averaged within answers and then across answers. We report observed exact
> pairwise label agreement with its denominator, annotation coverage, abstention
> and failure counts, generation latency and available token usage. Citation-ID
> validation establishes link integrity only, not semantic faithfulness.

Do not paste numerical conclusions until genuine results exist. Record the
actual corpus/index versions, query/reference set provenance, held-out split,
model returned by the provider, snapshot dates, top-k/budgets, repeat selection,
annotator protocol and incomplete coverage. This design does not control all
LLM nondeterminism or measure end-to-end live retrieval latency; retrieval
latency comes from the retrieval study and answer generation latency from this
study. Mixed-route robustness, routing accuracy, future-market predictions,
financial decision quality, and statistical significance require separate
experiments and must not be claimed from these ratings alone.

## Offline software verification

```bash
python -m unittest tests.test_answer_study -v
```

Tests inject fake completions and use temporary synthetic records. They cover
offline defaults, explicit generation, frozen live snapshots, complete four-way
joins, context budgets, blinding, safe provider failures, overwrite protection,
manual-score validation, agreement denominators, unique-question counts, genuine
pipeline ISO datetime shapes, invalid calendar dates, input/prompt hashes and
seeded execution order.
