# PrefPairs

[![CI](https://github.com/vipul21435/prefpairs/actions/workflows/ci.yml/badge.svg)](https://github.com/vipul21435/prefpairs/actions/workflows/ci.yml)

RLHF preference-data toolkit: collect pairwise judgments, catch low-quality
annotation before it reaches training, aggregate preferences with a sound
statistical model, and export training-ready DPO, KTO and reward-model datasets.

> Status: slice 1 of 8 is done (data model, SQLite store, JSONL import and the
> ground-truth simulator). The remaining stages are built in the ordered slices
> described in [PLAN.md](PLAN.md); rows marked *planned* below are not
> implemented yet.

## Why this exists

Preference data is only as good as the people and processes that produce it. In
practice the expensive failures are quiet ones: an annotator who always picks the
left response, a reviewer who rewards length rather than quality, a pool whose
judgments contradict themselves across repeated pairs, or a "winner" whose lead
disappears once you put a confidence interval on it. PrefPairs treats each of those
as a testable hypothesis with a named statistic, so a dataset ships with evidence
about its own quality instead of a raw vote count.

The design comes from hands-on post-training data work: running review workflows
and rubric-driven evaluation for LLM datasets (RLHF and SFT) at scale, and building
verifiable benchmark tasks where every grader is deterministic and every number is
reproducible.

## Pipeline

| Stage | What it does | Method | Status |
| --- | --- | --- | --- |
| Schema | Prompts, candidate responses with provenance, pairwise and ranked judgments, rationales, annotator sessions, gold pairs | pydantic v2 models over SQLite with versioned migrations | done |
| Simulate | Synthetic data with known model strengths and annotators with planted noise and bias | seeded generative model, analytic choice probabilities | done |
| Collect | Typer CLI and a server-rendered web UI for pairwise comparison, randomized left/right order, tie and skip, rationale field, injected repeat control pairs | FastAPI, Jinja2, HTMX | planned |
| Audit | Position bias, length/verbosity bias, inter-annotator agreement, self-consistency, transitivity violations, spammer detection | exact binomial test, logistic regression, Cohen/Fleiss kappa, cycle detection, gold accuracy plus Dawid-Skene EM | planned |
| Aggregate | Global ranking of models or responses with uncertainty | Bradley-Terry (MM algorithm, numpy only), Elo, bootstrap CIs, win-rate matrix | planned |
| Export | Training-ready datasets with quality filters and a dataset card | DPO, KTO and reward-model JSONL, deterministic splits | planned |

## Quickstart

Requires [uv](https://docs.astral.sh/uv/) and Git.

```bash
git clone https://github.com/vipul21435/prefpairs.git && cd prefpairs
make install                                           # uv sync --frozen + git hooks
make check                                             # ruff, mypy --strict, pytest + coverage
uv run prefpairs simulate --seed 0 --db .prefpairs/demo.db
uv run prefpairs stats --db .prefpairs/demo.db
```

`make demo` runs the last two commands. Output of the simulate step, copied from
a real run:

```text
wrote 1012 new records to .prefpairs/demo.db (seed 0)
true model order: model-a > model-f > model-b > model-e > model-d > model-c
planted defective annotators: ann-06 (adversarial), ann-08 (random_spammer), ann-10 (left_biased), ann-11 (length_biased)
```

and an excerpt of `stats`:

```text
Database .prefpairs/demo.db (schema v2, simulated, seed 0)

Records
  prompt                  40
  response               240
  annotator               12
  session                 12
  gold                    12
  pairwise               672
  ranked                  24

Pair kinds
  regular                480
  control                 48
  gold                   144
```

Every later stage is judged against that ground truth: the ranking must recover
the true model order and the audit must flag exactly the planted annotators.

## Data model

All records are immutable pydantic models (`src/prefpairs/schema.py`) and are
validated again by SQLite constraints when stored (`src/prefpairs/store.py`).

| Record | Key fields | Rules enforced |
| --- | --- | --- |
| `Prompt` | `id`, `text`, `category`, `tags` | slug ids, non-blank text |
| `Response` | `prompt_id`, `model`, `text`, `provenance` (source, generator, decoding params, timestamp, origin) | response belongs to an existing prompt |
| `Annotator`, `Session` | `kind`, `client`, `started_at`, `ended_at` | UTC-aware timestamps, session cannot end before it starts |
| `PairwiseJudgment` | displayed `left_response_id` / `right_response_id`, `choice` (left, right, tie, skip), `pair_kind` (regular, control, gold), `repeat_of`, `rationale`, `latency_ms` | both responses belong to the judgment's prompt, the session belongs to the annotator, a control must repeat an earlier judgment of the same annotator on the same pair, rationale capped at 2000 chars, judgments are append-only |
| `RankedJudgment` | `shown` order and `ranking` (best first) | the ranking is a permutation of what was shown |
| `GoldPair` | `better_response_id`, `worse_response_id`, `reason` | unique per pair in either orientation |

A judgment records what the annotator saw. Analysis uses its canonical form on
the pair sorted by response id (`a_wins`, `b_wins`, `tie`), which keeps position
effects separable from content effects.

The schema version lives in `PRAGMA user_version`; each migration runs in one
transaction with its version bump, so a failed upgrade leaves the file untouched.
Opening a database written by a newer version, or a SQLite file that is not a
PrefPairs store, is refused.

### JSONL import and export

One JSON object per line, tagged with `type`. A prompt may carry its responses
inline, the usual shape of a generation dump. Two lines from
[`examples/tiny.jsonl`](examples/tiny.jsonl), shortened:

```json
{"type":"prompt","id":"p2","text":"Give one tip for writing a clear commit message.","responses":[{"id":"p2-a","model":"model-a","text":"Start with a short imperative summary line.","provenance":{"source":"model"}},{"id":"p2-b","model":"model-b","text":"Write good messages.","provenance":{"source":"model"}}]}
{"type":"pairwise","id":"j2","session_id":"ann-1-s1","annotator_id":"ann-1","prompt_id":"p2","left_response_id":"p2-a","right_response_id":"p2-b","choice":"left","pair_kind":"gold","created_at":"2026-03-02T09:00:30Z"}
```

```bash
uv run prefpairs import examples/tiny.jsonl --db .prefpairs/tiny.db
# added prompt 2, response 5, annotator 2, session 2, gold 1, pairwise 5, ranked 1; 0 unchanged
uv run prefpairs import examples/tiny.jsonl --db .prefpairs/tiny.db
# added nothing; 18 unchanged
uv run prefpairs dump --db .prefpairs/tiny.db --out .prefpairs/tiny-copy.jsonl
```

Import is atomic and idempotent: records are inserted in dependency order,
unchanged records are skipped, and a record that conflicts with stored data or
breaks a reference aborts the whole file with the line number or record id.
`dump` writes canonical JSONL (sorted keys, ASCII) that re-imports into an
identical database.

## Synthetic ground truth

`prefpairs.simulate` generates data whose truth is known, so each statistic can
be tested for what it is supposed to find:

- Each model has a latent log-strength (evenly spaced, randomly assigned to
  names); a response's quality is its model's strength plus noise.
- Response length is log-normal with a per-model verbosity offset whose
  correlation with strength is set exactly (0 by default), so a quality
  preference cannot pass for a length preference by chance.
- An annotator skips, ties (more often on close pairs), or picks left with
  probability `sigmoid(beta * quality_gap + position_bias + length_weight * log_length_ratio)`.
  `AnnotatorProfile.choice_probabilities` returns this distribution exactly,
  and the tests check every simulated annotator's left-choice count against it.
- Work is scheduled like a real project: each pair is judged by several
  annotators with balanced load, everyone sees the gold pairs, and a share of
  each annotator's pairs comes back later as a control with the sides flipped.

| Archetype | beta | position bias | length weight | Should be flagged |
| --- | --- | --- | --- | --- |
| reliable | 3.0 | 0 | 0 | no |
| noisy | 0.9 | 0 | 0 | no (down-weighted, not flagged) |
| left_biased | 2.0 | +2.0 | 0 | yes |
| length_biased | 1.0 | 0 | 3.0 | yes |
| random_spammer | 0 | 0 | 0 | yes |
| adversarial | -2.0 | 0 | 0 | yes |

The same seed always yields the same database (checked row by row in
`tests/test_cli.py`), and independent random streams keep prompts, responses
and strengths fixed when only the annotator roster changes.

## CLI

| Command | Purpose |
| --- | --- |
| `prefpairs init --db PATH` | create a database or migrate it to the current schema |
| `prefpairs import FILE --db PATH` | atomic, idempotent JSONL import |
| `prefpairs simulate --seed N --db PATH [--roster reliable=6,left_biased=1] [--truth-out truth.json]` | write a simulated dataset and its ground truth |
| `prefpairs stats --db PATH [--json]` | record counts, choice and pair-kind mix, per-model responses, annotator load |
| `prefpairs dump --db PATH [--out FILE]` | canonical JSONL of every record |

`--db` defaults to `.prefpairs/prefpairs.db` and can be set with `PREFPAIRS_DB`.

## Development

| Command | Purpose |
| --- | --- |
| `make lint` | `ruff check` and `ruff format --check` |
| `make typecheck` | `mypy --strict` over `src/` |
| `make test` | `pytest -q` |
| `make cov` | tests with branch coverage, fails under 85% |
| `make demo` | simulate a seeded dataset and summarise it |

At the end of slice 1, `make cov` reports 200 tests passing with 99.85% branch
coverage.

Python 3.12, `src/` layout, dependencies pinned in `uv.lock`. No GPU and no paid API
is needed: every command runs offline with deterministic defaults. An
OpenAI-compatible endpoint can optionally be configured through environment
variables (see `.env.example`) for generating candidate responses.

## License

MIT. See [LICENSE](LICENSE).
