# PrefPairs implementation plan

PrefPairs is an RLHF preference-data toolkit: it collects pairwise judgments,
detects low-quality annotation, aggregates preferences with a sound statistical
model, and exports training-ready DPO, KTO and reward-model datasets. This file is
the working plan. Slices are built in order; each one is a coherent feature
delivered as 3-4 real Conventional Commits, each commit green on
`make check` (ruff, mypy --strict, pytest with branch coverage >= 85%).

## Decisions

- **Fresh repository, not a fork.** A 10-minute search (`gh search repos` for
  Bradley-Terry, Dawid-Skene, pairwise/preference annotation, DPO export) found
  only research-paper code, single-file scripts or repositories with no license.
  None was a base worth building on, so the project starts from an MIT scaffold.
- **numpy is the only numeric dependency.** Bradley-Terry (MM), Elo, logistic
  regression (Newton/IRLS), kappa, Dawid-Skene EM, bootstrap and the exact
  binomial test are implemented directly on numpy and `math`. No scipy, pandas,
  scikit-learn or torch: the disk budget is tight and the point of the project is
  to show the statistics, not to call them.
- **The synthetic generator lands in slice 1, not last.** Every statistical slice
  after it proves itself on data with known ground truth (true strengths, known
  biased annotators), so the generator has to exist first. The spec lists it as
  feature (6); it is delivered early and extended when later slices need it.
- **Canonical pair orientation.** A judgment stores what the annotator saw
  (`left_response_id`, `right_response_id`, `choice` in left/right/tie/skip). All
  analysis derives a canonical label against the pair sorted by response id
  (`a_wins`, `b_wins`, `tie`), so position effects and content effects stay
  separable.
- **Ties** count as half a win for each side in Bradley-Terry and Elo, are a
  third category in kappa, and are excluded from DPO/RM export. Skips are
  recorded but ignored by every statistic.
- **Bradley-Terry identifiability.** The MLE exists only if the comparison graph
  is strongly connected (Hunter 2004). A small pseudo-count prior (virtual games
  against a reference item, default 0.1) guarantees existence; strengths are
  reported on a log scale centred at mean zero.
- **Correlated judgments.** Judgments on the same prompt are not independent, so
  bootstrap CIs resample prompts (cluster bootstrap) by default, with an
  option to resample individual judgments.
- **Multiple testing.** Per-annotator bias tests use Holm-Bonferroni correction
  before an annotator is flagged.
- **Determinism.** Every random choice (left/right order, control-pair
  injection, bootstrap, simulation, splits) goes through a seeded
  `numpy.random.Generator`; exports are byte-identical across runs with the same
  seed, and the tests assert it.
- **Wording.** Avoid the words listed in the confidentiality rules of the build
  run (for example, say "anomalous annotator", never the usual statistics word
  for an extreme point). Run the grep in the push checklist before every push.

## Package layout (target)

```
src/prefpairs/
  schema.py            pydantic models: Prompt, Response, Provenance, Annotator,
                       Session, PairwiseJudgment, RankedJudgment, GoldPair
  store.py             SQLite repository, migrations via PRAGMA user_version
  simulate.py          ground-truth generator and annotator archetypes
  aggregate/           bradley_terry.py, elo.py, bootstrap.py, winrate.py
  quality/             position.py, length.py, agreement.py, consistency.py,
                       transitivity.py, spam.py (gold + Dawid-Skene), report.py
  collect/             scheduler.py, web/ (FastAPI app, templates, static)
  export/              filters.py, formats.py (DPO/KTO/RM), split.py, card.py
  cli.py               Typer app wiring every stage
```

## Slices

### Slice 1: Data model, SQLite store and synthetic ground-truth generator [x] done

Goal: define the pydantic schema for prompts, candidate responses with model and provenance metadata, pairwise and ranked judgments, rationales and annotator sessions; persist it in SQLite with versioned migrations and JSONL import; and add a seeded simulator that produces responses with known latent quality and annotators with known noise and bias, so every later slice can be tested against ground truth.

Commits:
1. `feat(schema)`: pydantic v2 models with validation (non-empty text, distinct
   left/right ids, choice enum, rationale length cap, ranked judgment is a
   permutation), canonical-label helper, JSON round-trip tests.
2. `feat(store)`: `Store` over stdlib `sqlite3` with foreign keys, indexes,
   `PRAGMA user_version` migrations, typed insert/query methods, JSONL import of
   prompts and responses; tests on a temp database including migration idempotency.
3. `feat(simulate)`: latent log-strength per model, per-response quality and
   length, annotator archetypes (reliable, noisy, left-biased, length-biased,
   random spammer, adversarial), gold pairs and repeated control pairs;
   determinism and distribution sanity tests.
4. `feat(cli)`: `prefpairs init`, `import`, `simulate`, `stats` commands with
   CliRunner tests.

Acceptance: `prefpairs simulate --seed 0 --db x.db` then `prefpairs stats --db x.db`
prints counts; two runs with the same seed produce identical databases (row dump
compared in a test).

Delivered (decisions taken while building it):

- JSONL lines are tagged with `type`, not `kind`, because `Annotator.kind`
  (human or simulated) is a real field; a test guards that no record ever has a
  field named `type`.
- Integrity lives in SQLite as well as pydantic: composite foreign keys tie a
  judgment's responses to its prompt and its session to its annotator, a
  trigger requires a control to repeat an earlier judgment of the same annotator
  on the same pair, judgments are append-only, and the file carries
  `PRAGMA application_id` so foreign SQLite files are never migrated.
- Import sorts records into dependency order (a control after the judgment it
  repeats, even in chains) and is idempotent; a conflicting id aborts the batch.
- Per-model verbosity is constructed to have an exact correlation with the true
  strengths (`length_quality_corr`, default 0). With six models, freely drawn
  offsets correlated with strength by chance (seed 0 made reliable annotators
  look anti-length), which would have made slice 3's length-bias check flag
  honest annotators.
- A database holds at most one simulated ground truth (migration 2,
  `simulation_truth`); rewriting the same simulation is a no-op.
- Added `prefpairs dump` (canonical JSONL) and `summary.py` for `stats`, beyond
  the four planned commands, so round trips can be checked from the CLI.

### Slice 2: Aggregation with Bradley-Terry, Elo, bootstrap CIs and win-rate matrix [x] done

Goal: turn judgments into a ranking with honest uncertainty: fit Bradley-Terry with the MM algorithm in numpy (ties as half wins, pseudo-count prior for identifiability), compute Elo averaged over seeded permutations, attach cluster-bootstrap confidence intervals, and report empirical and model-implied win-rate matrices, proven on simulated data to recover the true ordering.

Commits:
1. `feat(aggregate)`: Bradley-Terry MM fit (Hunter 2004) with convergence
   tolerance, iteration cap, log-likelihood trace; tests that the score equations
   hold at the solution, log-likelihood is monotone, and a hand-worked 3-item
   example matches.
2. `feat(aggregate)`: Elo with configurable K and scale, averaged over seeded
   permutations of judgment order; tests for symmetry and order-invariance of
   the averaged result.
3. `feat(aggregate)`: cluster bootstrap (by prompt, or by judgment) for BT and Elo
   with percentile CIs, plus empirical and predicted win-rate matrices.
4. `feat(cli)`: `prefpairs rank` (text table and `--json`); recovery test on
   simulated data: Kendall tau against true strengths above a fixed threshold and
   true strengths inside the 95% CIs for most items.

Delivered (decisions taken while building it):

- The bootstrap refits Bradley-Terry for many replicates at once (batched MM
  over `(B, n, n)` count arrays, chunked to bound memory) and reports how many
  replicates hit the iteration cap instead of hiding it.
- Intervals are reported for ranks as well as values (percentile interval of
  each replicate's competition rank), so close items show overlapping ranks.
- `rank` scores the estimated order against the truth with Kendall's tau when
  the database holds a simulation truth or `--truth FILE` is given; on the
  seed-0 simulation both Bradley-Terry and Elo reach tau 1.0.
- Known limitation: response-level ranking uses dense per-replicate matrices
  and is slow for hundreds of responses; a sparse fit belongs in slice 8.

### Slice 3: Annotator bias and agreement checks [x] done

Goal: detect the first family of low-quality annotation: position bias with an exact two-sided binomial test, length/verbosity bias with a numpy logistic regression of choice on length difference, inter-annotator agreement with Cohen and Fleiss kappa, and self-consistency on repeated control pairs; each check returns a typed result with its statistic, p-value or interval, and the evidence counts.

Commits:
1. `feat(quality)`: exact binomial test in log space (two-sided, sums outcomes no
   more likely than the observed one) with Holm-Bonferroni across annotators;
   tests against hand-computed p-values and the symmetric edge cases.
2. `feat(quality)`: logistic regression via Newton/IRLS with a tiny ridge term,
   Wald z-test and separation detection; per-annotator and pooled length-bias
   coefficient; exactness test (single binary covariate MLE equals the 2x2 log
   odds ratio) and recovery of a planted coefficient.
3. `feat(quality)`: Cohen kappa (pairwise annotators on shared items) and Fleiss
   kappa (items with a fixed number of raters) over the canonical labels, with
   hand-constructed examples checked to 1e-12.
4. `feat(quality)`: self-consistency rate on control repeats (sides flipped) with
   a Wilson interval; synthetic test that the left-biased and length-biased
   archetypes are flagged and reliable annotators are not.

Delivered (decisions taken while building it):

- Position bias counts every pair kind (gold and control pairs are shown in a
  randomised order too) and reports a pooled test next to the per-annotator
  ones. At the default simulation size (about 55 decisive choices per
  annotator) the model-free binomial test finds the planted left-biased
  annotator in 7 of 20 seeds, because that archetype still follows quality;
  at 60 prompts x 8 pairs it finds it in 20 of 20
  (`scripts/detection_rates.py`).
- Length bias needed three changes before it was calibrated on simulated
  data with no length-driven annotators: (1) a consensus quality covariate
  (leave-one-annotator-out Bradley-Terry model strengths), because model pairs
  tie length to quality in any finite sample; (2) a second round that refits
  the consensus without first-round flags, because a length-driven annotator
  pulls verbose models up and makes honest annotators look anti-length; and
  (3) a unit-variance normal prior on the slopes, because careful annotators
  follow consensus quality so closely that their data are nearly separated.
  The decision uses a likelihood-ratio test, which survives separation; Wald
  intervals are reported with a `separated` flag. Only regular pairs count by
  default (gold pairs are a small shared set picked for large quality gaps,
  controls are duplicates).
- Self-consistency flags when the upper Wilson bound is below 0.5, the rate
  of a coin flipper: only a position habit makes a flipped repeat reverse the
  first answer systematically. It catches an annotator who mostly clicks left
  (sharpness 0.3, position bias 3.0) in 10 of 10 seeds with a 0.5 control
  rate, but not the default left_biased archetype, whose repeats still agree
  more often than not because it also follows quality.
- Added `prefpairs checks` (text and `--json`) and extended `make demo` to
  require that it flags exactly the planted left_biased and length_biased
  annotators of the bundled sample, plus `scripts/detection_rates.py` for the
  power table in the README. Flags are per check; the combined verdict is
  slice 4.

### Slice 4: Transitivity, spammer detection and the audit report

Goal: complete the quality audit with transitivity violations (cycle detection in each annotator's per-prompt preference graph, intransitive-triad rate against the random baseline), spammer detection from gold-pair accuracy and a Dawid-Skene EM over pairwise labels (with a spammer score from the estimated confusion matrices), and a combined per-annotator report with flags, reasons and evidence, exposed as `prefpairs audit`.

Commits:
1. `feat(quality)`: per-annotator, per-prompt preference digraph; strongly
   connected components (iterative Tarjan) and 3-cycle counting; tests on
   hand-built tournaments with known cycle counts.
2. `feat(quality)`: gold-pair accuracy with Wilson lower bound; Dawid-Skene EM
   (majority-vote init, per-annotator confusion matrices, item posteriors,
   log-likelihood monotonicity) and a spammer score; tests that EM recovers
   planted annotator accuracies on simulated data.
3. `feat(quality)`: `AuditReport` combining every check into per-annotator
   flags with thresholds from a config model; JSON and plain-text renderers.
4. `feat(cli)`: `prefpairs audit` (exit code 1 when any annotator is flagged
   under `--strict`); end-to-end test on simulated data asserting exactly the
   planted bad annotators are flagged for a fixed seed.

### Slice 5: Collection via scheduler, CLI annotation and web UI

Goal: collect real judgments: a seeded scheduler that balances pair coverage, randomizes left/right order, injects repeated control pairs with sides flipped and gold pairs at configurable rates, tracked in annotator sessions; an interactive Typer `annotate` command; and a minimal server-rendered FastAPI + Jinja2 + HTMX UI with tie, skip and a rationale field.

Commits:
1. `feat(collect)`: pair scheduler (coverage balancing, seeded side
   randomization, control and gold injection, no immediate repeats) and session
   lifecycle in the store; tests for balance, flip-on-repeat and determinism.
2. `feat(cli)`: `prefpairs annotate` terminal loop (1/2/t/s keys, optional
   rationale) writing judgments with latency; CliRunner tests with scripted input.
3. `feat(web)`: FastAPI app with Jinja2 autoescaped templates, vendored HTMX
   (with its license), session start, compare card, submit and progress
   partials; `prefpairs serve`; TestClient tests for every route, including
   validation errors and a full session.
4. `test(collect)`: round-trip test that judgments collected through the web
   flow feed `audit` and `rank` unchanged.

### Slice 6: Export to DPO, KTO and reward-model datasets with a dataset card

Goal: produce training-ready data: aggregate votes per pair from unflagged annotators, apply filters (minimum votes, minimum agreement, exclude flagged annotators, drop ties), write DPO (prompt/chosen/rejected), KTO (desirable/undesirable) and reward-model JSONL with deterministic prompt-level train/validation/test splits, and emit a dataset card with provenance, filter settings, quality statistics and file checksums.

Commits:
1. `feat(export)`: vote aggregation and filter pipeline returning an auditable
   list of kept and dropped pairs with drop reasons.
2. `feat(export)`: DPO, KTO and reward-model writers (sorted keys, stable record
   order, newline-terminated) and hash-based prompt-level splits; tests for no
   prompt leakage across splits and byte-identical output on rerun.
3. `feat(export)`: dataset card (Markdown plus JSON) with source provenance,
   annotator and quality summary, ranking snapshot and sha256 of every file.
4. `feat(cli)`: `prefpairs export --format dpo|kto|rm`; golden-file tests on a
   small fixed simulated dataset.

### Slice 7: Docker, compose and end-to-end `make demo` [~] partly done

Goal: make the whole pipeline runnable in one command: a slim multi-stage Dockerfile on a digest-pinned Python 3.12 base (non-root user, labelled `project=prefpairs`), a compose file with the web UI and a one-shot pipeline service, and a `make demo` that runs simulate, audit, rank and export end to end and checks its outputs, verified from a fresh clone.

Commits:
1. `build(docker)`: multi-stage Dockerfile (uv export of the lockfile, wheel
   install, non-root, healthcheck), `.dockerignore`, image label for cleanup.
2. `build(compose)`: `web` and `pipeline` services sharing a volume; documented
   ports and environment.
3. `feat(demo)`: `scripts/demo.sh` plus `make demo` and `make docker-demo` that
   run the full pipeline on a seeded dataset and assert expected artifacts exist.
4. `ci`: CI job that builds the image and runs the containerised demo.

Delivered early (to make the repository runnable before the audit and export
slices): the digest-pinned multi-stage Dockerfile with uv and a non-root user,
`.dockerignore`, `make docker`, `scripts/demo.sh` with `make demo` and
`make docker-demo` on a bundled seed-0 sample, and a CI job that builds the
image and runs the demo in it. Still open: the compose file (commit 2), and
extending the demo to audit and export once those slices exist.

### Slice 8: Benchmarks and documentation polish

Goal: publish measured evidence and reviewer-ready docs: benchmark scripts for ranking recovery versus judgment budget, bootstrap CI coverage, audit detection power and false-flag rate across seeds, and Bradley-Terry fit time; a methods document explaining each statistic; and a final README with a 5-command quickstart and only numbers produced by the committed commands.

Commits:
1. `perf(bench)`: `benchmarks/` scripts writing JSON results (recovery, CI
   coverage, detection power and false-flag rate, fit time) with fixed seeds.
2. `docs`: `docs/methods.md` (models, estimators, tests, assumptions, references)
   and `docs/architecture.md`.
3. `docs(readme)`: final README with results tables, each with its reproduce
   command, CLI reference and dataset-card example.
4. `test`: fresh-clone check of the README quickstart recorded in CI.

## Push checklist

1. `make check` is green.
2. `git grep -niE "<banned-name pattern from the build rules>"` returns nothing.
3. ASCII only: `LC_ALL=C grep -rnP '[^\x00-\x7F]' --include='*.py' --include='*.md' .`
   returns nothing outside vendored files.
4. Every number in the README comes from a committed command.
