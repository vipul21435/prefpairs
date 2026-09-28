# PrefPairs

[![CI](https://github.com/vipul21435/prefpairs/actions/workflows/ci.yml/badge.svg)](https://github.com/vipul21435/prefpairs/actions/workflows/ci.yml)

RLHF preference-data toolkit: collect pairwise judgments, catch low-quality
annotation before it reaches training, aggregate preferences with a sound
statistical model, and export training-ready DPO, KTO and reward-model datasets.

> Status: early development. The scaffold (CLI, tooling, CI) is in place and the
> pipeline is being built in the ordered slices described in [PLAN.md](PLAN.md).
> Sections marked *(planned)* describe work that is not implemented yet.

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

## Pipeline *(planned)*

| Stage | What it does | Method |
| --- | --- | --- |
| Schema | Prompts, candidate responses with provenance, pairwise and ranked judgments, rationales, annotator sessions | pydantic models over SQLite |
| Collect | Typer CLI and a server-rendered web UI for pairwise comparison, randomized left/right order, tie and skip, rationale field, injected repeat control pairs | FastAPI, Jinja2, HTMX |
| Audit | Position bias, length/verbosity bias, inter-annotator agreement, self-consistency, transitivity violations, spammer detection | exact binomial test, logistic regression, Cohen/Fleiss kappa, cycle detection, gold accuracy plus Dawid-Skene EM |
| Aggregate | Global ranking of models or responses with uncertainty | Bradley-Terry (MM algorithm, numpy only), Elo, bootstrap CIs, win-rate matrix |
| Export | Training-ready datasets with quality filters and a dataset card | DPO, KTO and reward-model JSONL, deterministic splits |
| Simulate | Synthetic annotators with known ground truth and injected noise/bias | used by the tests to prove recovery and detection |

## Quickstart

Requires [uv](https://docs.astral.sh/uv/) and Git.

```bash
git clone https://github.com/vipul21435/prefpairs.git
cd prefpairs
make install      # uv sync --frozen + pre-commit hooks
make check        # ruff, mypy --strict, pytest with branch coverage
uv run prefpairs info
```

## Development

| Command | Purpose |
| --- | --- |
| `make lint` | `ruff check` and `ruff format --check` |
| `make typecheck` | `mypy --strict` over `src/` |
| `make test` | `pytest -q` |
| `make cov` | tests with branch coverage, fails under 85% |
| `make demo` | end-to-end demo (grows as slices land) |

Python 3.12, `src/` layout, dependencies pinned in `uv.lock`. No GPU and no paid API
is needed: every command runs offline with deterministic defaults. An
OpenAI-compatible endpoint can optionally be configured through environment
variables (see `.env.example`) for generating candidate responses.

## License

MIT. See [LICENSE](LICENSE).
