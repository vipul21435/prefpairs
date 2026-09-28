"""Synthetic preference data with known ground truth.

Every statistical claim PrefPairs makes (a ranking, a confidence interval, a
flagged annotator) is tested against data whose truth is known. This module
generates that data.

Generative model
----------------

* Each of ``n_models`` systems has a latent log-strength ``theta_m``, evenly
  spaced on ``[-strength_spread, strength_spread]`` and randomly assigned to
  model names, so the true ordering is never alphabetical by construction.
* Each model answers every prompt once. Response quality is
  ``q_r = theta_m + N(0, response_sd)``. Response length in words is
  log-normal with a per-model verbosity offset. The offsets are constructed so
  that their sample correlation with the true strengths is exactly
  ``length_quality_corr`` (default 0): with only a handful of models, freely
  drawn offsets would correlate with strength by chance, and a quality
  preference would then look like a length preference.
* An annotator shown ``left`` and ``right`` skips with probability ``s``, calls
  a tie with probability ``t * exp(-|q_left - q_right|)``, and otherwise picks
  left with probability ``sigmoid(beta * (q_left - q_right) + b_pos +
  w_len * log(words_left / words_right))``. The archetypes below are settings
  of ``(beta, b_pos, w_len, t, s)``; ``choice_probabilities`` evaluates the
  exact distribution so tests can compare empirical rates with it.
* Rankings follow a Plackett-Luce model with the same utilities (Gumbel-max
  sampling).

Work is scheduled the way a real project would be: each prompt contributes
``pairs_per_prompt`` pairs, each pair is judged by ``redundancy`` annotators
(balanced load), every annotator sees every gold pair, and a ``control_rate``
share of each annotator's pairs is repeated later with the sides flipped.

Randomness is split into independent streams (``numpy.random.SeedSequence``),
so for a fixed seed, changing the annotator roster does not change the
prompts, responses or true strengths.
"""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from itertools import combinations
from typing import Self

import numpy as np
from pydantic import Field, model_validator

from prefpairs.schema import (
    Annotator,
    AnnotatorKind,
    AnyRecord,
    Choice,
    GoldPair,
    PairKind,
    PairwiseJudgment,
    Prompt,
    Provenance,
    RankedJudgment,
    Record,
    Response,
    ResponseSource,
    Session,
    SessionClient,
    UtcDatetime,
)
from prefpairs.store import Store

GENERATOR_NAME = "prefpairs-simulator"
SESSION_BREAK = timedelta(minutes=30)
LATENCY_LOG_SD = 0.35
MIN_LATENCY_MS = 300


class SimulationError(ValueError):
    """The requested simulation cannot be generated (for example, too few gold pairs)."""


class Archetype(StrEnum):
    """Behaviour patterns of simulated annotators."""

    RELIABLE = "reliable"
    NOISY = "noisy"
    LEFT_BIASED = "left_biased"
    LENGTH_BIASED = "length_biased"
    RANDOM_SPAMMER = "random_spammer"
    ADVERSARIAL = "adversarial"

    @property
    def is_defective(self) -> bool:
        """True if a quality audit should flag this archetype.

        Noisy annotators are honest but less informative; they should be
        down-weighted by aggregation, not flagged.
        """
        return self not in (Archetype.RELIABLE, Archetype.NOISY)


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    z = math.exp(x)
    return z / (1.0 + z)


class AnnotatorProfile(Record):
    """Parameters of one annotator's choice model (see the module docstring)."""

    archetype: Archetype
    sharpness: float = Field(allow_inf_nan=False)
    position_bias: float = Field(default=0.0, allow_inf_nan=False)
    length_weight: float = Field(default=0.0, allow_inf_nan=False)
    tie_rate: float = Field(default=0.0, ge=0, le=1)
    skip_rate: float = Field(default=0.0, ge=0, le=1)
    rationale_rate: float = Field(default=0.0, ge=0, le=1)
    median_latency_ms: float = Field(gt=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def _rates_fit(self) -> Self:
        if self.tie_rate + self.skip_rate > 1:
            msg = "tie_rate + skip_rate must not exceed 1"
            raise ValueError(msg)
        return self

    def choice_probabilities(
        self, quality_gap: float, log_length_ratio: float
    ) -> dict[Choice, float]:
        """Exact outcome distribution for one displayed pair.

        ``quality_gap`` is ``q_left - q_right`` and ``log_length_ratio`` is
        ``log(words_left / words_right)``.
        """
        p_skip = self.skip_rate
        p_tie = self.tie_rate * math.exp(-abs(quality_gap))
        decisive = 1.0 - p_skip - p_tie
        logit = (
            self.sharpness * quality_gap
            + self.position_bias
            + self.length_weight * log_length_ratio
        )
        p_left = _sigmoid(logit)
        return {
            Choice.LEFT: decisive * p_left,
            Choice.RIGHT: decisive * (1.0 - p_left),
            Choice.TIE: p_tie,
            Choice.SKIP: p_skip,
        }


DEFAULT_PROFILES: Mapping[Archetype, AnnotatorProfile] = {
    Archetype.RELIABLE: AnnotatorProfile(
        archetype=Archetype.RELIABLE,
        sharpness=3.0,
        tie_rate=0.08,
        skip_rate=0.01,
        rationale_rate=0.5,
        median_latency_ms=24_000,
    ),
    Archetype.NOISY: AnnotatorProfile(
        archetype=Archetype.NOISY,
        sharpness=0.9,
        tie_rate=0.1,
        skip_rate=0.03,
        rationale_rate=0.3,
        median_latency_ms=15_000,
    ),
    Archetype.LEFT_BIASED: AnnotatorProfile(
        archetype=Archetype.LEFT_BIASED,
        sharpness=2.0,
        position_bias=2.0,
        tie_rate=0.04,
        skip_rate=0.01,
        rationale_rate=0.2,
        median_latency_ms=12_000,
    ),
    Archetype.LENGTH_BIASED: AnnotatorProfile(
        archetype=Archetype.LENGTH_BIASED,
        sharpness=1.0,
        length_weight=3.0,
        tie_rate=0.04,
        skip_rate=0.01,
        rationale_rate=0.3,
        median_latency_ms=18_000,
    ),
    Archetype.RANDOM_SPAMMER: AnnotatorProfile(
        archetype=Archetype.RANDOM_SPAMMER,
        sharpness=0.0,
        median_latency_ms=2_500,
    ),
    Archetype.ADVERSARIAL: AnnotatorProfile(
        archetype=Archetype.ADVERSARIAL,
        sharpness=-2.0,
        tie_rate=0.03,
        rationale_rate=0.2,
        median_latency_ms=20_000,
    ),
}
"""Default parameters per archetype. Apart from its planted defect, each biased
archetype is a competent annotator, so the defect is what a check must find."""

DEFAULT_ROSTER: Mapping[Archetype, int] = {
    Archetype.RELIABLE: 6,
    Archetype.NOISY: 2,
    Archetype.LEFT_BIASED: 1,
    Archetype.LENGTH_BIASED: 1,
    Archetype.RANDOM_SPAMMER: 1,
    Archetype.ADVERSARIAL: 1,
}


class SimulationConfig(Record):
    """Everything that determines a simulated dataset, together with the seed."""

    seed: int = Field(default=0, ge=0)
    n_prompts: int = Field(default=40, ge=1, le=100_000)
    n_models: int = Field(default=6, ge=2, le=26)
    strength_spread: float = Field(default=1.5, ge=0, allow_inf_nan=False)
    response_sd: float = Field(default=0.3, ge=0, allow_inf_nan=False)
    mean_words: float = Field(default=90.0, ge=5, le=5_000)
    verbosity_sd: float = Field(default=0.4, ge=0, le=3)
    length_quality_corr: float = Field(default=0.0, ge=-0.95, le=0.95)
    length_sd: float = Field(default=0.25, ge=0, le=3)
    pairs_per_prompt: int = Field(default=4, ge=1)
    redundancy: int = Field(default=3, ge=1)
    control_rate: float = Field(default=0.1, ge=0, le=1)
    n_gold: int = Field(default=12, ge=0)
    gold_min_gap: float = Field(default=1.0, gt=0, allow_inf_nan=False)
    ranked_per_annotator: int = Field(default=2, ge=0)
    ranked_size: int = Field(default=4, ge=2)
    session_size: int = Field(default=60, ge=1)
    roster: dict[Archetype, int] = Field(default_factory=lambda: dict(DEFAULT_ROSTER))
    profiles: dict[Archetype, AnnotatorProfile] = Field(
        default_factory=lambda: dict(DEFAULT_PROFILES)
    )
    start: UtcDatetime = datetime(2026, 1, 1, tzinfo=UTC)

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        problems = []
        if self.pairs_per_prompt > math.comb(self.n_models, 2):
            problems.append(
                f"pairs_per_prompt={self.pairs_per_prompt} exceeds the "
                f"{math.comb(self.n_models, 2)} pairs available with {self.n_models} models"
            )
        if self.ranked_per_annotator and self.ranked_size > self.n_models:
            problems.append(f"ranked_size={self.ranked_size} exceeds n_models={self.n_models}")
        if any(count < 0 for count in self.roster.values()):
            problems.append("roster counts must be non-negative")
        if self.n_annotators < 1:
            problems.append("the roster needs at least one annotator")
        if self.redundancy > self.n_annotators:
            problems.append(
                f"redundancy={self.redundancy} exceeds the {self.n_annotators} annotators"
            )
        for archetype, count in self.roster.items():
            if count and archetype not in self.profiles:
                problems.append(f"no profile for archetype {archetype.value}")
        for archetype, profile in self.profiles.items():
            if profile.archetype is not archetype:
                problems.append(f"profile under {archetype.value} is for {profile.archetype}")
        if problems:
            raise ValueError("; ".join(problems))
        return self

    @property
    def n_annotators(self) -> int:
        return sum(self.roster.values())


class SimulationTruth(Record):
    """The latent quantities behind a simulated dataset."""

    config: SimulationConfig
    model_strengths: dict[str, float]
    model_verbosity: dict[str, float]
    response_quality: dict[str, float]
    annotator_archetypes: dict[str, Archetype]

    @property
    def defective_annotators(self) -> tuple[str, ...]:
        """Annotators a correct audit should flag, sorted by id."""
        archetypes = self.annotator_archetypes
        return tuple(sorted(a for a, kind in archetypes.items() if kind.is_defective))

    def profile(self, annotator_id: str) -> AnnotatorProfile:
        return self.config.profiles[self.annotator_archetypes[annotator_id]]

    def model_ranking(self) -> list[str]:
        """Model names from strongest to weakest."""
        return sorted(self.model_strengths, key=lambda m: (-self.model_strengths[m], m))

    def better_response(self, first: str, second: str) -> str:
        """The response with the higher latent quality."""
        if self.response_quality[first] >= self.response_quality[second]:
            return first
        return second


@dataclass(frozen=True, slots=True)
class SimulatedDataset:
    """Records ready for the store, plus the truth that generated them."""

    truth: SimulationTruth
    prompts: tuple[Prompt, ...]
    responses: tuple[Response, ...]
    annotators: tuple[Annotator, ...]
    sessions: tuple[Session, ...]
    gold_pairs: tuple[GoldPair, ...]
    pairwise: tuple[PairwiseJudgment, ...]
    ranked: tuple[RankedJudgment, ...]

    def records(self) -> list[AnyRecord]:
        """All records in dependency order."""
        return [
            *self.prompts,
            *self.responses,
            *self.annotators,
            *self.sessions,
            *self.gold_pairs,
            *self.pairwise,
            *self.ranked,
        ]


# -- content --------------------------------------------------------------------

_TASKS = (
    ("explain", "Explain {topic} to a {audience}."),
    ("checklist", "Write a short checklist for someone getting started with {topic}."),
    ("compare", "Compare two common approaches to {topic} and say when each fits."),
    ("risks", "Summarise the main risks and trade-offs of {topic} for a {audience}."),
    ("example", "Give a small worked example of {topic}."),
)
_TOPICS = (
    "binary search",
    "database indexing",
    "unit testing",
    "rate limiting",
    "response caching",
    "public key encryption",
    "garbage collection",
    "load balancing",
    "gradient descent",
    "version control",
    "photosynthesis",
    "compound interest",
    "plate tectonics",
    "supply and demand",
    "time zones",
    "sourdough baking",
)
_AUDIENCES = ("new programmer", "product manager", "high school student", "senior engineer")
_VOCABULARY = (
    "the", "a", "this", "each", "value", "step", "result", "case", "input", "output",
    "simple", "careful", "common", "useful", "first", "next", "then", "because", "so",
    "when", "check", "keep", "use", "make", "note", "compare", "order", "small", "large",
    "rule", "idea", "example", "reason", "cost", "time", "space", "clear", "quick", "safe",
    "error", "test", "change", "state", "record", "update", "limit", "balance", "detail",
)  # fmt: skip
_RATIONALES = {
    Choice.LEFT: "The left answer is more accurate and easier to follow.",
    Choice.RIGHT: "The right answer is more accurate and easier to follow.",
    Choice.TIE: "Both answers are about equally good.",
}


def _model_names(n: int) -> list[str]:
    return [f"model-{chr(ord('a') + i)}" for i in range(n)]


def _synthetic_text(rng: np.random.Generator, n_words: int) -> str:
    """Filler prose with exactly ``n_words`` whitespace-separated words."""
    picks = rng.integers(0, len(_VOCABULARY), size=n_words)
    sentence_lengths = rng.integers(6, 15, size=n_words)
    words: list[str] = []
    sentence_start = 0
    for i, pick in enumerate(picks):
        word = _VOCABULARY[int(pick)]
        if i == sentence_start:
            word = word.capitalize()
        end_of_sentence = i - sentence_start + 1 >= int(sentence_lengths[sentence_start])
        if end_of_sentence or i == n_words - 1:
            word += "."
            sentence_start = i + 1
        words.append(word)
    return " ".join(words)


# -- schedule -------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _PairTask:
    time: float
    key: int
    prompt_id: str
    left: str
    right: str
    kind: PairKind
    original_key: int | None = None


@dataclass(frozen=True, slots=True)
class _RankTask:
    time: float
    key: int
    prompt_id: str
    shown: tuple[str, ...]


@dataclass(slots=True)
class _World:
    """Intermediate state shared by the generation steps."""

    config: SimulationConfig
    streams: dict[str, np.random.SeedSequence]
    prompts: list[Prompt] = field(default_factory=list)
    responses: list[Response] = field(default_factory=list)
    by_prompt: dict[str, list[str]] = field(default_factory=dict)
    quality: dict[str, float] = field(default_factory=dict)
    words: dict[str, int] = field(default_factory=dict)
    strengths: dict[str, float] = field(default_factory=dict)
    verbosity: dict[str, float] = field(default_factory=dict)
    annotator_ids: list[str] = field(default_factory=list)
    archetypes: dict[str, Archetype] = field(default_factory=dict)

    def rng(self, name: str) -> np.random.Generator:
        return np.random.default_rng(self.streams[name])


_STREAMS = (
    "models",
    "prompts",
    "responses",
    "annotators",
    "items",
    "gold",
    "assignment",
    "behaviour",
)


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 1e-12 else np.zeros_like(vector)


def _verbosity_offsets(
    rng: np.random.Generator, strengths: np.ndarray, sd: float, corr: float
) -> np.ndarray:
    """Mean-zero offsets with population sd ``sd`` and correlation ``corr`` with strengths.

    A random direction is made orthogonal to the (centred) strengths and mixed
    with them in proportion ``corr : sqrt(1 - corr^2)``, which fixes the sample
    correlation exactly. Degenerate cases (equal strengths, or two models where
    no orthogonal direction exists) fall back to whichever direction exists.
    """
    n = len(strengths)
    along = _unit(strengths - strengths.mean())
    raw = rng.standard_normal(n)
    raw = raw - raw.mean()
    across = _unit(raw - float(raw @ along) * along)
    direction = corr * along + math.sqrt(1.0 - corr * corr) * across
    return sd * math.sqrt(n) * _unit(direction)


def _make_models(world: _World) -> None:
    config = world.config
    rng = world.rng("models")
    names = _model_names(config.n_models)
    levels = np.linspace(-config.strength_spread, config.strength_spread, config.n_models)
    levels = rng.permutation(levels)
    levels = levels - levels.mean()
    verbosity = _verbosity_offsets(rng, levels, config.verbosity_sd, config.length_quality_corr)
    world.strengths = {m: float(v) for m, v in zip(names, levels, strict=True)}
    world.verbosity = {m: float(v) for m, v in zip(names, verbosity, strict=True)}


def _make_prompts(world: _World) -> None:
    config = world.config
    rng = world.rng("prompts")
    width = max(4, len(str(config.n_prompts)))
    for i in range(config.n_prompts):
        category, template = _TASKS[int(rng.integers(len(_TASKS)))]
        topic = _TOPICS[int(rng.integers(len(_TOPICS)))]
        audience = _AUDIENCES[int(rng.integers(len(_AUDIENCES)))]
        text = template.format(topic=topic, audience=audience)
        world.prompts.append(
            Prompt(id=f"p{i + 1:0{width}d}", text=text, category=category, tags=(topic,))
        )


def _make_responses(world: _World) -> None:
    config = world.config
    rng = world.rng("responses")
    provenance = Provenance(
        source=ResponseSource.SYNTHETIC,
        generator=GENERATOR_NAME,
        seed=config.seed,
        origin=f"simulate(seed={config.seed})",
    )
    log_mean = math.log(config.mean_words)
    for prompt in world.prompts:
        ids = []
        for model, strength in world.strengths.items():
            response_id = f"{prompt.id}-{model}"
            quality = strength + float(rng.normal(0.0, config.response_sd))
            log_words = log_mean + world.verbosity[model] + float(rng.normal(0.0, config.length_sd))
            n_words = max(3, round(math.exp(log_words)))
            text = _synthetic_text(rng, n_words)
            world.responses.append(
                Response(
                    id=response_id,
                    prompt_id=prompt.id,
                    model=model,
                    text=text,
                    provenance=provenance,
                )
            )
            world.quality[response_id] = quality
            world.words[response_id] = n_words
            ids.append(response_id)
        world.by_prompt[prompt.id] = ids


def _make_annotators(world: _World) -> list[Annotator]:
    config = world.config
    rng = world.rng("annotators")
    pool = [archetype for archetype in Archetype for _ in range(config.roster.get(archetype, 0))]
    order = rng.permutation(len(pool))
    width = max(2, len(str(len(pool))))
    annotators = []
    for i, index in enumerate(order):
        annotator_id = f"ann-{i + 1:0{width}d}"
        world.annotator_ids.append(annotator_id)
        world.archetypes[annotator_id] = pool[int(index)]
        annotators.append(Annotator(id=annotator_id, kind=AnnotatorKind.SIMULATED))
    return annotators


def _select_items(world: _World) -> tuple[list[tuple[str, str, str]], list[tuple[str, str, str]]]:
    """Regular pairs (prompt, a, b) per prompt, and the pairs left over."""
    rng = world.rng("items")
    regular: list[tuple[str, str, str]] = []
    leftover: list[tuple[str, str, str]] = []
    for prompt in world.prompts:
        pairs = list(combinations(world.by_prompt[prompt.id], 2))
        picks = rng.choice(len(pairs), size=world.config.pairs_per_prompt, replace=False)
        chosen = {int(i) for i in picks}
        for i, (a, b) in enumerate(pairs):
            (regular if i in chosen else leftover).append((prompt.id, a, b))
    return regular, leftover


def _make_gold(world: _World, leftover: Sequence[tuple[str, str, str]]) -> list[GoldPair]:
    config = world.config
    candidates = [
        (prompt_id, a, b)
        for prompt_id, a, b in leftover
        if abs(world.quality[a] - world.quality[b]) >= config.gold_min_gap
    ]
    if len(candidates) < config.n_gold:
        msg = (
            f"only {len(candidates)} unused pairs have a true quality gap >= "
            f"{config.gold_min_gap}; lower n_gold or gold_min_gap, or add prompts"
        )
        raise SimulationError(msg)
    rng = world.rng("gold")
    picks = sorted(int(i) for i in rng.choice(len(candidates), size=config.n_gold, replace=False))
    width = max(4, len(str(config.n_gold)))
    gold = []
    for number, index in enumerate(picks, start=1):
        prompt_id, a, b = candidates[index]
        better = a if world.quality[a] > world.quality[b] else b
        worse = b if better == a else a
        gap = abs(world.quality[a] - world.quality[b])
        gold.append(
            GoldPair(
                id=f"g{number:0{width}d}",
                prompt_id=prompt_id,
                better_response_id=better,
                worse_response_id=worse,
                reason=f"simulated: true quality gap {gap:.2f}",
            )
        )
    return gold


def _assign(world: _World, regular: Sequence[tuple[str, str, str]]) -> dict[str, list[int]]:
    """Give every regular pair to ``redundancy`` distinct annotators, balancing load."""
    rng = world.rng("assignment")
    n = len(world.annotator_ids)
    load = np.zeros(n, dtype=np.int64)
    assigned: dict[str, list[int]] = {a: [] for a in world.annotator_ids}
    for item in rng.permutation(len(regular)):
        tiebreak = rng.random(n)
        chosen = np.lexsort((tiebreak, load))[: world.config.redundancy]
        for annotator_index in chosen:
            load[annotator_index] += 1
            assigned[world.annotator_ids[int(annotator_index)]].append(int(item))
    return assigned


def _tasks_for(
    world: _World,
    rng: np.random.Generator,
    items: Sequence[tuple[str, str, str]],
    gold: Sequence[GoldPair],
) -> list[_PairTask | _RankTask]:
    """One annotator's work in the order it is done."""
    config = world.config
    regular: list[_PairTask] = []
    for prompt_id, a, b in items:
        left, right = (a, b) if rng.random() < 0.5 else (b, a)
        regular.append(
            _PairTask(float(rng.random()), len(regular), prompt_id, left, right, PairKind.REGULAR)
        )
    tasks: list[_PairTask | _RankTask] = list(regular)
    for pair in gold:
        a, b = pair.better_response_id, pair.worse_response_id
        left, right = (a, b) if rng.random() < 0.5 else (b, a)
        tasks.append(
            _PairTask(float(rng.random()), len(tasks), pair.prompt_id, left, right, PairKind.GOLD)
        )
    n_controls = round(config.control_rate * len(regular))
    for index in sorted(int(i) for i in rng.choice(len(regular), size=n_controls, replace=False)):
        original = regular[index]
        time = original.time + (1.0 - original.time) * float(rng.random())
        tasks.append(
            _PairTask(
                time,
                len(tasks),
                original.prompt_id,
                original.right,
                original.left,
                PairKind.CONTROL,
                original_key=original.key,
            )
        )
    prompt_ids = [p.id for p in world.prompts]
    for _ in range(config.ranked_per_annotator):
        prompt_id = prompt_ids[int(rng.integers(len(prompt_ids)))]
        members = world.by_prompt[prompt_id]
        picks = rng.choice(len(members), size=config.ranked_size, replace=False)
        shown = tuple(members[int(i)] for i in picks)
        tasks.append(_RankTask(float(rng.random()), len(tasks), prompt_id, shown))
    return sorted(tasks, key=lambda task: (task.time, task.key))


# -- behaviour ------------------------------------------------------------------


@dataclass(slots=True)
class _Output:
    sessions: list[Session] = field(default_factory=list)
    pairwise: list[PairwiseJudgment] = field(default_factory=list)
    ranked: list[RankedJudgment] = field(default_factory=list)


_CHOICES = (Choice.LEFT, Choice.RIGHT, Choice.TIE, Choice.SKIP)


def _draw_choice(rng: np.random.Generator, probabilities: Mapping[Choice, float]) -> Choice:
    u = float(rng.random())
    cumulative = 0.0
    for choice in _CHOICES[:-1]:
        cumulative += probabilities[choice]
        if u < cumulative:
            return choice
    return Choice.SKIP


def _plackett_luce(
    rng: np.random.Generator, world: _World, profile: AnnotatorProfile, shown: Sequence[str]
) -> tuple[str, ...]:
    utilities = np.array(
        [
            profile.sharpness * world.quality[r]
            + profile.length_weight * math.log(world.words[r])
            + (profile.position_bias if position == 0 else 0.0)
            for position, r in enumerate(shown)
        ]
    )
    gumbel = -np.log(-np.log(rng.random(len(shown))))
    order = np.argsort(-(utilities + gumbel), kind="stable")
    return tuple(shown[int(i)] for i in order)


def _annotate(
    world: _World,
    *,
    annotator_id: str,
    index: int,
    tasks: Sequence[_PairTask | _RankTask],
    rng: np.random.Generator,
    ids: dict[str, int],
    out: _Output,
) -> None:
    config = world.config
    profile = world.config.profiles[world.archetypes[annotator_id]]
    judgment_ids: dict[int, str] = {}
    clock = config.start + timedelta(hours=index)
    for chunk_start in range(0, len(tasks), config.session_size):
        session_id = f"{annotator_id}-s{chunk_start // config.session_size + 1:02d}"
        started = clock
        for task in tasks[chunk_start : chunk_start + config.session_size]:
            latency = max(
                MIN_LATENCY_MS,
                round(profile.median_latency_ms * math.exp(LATENCY_LOG_SD * rng.standard_normal())),
            )
            clock += timedelta(milliseconds=latency)
            wants_rationale = float(rng.random()) < profile.rationale_rate
            if isinstance(task, _RankTask):
                ids["ranked"] += 1
                ranking = _plackett_luce(rng, world, profile, task.shown)
                out.ranked.append(
                    RankedJudgment(
                        id=f"k{ids['ranked']:06d}",
                        session_id=session_id,
                        annotator_id=annotator_id,
                        prompt_id=task.prompt_id,
                        shown=task.shown,
                        ranking=ranking,
                        rationale="Ordered by accuracy, then clarity." if wants_rationale else None,
                        latency_ms=latency,
                        created_at=clock,
                    )
                )
                continue
            gap = world.quality[task.left] - world.quality[task.right]
            log_ratio = math.log(world.words[task.left] / world.words[task.right])
            choice = _draw_choice(rng, profile.choice_probabilities(gap, log_ratio))
            original = task.original_key
            ids["pairwise"] += 1
            judgment_id = f"j{ids['pairwise']:07d}"
            judgment_ids[task.key] = judgment_id
            out.pairwise.append(
                PairwiseJudgment(
                    id=judgment_id,
                    session_id=session_id,
                    annotator_id=annotator_id,
                    prompt_id=task.prompt_id,
                    left_response_id=task.left,
                    right_response_id=task.right,
                    choice=choice,
                    pair_kind=task.kind,
                    repeat_of=None if original is None else judgment_ids[original],
                    rationale=_RATIONALES.get(choice) if wants_rationale else None,
                    latency_ms=latency,
                    created_at=clock,
                )
            )
        out.sessions.append(
            Session(
                id=session_id,
                annotator_id=annotator_id,
                client=SessionClient.SIMULATOR,
                started_at=started,
                ended_at=clock,
            )
        )
        clock += SESSION_BREAK


def simulate(config: SimulationConfig | None = None) -> SimulatedDataset:
    """Generate a dataset and its ground truth. Same config, same output."""
    config = config or SimulationConfig()
    children = np.random.SeedSequence(config.seed).spawn(len(_STREAMS))
    world = _World(config=config, streams=dict(zip(_STREAMS, children, strict=True)))
    _make_models(world)
    _make_prompts(world)
    _make_responses(world)
    annotators = _make_annotators(world)
    regular, leftover = _select_items(world)
    gold = _make_gold(world, leftover)
    assigned = _assign(world, regular)

    behaviour = world.streams["behaviour"].spawn(len(world.annotator_ids))
    out = _Output()
    ids = {"pairwise": 0, "ranked": 0}
    for index, annotator_id in enumerate(world.annotator_ids):
        rng = np.random.default_rng(behaviour[index])
        items = [regular[i] for i in sorted(assigned[annotator_id])]
        tasks = _tasks_for(world, rng, items, gold)
        _annotate(
            world, annotator_id=annotator_id, index=index, tasks=tasks, rng=rng, ids=ids, out=out
        )

    truth = SimulationTruth(
        config=config,
        model_strengths=world.strengths,
        model_verbosity=world.verbosity,
        response_quality=world.quality,
        annotator_archetypes=world.archetypes,
    )
    return SimulatedDataset(
        truth=truth,
        prompts=tuple(world.prompts),
        responses=tuple(world.responses),
        annotators=tuple(annotators),
        sessions=tuple(out.sessions),
        gold_pairs=tuple(gold),
        pairwise=tuple(out.pairwise),
        ranked=tuple(out.ranked),
    )


def write_simulation(store: Store, dataset: SimulatedDataset) -> int:
    """Persist records and ground truth atomically; return the number of new records.

    Writing the same dataset twice is a no-op. Writing a different one into a
    database that already holds simulated data raises RecordConflictError.
    """
    with store.transaction():
        store.save_simulation_truth(dataset.truth.config.seed, dataset.truth.model_dump_json())
        result = store.import_records(dataset.records())
    return result.total_added


def load_truth(store: Store) -> SimulationTruth | None:
    """The ground truth stored with a simulated dataset, if any."""
    raw = store.simulation_truth_json()
    return None if raw is None else SimulationTruth.model_validate_json(raw)
