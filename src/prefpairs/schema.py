"""Typed data model for preference data.

Every record is an immutable pydantic model that validates itself on
construction. The models describe what an annotator actually saw and did; the
analysis layers derive everything else (canonical labels, implied pairs) from
these fields so that position effects and content effects stay separable.

Conventions:

* Identifiers are short ASCII slugs (see ``ID_PATTERN``) so they are safe in
  file names, URLs and log lines.
* Timestamps must be timezone-aware and are normalised to UTC.
* A pairwise judgment records the displayed order (``left_response_id``,
  ``right_response_id``) and the raw ``choice``. Its *canonical* form compares
  the pair sorted by response id: ``a`` is the smaller id, ``b`` the larger.
"""

from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated, Self

from pydantic import (
    AfterValidator,
    AwareDatetime,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    StringConstraints,
    model_validator,
)

ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$"
MAX_TEXT_CHARS = 200_000
MAX_RATIONALE_CHARS = 2_000
MAX_RANKED_ITEMS = 20


def _require_non_blank(value: str) -> str:
    if not value.strip():
        msg = "must contain at least one non-whitespace character"
        raise ValueError(msg)
    return value


def _to_utc(value: datetime) -> datetime:
    return value.astimezone(UTC)


def _blank_to_none(value: object) -> object:
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    return value


Id = Annotated[str, StringConstraints(pattern=ID_PATTERN)]
"""Record identifier: 1-128 chars, ASCII letters, digits and ``_.:-``."""

Text = Annotated[
    str,
    StringConstraints(min_length=1, max_length=MAX_TEXT_CHARS),
    AfterValidator(_require_non_blank),
]
"""Prompt or response body; stored verbatim, must not be blank."""

Label = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=200),
]
"""Short human-readable label such as a model name or a category."""

Rationale = Annotated[
    Annotated[str, StringConstraints(max_length=MAX_RATIONALE_CHARS)] | None,
    BeforeValidator(_blank_to_none),
]
"""Free-text justification. Whitespace is trimmed and a blank string becomes None."""

UtcDatetime = Annotated[AwareDatetime, AfterValidator(_to_utc)]
"""Timezone-aware timestamp normalised to UTC."""

LatencyMs = Annotated[int, Field(ge=0)] | None


class Choice(StrEnum):
    """What the annotator picked, relative to the displayed order."""

    LEFT = "left"
    RIGHT = "right"
    TIE = "tie"
    SKIP = "skip"


class CanonicalLabel(StrEnum):
    """Outcome relative to the pair sorted by response id (``a`` < ``b``)."""

    A_WINS = "a_wins"
    B_WINS = "b_wins"
    TIE = "tie"


class PairKind(StrEnum):
    """Why a pair was shown: regular work, a repeated control, or a gold check."""

    REGULAR = "regular"
    CONTROL = "control"
    GOLD = "gold"


class ResponseSource(StrEnum):
    """Where a candidate response came from."""

    MODEL = "model"
    HUMAN = "human"
    SYNTHETIC = "synthetic"


class AnnotatorKind(StrEnum):
    HUMAN = "human"
    SIMULATED = "simulated"


class SessionClient(StrEnum):
    """The front end that produced a session's judgments."""

    CLI = "cli"
    WEB = "web"
    SIMULATOR = "simulator"
    IMPORT = "import"


def canonical_label(left_id: str, right_id: str, choice: Choice) -> CanonicalLabel | None:
    """Map a displayed choice to a label on the id-sorted pair.

    Returns None for a skip. Raises ValueError when both sides are the same
    response, because such a comparison carries no information.
    """
    if left_id == right_id:
        msg = f"a pair needs two distinct responses, got {left_id!r} twice"
        raise ValueError(msg)
    if choice is Choice.SKIP:
        return None
    if choice is Choice.TIE:
        return CanonicalLabel.TIE
    winner = left_id if choice is Choice.LEFT else right_id
    return CanonicalLabel.A_WINS if winner == min(left_id, right_id) else CanonicalLabel.B_WINS


class Record(BaseModel):
    """Base class: immutable, strict about unknown fields."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class Prompt(Record):
    id: Id
    text: Text
    category: Label | None = None
    tags: tuple[Label, ...] = ()


class Provenance(Record):
    """How a candidate response was produced, for the dataset card."""

    source: ResponseSource
    generator: Label | None = None
    temperature: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    top_p: float | None = Field(default=None, gt=0, le=1, allow_inf_nan=False)
    max_tokens: int | None = Field(default=None, gt=0)
    seed: int | None = None
    generated_at: UtcDatetime | None = None
    origin: Label | None = None


class Response(Record):
    """A candidate response to one prompt, attributed to the system that wrote it."""

    id: Id
    prompt_id: Id
    model: Label
    text: Text
    provenance: Provenance

    @property
    def n_chars(self) -> int:
        return len(self.text)

    @property
    def n_words(self) -> int:
        return len(self.text.split())


class Annotator(Record):
    id: Id
    kind: AnnotatorKind = AnnotatorKind.HUMAN
    display_name: Label | None = None


class Session(Record):
    """A contiguous block of work by one annotator in one client."""

    id: Id
    annotator_id: Id
    client: SessionClient
    started_at: UtcDatetime
    ended_at: UtcDatetime | None = None

    @model_validator(mode="after")
    def _ends_after_start(self) -> Self:
        if self.ended_at is not None and self.ended_at < self.started_at:
            msg = "ended_at must not be earlier than started_at"
            raise ValueError(msg)
        return self


class PairwiseJudgment(Record):
    """One comparison of two responses to the same prompt, as displayed."""

    id: Id
    session_id: Id
    annotator_id: Id
    prompt_id: Id
    left_response_id: Id
    right_response_id: Id
    choice: Choice
    pair_kind: PairKind = PairKind.REGULAR
    repeat_of: Id | None = None
    rationale: Rationale = None
    latency_ms: LatencyMs = None
    created_at: UtcDatetime

    @model_validator(mode="after")
    def _check_pair(self) -> Self:
        if self.left_response_id == self.right_response_id:
            msg = "left_response_id and right_response_id must differ"
            raise ValueError(msg)
        if (self.pair_kind is PairKind.CONTROL) != (self.repeat_of is not None):
            msg = "repeat_of is required for control pairs and forbidden otherwise"
            raise ValueError(msg)
        if self.repeat_of == self.id:
            msg = "a judgment cannot repeat itself"
            raise ValueError(msg)
        return self

    @property
    def canonical_pair(self) -> tuple[str, str]:
        """The response ids sorted, so (a, b) with a < b."""
        a, b = sorted((self.left_response_id, self.right_response_id))
        return a, b

    @property
    def label(self) -> CanonicalLabel | None:
        """Canonical outcome, or None for a skip."""
        return canonical_label(self.left_response_id, self.right_response_id, self.choice)

    @property
    def winner_loser(self) -> tuple[str, str] | None:
        """(winner, loser) for a decisive choice, None for a tie or skip."""
        if self.choice is Choice.LEFT:
            return self.left_response_id, self.right_response_id
        if self.choice is Choice.RIGHT:
            return self.right_response_id, self.left_response_id
        return None


class RankedJudgment(Record):
    """A strict ranking of several responses to one prompt, best first."""

    id: Id
    session_id: Id
    annotator_id: Id
    prompt_id: Id
    shown: tuple[Id, ...] = Field(min_length=2, max_length=MAX_RANKED_ITEMS)
    ranking: tuple[Id, ...]
    rationale: Rationale = None
    latency_ms: LatencyMs = None
    created_at: UtcDatetime

    @model_validator(mode="after")
    def _ranking_is_permutation(self) -> Self:
        if len(set(self.shown)) != len(self.shown):
            msg = "shown must not contain duplicate response ids"
            raise ValueError(msg)
        if sorted(self.ranking) != sorted(self.shown):
            msg = "ranking must be a permutation of the shown response ids"
            raise ValueError(msg)
        return self

    def implied_pairs(self) -> list[tuple[str, str]]:
        """Every (winner, loser) pair implied by the ranking, n*(n-1)/2 of them."""
        return [
            (better, worse)
            for i, better in enumerate(self.ranking)
            for worse in self.ranking[i + 1 :]
        ]


class GoldPair(Record):
    """A pair with an agreed answer, used to measure annotator accuracy."""

    id: Id
    prompt_id: Id
    better_response_id: Id
    worse_response_id: Id
    reason: Rationale = None

    @model_validator(mode="after")
    def _distinct(self) -> Self:
        if self.better_response_id == self.worse_response_id:
            msg = "better_response_id and worse_response_id must differ"
            raise ValueError(msg)
        return self

    @property
    def canonical_pair(self) -> tuple[str, str]:
        a, b = sorted((self.better_response_id, self.worse_response_id))
        return a, b

    @property
    def expected_label(self) -> CanonicalLabel:
        if self.better_response_id < self.worse_response_id:
            return CanonicalLabel.A_WINS
        return CanonicalLabel.B_WINS


AnyRecord = Prompt | Response | Annotator | Session | PairwiseJudgment | RankedJudgment | GoldPair
"""Every record type the store persists."""

RECORD_KINDS: dict[str, type[AnyRecord]] = {
    "prompt": Prompt,
    "response": Response,
    "annotator": Annotator,
    "session": Session,
    "gold": GoldPair,
    "pairwise": PairwiseJudgment,
    "ranked": RankedJudgment,
}
"""JSONL ``kind`` tag for each record type, in dependency (insertion) order."""


def record_kind(record: AnyRecord) -> str:
    """The JSONL ``kind`` tag of a record."""
    for kind, cls in RECORD_KINDS.items():
        if type(record) is cls:
            return kind
    msg = f"not a PrefPairs record: {type(record).__name__}"
    raise TypeError(msg)
