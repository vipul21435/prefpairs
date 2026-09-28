"""Aggregate votes per response pair and filter them with recorded reasons.

Every judgment of a pair counts as one vote on the canonical (id-sorted)
pair, so ``a`` wins, ``b`` wins or tie, whatever side each response was shown
on. Skips carry no preference and are not votes. Votes from excluded
annotators (for example the ones the audit flagged) are counted separately
and never decide a pair, so the drop log can say how much evidence was set
aside.

A pair's winner is the response with more votes; equal decisive counts make
the pair a tie. Its agreement is the winner's share of the counted votes,
ties included, so a 2-1 split with one tie vote has agreement 2/4 = 0.5.

The filters never silently lose a pair: ``select_pairs`` returns one decision
per pair with every reason it was dropped, in a fixed order, so the counts in
a dataset card can be recomputed from the decisions alone.
"""

from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from enum import StrEnum

from pydantic import Field

from prefpairs.schema import CanonicalLabel, PairKind, PairwiseJudgment, Record


class DropReason(StrEnum):
    """Why a pair was left out, in the order the filters are applied."""

    NO_VOTES = "no_votes"
    """Every vote came from an excluded annotator, or every judgment was a skip."""
    TOO_FEW_VOTES = "too_few_votes"
    TIE = "tie"
    LOW_AGREEMENT = "low_agreement"


class FilterConfig(Record):
    """Which votes count and which pairs are kept."""

    min_votes: int = Field(default=1, ge=1)
    """Counted votes (decisive or tie, from included annotators) a pair needs."""
    min_agreement: float = Field(default=0.5, ge=0, le=1)
    """Smallest share of counted votes the winner must hold."""
    drop_ties: bool = True
    """Drop pairs whose decisive votes are split evenly (training needs a winner)."""
    exclude_annotators: tuple[str, ...] = ()
    pair_kinds: tuple[PairKind, ...] = (PairKind.REGULAR,)
    """Gold and control pairs are quality checks, so by default they are not data."""


class PairVotes(Record):
    """The votes on one canonical pair of responses to one prompt."""

    prompt_id: str
    a_id: str
    b_id: str
    n_a: int
    n_b: int
    n_tie: int
    n_skip: int
    n_excluded: int
    """Non-skip votes from excluded annotators (not counted)."""
    annotators: tuple[str, ...]
    """Included annotators who cast a counted vote, sorted."""

    @property
    def n_votes(self) -> int:
        return self.n_a + self.n_b + self.n_tie

    @property
    def is_tie(self) -> bool:
        return self.n_a == self.n_b

    @property
    def chosen_rejected(self) -> tuple[str, str] | None:
        """(chosen, rejected) response ids, or None for a tie."""
        if self.n_a > self.n_b:
            return self.a_id, self.b_id
        if self.n_b > self.n_a:
            return self.b_id, self.a_id
        return None

    @property
    def agreement(self) -> float | None:
        """The winner's share of counted votes (ties included); None without votes."""
        if self.n_votes == 0:
            return None
        return max(self.n_a, self.n_b) / self.n_votes


class PairDecision(Record):
    votes: PairVotes
    reasons: tuple[DropReason, ...]
    """Empty when the pair is kept."""

    @property
    def kept(self) -> bool:
        return not self.reasons


class Selection(Record):
    """Every pair with its keep/drop decision, sorted by prompt and pair."""

    config: FilterConfig
    decisions: tuple[PairDecision, ...]

    @property
    def kept(self) -> tuple[PairVotes, ...]:
        return tuple(d.votes for d in self.decisions if d.kept)

    def drop_counts(self) -> dict[str, int]:
        """How many pairs each reason dropped (a pair can have several reasons)."""
        counts = {reason.value: 0 for reason in DropReason}
        for decision in self.decisions:
            for reason in decision.reasons:
                counts[reason.value] += 1
        return counts


@dataclass(slots=True)
class _Tally:
    a: int = 0
    b: int = 0
    tie: int = 0
    skip: int = 0
    excluded: int = 0
    annotators: set[str] = field(default_factory=set)


def aggregate_votes(
    judgments: Iterable[PairwiseJudgment],
    *,
    exclude_annotators: Iterable[str] = (),
    pair_kinds: Sequence[PairKind] = (PairKind.REGULAR,),
) -> tuple[PairVotes, ...]:
    """One ``PairVotes`` per (prompt, canonical pair), sorted."""
    excluded = frozenset(exclude_annotators)
    kinds = frozenset(pair_kinds)
    tallies: dict[tuple[str, str, str], _Tally] = defaultdict(_Tally)
    for j in judgments:
        if j.pair_kind not in kinds:
            continue
        a, b = j.canonical_pair
        tally = tallies[(j.prompt_id, a, b)]
        label = j.label
        if label is None:
            tally.skip += 1
        elif j.annotator_id in excluded:
            tally.excluded += 1
        else:
            tally.annotators.add(j.annotator_id)
            if label is CanonicalLabel.A_WINS:
                tally.a += 1
            elif label is CanonicalLabel.B_WINS:
                tally.b += 1
            else:
                tally.tie += 1
    return tuple(
        PairVotes(
            prompt_id=prompt,
            a_id=a,
            b_id=b,
            n_a=t.a,
            n_b=t.b,
            n_tie=t.tie,
            n_skip=t.skip,
            n_excluded=t.excluded,
            annotators=tuple(sorted(t.annotators)),
        )
        for (prompt, a, b), t in sorted(tallies.items())
    )


def _reasons(votes: PairVotes, config: FilterConfig) -> tuple[DropReason, ...]:
    if votes.n_votes == 0:
        return (DropReason.NO_VOTES,)
    reasons = []
    if votes.n_votes < config.min_votes:
        reasons.append(DropReason.TOO_FEW_VOTES)
    if config.drop_ties and votes.is_tie:
        reasons.append(DropReason.TIE)
    agreement = votes.agreement
    if agreement is not None and agreement < config.min_agreement:
        reasons.append(DropReason.LOW_AGREEMENT)
    return tuple(reasons)


def select_pairs(
    judgments: Iterable[PairwiseJudgment], config: FilterConfig | None = None
) -> Selection:
    """Aggregate votes and decide, with reasons, which pairs become training data."""
    config = config or FilterConfig()
    votes = aggregate_votes(
        judgments, exclude_annotators=config.exclude_annotators, pair_kinds=config.pair_kinds
    )
    return Selection(
        config=config,
        decisions=tuple(PairDecision(votes=v, reasons=_reasons(v, config)) for v in votes),
    )
