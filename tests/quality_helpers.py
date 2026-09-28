"""Builders shared by the quality-check tests."""

from datetime import UTC, datetime, timedelta
from itertools import count

from prefpairs.schema import Choice, PairKind, PairwiseJudgment
from prefpairs.simulate import SimulationConfig

T0 = datetime(2026, 1, 1, tzinfo=UTC)
_ids = count(1)

BIG = SimulationConfig(n_prompts=60, pairs_per_prompt=8)
"""About 170 judgments per annotator: enough for every check to have power."""


def judgment(
    annotator: str,
    left: str,
    right: str,
    choice: Choice | str,
    *,
    prompt: str = "p1",
    kind: PairKind = PairKind.REGULAR,
    repeat_of: str | None = None,
    judgment_id: str | None = None,
) -> PairwiseJudgment:
    number = next(_ids)
    return PairwiseJudgment(
        id=judgment_id or f"j{number}",
        session_id=f"{annotator}-s1",
        annotator_id=annotator,
        prompt_id=prompt,
        left_response_id=left,
        right_response_id=right,
        choice=Choice(choice),
        pair_kind=kind,
        repeat_of=repeat_of,
        created_at=T0 + timedelta(seconds=number),
    )


def choices(annotator: str, sequence: str) -> list[PairwiseJudgment]:
    """One judgment per character: l(eft), r(ight), t(ie), s(kip)."""
    names = {"l": "left", "r": "right", "t": "tie", "s": "skip"}
    return [judgment(annotator, "a", "b", names[c]) for c in sequence]
