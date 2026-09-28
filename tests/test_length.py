"""Length-bias check: logistic regression of choice on log length ratio."""

import pytest

from prefpairs.quality import length_bias
from prefpairs.quality.length import consensus_strengths
from prefpairs.schema import PairKind, PairwiseJudgment, Provenance, Response
from prefpairs.simulate import Archetype, SimulationConfig, simulate
from tests.quality_helpers import BIG, judgment

PROVENANCE = Provenance.model_validate({"source": "synthetic"})


def response(response_id: str, words: int, model: str) -> Response:
    return Response(
        id=response_id,
        prompt_id="p1",
        model=model,
        text=" ".join(["word"] * words),
        provenance=PROVENANCE,
    )


def longer_wins(annotator: str, n: int) -> list[PairwiseJudgment]:
    """An annotator who always picks the longer of a 40-word and a 10-word answer."""
    out = []
    for i in range(n):
        left, right = ("long", "short") if i % 2 == 0 else ("short", "long")
        out.append(judgment(annotator, left, right, "left" if left == "long" else "right"))
    return out


RESPONSES = [response("long", 40, "model-a"), response("short", 10, "model-b")]


def test_always_choosing_the_longer_answer_is_separated_but_still_flagged() -> None:
    report = length_bias(longer_wins("ann-1", 30), RESPONSES, adjust_for_quality=False)
    (row,) = report.results
    assert (row.n_decisive, row.n_longer_chosen, row.n_shorter_chosen) == (30, 30, 0)
    assert row.separated
    assert row.coefficient is not None
    assert row.coefficient > 0
    assert row.p_value < 1e-6
    assert row.flagged
    assert report.flagged == ("ann-1",)
    assert report.distrusted == ()


def test_too_little_evidence_is_never_flagged() -> None:
    report = length_bias(longer_wins("ann-1", 5), RESPONSES)
    (row,) = report.results
    assert row.coefficient is None
    assert row.p_value == 1.0
    assert not row.flagged
    assert not report.pooled.flagged


def test_equal_lengths_cannot_be_estimated() -> None:
    same = [response("x", 10, "model-a"), response("y", 10, "model-b")]
    data = [judgment("ann-1", "x", "y", "left") for _ in range(20)]
    (row,) = length_bias(data, same, min_decisive=1).results
    assert row.n_equal_length == 20
    assert row.coefficient is None


def test_ties_skips_and_other_pair_kinds_are_left_out() -> None:
    data = [
        *longer_wins("ann-1", 12),
        judgment("ann-1", "long", "short", "tie"),
        judgment("ann-1", "long", "short", "skip"),
        judgment("ann-1", "long", "short", "right", kind=PairKind.GOLD),
    ]
    report = length_bias(data, RESPONSES)
    assert report.pair_kinds == (PairKind.REGULAR,)
    assert report.results[0].n_decisive == 12
    everything = length_bias(data, RESPONSES, pair_kinds=list(PairKind))
    assert everything.results[0].n_decisive == 13


def test_consensus_strengths() -> None:
    data = [judgment("ann-1", "long", "short", "left") for _ in range(4)]
    strengths = consensus_strengths(data, RESPONSES)
    assert strengths["model-a"] > strengths["model-b"]
    assert consensus_strengths([], RESPONSES) == {}


def test_quality_adjustment_reports_its_coefficient() -> None:
    dataset = simulate(SimulationConfig(seed=0))
    report = length_bias(dataset.pairwise, dataset.responses)
    assert report.adjust_for_quality
    reliable = [
        r
        for r in report.results
        if dataset.truth.annotator_archetypes[r.annotator_id or ""] is Archetype.RELIABLE
    ]
    # Careful annotators follow consensus quality: a clearly positive weight.
    assert all(r.quality_coefficient is not None and r.quality_coefficient > 1 for r in reliable)
    unadjusted = length_bias(dataset.pairwise, dataset.responses, adjust_for_quality=False)
    assert all(r.quality_coefficient is None for r in unadjusted.results)
    assert unadjusted.distrusted == ()


@pytest.mark.parametrize("seed", range(10))
def test_flags_exactly_the_length_biased_annotators(seed: int) -> None:
    dataset = simulate(BIG.model_copy(update={"seed": seed}))
    archetypes = dataset.truth.annotator_archetypes
    expected = tuple(sorted(a for a, k in archetypes.items() if k is Archetype.LENGTH_BIASED))
    report = length_bias(dataset.pairwise, dataset.responses)
    assert report.flagged == expected
    (planted,) = expected
    row = next(r for r in report.results if r.annotator_id == planted)
    # The planted weight is 3.0; the ridge shrinks the estimate but keeps its sign.
    assert row.coefficient is not None
    assert row.coefficient > 1.0
    assert row.n_longer_chosen > row.n_shorter_chosen


def mostly_longer(annotator: str, n: int, n_longer: int) -> list[PairwiseJudgment]:
    """Alternating sides; the longer answer wins the first ``n_longer`` judgments."""
    out = []
    for i in range(n):
        left, right = ("long", "short") if i % 2 == 0 else ("short", "long")
        winner = "long" if i < n_longer else "short"
        out.append(judgment(annotator, left, right, "left" if left == winner else "right"))
    return out


def test_untestable_annotators_do_not_join_the_holm_family() -> None:
    biased = mostly_longer("ann-x", 30, 24)
    alone = length_bias(biased, RESPONSES, adjust_for_quality=False)
    (row,) = alone.results
    assert row.flagged
    tail = [j for i in range(50) for j in mostly_longer(f"small-{i:02d}", 6, 3)]
    crowded = length_bias(biased + tail, RESPONSES, adjust_for_quality=False)
    rows = {r.annotator_id: r for r in crowded.results}
    assert rows["ann-x"].p_adjusted == pytest.approx(row.p_adjusted)
    assert rows["ann-x"].flagged
    assert crowded.flagged == ("ann-x",)
    small = [r for a, r in rows.items() if a != "ann-x"]
    assert all(r.coefficient is None and r.p_adjusted == 1.0 for r in small)
