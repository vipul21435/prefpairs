"""The slice-3 checks together, on simulated data with planted annotators."""

import pytest

from prefpairs.quality import length_bias, position_bias, self_consistency
from prefpairs.simulate import Archetype, simulate
from tests.quality_helpers import BIG


@pytest.mark.parametrize("seed", range(10))
def test_planted_biases_are_flagged_and_reliable_annotators_are_not(seed: int) -> None:
    dataset = simulate(BIG.model_copy(update={"seed": seed}))
    archetypes = dataset.truth.annotator_archetypes
    flagged = (
        set(position_bias(dataset.pairwise).flagged)
        | set(length_bias(dataset.pairwise, dataset.responses).flagged)
        | set(self_consistency(dataset.pairwise).flagged)
    )
    kinds = {archetypes[a] for a in flagged}
    assert kinds == {Archetype.LEFT_BIASED, Archetype.LENGTH_BIASED}
    assert not any(archetypes[a] is Archetype.RELIABLE for a in flagged)
