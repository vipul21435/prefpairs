"""Elo: hand-computed updates, conservation, symmetry and order invariance."""

import math

import numpy as np
import pytest
from pydantic import ValidationError

from prefpairs.aggregate import (
    ComparisonSet,
    EloConfig,
    build_comparisons,
    fit_bradley_terry,
    fit_elo,
    sample_bradley_terry,
)
from prefpairs.aggregate.elo import canonical_order, elo_passes, permuted_orders
from prefpairs.simulate import SimulationConfig, simulate

K32 = EloConfig(k=32.0, n_permutations=1)


def two_games() -> ComparisonSet:
    # Game 0: A beats B. Game 1: B beats A.
    return ComparisonSet.from_outcomes([("c", "A", "B", 1.0), ("c", "A", "B", 0.0)])


def expected_score(r_self: float, r_other: float) -> float:
    return 1.0 / (1.0 + 10 ** ((r_other - r_self) / 400.0))


class TestUpdateRule:
    def test_single_game_between_equals(self) -> None:
        data = ComparisonSet.from_outcomes([("c", "A", "B", 1.0)])
        fit = fit_elo(data, K32)
        assert fit.rating("A") == pytest.approx(1016.0)
        assert fit.rating("B") == pytest.approx(984.0)
        assert fit.ranking() == ["A", "B"]

    def test_tie_between_equals_changes_nothing(self) -> None:
        data = ComparisonSet.from_outcomes([("c", "A", "B", 0.5)])
        assert np.allclose(fit_elo(data, K32).ratings, 1000.0)

    def test_sequence_is_replayed_in_the_given_order(self) -> None:
        data = two_games()
        ratings = elo_passes(data, np.array([[0, 1], [1, 0]]), K32)
        # A wins first: 1016 / 984, then B (expected score E_B) beats A.
        e_b = expected_score(984.0, 1016.0)
        b_after = 984.0 + 32 * (1 - e_b)
        assert ratings[0] == pytest.approx([2000.0 - b_after, b_after])
        # The reverse order is the mirror image.
        assert ratings[1] == pytest.approx([b_after, 2000.0 - b_after])

    def test_padding_rows_skip_games(self) -> None:
        data = two_games()
        ratings = elo_passes(data, np.array([[0, -1], [-1, -1]]), K32)
        assert ratings[0] == pytest.approx([1016.0, 984.0])
        assert ratings[1] == pytest.approx([1000.0, 1000.0])

    def test_scale_and_base(self) -> None:
        data = ComparisonSet.from_outcomes([("c", "A", "B", 1.0), ("c", "A", "B", 1.0)])
        config = EloConfig(k=1.0, scale=1.0, base=math.e, initial=0.0, n_permutations=1)
        ratings = elo_passes(data, np.array([[0, 1]]), config)
        # First update: 0.5 each way; second: 1 - sigmoid(1).
        second = 1.0 - 1.0 / (1.0 + math.exp(-1.0))
        assert ratings[0] == pytest.approx([0.5 + second, -0.5 - second])


class TestInvariants:
    def test_ratings_are_zero_sum_in_every_pass(self) -> None:
        data = sample_bradley_terry([1.0, 0.2, -0.3, -1.0], n_clusters=20, seed=1)
        orders = permuted_orders(canonical_order(data), 7, np.random.default_rng(0))
        ratings = elo_passes(data, orders, EloConfig(k=16.0))
        assert np.allclose(ratings.mean(axis=1), 1000.0)

    def test_result_does_not_depend_on_storage_order(self) -> None:
        data = sample_bradley_terry([0.8, 0.0, -0.8], n_clusters=15, tie_rate=0.2, seed=2)
        rows = [
            (data.clusters[c], data.items[f], data.items[s], float(x))
            for f, s, x, c in zip(data.first, data.second, data.score, data.cluster, strict=True)
        ]
        shuffled = [rows[int(i)] for i in np.random.default_rng(9).permutation(len(rows))]
        # Also present each game with its items the other way round.
        mirrored = [(c, y, x, 1.0 - s) for c, x, y, s in shuffled]
        config = EloConfig(n_permutations=20, seed=4)
        base = fit_elo(data, config)
        for variant in (shuffled, mirrored):
            other = fit_elo(ComparisonSet.from_outcomes(variant), config)
            assert np.array_equal(base.ratings, other.ratings)

    def test_swapping_labels_swaps_ratings(self) -> None:
        rows = [("c", "A", "B", 1.0), ("c", "A", "C", 0.5), ("d", "B", "C", 0.0)]
        swap = {"A": "B", "B": "A", "C": "C"}
        config = EloConfig(n_permutations=50, seed=1)
        fit = fit_elo(ComparisonSet.from_outcomes(rows), config)
        swapped = fit_elo(
            ComparisonSet.from_outcomes([(c, swap[x], swap[y], s) for c, x, y, s in rows]), config
        )
        for item, image in swap.items():
            assert fit.rating(item) == pytest.approx(swapped.rating(image))

    def test_averaging_removes_most_of_the_order_noise(self) -> None:
        data = sample_bradley_terry([1.0, 0.5, 0.0, -0.5, -1.0], n_clusters=12, seed=3)
        config = EloConfig(k=16.0, n_permutations=400)
        first = fit_elo(data, config)
        second = fit_elo(data, config.model_copy(update={"seed": 1}))
        # A single order moves ratings by order_sd; the 400-order mean by about
        # order_sd / 20, so two seeds agree far more closely than one pass would.
        assert first.order_sd.min() > 5.0
        assert np.abs(first.ratings - second.ratings).max() < first.order_sd.min() / 4
        assert first.n_permutations == 400
        assert first.n_games == data.n_games

    def test_same_seed_same_result(self) -> None:
        data = sample_bradley_terry([0.3, 0.0, -0.3], n_clusters=5, seed=5)
        assert np.array_equal(fit_elo(data).ratings, fit_elo(data).ratings)

    def test_no_games_is_an_error(self) -> None:
        with pytest.raises(ValueError, match="no comparisons"):
            fit_elo(ComparisonSet.from_outcomes([]))

    def test_config_is_validated(self) -> None:
        with pytest.raises(ValidationError):
            EloConfig(base=1.0)


class TestAgreement:
    def test_elo_and_bradley_terry_agree_on_the_simulated_order(self) -> None:
        ds = simulate(SimulationConfig())
        data = build_comparisons(ds.pairwise, ds.responses)
        elo = fit_elo(data)
        assert elo.ranking() == fit_bradley_terry(data).ranking() == ds.truth.model_ranking()
