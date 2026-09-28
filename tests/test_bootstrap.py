"""Cluster bootstrap intervals and win-rate matrices."""

import numpy as np
import pytest
from pydantic import ValidationError

from prefpairs.aggregate import (
    BootstrapConfig,
    BradleyTerryConfig,
    ComparisonSet,
    EloConfig,
    NotIdentifiableError,
    ResampleUnit,
    bootstrap_bradley_terry,
    bootstrap_elo,
    empirical_win_rates,
    fit_bradley_terry,
    sample_bradley_terry,
    win_rates,
)
from prefpairs.aggregate.bootstrap import percentile_interval, ranks, replicate_weights

SMALL = BootstrapConfig(n_replicates=200)


class TestReplicateWeights:
    def test_prompt_bootstrap_keeps_prompts_whole(self) -> None:
        data = sample_bradley_terry([1.0, 0.0, -1.0], n_clusters=12, games_per_pair=2, seed=0)
        weights = replicate_weights(data, SMALL)
        assert weights.shape == (200, data.n_games)
        for members in data.cluster_members():
            block = weights[:, members]
            assert np.all(block == block[:, :1])
        # Each replicate draws exactly n_clusters prompts.
        per_cluster = np.stack([weights[:, m[0]] for m in data.cluster_members()], axis=1)
        assert np.all(per_cluster.sum(axis=1) == data.n_clusters)

    def test_judgment_bootstrap_draws_games(self) -> None:
        data = sample_bradley_terry([1.0, 0.0, -1.0], n_clusters=5, seed=0)
        config = BootstrapConfig(n_replicates=50, unit=ResampleUnit.JUDGMENT)
        weights = replicate_weights(data, config)
        assert np.all(weights.sum(axis=1) == data.n_games)
        assert not np.all(weights == weights[:, :1])

    def test_seeded(self) -> None:
        data = sample_bradley_terry([1.0, 0.0], n_clusters=8, seed=0)
        first = replicate_weights(data, SMALL)
        assert np.array_equal(first, replicate_weights(data, SMALL))
        other = replicate_weights(data, SMALL.model_copy(update={"seed": 1}))
        assert not np.array_equal(first, other)


class TestHelpers:
    def test_percentile_interval(self) -> None:
        values = np.arange(101, dtype=np.float64)[:, None]
        lower, upper = percentile_interval(values, 0.9)
        assert (lower[0], upper[0]) == pytest.approx((5.0, 95.0))

    def test_competition_ranks(self) -> None:
        assert ranks(np.array([[0.5, 2.0, 0.5, -1.0]])).tolist() == [[2, 1, 2, 4]]

    def test_config_is_validated(self) -> None:
        with pytest.raises(ValidationError):
            BootstrapConfig(level=1.0)
        with pytest.raises(ValidationError):
            BootstrapConfig(n_replicates=5)


class TestBradleyTerryIntervals:
    def test_intervals_bracket_the_estimate_and_are_seeded(self) -> None:
        data = sample_bradley_terry([1.0, 0.3, -0.2, -1.1], n_clusters=40, seed=1)
        result = bootstrap_bradley_terry(data, config=SMALL)
        assert result.n_replicates == 200
        assert result.unconverged == 0
        assert result.unit is ResampleUnit.PROMPT
        assert np.all(result.lower < result.estimate)
        assert np.all(result.estimate < result.upper)
        assert np.allclose(result.replicates.mean(axis=1), 0.0)
        assert result.rank_lower[0] == 1
        assert np.all(result.rank_lower <= result.rank_upper)
        lower, upper = result.interval("item-00")
        assert lower < upper
        again = bootstrap_bradley_terry(data, config=SMALL)
        assert np.array_equal(result.replicates, again.replicates)

    def test_intervals_shrink_like_one_over_root_n(self) -> None:
        truth = [0.8, 0.0, -0.8]
        small = bootstrap_bradley_terry(sample_bradley_terry(truth, n_clusters=25, seed=2))
        large = bootstrap_bradley_terry(sample_bradley_terry(truth, n_clusters=400, seed=2))
        ratio = np.mean(small.upper - small.lower) / np.mean(large.upper - large.lower)
        assert 3.0 < ratio < 5.5  # sqrt(400 / 25) = 4

    def test_coverage_is_close_to_nominal_on_data_from_the_model(self) -> None:
        truth = np.array([1.0, 0.5, 0.0, -0.3, -1.2])
        centred = truth - truth.mean()
        covered = []
        for seed in range(40):
            data = sample_bradley_terry(truth, n_clusters=30, seed=seed)
            config = BootstrapConfig(n_replicates=200, seed=seed)
            result = bootstrap_bradley_terry(data, config=config)
            covered.append((result.lower <= centred) & (centred <= result.upper))
        coverage = float(np.mean(covered))
        # 200 intervals at a nominal 95%; percentile intervals run slightly narrow.
        assert 0.85 <= coverage <= 1.0

    def test_cluster_bootstrap_widens_intervals_for_correlated_prompts(self) -> None:
        ratios = []
        for seed in range(5):
            data = sample_bradley_terry(
                [1.0, 0.0, -1.0, 0.5], n_clusters=30, games_per_pair=4, cluster_sd=1.0, seed=seed
            )
            by_prompt = bootstrap_bradley_terry(data, config=SMALL)
            by_judgment = bootstrap_bradley_terry(
                data, config=SMALL.model_copy(update={"unit": ResampleUnit.JUDGMENT})
            )
            ratios.append(
                np.mean(by_prompt.upper - by_prompt.lower)
                / np.mean(by_judgment.upper - by_judgment.lower)
            )
        assert min(ratios) > 1.25

    def test_plain_mle_refuses_disconnected_replicates(self) -> None:
        data = ComparisonSet.from_outcomes([("c1", "A", "B", 1.0), ("c2", "A", "B", 0.0)])
        with pytest.raises(NotIdentifiableError, match="replicate"):
            bootstrap_bradley_terry(data, BradleyTerryConfig(prior=0.0), SMALL)

    def test_plain_mle_on_connected_replicates(self) -> None:
        rows = [(f"c{i}", "A", "B", s) for i in range(5) for s in (1.0, 0.0)]
        data = ComparisonSet.from_outcomes(rows)
        result = bootstrap_bradley_terry(data, BradleyTerryConfig(prior=0.0), SMALL)
        assert np.allclose(result.replicates, 0.0)

    def test_iteration_cap_is_reported(self) -> None:
        data = sample_bradley_terry([2.0, 0.0, -2.0], n_clusters=10, seed=3)
        result = bootstrap_bradley_terry(data, BradleyTerryConfig(max_iter=1, tol=1e-15), SMALL)
        assert result.unconverged == 200

    def test_empty_input_is_an_error(self) -> None:
        with pytest.raises(ValueError, match="no comparisons"):
            bootstrap_bradley_terry(ComparisonSet.from_outcomes([]))


class TestEloIntervals:
    def test_intervals_bracket_the_estimate(self) -> None:
        data = sample_bradley_terry([1.0, 0.0, -1.0], n_clusters=40, seed=4)
        result = bootstrap_elo(data, EloConfig(k=8.0), SMALL)
        assert np.all(result.lower < result.estimate)
        assert np.all(result.estimate < result.upper)
        # Every replicate is zero-sum around the initial rating.
        assert np.allclose(result.replicates.mean(axis=1), 1000.0)
        # item-00 is clearly first; items 01 and 02 (true gap 1.0, Elo gap about
        # 40 points at K=8) swap places in some replicates, so their rank
        # intervals overlap instead of pretending to a strict order.
        assert result.rank_lower.tolist() == [1, 2, 2]
        assert result.rank_upper.tolist() == [1, 3, 3]

    def test_chunked_and_unchunked_runs_agree(self, monkeypatch: pytest.MonkeyPatch) -> None:
        data = sample_bradley_terry([0.5, 0.0, -0.5], n_clusters=10, seed=5)
        config = BootstrapConfig(n_replicates=20, elo_permutations=3)
        elo_whole = bootstrap_elo(data, config=config)
        bt_whole = bootstrap_bradley_terry(data, config=config)
        monkeypatch.setattr("prefpairs.aggregate.bootstrap._CELLS_PER_CHUNK", 1)
        elo_pieces = bootstrap_elo(data, config=config)
        bt_pieces = bootstrap_bradley_terry(data, config=config)
        assert np.allclose(elo_whole.replicates, elo_pieces.replicates)
        assert np.allclose(bt_whole.replicates, bt_pieces.replicates)

    def test_empty_input_is_an_error(self) -> None:
        with pytest.raises(ValueError, match="no comparisons"):
            bootstrap_elo(ComparisonSet.from_outcomes([]))


class TestWinRates:
    def test_empirical_rates(self) -> None:
        data = ComparisonSet.from_outcomes(
            [("c", "A", "B", 1.0), ("c", "A", "B", 0.5), ("c", "B", "C", 0.0), ("c", "A", "B", 0)]
        )
        rates, games = empirical_win_rates(data)
        assert rates[0, 1] == pytest.approx(0.5)
        assert rates[2, 1] == pytest.approx(1.0)
        assert np.isnan(rates[0, 2])
        assert np.isnan(np.diag(rates)).all()
        assert games[0, 1] == 3

    def test_plain_fit_reproduces_rates_on_a_tree(self) -> None:
        # With no prior and a tree-shaped comparison graph, the score equations
        # force the model to match every played pair exactly.
        data = ComparisonSet.from_outcomes(
            [("c", "A", "B", s) for s in (1.0, 1.0, 1.0, 0.0)]
            + [("c", "B", "C", s) for s in (1.0, 0.0, 0.5)]
        )
        fit = fit_bradley_terry(data, BradleyTerryConfig(prior=0.0, tol=1e-13))
        rates = win_rates(data, fit)
        assert rates.fit_gap == pytest.approx(0.0, abs=1e-9)
        played = rates.games > 0
        assert np.allclose(rates.empirical[played], rates.predicted[played])
        assert not np.isnan(rates.predicted[0, 2])

    def test_fit_gap_detects_a_rock_paper_scissors_cycle(self) -> None:
        rows = [("c", "A", "B", 1.0)] * 9 + [("c", "B", "C", 1.0)] * 9 + [("c", "C", "A", 1.0)] * 9
        data = ComparisonSet.from_outcomes(rows)
        rates = win_rates(data, fit_bradley_terry(data))
        # Equal strengths predict 0.5 everywhere; every played pair is 1.0 or 0.0.
        assert rates.fit_gap == pytest.approx(0.5, abs=1e-6)

    def test_fit_gap_without_games_and_mismatched_items(self) -> None:
        data = ComparisonSet.from_outcomes([("c", "A", "B", 1.0)])
        fit = fit_bradley_terry(data)
        rates = win_rates(data, fit)
        empty = type(rates)(
            items=rates.items,
            games=np.zeros((2, 2)),
            empirical=rates.empirical,
            predicted=rates.predicted,
        )
        assert empty.fit_gap == 0.0
        other = ComparisonSet.from_outcomes([("c", "A", "C", 1.0)])
        with pytest.raises(ValueError, match="different items"):
            win_rates(other, fit)
