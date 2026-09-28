"""Bradley-Terry MM fit: hand-worked answers, optimality, monotonicity, recovery."""

import math
from functools import cache

import numpy as np
import pytest
from pydantic import ValidationError

from prefpairs.aggregate import (
    BradleyTerryConfig,
    BradleyTerryFit,
    ComparisonSet,
    NotIdentifiableError,
    build_comparisons,
    fit_bradley_terry,
    predicted_win_rates,
    sample_bradley_terry,
)
from prefpairs.aggregate.bradley_terry import (
    best_shift,
    fit_counts_batch,
    is_strongly_connected,
    penalised_log_likelihood,
    sigmoid,
)
from prefpairs.simulate import SimulatedDataset, SimulationConfig, simulate

PLAIN = BradleyTerryConfig(prior=0.0, tol=1e-13)


def games(*rows: tuple[str, str, float], cluster: str = "c") -> ComparisonSet:
    return ComparisonSet.from_outcomes([(cluster, x, y, s) for x, y, s in rows])


@cache
def default_dataset() -> SimulatedDataset:
    return simulate(SimulationConfig())


class TestHandWorked:
    def test_chain_of_three(self) -> None:
        # A beats B 3-1 and B beats C 3-1; A and C never meet. The score
        # equations give p_A / (p_A + p_B) = 3/4 and p_C / (p_B + p_C) = 1/4,
        # so p_A = 3 p_B = 9 p_C and theta = (log 3, 0, -log 3).
        data = games(
            *[("A", "B", 1.0)] * 3,
            ("A", "B", 0.0),
            *[("B", "C", 1.0)] * 3,
            ("B", "C", 0.0),
        )
        fit = fit_bradley_terry(data, PLAIN)
        assert fit.converged
        assert np.allclose(fit.log_strengths, [math.log(3), 0.0, -math.log(3)], atol=1e-10)
        assert fit.ranking() == ["A", "B", "C"]
        assert fit.win_probability("A", "C") == pytest.approx(0.9)
        assert fit.predicted_win_rates()[0, 2] == pytest.approx(0.9)
        assert math.isnan(fit.reference_log_strength)

    def test_ties_count_as_half_a_win(self) -> None:
        # One win and one tie for A: 1.5 against 0.5, so p_A = 3 p_B.
        fit = fit_bradley_terry(games(("A", "B", 1.0), ("A", "B", 0.5)), PLAIN)
        assert fit.win_probability("A", "B") == pytest.approx(0.75, abs=1e-10)

    def test_symmetric_cycle_gives_equal_strengths(self) -> None:
        data = games(
            *[("A", "B", 1.0)] * 2,
            ("A", "B", 0.0),
            *[("B", "C", 1.0)] * 2,
            ("B", "C", 0.0),
            *[("C", "A", 1.0)] * 2,
            ("C", "A", 0.0),
        )
        fit = fit_bradley_terry(data, PLAIN)
        assert np.allclose(fit.log_strengths, 0.0, atol=1e-12)

    def test_single_item_pair_and_prior_shrinkage(self) -> None:
        data = games(("A", "B", 1.0))
        # Without a prior A would be infinitely strong; with one it is finite
        # and a stronger prior shrinks it more.
        with pytest.raises(NotIdentifiableError, match="strongly connected"):
            fit_bradley_terry(data, PLAIN)
        weak = fit_bradley_terry(data, BradleyTerryConfig(prior=0.1))
        strong = fit_bradley_terry(data, BradleyTerryConfig(prior=2.0))
        assert weak.strength("A") > strong.strength("A") > 0
        assert weak.strength("A") == pytest.approx(-weak.strength("B"))

    def test_no_games_is_an_error(self) -> None:
        with pytest.raises(ValueError, match="no comparisons"):
            fit_bradley_terry(ComparisonSet.from_outcomes([]))


class TestOptimality:
    @pytest.mark.parametrize("prior", [0.0, 0.1, 1.0])
    def test_score_equations_hold_at_the_solution(self, prior: float) -> None:
        data = sample_bradley_terry([1.2, 0.4, 0.0, -0.3, -1.3], n_clusters=30, seed=4)
        fit = fit_bradley_terry(data, BradleyTerryConfig(prior=prior, tol=1e-13))
        assert fit.converged
        theta = fit.log_strengths
        if prior > 0:
            theta = theta - fit.reference_log_strength  # reference strength back to 1
        p = np.exp(theta)
        expected = (fit.games * p[:, None] / (p[:, None] + p[None, :])).sum(axis=1)
        observed = fit.wins.sum(axis=1)
        if prior > 0:
            expected += 2 * prior * p / (p + 1)
            observed = observed + prior
        assert np.allclose(observed, expected, atol=1e-8)

    def test_log_likelihood_never_decreases(self) -> None:
        data = sample_bradley_terry([2.0, 1.0, 0.0, -1.0, -2.0], n_clusters=10, seed=1)
        for prior in (0.0, 0.1, 3.0):
            fit = fit_bradley_terry(data, BradleyTerryConfig(prior=prior, tol=1e-12))
            trace = np.array(fit.log_likelihood)
            assert len(trace) == fit.iterations
            assert np.all(np.diff(trace) >= -1e-9 * np.abs(trace[1:]))
            assert trace[-1] > trace[0]

    def test_solution_beats_nearby_points(self) -> None:
        data = sample_bradley_terry([1.0, 0.0, -0.5, 0.3], n_clusters=20, seed=2)
        config = BradleyTerryConfig(prior=0.5, tol=1e-13)
        fit = fit_bradley_terry(data, config)
        raw = (fit.log_strengths - fit.reference_log_strength)[None]
        best = penalised_log_likelihood(raw, fit.wins[None], 0.5)[0]
        rng = np.random.default_rng(0)
        for _ in range(50):
            nudged = raw + rng.normal(0, 1e-3, raw.shape)
            assert penalised_log_likelihood(nudged, fit.wins[None], 0.5)[0] <= best + 1e-12

    def test_iteration_cap_reports_non_convergence(self) -> None:
        data = sample_bradley_terry([1.0, 0.0, -1.0], n_clusters=10, seed=3)
        fit = fit_bradley_terry(data, BradleyTerryConfig(max_iter=2, tol=1e-15))
        assert (fit.iterations, fit.converged) == (2, False)

    def test_best_shift_zeroes_the_prior_gradient(self) -> None:
        rng = np.random.default_rng(5)
        log_p = rng.normal(0, 3, (7, 9))
        log_p[0] = [50.0, -50.0, 49.0, -49.0, 0.0, 1.0, -1.0, 2.0, 30.0]
        shifted = log_p + best_shift(log_p)[:, None]
        assert np.allclose(np.tanh(0.5 * shifted).sum(axis=1), 0.0, atol=1e-9)


class TestInvariance:
    def test_relabelling_items_permutes_strengths(self) -> None:
        rows = [("c", "A", "B", 1.0), ("c", "B", "C", 0.5), ("c", "C", "A", 0.0)]
        rows += [("d", "B", "A", 1.0), ("d", "C", "B", 1.0)]
        rename = {"A": "z", "B": "y", "C": "x"}
        fit = fit_bradley_terry(ComparisonSet.from_outcomes(rows))
        renamed = fit_bradley_terry(
            ComparisonSet.from_outcomes([(c, rename[a], rename[b], s) for c, a, b, s in rows])
        )
        for old, new in rename.items():
            assert fit.strength(old) == pytest.approx(renamed.strength(new), abs=1e-9)

    def test_integer_weights_equal_repeated_games(self) -> None:
        data = sample_bradley_terry([0.5, 0.0, -0.5], n_clusters=4, seed=6)
        weights = np.arange(data.n_games, dtype=np.float64) % 3
        repeated = ComparisonSet.from_outcomes(
            [
                (data.clusters[c], data.items[f], data.items[s], float(x))
                for f, s, x, c, w in zip(
                    data.first, data.second, data.score, data.cluster, weights, strict=True
                )
                for _ in range(int(w))
            ]
        )
        weighted = fit_bradley_terry(data, weights=weights)
        direct = fit_bradley_terry(repeated)
        assert np.allclose(weighted.log_strengths, direct.log_strengths, atol=1e-8)

    def test_batch_fit_matches_individual_fits(self) -> None:
        config = BradleyTerryConfig(tol=1e-12)
        sets = [sample_bradley_terry([1.0, 0.0, -1.0], n_clusters=5, seed=s) for s in range(4)]
        counts = [s.counts() for s in sets]
        wins = np.stack([c[0] for c in counts])
        played = np.stack([c[1] for c in counts])
        batch = fit_counts_batch(wins, played, config)
        for index, data in enumerate(sets):
            single = fit_bradley_terry(data, config)
            centred = batch.log_p[index] - batch.log_p[index].mean()
            assert np.allclose(centred, single.log_strengths, atol=1e-9)


class TestConnectivity:
    def test_strong_connectivity(self) -> None:
        assert is_strongly_connected(np.zeros((1, 1)))
        cycle = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=float)
        assert is_strongly_connected(cycle)
        chain = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=float)
        assert not is_strongly_connected(chain)
        split = np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=float)
        assert not is_strongly_connected(split)


class TestHelpers:
    def test_predicted_win_rates(self) -> None:
        rates = predicted_win_rates(np.array([1.0, 0.0, -2.0]))
        assert np.isnan(np.diag(rates)).all()
        off = ~np.eye(3, dtype=bool)
        assert np.allclose((rates + rates.T)[off], 1.0)
        assert rates[0, 2] == pytest.approx(sigmoid(3.0))
        assert sigmoid(-800.0) == 0.0
        assert sigmoid(800.0) == 1.0

    def test_sampler_validates_names(self) -> None:
        with pytest.raises(ValueError, match="names"):
            sample_bradley_terry([0.0, 1.0], n_clusters=1, names=["a", "a"])

    def test_sampler_produces_ties_and_cluster_effects(self) -> None:
        data = sample_bradley_terry(
            [0.0, 0.0], n_clusters=400, cluster_sd=1.0, tie_rate=0.3, seed=0, names=["u", "v"]
        )
        assert data.items == ("u", "v")
        assert 0.25 < data.n_ties / data.n_games < 0.35

    def test_config_is_validated(self) -> None:
        with pytest.raises(ValidationError):
            BradleyTerryConfig(prior=-1.0)


class TestRecovery:
    def test_recovers_known_strengths(self) -> None:
        truth = np.array([1.5, 0.7, 0.2, -0.4, -2.0])
        data = sample_bradley_terry(truth, n_clusters=400, seed=11)
        fit = fit_bradley_terry(data)
        assert fit.ranking() == [f"item-{i:02d}" for i in range(5)]
        assert np.allclose(fit.log_strengths, truth - truth.mean(), atol=0.15)

    def test_recovers_the_true_model_order_from_simulated_annotators(self) -> None:
        ds = default_dataset()
        fit = fit_bradley_terry(build_comparisons(ds.pairwise, ds.responses))
        assert isinstance(fit, BradleyTerryFit)
        assert fit.ranking() == ds.truth.model_ranking()
