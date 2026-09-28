"""Newton/IRLS logistic regression: exactness, recovery, separation and the LR test."""

import math

import numpy as np
import pytest

from prefpairs.quality import fit_logistic, likelihood_ratio
from prefpairs.quality.logistic import expit, log_likelihood, log_odds_ratio


def table_data(table: tuple[tuple[int, int], tuple[int, int]]) -> tuple[np.ndarray, np.ndarray]:
    """Rows ``(x=1: y1, y0)``, ``(x=0: y1, y0)`` of a 2x2 table as a design and labels."""
    rows: list[list[float]] = []
    labels: list[float] = []
    for x, (ones, zeros) in zip((1.0, 0.0), table, strict=True):
        rows += [[1.0, x]] * (ones + zeros)
        labels += [1.0] * ones + [0.0] * zeros
    return np.array(rows), np.array(labels)


class TestExactness:
    TABLE = ((30, 10), (15, 25))

    def test_binary_covariate_mle_is_the_log_odds_ratio(self) -> None:
        X, y = table_data(self.TABLE)
        fit = fit_logistic(X, y, ridge=0.0)
        assert fit.converged
        assert not fit.separated
        assert fit.coef[1] == pytest.approx(log_odds_ratio(self.TABLE), abs=1e-12)
        assert fit.coef[1] == pytest.approx(math.log(30 * 25 / (10 * 15)), abs=1e-12)
        # The intercept is the log odds in the x = 0 group.
        assert fit.coef[0] == pytest.approx(math.log(15 / 25), abs=1e-12)
        # Woolf's standard error of a log odds ratio is exact for this model.
        assert fit.std_error[1] == pytest.approx(math.sqrt(1 / 30 + 1 / 10 + 1 / 15 + 1 / 25))

    def test_tiny_default_ridge_changes_nothing_visible(self) -> None:
        X, y = table_data(self.TABLE)
        assert fit_logistic(X, y).coef == pytest.approx(fit_logistic(X, y, ridge=0.0).coef)

    def test_likelihood_ratio_matches_the_two_by_two_g_test(self) -> None:
        X, y = table_data(self.TABLE)
        statistic, p_value = likelihood_ratio(X, y, 1, ridge=0.0)
        observed = np.array(self.TABLE, dtype=float)
        expected = observed.sum(1, keepdims=True) * observed.sum(0, keepdims=True) / observed.sum()
        g = 2.0 * float(np.sum(observed * np.log(observed / expected)))
        assert statistic == pytest.approx(g, rel=1e-9)
        assert p_value == pytest.approx(math.erfc(math.sqrt(g / 2)), rel=1e-9)

    def test_intercept_only_reduced_model(self) -> None:
        X = np.ones((4, 1))
        y = np.array([1.0, 1.0, 1.0, 0.0])
        statistic, _ = likelihood_ratio(X, y, 0, ridge=0.0)
        full = fit_logistic(X, y, ridge=0.0)
        assert full.coef[0] == pytest.approx(math.log(3.0))
        assert statistic == pytest.approx(2 * (full.log_likelihood - 4 * math.log(0.5)))


def test_recovers_a_planted_coefficient() -> None:
    rng = np.random.default_rng(0)
    n = 20_000
    x = rng.standard_normal(n)
    z = rng.standard_normal(n)
    truth = np.array([0.4, 1.5, -0.8])
    X = np.column_stack([np.ones(n), x, z])
    y = (rng.random(n) < expit(X @ truth)).astype(float)
    fit = fit_logistic(X, y)
    assert fit.converged
    assert np.all(np.abs(fit.coef - truth) < 3 * fit.std_error)
    lower, upper = fit.wald_interval(1, 0.99)
    assert lower < 1.5 < upper
    assert fit.wald_p_value(1) < 1e-100
    assert fit.z[1] == pytest.approx(fit.coef[1] / fit.std_error[1])


def test_newton_increases_the_objective_monotonically() -> None:
    rng = np.random.default_rng(3)
    X = np.column_stack([np.ones(200), rng.standard_normal((200, 2)) * 4])
    y = (rng.random(200) < 0.5).astype(float)
    fit = fit_logistic(X, y)
    assert fit.penalised_log_likelihood >= log_likelihood(X, y, np.zeros(3)) - 1e-12
    capped = fit_logistic(X, y, max_iter=1)
    assert capped.n_iter == 1
    assert not capped.converged


class TestSeparation:
    def test_complete_separation_is_reported(self) -> None:
        X = np.column_stack([np.ones(6), [-3.0, -2.0, -1.0, 1.0, 2.0, 3.0]])
        y = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
        fit = fit_logistic(X, y)
        assert fit.separated
        assert fit.coef[1] > 10  # finite only because of the ridge
        # The Wald test collapses (huge standard error) but the LR test does not.
        assert fit.wald_p_value(1) > 0.3
        _, p_value = likelihood_ratio(X, y, 1, full=fit)
        assert p_value < 0.01

    def test_quasi_complete_separation_from_an_empty_cell(self) -> None:
        X, y = table_data(((20, 0), (10, 10)))
        assert fit_logistic(X, y).separated

    def test_overlapping_data_is_not_separated(self) -> None:
        X, y = table_data(((20, 5), (10, 10)))
        assert not fit_logistic(X, y).separated


class TestValidation:
    def test_shapes_labels_and_values(self) -> None:
        with pytest.raises(ValueError, match="shape"):
            fit_logistic(np.ones(3), np.ones(3))
        with pytest.raises(ValueError, match="at least one"):
            fit_logistic(np.ones((0, 1)), np.ones(0))
        with pytest.raises(ValueError, match="0 and 1"):
            fit_logistic(np.ones((2, 1)), np.array([0.0, 0.5]))
        with pytest.raises(ValueError, match="finite"):
            fit_logistic(np.array([[1.0], [np.inf]]), np.array([0.0, 1.0]))
        with pytest.raises(ValueError, match="ridge"):
            fit_logistic(np.ones((2, 1)), np.array([0.0, 1.0]), ridge=-1.0)

    def test_rank_deficient_design_without_ridge(self) -> None:
        X = np.column_stack([np.ones(4), np.ones(4)])
        with pytest.raises(np.linalg.LinAlgError):
            fit_logistic(X, np.array([0.0, 1.0, 0.0, 1.0]), ridge=0.0)

    def test_per_coefficient_ridge(self) -> None:
        X, y = table_data(TestExactness.TABLE)
        free = fit_logistic(X, y, ridge=np.array([0.0, 0.0]))
        shrunk = fit_logistic(X, y, ridge=np.array([0.0, 100.0]))
        assert abs(shrunk.coef[1]) < abs(free.coef[1])
