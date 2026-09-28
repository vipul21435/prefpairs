"""Logistic regression by Newton's method (iteratively reweighted least squares).

Model: ``P(y = 1 | x) = sigmoid(x @ beta)``. Each Newton step solves

    (X' W X + ridge * I) step = X' (y - p) - ridge * beta,   W = diag(p (1 - p))

which is exactly one weighted least-squares solve, hence "IRLS". A step that
would lower the (ridge-penalised) log-likelihood is halved until it does not,
so the objective never decreases.

The ridge term is tiny (``1e-8`` by default): it leaves well-posed fits
unchanged to many digits, keeps the Hessian invertible, and keeps the
coefficients finite when the data are *separated* (some combination of the
covariates predicts every label perfectly, so the plain MLE does not exist).
Separation is reported rather than hidden:

* complete separation: at the fit, every observation lies on its own side of
  the decision boundary, which is impossible for a finite MLE;
* quasi-complete separation: some observations are fitted with probability
  within ``1e-8`` of their label, which only happens as a coefficient runs off.

Under separation the Wald standard errors blow up (and Wald p-values drift
towards 1), so ``likelihood_ratio`` is the test to use; it stays well behaved.
"""

import math
from dataclasses import dataclass

import numpy as np

from prefpairs.aggregate.comparisons import FloatArray
from prefpairs.quality.stats import chi2_1_sf, normal_quantile, normal_two_sided_p

EXTREME_FIT = 1e-8


def expit(eta: FloatArray) -> FloatArray:
    """Numerically stable logistic function."""
    result: FloatArray = np.exp(-np.logaddexp(0.0, -eta))
    return result


def log_likelihood(X: FloatArray, y: FloatArray, coef: FloatArray) -> float:
    """Bernoulli log-likelihood ``sum(y * eta - log(1 + exp(eta)))``."""
    eta = X @ coef
    return float(np.sum(y * eta - np.logaddexp(0.0, eta)))


@dataclass(frozen=True, slots=True, eq=False)
class LogisticFit:
    """Coefficients, their standard errors, and how the fit went."""

    coef: FloatArray
    std_error: FloatArray
    log_likelihood: float
    """Unpenalised log-likelihood at the fitted coefficients."""
    penalised_log_likelihood: float
    """The maximised objective: log-likelihood minus the ridge penalty."""
    n_obs: int
    n_iter: int
    converged: bool
    separated: bool

    @property
    def z(self) -> FloatArray:
        """Wald statistics ``coef / std_error``."""
        result: FloatArray = self.coef / self.std_error
        return result

    def wald_p_value(self, j: int) -> float:
        return normal_two_sided_p(float(self.z[j]))

    def wald_interval(self, j: int, confidence: float = 0.95) -> tuple[float, float]:
        half = normal_quantile(0.5 + confidence / 2.0) * float(self.std_error[j])
        return float(self.coef[j]) - half, float(self.coef[j]) + half


def _check_design(X: FloatArray, y: FloatArray) -> None:
    if X.ndim != 2 or y.ndim != 1 or X.shape[0] != y.shape[0]:
        msg = f"need X of shape (n, k) and y of shape (n,), got {X.shape} and {y.shape}"
        raise ValueError(msg)
    if X.shape[0] == 0 or X.shape[1] == 0:
        msg = "need at least one observation and one covariate"
        raise ValueError(msg)
    if not np.all((y == 0.0) | (y == 1.0)):
        msg = "y must contain only 0 and 1"
        raise ValueError(msg)
    if not np.all(np.isfinite(X)):
        msg = "X must be finite"
        raise ValueError(msg)


def _ridge_vector(ridge: float | FloatArray, k: int) -> FloatArray:
    vector = np.broadcast_to(np.asarray(ridge, dtype=np.float64), (k,)).copy()
    if np.any(vector < 0) or not np.all(np.isfinite(vector)):
        msg = "ridge must be finite and non-negative"
        raise ValueError(msg)
    return vector


def fit_logistic(
    X: FloatArray,
    y: FloatArray,
    *,
    ridge: float | FloatArray = 1e-8,
    tol: float = 1e-10,
    max_iter: int = 100,
) -> LogisticFit:
    """Maximum-likelihood logistic regression (with a ridge) by Newton/IRLS.

    ``X`` must contain its own intercept column if one is wanted. ``ridge`` is
    one penalty for every coefficient or one per coefficient: the objective is
    ``log_likelihood - 0.5 * sum(ridge * beta**2)``, the MAP estimate under
    independent normal priors with variance ``1 / ridge``. With ``ridge=0`` a
    rank-deficient design raises ``numpy.linalg.LinAlgError``.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    _check_design(X, y)
    k = X.shape[1]
    lam = _ridge_vector(ridge, k)
    penalty = np.diag(lam)

    def objective(beta: FloatArray) -> float:
        return log_likelihood(X, y, beta) - 0.5 * float(lam @ (beta * beta))

    beta = np.zeros(k)
    current = objective(beta)
    converged = False
    n_iter = 0
    for n_iter in range(1, max_iter + 1):  # noqa: B007 - n_iter is reported
        p = expit(X @ beta)
        hessian = (X.T * (p * (1.0 - p))) @ X + penalty
        step = np.linalg.solve(hessian, X.T @ (y - p) - lam * beta)
        scale = 1.0
        while True:
            candidate = beta + scale * step
            value = objective(candidate)
            if value >= current - 1e-12 * abs(current) or scale < 1e-10:
                break
            scale /= 2.0
        beta, current = candidate, value
        if float(np.max(np.abs(scale * step))) < tol:
            converged = True
            break
    eta = X @ beta
    p = expit(eta)
    hessian = (X.T * (p * (1.0 - p))) @ X + penalty
    std_error = np.sqrt(np.diag(np.linalg.inv(hessian)))
    margin = (2.0 * y - 1.0) * eta
    separated = bool(np.all(margin > 0.0) or np.any(np.abs(y - p) < EXTREME_FIT))
    return LogisticFit(
        coef=beta,
        std_error=std_error,
        log_likelihood=log_likelihood(X, y, beta),
        penalised_log_likelihood=current,
        n_obs=X.shape[0],
        n_iter=n_iter,
        converged=converged,
        separated=separated,
    )


def likelihood_ratio(
    X: FloatArray,
    y: FloatArray,
    column: int,
    *,
    full: LogisticFit | None = None,
    ridge: float | FloatArray = 1e-8,
) -> tuple[float, float]:
    """Likelihood-ratio test that coefficient ``column`` is zero: ``(statistic, p_value)``.

    The statistic ``2 * (l_full - l_reduced)``, on the penalised objective
    (identical to the plain log-likelihood for a tiny ridge), is compared with
    a chi-square distribution with one degree of freedom. A penalty shrinks
    the tested coefficient towards zero, so the test errs on the side of
    not rejecting. Pass the same ``ridge`` that produced ``full``.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    lam = _ridge_vector(ridge, X.shape[1])
    full = full or fit_logistic(X, y, ridge=lam)
    reduced_columns = np.delete(X, column, axis=1)
    if reduced_columns.shape[1] == 0:
        reduced = log_likelihood(reduced_columns, y, np.zeros(0))
    else:
        reduced_fit = fit_logistic(reduced_columns, y, ridge=np.delete(lam, column))
        reduced = reduced_fit.penalised_log_likelihood
    statistic = max(0.0, 2.0 * (full.penalised_log_likelihood - reduced))
    return statistic, chi2_1_sf(statistic)


def log_odds_ratio(table: tuple[tuple[int, int], tuple[int, int]]) -> float:
    """``log(a d / (b c))`` of a 2x2 table ``((a, b), (c, d))`` with no empty cell."""
    (a, b), (c, d) = table
    return math.log(a * d / (b * c))
