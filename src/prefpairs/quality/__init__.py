"""Annotator quality checks: each returns a typed result with its evidence."""

from prefpairs.quality.length import LengthBiasReport, LengthBiasResult, length_bias
from prefpairs.quality.logistic import LogisticFit, fit_logistic, likelihood_ratio
from prefpairs.quality.position import PositionBiasReport, PositionBiasResult, position_bias
from prefpairs.quality.stats import binom_test, holm, wilson_interval

__all__ = [
    "LengthBiasReport",
    "LengthBiasResult",
    "LogisticFit",
    "PositionBiasReport",
    "PositionBiasResult",
    "binom_test",
    "fit_logistic",
    "holm",
    "length_bias",
    "likelihood_ratio",
    "position_bias",
    "wilson_interval",
]
