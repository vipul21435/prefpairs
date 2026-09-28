"""Annotator quality checks: each returns a typed result with its evidence."""

from prefpairs.quality.position import PositionBiasReport, PositionBiasResult, position_bias
from prefpairs.quality.stats import binom_test, holm, wilson_interval

__all__ = [
    "PositionBiasReport",
    "PositionBiasResult",
    "binom_test",
    "holm",
    "position_bias",
    "wilson_interval",
]
