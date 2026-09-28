"""Annotator quality checks: each returns a typed result with its evidence."""

from prefpairs.quality.agreement import (
    AgreementReport,
    AnnotatorAgreement,
    FleissAgreement,
    Kappa,
    PairAgreement,
    agreement,
    cohen_kappa,
    fleiss_kappa,
)
from prefpairs.quality.consistency import ConsistencyReport, ConsistencyResult, self_consistency
from prefpairs.quality.length import LengthBiasReport, LengthBiasResult, length_bias
from prefpairs.quality.logistic import LogisticFit, fit_logistic, likelihood_ratio
from prefpairs.quality.position import PositionBiasReport, PositionBiasResult, position_bias
from prefpairs.quality.stats import binom_test, holm, wilson_interval
from prefpairs.quality.transitivity import (
    TransitivityReport,
    TransitivityResult,
    count_cycles,
    strongly_connected_components,
    transitivity,
)

__all__ = [
    "AgreementReport",
    "AnnotatorAgreement",
    "ConsistencyReport",
    "ConsistencyResult",
    "FleissAgreement",
    "Kappa",
    "LengthBiasReport",
    "LengthBiasResult",
    "LogisticFit",
    "PairAgreement",
    "PositionBiasReport",
    "PositionBiasResult",
    "TransitivityReport",
    "TransitivityResult",
    "agreement",
    "binom_test",
    "cohen_kappa",
    "count_cycles",
    "fit_logistic",
    "fleiss_kappa",
    "holm",
    "length_bias",
    "likelihood_ratio",
    "position_bias",
    "self_consistency",
    "strongly_connected_components",
    "transitivity",
    "wilson_interval",
]
