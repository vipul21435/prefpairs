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
from prefpairs.quality.length import LengthBiasReport, LengthBiasResult, LengthUnit, length_bias
from prefpairs.quality.logistic import LogisticFit, fit_logistic, likelihood_ratio
from prefpairs.quality.position import PositionBiasReport, PositionBiasResult, position_bias
from prefpairs.quality.spam import (
    AnnotatorReliability,
    DawidSkeneFit,
    DawidSkeneReport,
    GoldAccuracy,
    GoldReport,
    annotator_reliability,
    dawid_skene,
    gold_accuracy,
    spammer_score,
)
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
    "AnnotatorReliability",
    "ConsistencyReport",
    "ConsistencyResult",
    "DawidSkeneFit",
    "DawidSkeneReport",
    "FleissAgreement",
    "GoldAccuracy",
    "GoldReport",
    "Kappa",
    "LengthBiasReport",
    "LengthBiasResult",
    "LengthUnit",
    "LogisticFit",
    "PairAgreement",
    "PositionBiasReport",
    "PositionBiasResult",
    "TransitivityReport",
    "TransitivityResult",
    "agreement",
    "annotator_reliability",
    "binom_test",
    "cohen_kappa",
    "count_cycles",
    "dawid_skene",
    "fit_logistic",
    "fleiss_kappa",
    "gold_accuracy",
    "holm",
    "length_bias",
    "likelihood_ratio",
    "position_bias",
    "self_consistency",
    "spammer_score",
    "strongly_connected_components",
    "transitivity",
    "wilson_interval",
]
