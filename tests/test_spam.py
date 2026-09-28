"""Gold accuracy, Dawid-Skene EM and the spammer score."""

import numpy as np
import pytest

from prefpairs.quality.spam import (
    annotator_reliability,
    dawid_skene,
    gold_accuracy,
    spammer_score,
)
from prefpairs.schema import GoldPair, PairKind
from prefpairs.simulate import Archetype, SimulationConfig, simulate
from tests.quality_helpers import judgment

GOLD = GoldPair(id="g1", prompt_id="p1", better_response_id="p1-b", worse_response_id="p1-a")


class TestGoldAccuracy:
    def test_counts_ties_as_misses_and_leaves_skips_out(self) -> None:
        judgments = [
            judgment("x", "p1-a", "p1-b", "right", kind=PairKind.GOLD),
            judgment("x", "p1-b", "p1-a", "left", kind=PairKind.GOLD),
            judgment("x", "p1-b", "p1-a", "right", kind=PairKind.GOLD),
            judgment("x", "p1-a", "p1-b", "tie", kind=PairKind.GOLD),
            judgment("x", "p1-a", "p1-b", "skip", kind=PairKind.GOLD),
            judgment("x", "p1-a", "p1-b", "left"),
        ]
        report = gold_accuracy(judgments, [GOLD])
        (result,) = report.results
        assert (result.n_gold, result.n_answered, result.n_correct) == (5, 4, 2)
        assert (result.n_tie, result.n_skip) == (1, 1)
        assert result.accuracy == 0.5
        assert result.ci_lower < 0.5 < result.ci_upper
        assert not result.flagged
        assert report.n_gold_pairs == 1

    def test_flags_when_the_upper_bound_is_below_the_minimum(self) -> None:
        wrong = [judgment("x", "p1-a", "p1-b", "left", kind=PairKind.GOLD) for _ in range(12)]
        right = [judgment("y", "p1-a", "p1-b", "right", kind=PairKind.GOLD) for _ in range(12)]
        report = gold_accuracy([*wrong, *right], [GOLD])
        by_id = {r.annotator_id: r for r in report.results}
        assert by_id["x"].accuracy == 0.0
        assert by_id["x"].ci_upper == pytest.approx(0.2425, abs=1e-4)
        assert by_id["y"].ci_lower == pytest.approx(0.7575, abs=1e-4)
        assert report.flagged == ("x",)

    def test_all_skips_are_never_flagged(self) -> None:
        skips = [judgment("x", "p1-a", "p1-b", "skip", kind=PairKind.GOLD) for _ in range(3)]
        (result,) = gold_accuracy(skips, [GOLD]).results
        assert result.accuracy is None
        assert not result.flagged

    def test_unknown_gold_pair_is_an_error(self) -> None:
        with pytest.raises(KeyError):
            gold_accuracy([judgment("x", "p1-a", "p1-c", "left", kind=PairKind.GOLD)], [GOLD])


def _planted(seed: int, accuracies: list[float], n_items: int = 400) -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(seed)
    truth = (rng.random(n_items) < 0.3).astype(np.int64)
    items, annotators, labels = [], [], []
    for a, accuracy in enumerate(accuracies):
        correct = rng.random(n_items) < accuracy
        items.extend(range(n_items))
        annotators.extend([a] * n_items)
        labels.extend(np.where(correct, truth, 1 - truth).tolist())
    return truth, np.array(items), np.array(annotators), np.array(labels)


class TestDawidSkene:
    def test_recovers_planted_accuracies_and_truth(self) -> None:
        accuracies = [0.95, 0.9, 0.8, 0.7, 0.5, 0.2]
        truth, items, annotators, labels = _planted(0, accuracies)
        fit = dawid_skene(
            items, annotators, labels, n_items=400, n_annotators=6, n_classes=2, symmetric=True
        )
        estimated = fit.confusion[:, 0, 0]
        assert np.abs(estimated - accuracies).max() < 0.05
        assert fit.converged
        assert np.mean(fit.posteriors.argmax(axis=1) == truth) > 0.97
        assert fit.priors[1] == pytest.approx(0.3, abs=0.05)

    @pytest.mark.parametrize("symmetric", [False, True])
    @pytest.mark.parametrize("smoothing", [0.0, 0.5])
    def test_objective_never_decreases(self, symmetric: bool, smoothing: float) -> None:
        _, items, annotators, labels = _planted(1, [0.85, 0.75, 0.6, 0.55, 0.3], n_items=150)
        fit = dawid_skene(
            items,
            annotators,
            labels,
            n_items=150,
            n_annotators=5,
            n_classes=2,
            symmetric=symmetric,
            smoothing=smoothing,
        )
        steps = np.diff(fit.objective)
        assert len(fit.objective) > 3
        assert np.all(steps >= -1e-9 * np.abs(np.array(fit.objective[1:])))
        assert fit.log_likelihood == fit.objective[-1]
        np.testing.assert_allclose(fit.confusion.sum(axis=2), 1.0)
        np.testing.assert_allclose(fit.posteriors.sum(axis=1), 1.0)

    def test_full_confusion_recovers_asymmetric_annotator(self) -> None:
        rng = np.random.default_rng(2)
        truth = (rng.random(600) < 0.5).astype(np.int64)
        items, annotators, labels = [], [], []
        for a in range(4):
            correct = rng.random(600) < 0.9
            items.extend(range(600))
            annotators.extend([a] * 600)
            labels.extend(np.where(correct, truth, 1 - truth).tolist())
        # Annotator 4 says class 0 whenever the truth is 0, and a coin flip otherwise.
        items.extend(range(600))
        annotators.extend([4] * 600)
        labels.extend(np.where(truth == 0, 0, rng.integers(0, 2, 600)).tolist())
        fit = dawid_skene(
            np.array(items),
            np.array(annotators),
            np.array(labels),
            n_items=600,
            n_annotators=5,
            n_classes=2,
        )
        assert fit.confusion[4, 0, 0] > 0.97
        assert fit.confusion[4, 1, 1] == pytest.approx(0.5, abs=0.07)

    def test_input_validation(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            dawid_skene([0], [0, 1], [0], n_items=1, n_annotators=2, n_classes=2)
        with pytest.raises(ValueError, match="out of range"):
            dawid_skene([0], [0], [2], n_items=1, n_annotators=1, n_classes=2)
        with pytest.raises(ValueError, match="two or more classes"):
            dawid_skene([0], [0], [0], n_items=1, n_annotators=1, n_classes=1)

    def test_iteration_cap_is_reported(self) -> None:
        _, items, annotators, labels = _planted(3, [0.8, 0.7, 0.6], n_items=50)
        fit = dawid_skene(
            items, annotators, labels, n_items=50, n_annotators=3, n_classes=2, max_iter=2
        )
        assert fit.iterations == 2
        assert not fit.converged


class TestSpammerScore:
    def test_two_class_score_is_squared_youden_index(self) -> None:
        sensitivity, specificity = 0.8, 0.7
        matrix = np.array([[sensitivity, 1 - sensitivity], [1 - specificity, specificity]])
        assert spammer_score(matrix) == pytest.approx((sensitivity + specificity - 1) ** 2)

    def test_rank_one_is_zero_and_permutation_is_one(self) -> None:
        assert spammer_score(np.array([[0.3, 0.7], [0.3, 0.7]])) == 0.0
        assert spammer_score(np.eye(3)) == pytest.approx(1.0)
        assert spammer_score(np.eye(3)[::-1]) == pytest.approx(1.0)


class TestAnnotatorReliability:
    def test_simulated_spammer_and_adversary_are_told_apart(self) -> None:
        dataset = simulate(SimulationConfig(seed=0))
        report = annotator_reliability(dataset.pairwise)
        archetypes = dataset.truth.annotator_archetypes
        assert [archetypes[a] for a in report.spammers] == [Archetype.RANDOM_SPAMMER]
        assert [archetypes[a] for a in report.reversed] == [Archetype.ADVERSARIAL]
        assert report.symmetric
        assert report.converged
        for result in report.results:
            if archetypes[result.annotator_id] is Archetype.RELIABLE:
                assert result.accuracy > 0.85
                assert result.spammer_score > 0.4
            assert result.sensitivity == result.specificity

    def test_too_few_labels_are_never_flagged(self) -> None:
        judgments = [judgment(a, "p1-a", "p1-b", "left") for a in ("x", "y", "z")]
        report = annotator_reliability(judgments, min_labels=2, symmetric=False)
        assert report.n_items == 1
        assert report.n_labels == 3
        assert report.spammers == ()
        assert report.reversed == ()
