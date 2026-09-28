"""Spammer and adversarial-annotator detection: gold accuracy and Dawid-Skene.

Two independent views of whether an annotator's labels carry information:

* **Gold accuracy.** Every annotator sees the gold pairs (pairs with an agreed
  better response). Accuracy is the share of non-skip gold judgments that
  pick the agreed response; a tie counts as a miss, because gold pairs are
  chosen to have a clear answer. A Wilson interval bounds it, and an
  annotator is flagged when the *upper* bound is below ``min_gold_accuracy``:
  even the most favourable reading of the evidence says they are wrong too
  often.
* **Dawid-Skene EM.** Treat the true label of every item (a canonical pair on
  a prompt) as a latent class, and every annotator as a confusion matrix
  ``pi[a, k, l] = P(annotator a says l | truth k)``. EM alternates item
  posteriors (E-step) and class priors plus confusion matrices (M-step),
  starting from majority-vote shares. A symmetric Dirichlet pseudo-count
  keeps every estimate inside (0, 1); the objective it maximises is then the
  log-likelihood plus the log prior, which EM never decreases (the tests
  check this on every iteration).

The spammer score of an annotator (Raykar and Yu, 2012) measures how far
their confusion matrix is from rank one, i.e. from labels that do not depend
on the truth: ``S = 1 / (K (K - 1)) * sum_{k < k'} ||pi[k] - pi[k']||^2``.
For two classes it equals ``(sensitivity + specificity - 1)^2``: 0 for a coin
flipper, 1 for a perfect (or perfectly reversed) annotator. Reversal is
reported separately as the estimated accuracy ``sum_k prior_k pi[k, k]``,
which is below 0.5 for an annotator who systematically picks the worse
response.
"""

from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
from pydantic import Field

from prefpairs.quality.agreement import item_labels
from prefpairs.quality.stats import wilson_interval
from prefpairs.schema import CanonicalLabel, GoldPair, PairKind, PairwiseJudgment, Record

type FloatArray = np.ndarray[tuple[int, ...], np.dtype[np.float64]]
type IntArray = np.ndarray[tuple[int, ...], np.dtype[np.int64]]

DEFAULT_MIN_GOLD_ACCURACY = 0.7
DEFAULT_MIN_SPAMMER_SCORE = 0.1
DEFAULT_MIN_ACCURACY = 0.5
DECISIVE_CLASSES = (CanonicalLabel.A_WINS, CanonicalLabel.B_WINS)


# -- gold accuracy ----------------------------------------------------------------


class GoldAccuracy(Record):
    """One annotator's answers on gold pairs."""

    annotator_id: str
    n_gold: int
    """Gold judgments, skips included."""
    n_answered: int
    n_correct: int
    n_tie: int
    n_skip: int
    accuracy: float | None
    ci_lower: float
    ci_upper: float
    flagged: bool


class GoldReport(Record):
    confidence: float = Field(gt=0, lt=1)
    min_accuracy: float = Field(ge=0, le=1)
    n_gold_pairs: int
    results: tuple[GoldAccuracy, ...]

    @property
    def flagged(self) -> tuple[str, ...]:
        return tuple(sorted(r.annotator_id for r in self.results if r.flagged))


def gold_accuracy(
    judgments: Iterable[PairwiseJudgment],
    gold_pairs: Iterable[GoldPair],
    *,
    confidence: float = 0.95,
    min_accuracy: float = DEFAULT_MIN_GOLD_ACCURACY,
) -> GoldReport:
    """Score every gold judgment against the gold pair on the same prompt and responses.

    Raises KeyError for a gold judgment that matches no gold pair.
    """
    expected = {(g.prompt_id, *g.canonical_pair): g.expected_label for g in gold_pairs}
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for j in judgments:
        if j.pair_kind is not PairKind.GOLD:
            continue
        answer = expected[(j.prompt_id, *j.canonical_pair)]
        tally = counts[j.annotator_id]
        tally["gold"] += 1
        label = j.label
        if label is None:
            tally["skip"] += 1
            continue
        tally["answered"] += 1
        tally["tie"] += label is CanonicalLabel.TIE
        tally["correct"] += label is answer
    results = []
    for annotator in sorted(counts):
        t = counts[annotator]
        lower, upper = wilson_interval(t["correct"], t["answered"], confidence)
        results.append(
            GoldAccuracy(
                annotator_id=annotator,
                n_gold=t["gold"],
                n_answered=t["answered"],
                n_correct=t["correct"],
                n_tie=t["tie"],
                n_skip=t["skip"],
                accuracy=t["correct"] / t["answered"] if t["answered"] else None,
                ci_lower=lower,
                ci_upper=upper,
                flagged=t["answered"] > 0 and upper < min_accuracy,
            )
        )
    return GoldReport(
        confidence=confidence,
        min_accuracy=min_accuracy,
        n_gold_pairs=len(expected),
        results=tuple(results),
    )


# -- Dawid-Skene EM ---------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DawidSkeneFit:
    """Estimated class priors, confusion matrices and item posteriors."""

    priors: FloatArray
    """``(K,)`` class probabilities."""
    confusion: FloatArray
    """``(annotators, K, K)``: row k is the label distribution given true class k."""
    posteriors: FloatArray
    """``(items, K)`` probability of each true class per item."""
    objective: tuple[float, ...]
    """Log-likelihood plus log Dirichlet prior after every E-step."""
    iterations: int
    converged: bool

    @property
    def log_likelihood(self) -> float:
        return self.objective[-1]


def _majority_init(items: IntArray, labels: IntArray, n_items: int, n_classes: int) -> FloatArray:
    votes = np.zeros((n_items, n_classes))
    np.add.at(votes, (items, labels), 1.0)
    totals = votes.sum(axis=1, keepdims=True)
    uniform = np.full_like(votes, 1.0 / n_classes)
    shares: FloatArray = np.divide(votes, totals, out=uniform, where=totals > 0)
    return shares


def _m_step(
    item_posteriors: FloatArray,
    label_onehot: FloatArray,
    annotator_onehot: FloatArray,
    *,
    symmetric: bool,
    smoothing: float,
) -> tuple[FloatArray, FloatArray]:
    """Confusion matrices given each observation's item posterior, plus the log prior terms."""
    tiny = np.finfo(np.float64).tiny
    n_annotators = annotator_onehot.shape[1]
    n_classes = label_onehot.shape[1]
    if not symmetric:
        counts = np.einsum("na,nk,nl->akl", annotator_onehot, item_posteriors, label_onehot)
        counts += smoothing
        confusion: FloatArray = counts / counts.sum(axis=2, keepdims=True)
        return confusion, np.log(np.maximum(confusion, tiny))
    correct = annotator_onehot.T @ np.sum(item_posteriors * label_onehot, axis=1)
    total = annotator_onehot.sum(axis=0) + 2.0 * smoothing
    accuracy = np.divide(
        correct + smoothing, total, out=np.full(n_annotators, 0.5), where=total > 0
    )
    off = (1.0 - accuracy) / (n_classes - 1)
    confusion = np.tile(off[:, None, None], (1, n_classes, n_classes))
    diagonal = np.arange(n_classes)
    confusion[:, diagonal, diagonal] = accuracy[:, None]
    penalty = np.log(np.maximum(accuracy, tiny)) + np.log(np.maximum(1.0 - accuracy, tiny))
    return confusion, penalty


def dawid_skene(
    items: Sequence[int] | IntArray,
    annotators: Sequence[int] | IntArray,
    labels: Sequence[int] | IntArray,
    *,
    n_items: int,
    n_annotators: int,
    n_classes: int,
    symmetric: bool = False,
    smoothing: float = 0.5,
    max_iter: int = 500,
    tol: float = 1e-9,
) -> DawidSkeneFit:
    """Fit the Dawid-Skene model to observations ``(item, annotator, label)``.

    With ``symmetric`` each annotator has a single accuracy ``p`` (the
    "one-coin" model): ``P(label = truth) = p`` and every wrong label has
    probability ``(1 - p) / (K - 1)``. ``smoothing`` is the pseudo-count added
    to every class prior and confusion cell (to the correct and wrong counts in
    the symmetric model); 0 gives plain maximum-likelihood EM.
    """
    item = np.asarray(items, dtype=np.int64)
    annotator = np.asarray(annotators, dtype=np.int64)
    label = np.asarray(labels, dtype=np.int64)
    if not item.shape == annotator.shape == label.shape or item.ndim != 1:
        msg = "items, annotators and labels must be 1-d arrays of the same length"
        raise ValueError(msg)
    bounds = ((item, n_items), (annotator, n_annotators), (label, n_classes))
    if any(values.size and (values.min() < 0 or values.max() >= n) for values, n in bounds):
        msg = "an index is out of range"
        raise ValueError(msg)
    if n_classes < 2 or smoothing < 0 or max_iter < 1:
        msg = "need two or more classes, a non-negative smoothing and max_iter >= 1"
        raise ValueError(msg)
    tiny = np.finfo(np.float64).tiny
    label_onehot = np.eye(n_classes)[label]
    annotator_onehot = np.eye(n_annotators)[annotator]
    posteriors = _majority_init(item, label, n_items, n_classes)
    objective: list[float] = []
    converged = False
    iterations = 0
    for iterations in range(1, max_iter + 1):  # noqa: B007 - reported after the loop
        # M-step: class priors and confusion matrices from the current posteriors.
        priors = (posteriors.sum(axis=0) + smoothing) / (n_items + n_classes * smoothing)
        confusion, penalty = _m_step(
            posteriors[item],
            label_onehot,
            annotator_onehot,
            symmetric=symmetric,
            smoothing=smoothing,
        )
        # E-step: item posteriors given the parameters.
        log_confusion = np.log(np.maximum(confusion, tiny))
        log_joint = np.tile(np.log(np.maximum(priors, tiny)), (n_items, 1))
        np.add.at(log_joint, item, log_confusion[annotator, :, label])
        top = log_joint.max(axis=1, keepdims=True)
        log_norm = top + np.log(np.exp(log_joint - top).sum(axis=1, keepdims=True))
        posteriors = np.exp(log_joint - log_norm)
        value = float(log_norm.sum())
        if smoothing:
            value += smoothing * float(np.log(np.maximum(priors, tiny)).sum())
            value += smoothing * float(penalty.sum())
        objective.append(value)
        if len(objective) > 1 and abs(objective[-1] - objective[-2]) <= tol * (
            1.0 + abs(objective[-1])
        ):
            converged = True
            break
    return DawidSkeneFit(
        priors=priors,
        confusion=confusion,
        posteriors=posteriors,
        objective=tuple(objective),
        iterations=iterations,
        converged=converged,
    )


def spammer_score(confusion: FloatArray) -> float:
    """Raykar-Yu spammer score of one ``(K, K)`` confusion matrix, in [0, 1]."""
    k = confusion.shape[0]
    total = sum(
        float(np.sum((confusion[i] - confusion[j]) ** 2)) for i in range(k) for j in range(i + 1, k)
    )
    return total / (k * (k - 1))


class AnnotatorReliability(Record):
    """Dawid-Skene estimates for one annotator."""

    annotator_id: str
    n_labels: int
    accuracy: float
    """``sum_k prior_k * P(label k | truth k)``: below 0.5 means reversed."""
    sensitivity: float
    """P(says a_wins | a is better)."""
    specificity: float
    """P(says b_wins | b is better)."""
    spammer_score: float
    spammer: bool
    reversed: bool


class DawidSkeneReport(Record):
    pair_kinds: tuple[PairKind, ...]
    symmetric: bool
    smoothing: float
    min_labels: int
    min_spammer_score: float
    min_accuracy: float
    n_items: int
    n_labels: int
    iterations: int
    converged: bool
    log_likelihood: float
    prior_a_wins: float
    results: tuple[AnnotatorReliability, ...]

    @property
    def spammers(self) -> tuple[str, ...]:
        return tuple(sorted(r.annotator_id for r in self.results if r.spammer))

    @property
    def reversed(self) -> tuple[str, ...]:
        return tuple(sorted(r.annotator_id for r in self.results if r.reversed))


def annotator_reliability(
    judgments: Iterable[PairwiseJudgment],
    *,
    pair_kinds: Sequence[PairKind] = (PairKind.GOLD, PairKind.REGULAR),
    symmetric: bool = True,
    smoothing: float = 0.5,
    min_labels: int = 20,
    min_spammer_score: float = DEFAULT_MIN_SPAMMER_SCORE,
    min_accuracy: float = DEFAULT_MIN_ACCURACY,
) -> DawidSkeneReport:
    """Binary Dawid-Skene over decisive canonical labels (ties and skips left out).

    An annotator with at least ``min_labels`` labels is a *spammer* when the
    spammer score is below ``min_spammer_score`` (labels nearly independent of
    the truth), and *reversed* when the score is at least that but the
    estimated accuracy is below ``min_accuracy`` (informative, but backwards).
    The symmetric one-coin model is the default because the canonical
    orientation of a pair is arbitrary: with a full confusion matrix, a strong
    model sorting first by id leaves few items of the minority class, and
    their row is estimated from a handful of labels.
    """
    kinds = tuple(sorted(set(pair_kinds)))
    labels = item_labels(judgments, kinds)
    names = sorted({a for by_annotator in labels.values() for a in by_annotator})
    index = {a: i for i, a in enumerate(names)}
    classes = {c: i for i, c in enumerate(DECISIVE_CLASSES)}
    rows = [
        (n, index[a], classes[label])
        for n, key in enumerate(sorted(labels))
        for a, label in sorted(labels[key].items())
        if label in classes
    ]
    observed = np.array(rows, dtype=np.int64).reshape(-1, 3)
    fit = dawid_skene(
        observed[:, 0],
        observed[:, 1],
        observed[:, 2],
        n_items=len(labels),
        n_annotators=len(names),
        n_classes=len(DECISIVE_CLASSES),
        symmetric=symmetric,
        smoothing=smoothing,
    )
    per_annotator = Counter(int(a) for a in observed[:, 1])
    results = []
    for i, name in enumerate(names):
        matrix = fit.confusion[i]
        accuracy = float(np.sum(fit.priors * np.diag(matrix)))
        score = spammer_score(matrix)
        enough = per_annotator[i] >= min_labels
        results.append(
            AnnotatorReliability(
                annotator_id=name,
                n_labels=per_annotator[i],
                accuracy=accuracy,
                sensitivity=float(matrix[0, 0]),
                specificity=float(matrix[1, 1]),
                spammer_score=score,
                spammer=enough and score < min_spammer_score,
                reversed=enough and accuracy < min_accuracy and score >= min_spammer_score,
            )
        )
    return DawidSkeneReport(
        pair_kinds=kinds,
        symmetric=symmetric,
        smoothing=smoothing,
        min_labels=min_labels,
        min_spammer_score=min_spammer_score,
        min_accuracy=min_accuracy,
        n_items=len(labels),
        n_labels=len(rows),
        iterations=fit.iterations,
        converged=fit.converged,
        log_likelihood=fit.log_likelihood,
        prior_a_wins=float(fit.priors[0]),
        results=tuple(results),
    )
