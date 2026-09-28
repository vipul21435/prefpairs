"""Inter-annotator agreement on canonical labels: Cohen's and Fleiss' kappa.

Agreement is measured on what a judgment *means*, not on what was clicked:
each non-skip judgment becomes a canonical label on the id-sorted pair
(``a_wins``, ``b_wins`` or ``tie``), so two annotators who saw the pair in
opposite orders and preferred the same response agree. Ties are a third
category; skips carry no label and are left out.

* Cohen's kappa compares two annotators on the items both labelled:
  ``(p_o - p_e) / (1 - p_e)`` with ``p_o`` the share of identical labels and
  ``p_e`` the agreement expected from each annotator's own label frequencies.
* Fleiss' kappa generalises this to many raters, but needs the same number of
  raters ``m`` on every item. Items with a different number of labels are
  excluded and counted, and ``m`` defaults to the most common count.

Kappa is undefined (None) when chance agreement is already perfect, for
example when every label in the sample is the same.
"""

from collections import Counter, defaultdict
from collections.abc import Hashable, Iterable, Sequence
from itertools import combinations

import numpy as np
from numpy.typing import ArrayLike

from prefpairs.schema import CanonicalLabel, PairKind, PairwiseJudgment, Record

DEFAULT_PAIR_KINDS = (PairKind.GOLD, PairKind.REGULAR)
"""Controls are left out: they repeat a judgment the annotator already made."""


class Kappa(Record):
    """Observed agreement, chance agreement and kappa over ``n_items`` items."""

    n_items: int
    observed: float
    expected: float
    kappa: float | None


def _kappa(observed: float, expected: float) -> float | None:
    if expected >= 1.0:
        return None
    return (observed - expected) / (1.0 - expected)


def cohen_kappa(first: Sequence[Hashable], second: Sequence[Hashable]) -> Kappa:
    """Cohen's kappa of two raters' labels on the same items, in the same order."""
    n = len(first)
    if n != len(second):
        msg = f"both raters need a label for every item, got {n} and {len(second)}"
        raise ValueError(msg)
    if n == 0:
        msg = "need at least one item"
        raise ValueError(msg)
    observed = sum(x == y for x, y in zip(first, second, strict=True)) / n
    count_first, count_second = Counter(first), Counter(second)
    expected = sum(count_first[c] * count_second[c] for c in count_first) / (n * n)
    return Kappa(n_items=n, observed=observed, expected=expected, kappa=_kappa(observed, expected))


def fleiss_kappa(counts: ArrayLike) -> Kappa:
    """Fleiss' kappa from an ``(items, categories)`` table of label counts.

    Every row must sum to the same number of raters ``m >= 2``.
    """
    table = np.asarray(counts, dtype=np.float64)
    if table.ndim != 2 or table.shape[0] == 0 or table.shape[1] == 0:
        msg = f"need a non-empty (items, categories) table, got shape {table.shape}"
        raise ValueError(msg)
    if np.any(table < 0):
        msg = "counts must be non-negative"
        raise ValueError(msg)
    raters = table.sum(axis=1)
    m = float(raters[0])
    if m < 2 or not np.all(raters == m):
        msg = "every item needs the same number of raters, at least 2"
        raise ValueError(msg)
    n_items = table.shape[0]
    per_item = (np.sum(table * table, axis=1) - m) / (m * (m - 1.0))
    observed = float(per_item.mean())
    shares = table.sum(axis=0) / (n_items * m)
    expected = float(np.sum(shares * shares))
    return Kappa(
        n_items=n_items, observed=observed, expected=expected, kappa=_kappa(observed, expected)
    )


class PairAgreement(Record):
    """Cohen's kappa of two annotators on the items both labelled."""

    annotator_a: str
    annotator_b: str
    n_items: int
    observed: float
    expected: float
    kappa: float | None


class AnnotatorAgreement(Record):
    """How one annotator agrees with everyone they share items with."""

    annotator_id: str
    n_partners: int
    n_shared_items: int
    """Items shared with partners, summed over partners."""
    mean_kappa: float | None
    """Cohen's kappa averaged over partners, weighted by shared items."""


class FleissAgreement(Record):
    """Fleiss' kappa over the items that have exactly ``n_raters`` labels."""

    n_raters: int
    n_items: int
    n_items_excluded: int
    """Items labelled by a different number of annotators."""
    observed: float
    expected: float
    kappa: float | None


class AgreementReport(Record):
    pair_kinds: tuple[PairKind, ...]
    min_shared: int
    n_items: int
    """Items with at least one label."""
    pairs: tuple[PairAgreement, ...]
    annotators: tuple[AnnotatorAgreement, ...]
    fleiss: FleissAgreement | None


type ItemKey = tuple[str, str, str]


def item_labels(
    judgments: Iterable[PairwiseJudgment], pair_kinds: Sequence[PairKind] = DEFAULT_PAIR_KINDS
) -> dict[ItemKey, dict[str, CanonicalLabel]]:
    """``{(prompt, a, b): {annotator: label}}``, keeping each annotator's first label."""
    kinds = set(pair_kinds)
    labels: dict[ItemKey, dict[str, CanonicalLabel]] = defaultdict(dict)
    for j in judgments:
        label = j.label
        if j.pair_kind not in kinds or label is None:
            continue
        a, b = j.canonical_pair
        labels[(j.prompt_id, a, b)].setdefault(j.annotator_id, label)
    return dict(labels)


def _fleiss(
    labels: dict[ItemKey, dict[str, CanonicalLabel]], n_raters: int | None
) -> FleissAgreement | None:
    sizes = Counter(len(by_annotator) for by_annotator in labels.values())
    candidates = {size: count for size, count in sizes.items() if size >= 2}
    if n_raters is None:
        if not candidates:
            return None
        n_raters = max(candidates, key=lambda size: (candidates[size], size))
    rows = [
        [Counter(by_annotator.values())[c] for c in CanonicalLabel]
        for by_annotator in labels.values()
        if len(by_annotator) == n_raters
    ]
    if not rows or n_raters < 2:
        return None
    kappa = fleiss_kappa(rows)
    return FleissAgreement(
        n_raters=n_raters,
        n_items=kappa.n_items,
        n_items_excluded=len(labels) - kappa.n_items,
        observed=kappa.observed,
        expected=kappa.expected,
        kappa=kappa.kappa,
    )


def agreement(
    judgments: Iterable[PairwiseJudgment],
    *,
    pair_kinds: Sequence[PairKind] = DEFAULT_PAIR_KINDS,
    min_shared: int = 1,
    n_raters: int | None = None,
) -> AgreementReport:
    """Pairwise Cohen's kappa, a per-annotator summary and Fleiss' kappa."""
    kinds = tuple(sorted(set(pair_kinds)))
    labels = item_labels(judgments, kinds)
    shared: dict[tuple[str, str], list[ItemKey]] = defaultdict(list)
    for key, by_annotator in labels.items():
        for pair in combinations(sorted(by_annotator), 2):
            shared[pair].append(key)
    pairs = []
    for (first, second), keys in sorted(shared.items()):
        if len(keys) < min_shared:
            continue
        kappa = cohen_kappa([labels[k][first] for k in keys], [labels[k][second] for k in keys])
        pairs.append(
            PairAgreement(
                annotator_a=first,
                annotator_b=second,
                n_items=kappa.n_items,
                observed=kappa.observed,
                expected=kappa.expected,
                kappa=kappa.kappa,
            )
        )
    annotators = sorted({a for by_annotator in labels.values() for a in by_annotator})
    summaries = []
    for annotator in annotators:
        mine = [p for p in pairs if annotator in (p.annotator_a, p.annotator_b)]
        defined = [p for p in mine if p.kappa is not None]
        weight = sum(p.n_items for p in defined)
        mean = sum(p.n_items * (p.kappa or 0.0) for p in defined) / weight if weight else None
        summaries.append(
            AnnotatorAgreement(
                annotator_id=annotator,
                n_partners=len(mine),
                n_shared_items=sum(p.n_items for p in mine),
                mean_kappa=mean,
            )
        )
    return AgreementReport(
        pair_kinds=kinds,
        min_shared=min_shared,
        n_items=len(labels),
        pairs=tuple(pairs),
        annotators=tuple(summaries),
        fleiss=_fleiss(labels, n_raters),
    )
