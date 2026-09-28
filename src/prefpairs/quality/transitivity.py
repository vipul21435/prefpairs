"""Transitivity of each annotator's preferences.

A careful annotator's preferences are close to a linear order: if they prefer
A to B and B to C, they rarely prefer C to A. The check turns each
annotator's decisive judgments into a *preference digraph* and measures how
far it is from acyclic.

* Nodes are models (``level=model``, the default) or individual responses
  (``level=response``). Responses belong to one prompt, so the response-level
  graph is the disjoint union of the annotator's per-prompt graphs. At model
  level every prompt contributes, which is what makes the check usable when an
  annotator sees only one or two pairs per prompt.
* For every pair of nodes the annotator compared, the edge points from the
  node they preferred more often to the other one; a net zero (including
  ties) leaves no edge. So between two nodes there is at most one edge.
* Strongly connected components are found with an iterative Tarjan
  algorithm: the graph has a cycle exactly when some component has two or
  more nodes.
* A *complete triad* is three nodes with an edge between each pair; it is
  *intransitive* when the three edges form a directed 3-cycle. If every edge
  were oriented by a coin flip, each complete triad would be intransitive
  with probability exactly 1/4 (2 of the 8 orientations are cycles). The
  p-value is the share of seeded random re-orientations of the annotator's
  own edges that produce at most as many 3-cycles as observed: small means
  clearly more transitive than chance.

An annotator is flagged when they have at least ``min_triads`` complete
triads and the p-value exceeds ``chance_level`` (default 0.5): their graph
has at least as many 3-cycles as the median coin flipper with the same
comparisons. This is deliberately not a significance test of "random": with
six models there are only 20 triads, so even a perfectly transitive
annotator cannot reach a small p-value when a few pairs are missing, and a
"not significantly transitive" rule would flag careful annotators.
"""

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence

import numpy as np
from pydantic import Field

from prefpairs.aggregate.comparisons import Level
from prefpairs.schema import PairKind, PairwiseJudgment, Record, Response

RANDOM_TRIAD_RATE = 0.25
"""Probability that a complete triad is a 3-cycle when each edge is a coin flip."""

DEFAULT_PAIR_KINDS = (PairKind.GOLD, PairKind.REGULAR)
"""Controls are left out: they repeat a judgment the annotator already made."""

type Graph = dict[str, set[str]]


def _successors(graph: Mapping[str, Iterable[str]]) -> dict[str, list[str]]:
    """Sorted successor lists, with a (possibly empty) entry for every node."""
    successors: dict[str, list[str]] = defaultdict(list)
    for node, targets in graph.items():
        successors[node].extend(sorted(targets))
        for target in targets:
            successors.setdefault(target, [])
    return dict(successors)


def strongly_connected_components(graph: Mapping[str, Iterable[str]]) -> list[tuple[str, ...]]:
    """Tarjan's algorithm without recursion; components in reverse topological order.

    Every node that appears as a key or as a successor is placed in exactly
    one component, and the nodes inside a component are sorted.
    """
    successors = _successors(graph)
    index: dict[str, int] = {}
    low: dict[str, int] = {}
    stack: list[str] = []
    on_stack: set[str] = set()
    components: list[tuple[str, ...]] = []
    counter = 0
    for root in sorted(successors):
        if root in index:
            continue
        work: list[tuple[str, int]] = [(root, 0)]
        while work:
            node, position = work.pop()
            if position == 0:
                index[node] = low[node] = counter
                counter += 1
                stack.append(node)
                on_stack.add(node)
            targets = successors[node]
            if position < len(targets):
                work.append((node, position + 1))
                target = targets[position]
                if target not in index:
                    work.append((target, 0))
                elif target in on_stack:
                    low[node] = min(low[node], index[target])
                continue
            if low[node] == index[node]:
                members = []
                while True:
                    member = stack.pop()
                    on_stack.discard(member)
                    members.append(member)
                    if member == node:
                        break
                components.append(tuple(sorted(members)))
            if work:
                parent = work[-1][0]
                low[parent] = min(low[parent], low[node])
    return components


def _edge_signs(graph: Mapping[str, Iterable[str]]) -> dict[tuple[str, str], int]:
    """``{(u, v): +1 if u -> v else -1}`` for ``u < v``; rejects two-way edges."""
    signs: dict[tuple[str, str], int] = {}
    for source, targets in graph.items():
        for target in targets:
            if source == target:
                msg = f"self-loop on {source!r}"
                raise ValueError(msg)
            key = (source, target) if source < target else (target, source)
            sign = 1 if source < target else -1
            if signs.setdefault(key, sign) != sign:
                msg = f"both {source!r} -> {target!r} and the reverse edge are present"
                raise ValueError(msg)
    return signs


def complete_triads(signs: Mapping[tuple[str, str], int]) -> list[tuple[int, int, int]]:
    """Signs ``(s_uv, s_vw, s_uw)`` of every triad ``u < v < w`` with all three edges."""
    neighbours: dict[str, set[str]] = defaultdict(set)
    for u, v in signs:
        neighbours[u].add(v)
        neighbours[v].add(u)
    triads = []
    for u, v in sorted(signs):
        for w in sorted(neighbours[u] & neighbours[v]):
            if w > v:
                triads.append((signs[(u, v)], signs[(v, w)], signs[(u, w)]))
    return triads


def _is_cycle(s_uv: int, s_vw: int, s_uw: int) -> bool:
    """u -> v -> w -> u, or the reverse, in terms of the sorted-pair signs."""
    return s_uv == s_vw == -s_uw


def count_cycles(graph: Mapping[str, Iterable[str]]) -> tuple[int, int]:
    """``(complete triads, directed 3-cycles)`` of a graph with at most one edge per pair."""
    triads = complete_triads(_edge_signs(graph))
    return len(triads), sum(_is_cycle(*t) for t in triads)


def random_cycle_p_value(
    graph: Mapping[str, Iterable[str]], observed: int, *, n_draws: int, seed: int
) -> float:
    """Monte Carlo ``P(cycles <= observed)`` when every edge of ``graph`` is re-oriented at random.

    Uses the add-one estimator ``(1 + hits) / (1 + n_draws)``, which never
    reports an impossible p-value of zero.
    """
    signs = _edge_signs(graph)
    edges = sorted(signs)
    position = {edge: i for i, edge in enumerate(edges)}
    neighbours: dict[str, set[str]] = defaultdict(set)
    for u, v in edges:
        neighbours[u].add(v)
        neighbours[v].add(u)
    triads = [
        (position[(u, v)], position[(v, w)], position[(u, w)])
        for u, v in edges
        for w in sorted(neighbours[u] & neighbours[v])
        if w > v
    ]
    if not triads:
        return 1.0
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, 2, size=(n_draws, len(edges)), dtype=np.int8) * 2 - 1
    index = np.array(triads, dtype=np.int64)
    uv, vw, uw = draws[:, index[:, 0]], draws[:, index[:, 1]], draws[:, index[:, 2]]
    cycles = ((uv == vw) & (uw == -uv)).sum(axis=1)
    hits = int((cycles <= observed).sum())
    return (1 + hits) / (1 + n_draws)


class TransitivityResult(Record):
    """Cycle structure of one annotator's preference digraph."""

    annotator_id: str
    n_nodes: int
    n_edges: int
    n_level_pairs: int
    """Node pairs compared with no net preference (ties or even splits)."""
    n_cyclic_components: int
    """Strongly connected components with two or more nodes."""
    largest_component: int
    n_triads: int
    """Complete triads: three nodes with an edge between every pair."""
    n_cycles: int
    """Complete triads that are directed 3-cycles."""
    cycle_rate: float | None
    random_rate: float
    p_value: float
    """Monte Carlo P(at most ``n_cycles`` cycles) under random edge orientation."""
    flagged: bool


class TransitivityReport(Record):
    level: Level
    pair_kinds: tuple[PairKind, ...]
    chance_level: float = Field(gt=0, lt=1)
    min_triads: int
    n_draws: int
    seed: int
    results: tuple[TransitivityResult, ...]

    @property
    def flagged(self) -> tuple[str, ...]:
        return tuple(sorted(r.annotator_id for r in self.results if r.flagged))


def preference_graphs(
    judgments: Iterable[PairwiseJudgment],
    *,
    level: Level = Level.MODEL,
    response_models: Mapping[str, str] | None = None,
    pair_kinds: Sequence[PairKind] = DEFAULT_PAIR_KINDS,
) -> dict[str, tuple[Graph, int]]:
    """``{annotator: (graph, level pairs)}`` from net decisive wins per node pair.

    At model level ``response_models`` maps response ids to models, and
    comparisons of two responses by the same model are ignored.
    """
    if level is Level.MODEL and response_models is None:
        msg = "model level needs the model of every response"
        raise ValueError(msg)
    if level is Level.RESPONSE:
        response_models = None
    kinds = set(pair_kinds)

    def node(response_id: str) -> str:
        return response_models[response_id] if response_models is not None else response_id

    net: dict[str, Counter[tuple[str, str]]] = defaultdict(Counter)
    for judgment in judgments:
        if judgment.pair_kind not in kinds:
            continue
        first, second = (node(r) for r in judgment.canonical_pair)
        if first == second:
            continue
        key = (first, second) if first < second else (second, first)
        seen = net[judgment.annotator_id]
        seen[key] += 0
        outcome = judgment.winner_loser
        if outcome is not None:
            seen[key] += 1 if node(outcome[0]) == key[0] else -1
    graphs: dict[str, tuple[Graph, int]] = {}
    for annotator in sorted(net):
        graph: Graph = {}
        level_pairs = 0
        for (u, v), balance in sorted(net[annotator].items()):
            graph.setdefault(u, set())
            graph.setdefault(v, set())
            if balance > 0:
                graph[u].add(v)
            elif balance < 0:
                graph[v].add(u)
            else:
                level_pairs += 1
        graphs[annotator] = (graph, level_pairs)
    return graphs


def transitivity(
    judgments: Iterable[PairwiseJudgment],
    responses: Iterable[Response] = (),
    *,
    level: Level = Level.MODEL,
    pair_kinds: Sequence[PairKind] = DEFAULT_PAIR_KINDS,
    chance_level: float = 0.5,
    min_triads: int = 10,
    n_draws: int = 2000,
    seed: int = 0,
) -> TransitivityReport:
    """Cycle counts and the random-orientation test for every annotator."""
    models = {r.id: r.model for r in responses} if level is Level.MODEL else None
    kinds = tuple(sorted(set(pair_kinds)))
    graphs = preference_graphs(judgments, level=level, response_models=models, pair_kinds=kinds)
    results = []
    for annotator, (graph, level_pairs) in graphs.items():
        components = strongly_connected_components(graph)
        cyclic = [c for c in components if len(c) > 1]
        n_triads, n_cycles = count_cycles(graph)
        p_value = random_cycle_p_value(graph, n_cycles, n_draws=n_draws, seed=seed)
        results.append(
            TransitivityResult(
                annotator_id=annotator,
                n_nodes=len(graph),
                n_edges=sum(len(t) for t in graph.values()),
                n_level_pairs=level_pairs,
                n_cyclic_components=len(cyclic),
                largest_component=max((len(c) for c in components), default=0),
                n_triads=n_triads,
                n_cycles=n_cycles,
                cycle_rate=n_cycles / n_triads if n_triads else None,
                random_rate=RANDOM_TRIAD_RATE,
                p_value=p_value,
                flagged=n_triads >= min_triads and p_value > chance_level,
            )
        )
    return TransitivityReport(
        level=level,
        pair_kinds=kinds,
        chance_level=chance_level,
        min_triads=min_triads,
        n_draws=n_draws,
        seed=seed,
        results=tuple(results),
    )
