"""Preference digraphs, Tarjan components and 3-cycle counts."""

from itertools import combinations, pairwise, permutations

import numpy as np
import pytest

from prefpairs.aggregate import Level
from prefpairs.quality.transitivity import (
    RANDOM_TRIAD_RATE,
    count_cycles,
    preference_graphs,
    random_cycle_p_value,
    strongly_connected_components,
    transitivity,
)
from prefpairs.schema import PairKind, PairwiseJudgment
from prefpairs.simulate import Archetype, SimulationConfig, simulate
from tests.quality_helpers import judgment


def tournament(nodes: str, beats: set[tuple[str, str]]) -> dict[str, set[str]]:
    """Complete tournament: ``(u, v)`` in ``beats`` means u -> v, otherwise v -> u."""
    graph: dict[str, set[str]] = {n: set() for n in nodes}
    for u, v in combinations(nodes, 2):
        if (u, v) in beats:
            graph[u].add(v)
        else:
            graph[v].add(u)
    return graph


def brute_force_cycles(graph: dict[str, set[str]]) -> int:
    return sum(
        1
        for trio in combinations(sorted(graph), 3)
        if any(b in graph[a] and c in graph[b] and a in graph[c] for a, b, c in permutations(trio))
    )


class TestComponents:
    def test_single_cycle_is_one_component(self) -> None:
        assert strongly_connected_components({"a": {"b"}, "b": {"c"}, "c": {"a"}}) == [
            ("a", "b", "c")
        ]

    def test_dag_has_only_singletons_in_reverse_topological_order(self) -> None:
        components = strongly_connected_components({"a": {"b", "c"}, "b": {"c"}})
        assert components == [("c",), ("b",), ("a",)]

    def test_two_cycles_joined_by_a_bridge(self) -> None:
        graph = {"a": {"b"}, "b": {"a", "c"}, "c": {"d"}, "d": {"c"}, "e": set()}
        components = strongly_connected_components(graph)
        assert sorted(components) == [("a", "b"), ("c", "d"), ("e",)]
        assert components.index(("c", "d")) < components.index(("a", "b"))

    def test_deep_chain_does_not_recurse(self) -> None:
        nodes = [f"n{i:05d}" for i in range(5000)]
        graph = {u: {v} for u, v in pairwise(nodes)}
        graph[nodes[-1]] = {nodes[0]}
        assert strongly_connected_components(graph) == [tuple(nodes)]


class TestCycleCounts:
    def test_transitive_tournament_has_no_cycles(self) -> None:
        graph = tournament("abcde", set(combinations("abcde", 2)))
        assert count_cycles(graph) == (10, 0)

    def test_regular_five_tournament_has_five_cycles(self) -> None:
        # Each node beats the next two (mod 5): every score is 2, and Kendall's
        # formula C(5,3) - sum C(s_i, 2) = 10 - 5 = 5 gives the cycle count.
        nodes = "abcde"
        graph = {n: {nodes[(i + 1) % 5], nodes[(i + 2) % 5]} for i, n in enumerate(nodes)}
        assert count_cycles(graph) == (10, 5)

    def test_matches_brute_force_on_random_tournaments(self) -> None:
        rng = np.random.default_rng(3)
        nodes = "abcdefg"
        for _ in range(20):
            beats = {p for p in combinations(nodes, 2) if rng.random() < 0.5}
            graph = tournament(nodes, beats)
            scores = [len(graph[n]) for n in nodes]
            kendall = 35 - sum(s * (s - 1) // 2 for s in scores)
            assert brute_force_cycles(graph) == kendall
            assert count_cycles(graph) == (35, kendall)

    def test_missing_edges_leave_incomplete_triads_out(self) -> None:
        assert count_cycles({"a": {"b"}, "b": {"c"}}) == (0, 0)

    def test_two_way_edge_and_self_loop_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="reverse edge"):
            count_cycles({"a": {"b"}, "b": {"a"}})
        with pytest.raises(ValueError, match="self-loop"):
            count_cycles({"a": {"a"}})


class TestRandomBaseline:
    def test_one_cycle_in_one_triad_is_typical_of_chance(self) -> None:
        graph = {"a": {"b"}, "b": {"c"}, "c": {"a"}}
        assert random_cycle_p_value(graph, 1, n_draws=4000, seed=0) == 1.0
        p_zero = random_cycle_p_value(graph, 0, n_draws=4000, seed=0)
        assert p_zero == pytest.approx(1 - RANDOM_TRIAD_RATE, abs=0.03)

    def test_transitive_tournament_is_rare_under_chance(self) -> None:
        graph = tournament("abcdefg", set(combinations("abcdefg", 2)))
        assert random_cycle_p_value(graph, 0, n_draws=2000, seed=1) < 0.01

    def test_no_triads_gives_one(self) -> None:
        assert random_cycle_p_value({"a": {"b"}}, 0, n_draws=10, seed=0) == 1.0


def _cycle_judgments(annotator: str, prompt: str = "p1") -> list[PairwiseJudgment]:
    return [
        judgment(annotator, f"{prompt}-a", f"{prompt}-b", "left", prompt=prompt),
        judgment(annotator, f"{prompt}-b", f"{prompt}-c", "left", prompt=prompt),
        judgment(annotator, f"{prompt}-a", f"{prompt}-c", "right", prompt=prompt),
    ]


class TestGraphs:
    def test_response_level_graph_uses_net_wins_and_counts_level_pairs(self) -> None:
        judgments = [
            *_cycle_judgments("x"),
            judgment("x", "p1-a", "p1-d", "left"),
            judgment("x", "p1-d", "p1-a", "left"),
            judgment("x", "p1-c", "p1-d", "tie"),
            judgment("x", "p1-c", "p1-d", "skip"),
        ]
        graphs = preference_graphs(judgments, level=Level.RESPONSE)
        graph, level_pairs = graphs["x"]
        assert graph == {"p1-a": {"p1-b"}, "p1-b": {"p1-c"}, "p1-c": {"p1-a"}, "p1-d": set()}
        assert level_pairs == 2

    def test_model_level_merges_prompts_and_drops_same_model_pairs(self) -> None:
        models = {f"{p}-{m}": m for p in ("p1", "p2") for m in "abc"}
        models["p1-c2"] = "c"
        judgments = [
            judgment("x", "p1-a", "p1-b", "left", prompt="p1"),
            judgment("x", "p2-b", "p2-c", "left", prompt="p2"),
            judgment("x", "p2-c", "p2-a", "left", prompt="p2"),
            judgment("x", "p1-c", "p1-c2", "left", prompt="p1"),
        ]
        graph, _ = preference_graphs(judgments, response_models=models)["x"]
        assert graph == {"a": {"b"}, "b": {"c"}, "c": {"a"}}

    def test_model_level_needs_models_and_controls_are_ignored(self) -> None:
        with pytest.raises(ValueError, match="model of every response"):
            preference_graphs([], level=Level.MODEL)
        original = judgment("x", "a", "b", "left", judgment_id="orig")
        control = judgment("x", "b", "a", "left", kind=PairKind.CONTROL, repeat_of="orig")
        graph, level_pairs = preference_graphs([original, control], level=Level.RESPONSE)["x"]
        assert graph == {"a": {"b"}, "b": set()}
        assert level_pairs == 0


class TestTransitivityReport:
    def test_cycle_on_one_prompt_is_found(self) -> None:
        report = transitivity(
            [*_cycle_judgments("x"), *_cycle_judgments("y", "p2")[:2]],
            level=Level.RESPONSE,
            min_triads=1,
        )
        by_id = {r.annotator_id: r for r in report.results}
        assert by_id["x"].n_cycles == 1
        assert by_id["x"].n_cyclic_components == 1
        assert by_id["x"].largest_component == 3
        assert by_id["x"].cycle_rate == 1.0
        assert by_id["x"].flagged
        assert by_id["y"].n_triads == 0
        assert by_id["y"].cycle_rate is None
        assert not by_id["y"].flagged
        assert report.flagged == ("x",)

    def test_simulated_spammer_is_flagged_and_consistent_annotators_are_not(self) -> None:
        # Ten models give each annotator 40-90 complete triads; with six models
        # there are at most 20 and the check has little power (see the module).
        config = SimulationConfig(seed=0, n_models=10, n_prompts=60, pairs_per_prompt=8)
        dataset = simulate(config)
        report = transitivity(dataset.pairwise, dataset.responses)
        archetypes = dataset.truth.annotator_archetypes
        assert [archetypes[a] for a in report.flagged] == [Archetype.RANDOM_SPAMMER]
        for result in report.results:
            assert result.n_triads >= 30
            if archetypes[result.annotator_id] is Archetype.RANDOM_SPAMMER:
                assert result.cycle_rate is not None
                assert result.cycle_rate >= 0.2
            elif archetypes[result.annotator_id] is not Archetype.NOISY:
                # Consistent preferences, right or wrong, are far from random.
                assert result.p_value < 0.05
