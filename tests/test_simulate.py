"""The synthetic generator: determinism, structure and statistical sanity.

Later slices prove their statistics against this generator, so the generator
itself is checked here against its own analytic model.
"""

import math
from collections import Counter, defaultdict
from functools import cache
from itertools import pairwise

import numpy as np
import pytest
from pydantic import ValidationError

from prefpairs.jsonl import record_to_json
from prefpairs.schema import Choice, PairKind
from prefpairs.simulate import (
    DEFAULT_PROFILES,
    AnnotatorProfile,
    Archetype,
    SimulatedDataset,
    SimulationConfig,
    SimulationError,
    load_truth,
    simulate,
    write_simulation,
)
from prefpairs.store import RecordConflictError, Store


@cache
def default_dataset() -> SimulatedDataset:
    return simulate(SimulationConfig())


@cache
def large_dataset() -> SimulatedDataset:
    """Two annotators of every archetype, enough judgments for tight rates."""
    return simulate(SimulationConfig(seed=7, n_prompts=300, roster=dict.fromkeys(Archetype, 2)))


def judgments_by_archetype(ds: SimulatedDataset) -> dict[Archetype, list[tuple[str, str, Choice]]]:
    grouped: dict[Archetype, list[tuple[str, str, Choice]]] = defaultdict(list)
    for j in ds.pairwise:
        archetype = ds.truth.annotator_archetypes[j.annotator_id]
        grouped[archetype].append((j.left_response_id, j.right_response_id, j.choice))
    return grouped


class TestDeterminism:
    def test_same_seed_same_dataset(self) -> None:
        first, second = simulate(SimulationConfig(seed=3)), simulate(SimulationConfig(seed=3))
        assert first == second
        assert [record_to_json(r) for r in first.records()] == [
            record_to_json(r) for r in second.records()
        ]
        assert first.truth.model_dump_json() == second.truth.model_dump_json()

    def test_different_seeds_differ(self) -> None:
        first, second = simulate(SimulationConfig(seed=1)), simulate(SimulationConfig(seed=2))
        assert first.truth.model_strengths != second.truth.model_strengths
        assert [j.choice for j in first.pairwise] != [j.choice for j in second.pairwise]

    def test_roster_changes_do_not_move_the_world(self) -> None:
        base = simulate(SimulationConfig(seed=5))
        other = simulate(
            SimulationConfig(seed=5, roster={Archetype.RELIABLE: 4}, n_gold=3, control_rate=0.5)
        )
        assert base.prompts == other.prompts
        assert base.responses == other.responses
        assert base.truth.response_quality == other.truth.response_quality
        assert base.truth.model_strengths == other.truth.model_strengths

    def test_default_config_is_used_when_omitted(self) -> None:
        assert simulate() == default_dataset()


class TestStructure:
    def test_counts_follow_the_config(self) -> None:
        config = SimulationConfig()
        ds = default_dataset()
        kinds = Counter(j.pair_kind for j in ds.pairwise)
        n_regular = config.n_prompts * config.pairs_per_prompt * config.redundancy
        assert len(ds.prompts) == config.n_prompts
        assert len(ds.responses) == config.n_prompts * config.n_models
        assert len(ds.annotators) == config.n_annotators == 12
        assert len(ds.gold_pairs) == config.n_gold
        assert kinds[PairKind.REGULAR] == n_regular
        assert kinds[PairKind.GOLD] == config.n_gold * config.n_annotators
        per_annotator = Counter(
            j.annotator_id for j in ds.pairwise if j.pair_kind is PairKind.REGULAR
        )
        controls = Counter(j.annotator_id for j in ds.pairwise if j.pair_kind is PairKind.CONTROL)
        for annotator, count in per_annotator.items():
            assert controls[annotator] == round(config.control_rate * count)
        assert len(ds.ranked) == config.ranked_per_annotator * config.n_annotators

    def test_every_regular_pair_has_distinct_annotators_and_load_is_balanced(self) -> None:
        ds = default_dataset()
        raters: dict[tuple[str, str], list[str]] = defaultdict(list)
        for j in ds.pairwise:
            if j.pair_kind is PairKind.REGULAR:
                raters[j.canonical_pair].append(j.annotator_id)
        assert len(raters) == 40 * 4
        assert all(len(set(r)) == len(r) == 3 for r in raters.values())
        load = Counter(a for r in raters.values() for a in r)
        assert max(load.values()) - min(load.values()) <= 1

    def test_controls_repeat_an_earlier_judgment_with_sides_flipped(self) -> None:
        ds = default_dataset()
        by_id = {j.id: j for j in ds.pairwise}
        controls = [j for j in ds.pairwise if j.pair_kind is PairKind.CONTROL]
        assert controls
        for control in controls:
            assert control.repeat_of is not None
            original = by_id[control.repeat_of]
            assert original.pair_kind is PairKind.REGULAR
            assert original.annotator_id == control.annotator_id
            assert (original.left_response_id, original.right_response_id) == (
                control.right_response_id,
                control.left_response_id,
            )
            assert original.created_at < control.created_at

    def test_gold_pairs_have_a_clear_true_winner_and_are_not_regular_work(self) -> None:
        ds = default_dataset()
        quality = ds.truth.response_quality
        regular = {j.canonical_pair for j in ds.pairwise if j.pair_kind is PairKind.REGULAR}
        for gold in ds.gold_pairs:
            gap = quality[gold.better_response_id] - quality[gold.worse_response_id]
            assert gap >= SimulationConfig().gold_min_gap
            assert gold.canonical_pair not in regular
            assert ds.truth.better_response(gold.worse_response_id, gold.better_response_id) == (
                gold.better_response_id
            )

    def test_sessions_are_chronological_and_bounded(self) -> None:
        config = SimulationConfig(session_size=25)
        ds = simulate(config)
        sessions = {s.id: s for s in ds.sessions}
        per_session = Counter(j.session_id for j in ds.pairwise) + Counter(
            k.session_id for k in ds.ranked
        )
        assert set(per_session) == set(sessions)
        assert max(per_session.values()) <= 25
        for annotator in ds.annotators:
            times = sorted(
                [(j.created_at, j.id) for j in ds.pairwise if j.annotator_id == annotator.id]
                + [(k.created_at, k.id) for k in ds.ranked if k.annotator_id == annotator.id]
            )
            stamps = [t for t, _ in times]
            assert all(a < b for a, b in pairwise(stamps))
        for j in ds.pairwise:
            session = sessions[j.session_id]
            assert session.annotator_id == j.annotator_id
            assert session.ended_at is not None
            assert session.started_at < j.created_at <= session.ended_at

    def test_response_text_matches_the_planted_length(self) -> None:
        ds = default_dataset()
        assert all(r.text.endswith(".") for r in ds.responses)
        assert all(r.text.isascii() for r in ds.responses)
        words = [r.n_words for r in ds.responses]
        assert min(words) >= 3
        assert 40 < float(np.median(words)) < 200

    def test_annotator_ids_do_not_reveal_archetypes(self) -> None:
        ds = default_dataset()
        order = [ds.truth.annotator_archetypes[a.id] for a in ds.annotators]
        assert order != sorted(order, key=list(Archetype).index)
        assert set(ds.truth.defective_annotators) == {
            a for a, arch in ds.truth.annotator_archetypes.items() if arch.is_defective
        }
        assert len(ds.truth.defective_annotators) == 4

    def test_ranked_judgments_are_valid_permutations_of_one_prompt(self) -> None:
        ds = default_dataset()
        prompt_of = {r.id: r.prompt_id for r in ds.responses}
        for ranked in ds.ranked:
            assert len(ranked.shown) == SimulationConfig().ranked_size
            assert {prompt_of[r] for r in ranked.shown} == {ranked.prompt_id}


class TestLatentModel:
    def test_strengths_are_centred_and_evenly_spaced(self) -> None:
        config = SimulationConfig(n_models=5, strength_spread=2.0)
        truth = simulate(config).truth
        values = sorted(truth.model_strengths.values())
        assert math.isclose(sum(values), 0.0, abs_tol=1e-12)
        assert np.allclose(values, np.linspace(-2.0, 2.0, 5))
        assert truth.model_ranking()[0] == max(truth.model_strengths, key=truth.model_strengths.get)

    @pytest.mark.parametrize("corr", [0.0, 0.6, -0.9])
    def test_verbosity_has_exactly_the_requested_correlation(self, corr: float) -> None:
        truth = simulate(SimulationConfig(length_quality_corr=corr, verbosity_sd=0.5)).truth
        models = sorted(truth.model_strengths)
        strength = np.array([truth.model_strengths[m] for m in models])
        verbosity = np.array([truth.model_verbosity[m] for m in models])
        assert math.isclose(float(verbosity.mean()), 0.0, abs_tol=1e-12)
        assert math.isclose(float(verbosity.std()), 0.5, rel_tol=1e-9)
        assert math.isclose(float(np.corrcoef(strength, verbosity)[0, 1]), corr, abs_tol=1e-9)

    def test_degenerate_worlds_still_generate(self) -> None:
        flat = simulate(
            SimulationConfig(strength_spread=0.0, n_gold=0, response_sd=0.0, n_prompts=3)
        )
        assert set(flat.truth.model_strengths.values()) == {0.0}
        two = simulate(
            SimulationConfig(
                n_models=2, pairs_per_prompt=1, n_gold=0, ranked_size=2, length_quality_corr=0.5
            )
        )
        assert len(two.responses) == 80

    def test_too_few_gold_candidates_is_reported(self) -> None:
        with pytest.raises(SimulationError, match="lower n_gold"):
            simulate(SimulationConfig(n_prompts=2, n_gold=50))


class TestChoiceModel:
    @pytest.mark.parametrize("archetype", list(Archetype))
    @pytest.mark.parametrize(("gap", "log_ratio"), [(0.0, 0.0), (1.3, -0.4), (-2.0, 0.7)])
    def test_probabilities_are_a_distribution(
        self, archetype: Archetype, gap: float, log_ratio: float
    ) -> None:
        probs = DEFAULT_PROFILES[archetype].choice_probabilities(gap, log_ratio)
        assert set(probs) == set(Choice)
        assert all(p >= 0 for p in probs.values())
        assert math.isclose(sum(probs.values()), 1.0, abs_tol=1e-12)

    def test_unbiased_profiles_are_mirror_symmetric(self) -> None:
        profile = DEFAULT_PROFILES[Archetype.RELIABLE]
        forward = profile.choice_probabilities(0.8, 0.3)
        mirrored = profile.choice_probabilities(-0.8, -0.3)
        assert math.isclose(forward[Choice.LEFT], mirrored[Choice.RIGHT])
        assert math.isclose(forward[Choice.TIE], mirrored[Choice.TIE])

    def test_archetype_signatures(self) -> None:
        p = {a: DEFAULT_PROFILES[a] for a in Archetype}
        assert p[Archetype.RANDOM_SPAMMER].choice_probabilities(3.0, 1.0)[Choice.LEFT] == 0.5
        assert p[Archetype.LEFT_BIASED].choice_probabilities(0.0, 0.0)[Choice.LEFT] > 0.8
        assert p[Archetype.ADVERSARIAL].choice_probabilities(1.0, 0.0)[Choice.LEFT] < 0.2
        longer_left = p[Archetype.LENGTH_BIASED].choice_probabilities(0.0, 0.5)
        assert longer_left[Choice.LEFT] > 0.75
        assert p[Archetype.RELIABLE].choice_probabilities(0.0, 0.5)[Choice.LEFT] < 0.5

    def test_extreme_logits_do_not_overflow(self) -> None:
        profile = DEFAULT_PROFILES[Archetype.RELIABLE].model_copy(update={"sharpness": 1e6})
        assert profile.choice_probabilities(-10.0, 0.0)[Choice.LEFT] == 0.0
        assert profile.choice_probabilities(10.0, 0.0)[Choice.RIGHT] == 0.0


class TestEmpiricalBehaviour:
    """Sampled choices agree with the analytic model and show each archetype's signature."""

    def test_left_rate_matches_the_model_for_every_annotator(self) -> None:
        ds = large_dataset()
        words = {r.id: r.n_words for r in ds.responses}
        quality = ds.truth.response_quality
        expected: dict[str, float] = defaultdict(float)
        variance: dict[str, float] = defaultdict(float)
        observed: Counter[str] = Counter()
        for j in ds.pairwise:
            probs = ds.truth.profile(j.annotator_id).choice_probabilities(
                quality[j.left_response_id] - quality[j.right_response_id],
                math.log(words[j.left_response_id] / words[j.right_response_id]),
            )
            p_left = probs[Choice.LEFT]
            expected[j.annotator_id] += p_left
            variance[j.annotator_id] += p_left * (1 - p_left)
            observed[j.annotator_id] += j.choice is Choice.LEFT
        for annotator, mean in expected.items():
            z = (observed[annotator] - mean) / math.sqrt(variance[annotator])
            assert abs(z) < 4, (annotator, z)

    def test_archetype_signatures_show_up_in_the_data(self) -> None:
        ds = large_dataset()
        words = {r.id: r.n_words for r in ds.responses}
        rates = {}
        for archetype, rows in judgments_by_archetype(ds).items():
            decisive = [row for row in rows if row[2] in (Choice.LEFT, Choice.RIGHT)]
            winners = [(lft, rgt) if c is Choice.LEFT else (rgt, lft) for lft, rgt, c in decisive]
            accuracy = np.mean([ds.truth.better_response(w, x) == w for w, x in winners])
            left = np.mean([c is Choice.LEFT for _, _, c in decisive])
            longer = np.mean([words[w] > words[x] for w, x in winners if words[w] != words[x]])
            rates[archetype] = (float(accuracy), float(left), float(longer))
        assert rates[Archetype.RELIABLE][0] > 0.85
        assert 0.6 < rates[Archetype.NOISY][0] < rates[Archetype.RELIABLE][0]
        assert 0.4 < rates[Archetype.RANDOM_SPAMMER][0] < 0.6
        assert rates[Archetype.ADVERSARIAL][0] < 0.25
        assert rates[Archetype.LEFT_BIASED][1] > 0.6
        assert 0.44 < rates[Archetype.RELIABLE][1] < 0.56
        assert rates[Archetype.LENGTH_BIASED][2] > 0.65
        assert 0.44 < rates[Archetype.RELIABLE][2] < 0.56

    def test_spammers_are_fast_and_silent(self) -> None:
        ds = large_dataset()
        latency: dict[Archetype, list[int]] = defaultdict(list)
        rationales: Counter[Archetype] = Counter()
        for j in ds.pairwise:
            archetype = ds.truth.annotator_archetypes[j.annotator_id]
            assert j.latency_ms is not None
            latency[archetype].append(j.latency_ms)
            rationales[archetype] += j.rationale is not None
        assert np.median(latency[Archetype.RANDOM_SPAMMER]) < np.median(latency[Archetype.RELIABLE])
        assert rationales[Archetype.RANDOM_SPAMMER] == 0
        assert rationales[Archetype.RELIABLE] > 0

    def test_reliable_rankings_put_the_best_response_first_most_often(self) -> None:
        ds = simulate(SimulationConfig(roster={Archetype.RELIABLE: 3}, ranked_per_annotator=40))
        quality = ds.truth.response_quality
        top_is_best = [k.ranking[0] == max(k.shown, key=quality.__getitem__) for k in ds.ranked]
        assert np.mean(top_is_best) > 0.6


class TestConfigValidation:
    @pytest.mark.parametrize(
        ("overrides", "message"),
        [
            ({"pairs_per_prompt": 16}, "pairs_per_prompt=16 exceeds"),
            ({"redundancy": 13}, "redundancy=13 exceeds"),
            ({"ranked_size": 7}, "ranked_size=7 exceeds"),
            ({"roster": {}}, "at least one annotator"),
            ({"roster": {Archetype.RELIABLE: -1, Archetype.NOISY: 2}}, "non-negative"),
            ({"profiles": {}}, "no profile for archetype reliable"),
            (
                {
                    "profiles": {
                        **DEFAULT_PROFILES,
                        Archetype.NOISY: DEFAULT_PROFILES[Archetype.RELIABLE],
                    }
                },
                "profile under noisy",
            ),
        ],
    )
    def test_inconsistent_configs_are_rejected(
        self, overrides: dict[str, object], message: str
    ) -> None:
        with pytest.raises(ValidationError, match=message):
            SimulationConfig.model_validate(overrides)

    def test_rates_must_leave_room_for_a_decision(self) -> None:
        with pytest.raises(ValidationError, match="must not exceed 1"):
            AnnotatorProfile(
                archetype=Archetype.NOISY,
                sharpness=1.0,
                tie_rate=0.6,
                skip_rate=0.5,
                median_latency_ms=1000,
            )


class TestPersistence:
    def test_write_and_load_round_trip(self, store: Store) -> None:
        ds = default_dataset()
        added = write_simulation(store, ds)
        assert added == len(ds.records())
        assert load_truth(store) == ds.truth
        assert list(store.iter_records()) == ds.records()

    def test_writing_the_same_dataset_twice_is_a_no_op(self, store: Store) -> None:
        ds = default_dataset()
        write_simulation(store, ds)
        before = store.dump_rows()
        assert write_simulation(store, ds) == 0
        assert store.dump_rows() == before

    def test_a_second_simulation_is_refused_atomically(self, store: Store) -> None:
        write_simulation(store, default_dataset())
        before = store.dump_rows()
        with pytest.raises(RecordConflictError, match="different simulated dataset"):
            write_simulation(store, simulate(SimulationConfig(seed=99)))
        assert store.dump_rows() == before

    def test_plain_databases_have_no_truth(self, store: Store) -> None:
        assert load_truth(store) is None

    def test_same_seed_gives_identical_databases(self) -> None:
        with Store.open(":memory:") as first, Store.open(":memory:") as second:
            write_simulation(first, simulate(SimulationConfig(seed=11)))
            write_simulation(second, simulate(SimulationConfig(seed=11)))
            assert first.dump_rows() == second.dump_rows()
