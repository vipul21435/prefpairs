"""Validation and helper behaviour of the pydantic data model."""

from datetime import UTC, datetime, timedelta, timezone
from itertools import pairwise

import pytest
from pydantic import ValidationError

from prefpairs.schema import (
    MAX_RATIONALE_CHARS,
    RECORD_KINDS,
    Annotator,
    AnnotatorKind,
    CanonicalLabel,
    Choice,
    GoldPair,
    PairKind,
    PairwiseJudgment,
    Prompt,
    Provenance,
    RankedJudgment,
    Response,
    ResponseSource,
    Session,
    SessionClient,
    canonical_label,
    record_kind,
)

T0 = datetime(2026, 1, 1, tzinfo=UTC)


def judgment(**overrides: object) -> PairwiseJudgment:
    fields: dict[str, object] = {
        "id": "j1",
        "session_id": "s1",
        "annotator_id": "ann-1",
        "prompt_id": "p1",
        "left_response_id": "r-b",
        "right_response_id": "r-a",
        "choice": "left",
        "created_at": T0,
    }
    fields.update(overrides)
    return PairwiseJudgment.model_validate(fields)


class TestCanonicalLabel:
    @pytest.mark.parametrize(
        ("left", "right", "choice", "expected"),
        [
            ("a", "b", Choice.LEFT, CanonicalLabel.A_WINS),
            ("a", "b", Choice.RIGHT, CanonicalLabel.B_WINS),
            ("b", "a", Choice.LEFT, CanonicalLabel.B_WINS),
            ("b", "a", Choice.RIGHT, CanonicalLabel.A_WINS),
            ("a", "b", Choice.TIE, CanonicalLabel.TIE),
            ("b", "a", Choice.TIE, CanonicalLabel.TIE),
            ("a", "b", Choice.SKIP, None),
        ],
    )
    def test_maps_displayed_choice_to_sorted_pair(
        self, left: str, right: str, choice: Choice, expected: CanonicalLabel | None
    ) -> None:
        assert canonical_label(left, right, choice) is expected

    def test_flipping_sides_and_choice_gives_the_same_label(self) -> None:
        for choice, flipped in [(Choice.LEFT, Choice.RIGHT), (Choice.RIGHT, Choice.LEFT)]:
            assert canonical_label("x", "y", choice) is canonical_label("y", "x", flipped)

    def test_rejects_identical_sides(self) -> None:
        with pytest.raises(ValueError, match="distinct"):
            canonical_label("a", "a", Choice.LEFT)


class TestIdentifiersAndText:
    @pytest.mark.parametrize("bad_id", ["", " p1", "p 1", "-p1", "p/1", "p\u00e91", "x" * 129])
    def test_rejects_malformed_ids(self, bad_id: str) -> None:
        with pytest.raises(ValidationError):
            Prompt(id=bad_id, text="hello")

    @pytest.mark.parametrize("good_id", ["p1", "P-0001", "a.b:c_d-e", "x" * 128])
    def test_accepts_slug_ids(self, good_id: str) -> None:
        assert Prompt(id=good_id, text="hello").id == good_id

    @pytest.mark.parametrize("blank", ["", "   ", "\n\t"])
    def test_rejects_blank_text(self, blank: str) -> None:
        with pytest.raises(ValidationError, match=r"non-whitespace|at least 1"):
            Prompt(id="p1", text=blank)

    def test_text_is_stored_verbatim(self) -> None:
        text = "  leading and trailing space matter for length  \n"
        assert Prompt(id="p1", text=text).text == text

    def test_labels_are_trimmed(self) -> None:
        prompt = Prompt(id="p1", text="t", category="  coding ", tags=(" a ", "b"))
        assert prompt.category == "coding"
        assert prompt.tags == ("a", "b")

    def test_unknown_fields_are_rejected(self) -> None:
        with pytest.raises(ValidationError, match="Extra inputs"):
            Prompt.model_validate({"id": "p1", "text": "t", "colour": "red"})

    def test_records_are_immutable(self) -> None:
        prompt = Prompt(id="p1", text="t")
        with pytest.raises(ValidationError, match="frozen"):
            prompt.text = "changed"  # type: ignore[misc]


class TestResponse:
    def test_length_helpers(self) -> None:
        response = Response(
            id="r1",
            prompt_id="p1",
            model="model-a",
            text="one two  three",
            provenance=Provenance(source=ResponseSource.MODEL),
        )
        assert response.n_chars == 14
        assert response.n_words == 3

    @pytest.mark.parametrize(
        "bad",
        [
            {"temperature": -0.1},
            {"temperature": float("inf")},
            {"top_p": 0.0},
            {"top_p": 1.5},
            {"max_tokens": 0},
        ],
    )
    def test_provenance_rejects_invalid_decoding_parameters(self, bad: dict[str, float]) -> None:
        with pytest.raises(ValidationError):
            Provenance.model_validate({"source": "model", **bad})

    def test_provenance_timestamp_is_normalised_to_utc(self) -> None:
        ist = timezone(timedelta(hours=5, minutes=30))
        provenance = Provenance(
            source=ResponseSource.MODEL, generated_at=datetime(2026, 1, 1, 5, 30, tzinfo=ist)
        )
        assert provenance.generated_at == T0
        assert provenance.generated_at is not None
        assert provenance.generated_at.tzinfo is UTC

    def test_naive_timestamps_are_rejected(self) -> None:
        with pytest.raises(ValidationError, match="timezone"):
            Provenance.model_validate({"source": "model", "generated_at": "2026-01-01T00:00:00"})


class TestSession:
    def test_end_before_start_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="ended_at"):
            Session(
                id="s1",
                annotator_id="ann-1",
                client=SessionClient.WEB,
                started_at=T0,
                ended_at=T0 - timedelta(seconds=1),
            )

    def test_open_session_has_no_end(self) -> None:
        session = Session(id="s1", annotator_id="ann-1", client=SessionClient.CLI, started_at=T0)
        assert session.ended_at is None


class TestPairwiseJudgment:
    def test_derived_views(self) -> None:
        j = judgment(choice="left")
        assert j.canonical_pair == ("r-a", "r-b")
        assert j.label is CanonicalLabel.B_WINS
        assert j.winner_loser == ("r-b", "r-a")
        assert judgment(choice="right").winner_loser == ("r-a", "r-b")
        assert judgment(choice="tie").winner_loser is None
        assert judgment(choice="skip").label is None

    def test_same_response_on_both_sides_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="must differ"):
            judgment(right_response_id="r-b")

    def test_unknown_choice_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            judgment(choice="both")

    def test_control_pairs_require_repeat_of(self) -> None:
        with pytest.raises(ValidationError, match="repeat_of"):
            judgment(pair_kind=PairKind.CONTROL)
        with pytest.raises(ValidationError, match="repeat_of"):
            judgment(repeat_of="j0")
        control = judgment(pair_kind=PairKind.CONTROL, repeat_of="j0")
        assert control.repeat_of == "j0"

    def test_judgment_cannot_repeat_itself(self) -> None:
        with pytest.raises(ValidationError, match="itself"):
            judgment(pair_kind=PairKind.CONTROL, repeat_of="j1")

    def test_rationale_is_trimmed_and_blank_becomes_none(self) -> None:
        assert judgment(rationale="  clearer answer \n").rationale == "clearer answer"
        assert judgment(rationale="   ").rationale is None

    def test_rationale_length_is_capped(self) -> None:
        assert judgment(rationale="x" * MAX_RATIONALE_CHARS).rationale is not None
        with pytest.raises(ValidationError, match="at most"):
            judgment(rationale="x" * (MAX_RATIONALE_CHARS + 1))

    def test_negative_latency_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            judgment(latency_ms=-1)


class TestRankedJudgment:
    def ranked(self, shown: tuple[str, ...], ranking: tuple[str, ...]) -> RankedJudgment:
        return RankedJudgment(
            id="k1",
            session_id="s1",
            annotator_id="ann-1",
            prompt_id="p1",
            shown=shown,
            ranking=ranking,
            created_at=T0,
        )

    def test_implied_pairs_follow_the_ranking(self) -> None:
        ranked = self.ranked(("r1", "r2", "r3"), ("r3", "r1", "r2"))
        assert ranked.implied_pairs() == [("r3", "r1"), ("r3", "r2"), ("r1", "r2")]

    @pytest.mark.parametrize("n", [2, 5, 20])
    def test_implied_pair_count(self, n: int) -> None:
        ids = tuple(f"r{i}" for i in range(n))
        assert len(self.ranked(ids, ids[::-1]).implied_pairs()) == n * (n - 1) // 2

    @pytest.mark.parametrize(
        ("shown", "ranking"),
        [
            (("r1", "r2", "r3"), ("r1", "r2")),
            (("r1", "r2", "r3"), ("r1", "r2", "r4")),
            (("r1", "r2", "r3"), ("r1", "r1", "r2")),
            (("r1", "r1", "r2"), ("r1", "r1", "r2")),
        ],
    )
    def test_ranking_must_be_a_permutation_of_shown(
        self, shown: tuple[str, ...], ranking: tuple[str, ...]
    ) -> None:
        with pytest.raises(ValidationError, match=r"permutation|duplicate"):
            self.ranked(shown, ranking)

    def test_needs_at_least_two_items(self) -> None:
        with pytest.raises(ValidationError):
            self.ranked(("r1",), ("r1",))


class TestGoldPair:
    def test_expected_label_uses_sorted_order(self) -> None:
        gold = GoldPair(id="g1", prompt_id="p1", better_response_id="r-b", worse_response_id="r-a")
        assert gold.canonical_pair == ("r-a", "r-b")
        assert gold.expected_label is CanonicalLabel.B_WINS
        flipped = gold.model_copy(update={"better_response_id": "r-a", "worse_response_id": "r-b"})
        assert flipped.expected_label is CanonicalLabel.A_WINS

    def test_distinct_sides(self) -> None:
        with pytest.raises(ValidationError, match="must differ"):
            GoldPair(id="g1", prompt_id="p1", better_response_id="r", worse_response_id="r")


def test_every_record_type_round_trips_through_json() -> None:
    records = [
        Prompt(id="p1", text="Explain recursion.", category="cs", tags=("intro",)),
        Response(
            id="r1",
            prompt_id="p1",
            model="model-a",
            text="A function that calls itself.",
            provenance=Provenance(
                source=ResponseSource.MODEL,
                generator="stub 1.0",
                temperature=0.7,
                top_p=0.9,
                max_tokens=256,
                seed=3,
                generated_at=T0,
            ),
        ),
        Annotator(id="ann-1", kind=AnnotatorKind.SIMULATED, display_name="Sim 1"),
        Session(id="s1", annotator_id="ann-1", client=SessionClient.WEB, started_at=T0),
        GoldPair(id="g1", prompt_id="p1", better_response_id="r1", worse_response_id="r2"),
        judgment(rationale="more precise", latency_ms=1200),
        RankedJudgment(
            id="k1",
            session_id="s1",
            annotator_id="ann-1",
            prompt_id="p1",
            shown=("r1", "r2", "r3"),
            ranking=("r2", "r3", "r1"),
            created_at=T0,
        ),
    ]
    for record in records:
        restored = type(record).model_validate_json(record.model_dump_json())
        assert restored == record
    assert [record_kind(r) for r in records] == [
        "prompt",
        "response",
        "annotator",
        "session",
        "gold",
        "pairwise",
        "ranked",
    ]


def test_record_kinds_are_in_dependency_order() -> None:
    order = list(RECORD_KINDS)
    for before, after in pairwise(["prompt", "response", "annotator", "session", "pairwise"]):
        assert order.index(before) < order.index(after)


def test_record_kind_rejects_foreign_objects() -> None:
    with pytest.raises(TypeError, match="not a PrefPairs record"):
        record_kind(object())  # type: ignore[arg-type]
