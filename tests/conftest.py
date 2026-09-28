"""Shared fixtures: an in-memory store and a small, fully linked record set."""

from collections.abc import Iterator
from datetime import UTC, datetime, timedelta

import pytest

from prefpairs.schema import (
    Annotator,
    AnnotatorKind,
    AnyRecord,
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
)
from prefpairs.store import Store

T0 = datetime(2026, 1, 1, tzinfo=UTC)


def build_records() -> list[AnyRecord]:
    """Two prompts, five responses, two annotators and one of every judgment type."""
    provenance = Provenance(source=ResponseSource.MODEL, generator="stub", temperature=0.7)
    return [
        Prompt(id="p1", text="Explain recursion to a new programmer.", category="cs"),
        Prompt(id="p2", text="Summarise the water cycle.", tags=("science", "short")),
        Response(
            id="p1-a",
            prompt_id="p1",
            model="model-a",
            text="It calls itself.",
            provenance=provenance,
        ),
        Response(
            id="p1-b",
            prompt_id="p1",
            model="model-b",
            text="A function that solves a smaller copy of the problem.",
            provenance=provenance,
        ),
        Response(
            id="p1-c",
            prompt_id="p1",
            model="model-c",
            text="Loops, but fancier.",
            provenance=Provenance(source=ResponseSource.HUMAN),
        ),
        Response(
            id="p2-a",
            prompt_id="p2",
            model="model-a",
            text="Evaporation, condensation, rain.",
            provenance=provenance,
        ),
        Response(
            id="p2-b",
            prompt_id="p2",
            model="model-b",
            text="Water moves around.",
            provenance=provenance,
        ),
        Annotator(id="ann-1", display_name="First"),
        Annotator(id="ann-2", kind=AnnotatorKind.SIMULATED),
        Session(
            id="s1",
            annotator_id="ann-1",
            client=SessionClient.WEB,
            started_at=T0,
            ended_at=T0 + timedelta(minutes=5),
        ),
        Session(id="s2", annotator_id="ann-2", client=SessionClient.CLI, started_at=T0),
        GoldPair(
            id="g1",
            prompt_id="p2",
            better_response_id="p2-a",
            worse_response_id="p2-b",
            reason="complete",
        ),
        PairwiseJudgment(
            id="j1",
            session_id="s1",
            annotator_id="ann-1",
            prompt_id="p1",
            left_response_id="p1-a",
            right_response_id="p1-b",
            choice=Choice.RIGHT,
            rationale="more precise",
            latency_ms=4200,
            created_at=T0 + timedelta(seconds=10),
        ),
        PairwiseJudgment(
            id="j2",
            session_id="s1",
            annotator_id="ann-1",
            prompt_id="p2",
            left_response_id="p2-b",
            right_response_id="p2-a",
            choice=Choice.RIGHT,
            pair_kind=PairKind.GOLD,
            created_at=T0 + timedelta(seconds=20),
        ),
        PairwiseJudgment(
            id="j3",
            session_id="s1",
            annotator_id="ann-1",
            prompt_id="p1",
            left_response_id="p1-b",
            right_response_id="p1-a",
            choice=Choice.LEFT,
            pair_kind=PairKind.CONTROL,
            repeat_of="j1",
            created_at=T0 + timedelta(seconds=30),
        ),
        PairwiseJudgment(
            id="j4",
            session_id="s2",
            annotator_id="ann-2",
            prompt_id="p1",
            left_response_id="p1-c",
            right_response_id="p1-a",
            choice=Choice.TIE,
            created_at=T0 + timedelta(seconds=40),
        ),
        PairwiseJudgment(
            id="j5",
            session_id="s2",
            annotator_id="ann-2",
            prompt_id="p2",
            left_response_id="p2-a",
            right_response_id="p2-b",
            choice=Choice.SKIP,
            created_at=T0 + timedelta(seconds=50, microseconds=1),
        ),
        RankedJudgment(
            id="k1",
            session_id="s2",
            annotator_id="ann-2",
            prompt_id="p1",
            shown=("p1-a", "p1-b", "p1-c"),
            ranking=("p1-b", "p1-c", "p1-a"),
            latency_ms=9000,
            created_at=T0 + timedelta(minutes=1),
        ),
    ]


@pytest.fixture
def records() -> list[AnyRecord]:
    return build_records()


@pytest.fixture
def store() -> Iterator[Store]:
    with Store.open(":memory:") as db:
        yield db


@pytest.fixture
def filled(store: Store, records: list[AnyRecord]) -> Store:
    store.add_all(records)
    return store
