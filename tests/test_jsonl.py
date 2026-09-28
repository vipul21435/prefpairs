"""JSONL parsing and canonical serialisation."""

import io
import json
from pathlib import Path

import pytest

from prefpairs.jsonl import (
    TYPE_KEY,
    JsonlError,
    parse_records,
    read_records,
    record_to_json,
    write_records,
)
from prefpairs.schema import RECORD_KINDS, AnyRecord, Prompt, Response
from prefpairs.store import Store


def test_every_record_round_trips(records: list[AnyRecord]) -> None:
    lines = [record_to_json(r) for r in records]
    assert list(parse_records(lines)) == records


def test_serialisation_is_canonical() -> None:
    prompt = Prompt(id="p1", text="caf\u00e9 \u2013 menu", tags=("b", "a"))
    line = record_to_json(prompt)
    assert line.isascii()
    assert " " not in line.replace("caf\\u00e9 \\u2013 menu", "")
    assert list(json.loads(line)) == sorted(json.loads(line))
    assert json.loads(line)["type"] == "prompt"
    assert "category" not in json.loads(line), "None fields are omitted"
    assert next(parse_records([line])) == prompt


def test_file_round_trip(tmp_path: Path, records: list[AnyRecord]) -> None:
    path = tmp_path / "data.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        assert write_records(records, handle) == len(records)
    assert read_records(path) == records
    assert path.read_text(encoding="utf-8").endswith("\n")


def test_inline_responses_inherit_the_prompt_id() -> None:
    line = json.dumps(
        {
            "type": "prompt",
            "id": "p1",
            "text": "Name a prime.",
            "responses": [
                {"id": "p1-a", "model": "m-a", "text": "7", "provenance": {"source": "model"}},
                {
                    "id": "p1-b",
                    "prompt_id": "p1",
                    "model": "m-b",
                    "text": "9",
                    "provenance": {"source": "human"},
                },
            ],
        }
    )
    prompt, first, second = parse_records([line])
    assert isinstance(prompt, Prompt)
    assert isinstance(first, Response)
    assert isinstance(second, Response)
    assert (first.prompt_id, second.prompt_id) == ("p1", "p1")


def test_blank_lines_are_ignored() -> None:
    line = record_to_json(Prompt(id="p1", text="t"))
    assert len(list(parse_records(["", line, "   \n", line]))) == 2


@pytest.mark.parametrize(
    ("line", "message"),
    [
        ("{not json", "invalid JSON"),
        ("[1, 2]", "expected a JSON object"),
        ('{"id": "p1", "text": "t"}', "missing type None"),
        ('{"type": "banana", "id": "p1"}', "unknown or missing type 'banana'"),
        ('{"type": "prompt", "id": "p 1", "text": "t"}', "invalid prompt: id:"),
        ('{"type": "prompt", "id": "p1", "text": "t", "extra": 1}', "extra: Extra inputs"),
        ('{"type": "prompt", "id": "p1", "text": "t", "responses": {}}', "must be a list"),
        ('{"type": "prompt", "id": "p1", "text": "t", "responses": [3]}', r"responses\[0\]"),
        (
            '{"type": "prompt", "id": "p1", "text": "t", "responses": '
            '[{"id": "r", "prompt_id": "p2", "model": "m", "text": "x", '
            '"provenance": {"source": "model"}}]}',
            "does not match the enclosing prompt",
        ),
        (
            '{"type": "prompt", "id": "p1", "text": "t", "responses": '
            '[{"id": "r", "model": "m", "text": " ", "provenance": {"source": "model"}}]}',
            r"invalid response \(responses\[0\]\): text",
        ),
    ],
)
def test_errors_name_the_line_and_the_problem(line: str, message: str) -> None:
    good = record_to_json(Prompt(id="p0", text="fine"))
    with pytest.raises(JsonlError, match=message) as excinfo:
        list(parse_records([good, "", line]))
    assert excinfo.value.line_no == 3
    assert str(excinfo.value).startswith("line 3: ")


def test_imported_file_populates_a_store(tmp_path: Path, records: list[AnyRecord]) -> None:
    buffer = io.StringIO()
    write_records(records, buffer)
    path = tmp_path / "in.jsonl"
    path.write_text(buffer.getvalue(), encoding="utf-8")
    with Store.open(":memory:") as db:
        result = db.import_records(read_records(path))
        assert result.total_added == len(records)
        assert list(db.iter_records()) == records


def test_type_key_never_collides_with_a_record_field() -> None:
    for cls in RECORD_KINDS.values():
        assert TYPE_KEY not in cls.model_fields, cls.__name__
