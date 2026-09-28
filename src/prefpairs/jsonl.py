"""JSON Lines interchange format for PrefPairs records.

Each line is one JSON object whose ``type`` names its record kind, one of the
keys of ``RECORD_KINDS`` (``type`` is reserved: no record has a field of that
name)::

    {"type": "prompt", "id": "p1", "text": "Explain recursion."}
    {"type": "response", "id": "p1-a", "prompt_id": "p1", "model": "model-a", ...}

A prompt line may carry its candidate responses inline, which is the common
shape of a generation dump; ``prompt_id`` is then filled in from the prompt::

    {"type": "prompt", "id": "p1", "text": "...", "responses": [
        {"id": "p1-a", "model": "model-a", "text": "...", "provenance": {"source": "model"}}]}

Parsing is strict: the first malformed line raises ``JsonlError`` with its line
number, so an import either succeeds completely or reports exactly where it
failed. Writing is canonical (sorted keys, ASCII, no insignificant whitespace),
so the same records always serialise to the same bytes.
"""

import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import IO, Any

from pydantic import ValidationError

from prefpairs.schema import RECORD_KINDS, AnyRecord, record_kind

TYPE_KEY = "type"
"""JSON key that tags each line with its record kind."""


class JsonlError(ValueError):
    """A line could not be parsed into a record."""

    def __init__(self, line_no: int, message: str) -> None:
        super().__init__(f"line {line_no}: {message}")
        self.line_no = line_no
        self.message = message


def _summarise(exc: ValidationError) -> str:
    parts = []
    for error in exc.errors():
        location = ".".join(str(part) for part in error["loc"]) or "record"
        parts.append(f"{location}: {error['msg']}")
    return "; ".join(parts)


def _validate(line_no: int, kind: str, data: dict[str, Any], where: str = "") -> AnyRecord:
    try:
        return RECORD_KINDS[kind].model_validate(data)
    except ValidationError as exc:
        raise JsonlError(line_no, f"invalid {kind}{where}: {_summarise(exc)}") from exc


def _inline_responses(line_no: int, prompt_id: str, nested: object) -> Iterator[AnyRecord]:
    if not isinstance(nested, list):
        raise JsonlError(line_no, "responses must be a list of objects")
    for index, item in enumerate(nested):
        where = f" (responses[{index}])"
        if not isinstance(item, dict):
            raise JsonlError(line_no, f"expected an object{where}")
        declared = item.get("prompt_id", prompt_id)
        if declared != prompt_id:
            msg = f"prompt_id {declared!r} does not match the enclosing prompt {prompt_id!r}{where}"
            raise JsonlError(line_no, msg)
        yield _validate(line_no, "response", {**item, "prompt_id": prompt_id}, where)


def parse_records(lines: Iterable[str]) -> Iterator[AnyRecord]:
    """Parse JSONL text into validated records. Blank lines are ignored."""
    for line_no, raw in enumerate(lines, start=1):
        line = raw.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise JsonlError(line_no, f"invalid JSON: {exc.msg} (column {exc.colno})") from exc
        if not isinstance(obj, dict):
            raise JsonlError(line_no, "expected a JSON object")
        kind = obj.pop(TYPE_KEY, None)
        if kind not in RECORD_KINDS:
            expected = ", ".join(RECORD_KINDS)
            raise JsonlError(
                line_no, f"unknown or missing type {kind!r}; expected one of {expected}"
            )
        nested = obj.pop("responses", None) if kind == "prompt" else None
        record = _validate(line_no, kind, obj)
        yield record
        if nested is not None:
            yield from _inline_responses(line_no, record.id, nested)


def read_records(path: Path) -> list[AnyRecord]:
    """Parse a JSONL file; raises JsonlError on the first bad line."""
    with path.open(encoding="utf-8") as handle:
        return list(parse_records(handle))


def record_to_json(record: AnyRecord) -> str:
    """Canonical one-line JSON for a record, tagged with its type."""
    payload = record.model_dump(mode="json", exclude_none=True)
    return json.dumps(
        {TYPE_KEY: record_kind(record), **payload},
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
    )


def write_records(records: Iterable[AnyRecord], handle: IO[str]) -> int:
    """Write records as JSONL; return how many were written."""
    count = 0
    for record in records:
        handle.write(record_to_json(record))
        handle.write("\n")
        count += 1
    return count


__all__ = [
    "TYPE_KEY",
    "JsonlError",
    "parse_records",
    "read_records",
    "record_to_json",
    "write_records",
]
