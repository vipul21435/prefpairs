"""DPO, KTO and reward-model JSONL with hash-based prompt-level splits.

Splits are decided per prompt, never per pair, so no prompt (and therefore no
response) appears in two splits: a model cannot be evaluated on a prompt it
was trained on. A prompt's split comes from ``sha256(salt + ":" + prompt_id)``
read as a number in [0, 1), which depends on nothing else, so adding data
never moves an existing prompt and reruns are byte-identical.

Formats (one JSON object per line, keys sorted, rows in prompt/pair order):

* ``dpo``: ``prompt``, ``chosen``, ``rejected`` plus ids and vote counts.
* ``kto``: one row per response in a kept pair, ``completion`` and a boolean
  ``label`` (desirable). A response that won and lost the same number of kept
  pairs has no clear label and is left out.
* ``rm``: the canonical pair ``response_a`` / ``response_b`` with the soft
  label ``p_a`` = share of counted votes for ``a`` (a tie vote counts half),
  for reward models trained on graded preferences.
"""

import hashlib
import json
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import Field, model_validator

from prefpairs.export.votes import PairVotes
from prefpairs.schema import Prompt, Record, Response

SPLITS = ("train", "validation", "test")


class ExportFormat(StrEnum):
    DPO = "dpo"
    KTO = "kto"
    RM = "rm"


class SplitConfig(Record):
    """Fractions of prompts per split and the salt of the split hash."""

    train: float = Field(default=0.8, ge=0, le=1)
    validation: float = Field(default=0.1, ge=0, le=1)
    test: float = Field(default=0.1, ge=0, le=1)
    salt: str = "prefpairs"

    @model_validator(mode="after")
    def _sums_to_one(self) -> "SplitConfig":
        if abs(self.train + self.validation + self.test - 1.0) > 1e-9:
            msg = "split fractions must sum to 1"
            raise ValueError(msg)
        return self


def split_of(prompt_id: str, config: SplitConfig) -> str:
    """The split of one prompt: a pure function of the salt and the prompt id."""
    digest = hashlib.sha256(f"{config.salt}:{prompt_id}".encode()).digest()
    u = int.from_bytes(digest[:8], "big") / 2**64
    if u < config.train:
        return "train"
    if u < config.train + config.validation:
        return "validation"
    return "test"


def _dpo(
    pairs: Sequence[PairVotes], prompts: Mapping[str, Prompt], responses: Mapping[str, Response]
) -> list[dict[str, Any]]:
    rows = []
    for v in pairs:
        outcome = v.chosen_rejected
        if outcome is None:
            continue
        chosen, rejected = outcome
        rows.append(
            {
                "prompt_id": v.prompt_id,
                "prompt": prompts[v.prompt_id].text,
                "chosen_id": chosen,
                "chosen": responses[chosen].text,
                "rejected_id": rejected,
                "rejected": responses[rejected].text,
                "n_votes": v.n_votes,
                "agreement": v.agreement,
            }
        )
    return rows


def _kto(
    pairs: Sequence[PairVotes], prompts: Mapping[str, Prompt], responses: Mapping[str, Response]
) -> list[dict[str, Any]]:
    wins: dict[tuple[str, str], int] = defaultdict(int)
    losses: dict[tuple[str, str], int] = defaultdict(int)
    for v in pairs:
        outcome = v.chosen_rejected
        if outcome is None:
            continue
        wins[(v.prompt_id, outcome[0])] += 1
        losses[(v.prompt_id, outcome[1])] += 1
    rows = []
    for prompt_id, response_id in sorted({*wins, *losses}):
        n_won, n_lost = wins[(prompt_id, response_id)], losses[(prompt_id, response_id)]
        if n_won == n_lost:
            continue
        rows.append(
            {
                "prompt_id": prompt_id,
                "prompt": prompts[prompt_id].text,
                "response_id": response_id,
                "completion": responses[response_id].text,
                "label": n_won > n_lost,
                "n_pairs_won": n_won,
                "n_pairs_lost": n_lost,
            }
        )
    return rows


def _rm(
    pairs: Sequence[PairVotes], prompts: Mapping[str, Prompt], responses: Mapping[str, Response]
) -> list[dict[str, Any]]:
    return [
        {
            "prompt_id": v.prompt_id,
            "prompt": prompts[v.prompt_id].text,
            "response_a_id": v.a_id,
            "response_a": responses[v.a_id].text,
            "response_b_id": v.b_id,
            "response_b": responses[v.b_id].text,
            "n_a": v.n_a,
            "n_b": v.n_b,
            "n_tie": v.n_tie,
            "p_a": (v.n_a + 0.5 * v.n_tie) / v.n_votes,
        }
        for v in pairs
        if v.n_votes > 0
    ]


_BUILDERS = {ExportFormat.DPO: _dpo, ExportFormat.KTO: _kto, ExportFormat.RM: _rm}


def build_rows(
    fmt: ExportFormat,
    pairs: Iterable[PairVotes],
    prompts: Iterable[Prompt],
    responses: Iterable[Response],
    split: SplitConfig | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Rows of every split for one format, in a stable order."""
    split = split or SplitConfig()
    by_split: dict[str, list[PairVotes]] = defaultdict(list)
    for v in pairs:
        by_split[split_of(v.prompt_id, split)].append(v)
    prompt_map = {p.id: p for p in prompts}
    response_map = {r.id: r for r in responses}
    ordered = {
        name: sorted(by_split.get(name, []), key=lambda v: (v.prompt_id, v.a_id, v.b_id))
        for name in SPLITS
    }
    return {name: _BUILDERS[fmt](ordered[name], prompt_map, response_map) for name in SPLITS}


def to_jsonl(rows: Iterable[Mapping[str, Any]]) -> bytes:
    """Sorted keys, one object per line, every line newline-terminated."""
    return b"".join(
        (json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n").encode() for row in rows
    )


class WrittenFile(Record):
    path: str
    """Relative to the export directory."""
    split: str
    n_rows: int
    sha256: str


def write_splits(
    out_dir: Path, fmt: ExportFormat, rows: Mapping[str, Sequence[Mapping[str, Any]]]
) -> tuple[WrittenFile, ...]:
    """Write ``<out_dir>/<fmt>/<split>.jsonl`` for every split (empty ones too)."""
    target = out_dir / fmt.value
    target.mkdir(parents=True, exist_ok=True)
    written = []
    for name in SPLITS:
        data = to_jsonl(rows[name])
        (target / f"{name}.jsonl").write_bytes(data)
        written.append(
            WrittenFile(
                path=f"{fmt.value}/{name}.jsonl",
                split=name,
                n_rows=len(rows[name]),
                sha256=hashlib.sha256(data).hexdigest(),
            )
        )
    return tuple(written)
