"""DPO, KTO and reward-model rows, prompt-level splits and byte-stable JSONL."""

import json
from pathlib import Path

import pytest

from prefpairs.export import (
    SPLITS,
    ExportFormat,
    FilterConfig,
    SplitConfig,
    build_rows,
    select_pairs,
    split_of,
    to_jsonl,
    write_splits,
)
from prefpairs.schema import Prompt, Provenance, Response
from prefpairs.simulate import SimulationConfig, simulate
from tests.quality_helpers import judgment

PROVENANCE = Provenance.model_validate({"source": "synthetic"})
PROMPTS = [Prompt(id="p1", text="Say hi.")]
RESPONSES = [
    Response(id=r, prompt_id="p1", model=f"m-{r}", text=f"text {r}", provenance=PROVENANCE)
    for r in ("a", "b", "c")
]
ALL_TRAIN = SplitConfig(train=1.0, validation=0.0, test=0.0)


def rows(fmt: ExportFormat, data: list, config: FilterConfig | None = None) -> list[dict]:
    kept = select_pairs(data, config).kept
    return build_rows(fmt, kept, PROMPTS, RESPONSES, ALL_TRAIN)["train"]


def test_dpo_rows_name_the_majority_winner() -> None:
    data = [
        judgment("ann-1", "a", "b", "right"),
        judgment("ann-2", "b", "a", "left"),
        judgment("ann-3", "a", "b", "left"),
    ]
    (row,) = rows(ExportFormat.DPO, data)
    assert row["prompt"] == "Say hi."
    assert (row["chosen_id"], row["chosen"]) == ("b", "text b")
    assert (row["rejected_id"], row["rejected"]) == ("a", "text a")
    assert row["n_votes"] == 3
    assert row["agreement"] == pytest.approx(2 / 3)


def test_kto_labels_each_response_by_its_kept_pairs() -> None:
    data = [
        judgment("ann-1", "a", "b", "left"),  # a beats b
        judgment("ann-1", "b", "c", "left"),  # b beats c
    ]
    labels = {r["response_id"]: r["label"] for r in rows(ExportFormat.KTO, data)}
    # b won one pair and lost one: no clear label, left out
    assert labels == {"a": True, "c": False}


def test_rm_rows_carry_a_soft_label_and_can_keep_ties() -> None:
    data = [
        judgment("ann-1", "a", "b", "left"),
        judgment("ann-2", "a", "b", "tie"),
        judgment("ann-1", "b", "c", "tie"),
    ]
    config = FilterConfig(drop_ties=False, min_agreement=0.0)
    out = rows(ExportFormat.RM, data, config)
    assert [(r["response_a_id"], r["response_b_id"], r["p_a"]) for r in out] == [
        ("a", "b", 0.75),
        ("b", "c", 0.5),
    ]
    assert rows(ExportFormat.DPO, data, config)[0]["chosen_id"] == "a"
    assert len(rows(ExportFormat.DPO, data, config)) == 1


def test_split_is_a_pure_function_of_salt_and_prompt() -> None:
    config = SplitConfig()
    ids = [f"prompt-{i:04d}" for i in range(2000)]
    first = [split_of(p, config) for p in ids]
    assert first == [split_of(p, config) for p in ids]
    shares = {name: first.count(name) / len(ids) for name in SPLITS}
    assert shares["train"] == pytest.approx(0.8, abs=0.03)
    assert shares["validation"] == pytest.approx(0.1, abs=0.03)
    other = [split_of(p, SplitConfig(salt="other")) for p in ids]
    assert other != first


def test_split_fractions_must_sum_to_one() -> None:
    with pytest.raises(ValueError, match="sum to 1"):
        SplitConfig(train=0.5, validation=0.1, test=0.1)


def test_no_prompt_appears_in_two_splits_and_reruns_are_byte_identical(tmp_path: Path) -> None:
    dataset = simulate(SimulationConfig(seed=1, n_prompts=40))
    kept = select_pairs(dataset.pairwise).kept
    for fmt in ExportFormat:
        split_rows = build_rows(fmt, kept, dataset.prompts, dataset.responses)
        seen: dict[str, str] = {}
        for name, split in split_rows.items():
            for row in split:
                assert seen.setdefault(row["prompt_id"], name) == name
        assert all(split_rows[name] for name in SPLITS)
        first = write_splits(tmp_path / "one", fmt, split_rows)
        shuffled = build_rows(fmt, list(reversed(kept)), dataset.prompts, dataset.responses)
        second = write_splits(tmp_path / "two", fmt, shuffled)
        assert first == second
        for written in first:
            one = (tmp_path / "one" / written.path).read_bytes()
            assert one == (tmp_path / "two" / written.path).read_bytes()
            assert one.count(b"\n") == written.n_rows


def test_jsonl_sorts_keys_and_terminates_every_line() -> None:
    data = to_jsonl([{"b": 1, "a": "é"}, {"z": None}])
    assert data == '{"a": "é", "b": 1}\n{"z": null}\n'.encode()
    assert [json.loads(line) for line in data.decode().splitlines()] == [
        {"a": "é", "b": 1},
        {"z": None},
    ]
    assert to_jsonl([]) == b""
