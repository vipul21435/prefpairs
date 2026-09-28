"""How often each audit flag fires for each simulated archetype.

Runs the full audit (the same code path as ``prefpairs audit``) on seeded
simulations at two sizes and prints, per archetype, how many simulated
annotators each flag caught. Planted defects should be flagged often;
reliable and noisy annotators should be flagged rarely.

    uv run python scripts/detection_rates.py --seeds 20
"""

import argparse
from collections import Counter
from collections.abc import Sequence

from prefpairs.quality.report import FLAG_ORDER, run_audit
from prefpairs.simulate import Archetype, SimulationConfig, simulate, write_simulation
from prefpairs.store import Store

SIZES = ((40, 4), (60, 8))
COLUMNS = (*FLAG_ORDER, "any")
SHORT = {"consistency": "consist", "intransitive": "intrans"}


def rates(n_prompts: int, pairs_per_prompt: int, seeds: int) -> dict[Archetype, Counter[str]]:
    table: dict[Archetype, Counter[str]] = {a: Counter() for a in Archetype}
    for seed in range(seeds):
        config = SimulationConfig(seed=seed, n_prompts=n_prompts, pairs_per_prompt=pairs_per_prompt)
        dataset = simulate(config)
        with Store.open(":memory:") as store:
            write_simulation(store, dataset)
            report = run_audit(store)
        verdicts = {a.annotator_id: a for a in report.annotators}
        for annotator, archetype in dataset.truth.annotator_archetypes.items():
            table[archetype]["annotators"] += 1
            flags = set(verdicts[annotator].flags)
            for column in FLAG_ORDER:
                table[archetype][column] += column in flags
            table[archetype]["any"] += bool(flags)
    return table


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", type=int, default=20, help="seeds 0..N-1 per size")
    args = parser.parse_args(argv)
    print(f"flagged / simulated annotators over seeds 0-{args.seeds - 1}, default audit config")
    header = f"{'size':<21}  {'archetype':<14}" + "".join(
        f"  {SHORT.get(c, c):>8}" for c in COLUMNS
    )
    print(header)
    for n_prompts, pairs in SIZES:
        size = f"{n_prompts} prompts x {pairs} pairs"
        for archetype, counts in rates(n_prompts, pairs, args.seeds).items():
            total = counts["annotators"]
            cells = "".join(f"  {f'{counts[c]}/{total}':>8}" for c in COLUMNS)
            print(f"{size:<21}  {archetype.value:<14}{cells}")


if __name__ == "__main__":
    main()
