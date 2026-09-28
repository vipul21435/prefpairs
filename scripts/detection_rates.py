"""How often each annotator check flags each simulated archetype.

Runs the position, length and self-consistency checks on seeded simulations
at two sizes and prints, per archetype, how many simulated annotators each
check flagged. Planted defects should be flagged often; reliable and noisy
annotators should be flagged about as rarely as the alpha level allows.

    uv run python scripts/detection_rates.py --seeds 20
"""

import argparse
from collections import Counter
from collections.abc import Sequence

from prefpairs.quality import length_bias, position_bias, self_consistency
from prefpairs.simulate import Archetype, SimulationConfig, simulate

SIZES = ((40, 4), (60, 8))
CHECKS = ("position", "length", "consistency", "any")


def rates(n_prompts: int, pairs_per_prompt: int, seeds: int) -> dict[Archetype, Counter[str]]:
    table: dict[Archetype, Counter[str]] = {a: Counter() for a in Archetype}
    for seed in range(seeds):
        config = SimulationConfig(seed=seed, n_prompts=n_prompts, pairs_per_prompt=pairs_per_prompt)
        dataset = simulate(config)
        flagged = {
            "position": set(position_bias(dataset.pairwise).flagged),
            "length": set(length_bias(dataset.pairwise, dataset.responses).flagged),
            "consistency": set(self_consistency(dataset.pairwise).flagged),
        }
        flagged["any"] = set().union(*flagged.values())
        for annotator, archetype in dataset.truth.annotator_archetypes.items():
            table[archetype]["annotators"] += 1
            for check in CHECKS:
                table[archetype][check] += annotator in flagged[check]
    return table


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", type=int, default=20, help="seeds 0..N-1 per size")
    args = parser.parse_args(argv)
    print(f"flagged / simulated annotators over seeds 0-{args.seeds - 1}, alpha 0.05 (Holm)")
    header = f"{'size':<22}  {'archetype':<15}" + "".join(f"  {c:>11}" for c in CHECKS)
    print(header)
    for n_prompts, pairs in SIZES:
        size = f"{n_prompts} prompts x {pairs} pairs"
        for archetype, counts in rates(n_prompts, pairs, args.seeds).items():
            total = counts["annotators"]
            cells = "".join(f"  {f'{counts[c]}/{total}':>11}" for c in CHECKS)
            print(f"{size:<22}  {archetype.value:<15}{cells}")


if __name__ == "__main__":
    main()
