#!/usr/bin/env python3
"""
Generate reproducible data streams for LogLog/HLL/HLL++ benchmarks.

By default the script emits a suite of heterogeneous scenarios (uniform, zipfian,
bounded-range, clustered) so the benchmarking pipeline can cover multiple
cardinality regimes. Passing ``--sizes`` reverts to the simpler
uniform+zipf generation for quick smoke tests.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np


def write_stream(path: Path, data: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for value in data:
            fh.write(f"{int(value)}\n")


def build_uniform(rng: np.random.Generator, size: int, *, high: int = 1 << 62) -> np.ndarray:
    return rng.integers(0, high, size=size, dtype=np.int64)


def build_zipf(rng: np.random.Generator, size: int, a: float = 1.2) -> np.ndarray:
    base = rng.zipf(a=a, size=size)
    return base.astype(np.int64)


def build_bounded(rng: np.random.Generator, size: int, limit: int) -> np.ndarray:
    return rng.integers(0, limit, size=size, dtype=np.int64)


def build_clustered(rng: np.random.Generator, size: int, burst: int = 50_000) -> np.ndarray:
    """
    Create bursts of sequential IDs to emulate cache-friendly streams that still
    grow monotonically overall.
    """
    data = []
    generated = 0
    block_id = 0
    while generated < size:
        block_size = min(burst, size - generated)
        start = block_id * burst
        block_vals = np.arange(start, start + block_size, dtype=np.int64)
        rng.shuffle(block_vals)
        data.append(block_vals)
        generated += block_size
        block_id += 1
    if not data:
        return np.array([], dtype=np.int64)
    return np.concatenate(data)[:size]


@dataclass(frozen=True)
class Scenario:
    name: str
    size: int
    builder: Callable[[np.random.Generator, int], np.ndarray]
    description: str


DEFAULT_SCENARIOS: tuple[Scenario, ...] = (
    Scenario(
        name="uniform_small_500k",
        size=500_000,
        builder=lambda rng, size: build_uniform(rng, size),
        description="Small uniform stream to validate low-volume behaviour.",
    ),
    Scenario(
        name="uniform_2m",
        size=2_000_000,
        builder=lambda rng, size: build_uniform(rng, size),
        description="High-cardinality uniform stream to stress estimator linearity.",
    ),
    Scenario(
        name="uniform_5m",
        size=5_000_000,
        builder=lambda rng, size: build_uniform(rng, size),
        description="Very large uniform stream for near-capacity evaluation.",
    ),
    Scenario(
        name="zipf_light_500k",
        size=500_000,
        builder=lambda rng, size: build_zipf(rng, size, a=1.2),
        description="Moderate skew workload.",
    ),
    Scenario(
        name="zipf_heavy_2m",
        size=2_000_000,
        builder=lambda rng, size: build_zipf(rng, size, a=1.05),
        description="Power-law workload representing skewed real-world IDs.",
    ),
    Scenario(
        name="bounded_dense_500k",
        size=500_000,
        builder=lambda rng, size: build_bounded(rng, size, limit=1_000),
        description="Extremely low cardinality to test sparse regimes quickly.",
    ),
    Scenario(
        name="bounded_dense_1m",
        size=1_000_000,
        builder=lambda rng, size: build_bounded(rng, size, limit=2_000),
        description="Low-cardinality stream (heavy duplicates) to test sparse mode.",
    ),
    Scenario(
        name="clustered_wave_1m",
        size=1_000_000,
        builder=lambda rng, size: build_clustered(rng, size, burst=40_000),
        description="Sequential bursts with moderate duration.",
    ),
    Scenario(
        name="clustered_wave_3m",
        size=3_000_000,
        builder=lambda rng, size: build_clustered(rng, size, burst=60_000),
        description="Sequential bursts to mimic temporal locality.",
    ),
)


def generate_default_scenarios(
    out_root: Path,
    seed: int,
    scenarios: Iterable[Scenario],
) -> None:
    for idx, scenario in enumerate(scenarios):
        rng = np.random.default_rng(seed + idx * 17 + scenario.size)
        path = out_root / f"{scenario.name}_{scenario.size}.txt"
        print(f"[datasets] {scenario.name}: size={scenario.size:,} -> {path}")
        write_stream(path, scenario.builder(rng, scenario.size))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate datasets for LL/HLL/HLL++ benchmarks.")
    parser.add_argument("--out-dir", default="datasets/cardinality", help="Destination folder.")
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        help="Optional list of sizes for classic uniform+zipf generation.",
    )
    parser.add_argument("--seed", type=int, default=7, help="Base RNG seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_root = Path(args.out_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    if args.sizes:
        for size in args.sizes:
            rng = np.random.default_rng(args.seed + size)
            uniform_path = out_root / f"uniform_{size}.txt"
            zipf_path = out_root / f"zipf_{size}.txt"
            print(f"[datasets] Generating uniform stream ({size:,}) -> {uniform_path}")
            write_stream(uniform_path, build_uniform(rng, size))
            print(f"[datasets] Generating zipfian stream ({size:,}) -> {zipf_path}")
            write_stream(zipf_path, build_zipf(rng, size))
    else:
        generate_default_scenarios(out_root, args.seed, DEFAULT_SCENARIOS)

    print("Dataset generation complete.")


if __name__ == "__main__":
    main()
