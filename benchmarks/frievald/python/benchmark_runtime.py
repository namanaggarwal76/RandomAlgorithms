"""Runtime benchmarking: random sampled matrix sizes.

Instead of fixed size list and repeated averages, this benchmark samples a
random size n in [n_min, n_max] for each sample and records raw timing for all
algorithms:
    - numpy_matmul (deterministic BLAS)
    - strassen (threshold recursion)
    - triple_loop (optional skip above threshold)
    - frievald (each k supplied recorded separately)

Each row corresponds to one sampled matrix pair. Downstream visualization uses
scatter plots; no aggregation performed here.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
import sys
from typing import List

import numpy as np
import pandas as pd

# Ensure project root on path when executed directly.
# Add repository src directory (contains algorithms/utils) to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[3] / "src/frievald/python"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from algorithms import frievald_verify, matmul_numpy, matmul_triple_loop, matmul_strassen
from utils import generate_random_matrix

RESULTS_DIR = Path("results/frievald")
RESULTS_PATH = RESULTS_DIR / "runtime.csv"


def parse_int_list(csv: str) -> List[int]:
    return [int(x) for x in csv.split(',') if x.strip()]


def time_function(func, *args) -> float:
    start = time.perf_counter()
    func(*args)
    return time.perf_counter() - start


def benchmark_sample(n: int, frievald_iters: List[int], skip_triple_above: int, rng: np.random.Generator, strassen_threshold: int, sample_id: int):
    records = []
    A = generate_random_matrix(n, rng=rng)
    B = generate_random_matrix(n, rng=rng)
    t_np = time_function(matmul_numpy, A, B)
    C = matmul_numpy(A, B)
    records.append({"n": n, "algorithm": "numpy_matmul", "k": 0, "sample": sample_id, "seconds": t_np})

    if n <= 3000:
        t_strassen = time_function(matmul_strassen, A, B, strassen_threshold)
        records.append({"n": n, "algorithm": "strassen", "k": 0, "sample": sample_id, "seconds": t_strassen})
    else:
        records.append({"n": n, "algorithm": "strassen_skipped", "k": 0, "sample": sample_id, "seconds": float('nan')})

    if n <= skip_triple_above:
        t_triple = time_function(matmul_triple_loop, A, B)
        records.append({"n": n, "algorithm": "triple_loop", "k": 0, "sample": sample_id, "seconds": t_triple})
    else:
        records.append({"n": n, "algorithm": "triple_loop_skipped", "k": 0, "sample": sample_id, "seconds": float('nan')})

    for k in frievald_iters:
        t_f = time_function(frievald_verify, A, B, C, k)
        records.append({"n": n, "algorithm": "frievald", "k": k, "sample": sample_id, "seconds": t_f})
    return records


def run_runtime_benchmark(args: argparse.Namespace):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    all_records = []
    for sample_id in range(args.samples):
        n = rng.integers(args.n_min, args.n_max + 1)
        recs = benchmark_sample(n, args.frievald_iters, args.skip_triple_loop_above, rng, args.strassen_threshold, sample_id)
        all_records.extend(recs)
        if (sample_id + 1) % max(1, args.progress_interval) == 0:
            print(f"Sample {sample_id+1}/{args.samples} (n={n}) recorded {len(recs)} rows")
    df = pd.DataFrame(all_records)
    
    if args.append and RESULTS_PATH.exists():
        df.to_csv(RESULTS_PATH, mode='a', header=False, index=False)
        print(f"Appended sampled runtime benchmark to {RESULTS_PATH.resolve()} ({len(df)} rows)")
    else:
        df.to_csv(RESULTS_PATH, index=False)
        print(f"Saved sampled runtime benchmark to {RESULTS_PATH.resolve()} ({len(df)} rows)")


def build_parser():
    p = argparse.ArgumentParser(description="Random-sampled runtime benchmark for matrix algorithms")
    p.add_argument("--samples", type=int, default=100, help="Number of random size samples")
    p.add_argument("--n-min", type=int, default=4, help="Minimum matrix size (inclusive)")
    p.add_argument("--n-max", type=int, default=100, help="Maximum matrix size (inclusive)")
    p.add_argument("--frievald-iters", type=parse_int_list, default=parse_int_list("1,10,20"), help="Comma-separated k values for Frievald")
    p.add_argument("--skip-triple-loop-above", type=int, default=2000, help="Skip naive triple loop above this size (record NaN)")
    p.add_argument("--strassen-threshold", type=int, default=32, help="Base-case threshold for Strassen recursion")
    p.add_argument("--seed", type=int, default=42, help="RNG seed")
    p.add_argument("--progress-interval", type=int, default=25, help="Print progress every this many samples")
    p.add_argument("--append", action="store_true", help="Append to existing CSV instead of overwriting")
    return p


if __name__ == "__main__":
    parser = build_parser()
    run_runtime_benchmark(parser.parse_args())
