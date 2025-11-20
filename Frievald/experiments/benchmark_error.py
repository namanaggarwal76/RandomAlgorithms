"""Empirical failure probability / detection curve study for Frievald's algorithm.

Modes:
1. fixed_k (default): replicate prior behavior—evaluate false positive rate for
    each k over many matrices and corruptions.
2. detection_curve: for each trial create a *single* corrupted matrix and run
    Frievald one iteration at a time until rejection; record the iteration count
    required to detect the error. From these counts compute survival probability
    P[still undetected after k] which should approximate (1/2)^k.
"""

from __future__ import annotations

import argparse
from math import sqrt
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.algorithms import frievald_verify, matmul_numpy
from src.utils import generate_random_matrix, inject_error

RESULTS_DIR = Path("experiments/results")
RESULTS_PATH = RESULTS_DIR / "error.csv"


def parse_int_list(csv: str):
    return [int(x) for x in csv.split(',') if x.strip()]


def ci_binomial(p_hat: float, n: int, z: float = 1.96):
    if n == 0:
        return (float('nan'), float('nan'))
    radius = z * sqrt(p_hat * (p_hat - p_hat**2) / n) if p_hat not in (0.0, 1.0) else z * sqrt((p_hat * (1 - p_hat) + 1/(2*n)) / n)
    lower = max(0.0, p_hat - radius)
    upper = min(1.0, p_hat + radius)
    return lower, upper


def run_error_benchmark_fixed(args: argparse.Namespace):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    records = []

    ks = range(args.k_min, args.k_max + 1) if args.k_range is None else parse_int_list(args.k_range)

    for k in ks:
        false_positives = 0
        total_trials = 0
        for m in range(args.matrices_per_k):
            A = generate_random_matrix(args.matrix_size, rng=rng)
            B = generate_random_matrix(args.matrix_size, rng=rng)
            C_true = matmul_numpy(A, B)
            for _ in range(args.corruptions_per_matrix):
                C_bad = inject_error(C_true, mode=args.error_mode, rng=rng)
                if frievald_verify(A, B, C_bad, k):
                    false_positives += 1
                total_trials += 1
        observed = false_positives / total_trials
        theoretical = 0.5 ** k
        ci_low, ci_high = ci_binomial(observed, total_trials)
        records.append({
            "k": k,
            "n": args.matrix_size,
            "matrices": args.matrices_per_k,
            "corruptions_per_matrix": args.corruptions_per_matrix,
            "trials": total_trials,
            "false_positives": false_positives,
            "observed_failure": observed,
            "theoretical_failure": theoretical,
            "ci_lower": ci_low,
            "ci_upper": ci_high,
            "error_mode": args.error_mode,
            "seed": args.seed,
        })
        print(f"k={k:2d}: observed={observed:.6f} theoretical={theoretical:.6f} CI=[{ci_low:.6f}, {ci_high:.6f}]")
    
    df = pd.DataFrame(records)
    if args.append and RESULTS_PATH.exists():
        df.to_csv(RESULTS_PATH, mode='a', header=False, index=False)
        print(f"Appended fixed-k benchmark to {RESULTS_PATH.resolve()}")
    else:
        df.to_csv(RESULTS_PATH, index=False)
        print(f"Saved fixed-k benchmark to {RESULTS_PATH.resolve()}")


def run_error_benchmark_detection(args: argparse.Namespace):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    detection_counts = []  # iterations until detection per trial
    trials = args.detection_trials
    n = args.matrix_size
    for t in range(trials):
        A = generate_random_matrix(n, rng=rng)
        B = generate_random_matrix(n, rng=rng)
        C_true = matmul_numpy(A, B)
        C_bad = inject_error(C_true, mode=args.error_mode, rng=rng)
        iterations = 0
        while True:
            iterations += 1
            if not frievald_verify(A, B, C_bad, 1):  # single iteration per step
                break
            if iterations >= args.max_detection_iterations:
                break  # safety cap
        detection_counts.append(iterations)
    # Compute survival probability for k = 1..max_k
    max_k = min(args.max_detection_iterations, args.k_max)
    records = []
    for k in range(1, max_k + 1):
        # cumulative detection: number of trials detected by iteration k
        detected_by_k = sum(1 for d in detection_counts if d <= k)
        cumulative_detection_prob = detected_by_k / trials
        survival_prob = 1.0 - cumulative_detection_prob  # still undetected after k
        theoretical_survival = 0.5 ** k
        theoretical_cumulative = 1.0 - theoretical_survival
        records.append({
            "k": k,
            "n": n,
            "trials": trials,
            "cumulative_detection_prob": cumulative_detection_prob,
            "survival_prob": survival_prob,
            "theoretical_survival": theoretical_survival,
            "theoretical_cumulative": theoretical_cumulative,
            "error_mode": args.error_mode,
            "seed": args.seed,
        })
        if k <= 10 or k % 5 == 0:
            print(f"k={k:2d}: cum_detect={cumulative_detection_prob:.6e} survival={survival_prob:.6e} theory_survival={theoretical_survival:.6e}")
    df = pd.DataFrame(records)
    if args.append and RESULTS_PATH.exists():
        df.to_csv(RESULTS_PATH, mode='a', header=False, index=False)
        print(f"Appended detection curve benchmark to {RESULTS_PATH.resolve()}")
    else:
        df.to_csv(RESULTS_PATH, index=False)
        print(f"Saved detection curve benchmark to {RESULTS_PATH.resolve()}")


def build_parser():
    p = argparse.ArgumentParser(description="Empirical failure probability benchmark for Frievald's algorithm")
    p.add_argument("--matrix-size", type=int, default=500, help="Matrix dimension n")
    p.add_argument("--matrices-per-k", type=int, default=25, help="Distinct (A,B) pairs per k")
    p.add_argument("--corruptions-per-matrix", type=int, default=20, help="Corrupted versions per matrix pair")
    p.add_argument("--k-min", type=int, default=1, help="Minimum k (inclusive) if --k-range not supplied")
    p.add_argument("--k-max", type=int, default=20, help="Maximum k (inclusive) if --k-range not supplied")
    p.add_argument("--k-range", type=str, default=None, help="Explicit comma-separated list of k values (overrides k-min/max)")
    p.add_argument("--error-mode", type=str, default="sparse", choices=["sparse", "dense"], help="Error injection mode")
    p.add_argument("--seed", type=int, default=123, help="RNG seed")
    p.add_argument("--mode", type=str, default="fixed_k", choices=["fixed_k", "detection_curve"], help="Benchmark mode")
    p.add_argument("--detection-trials", type=int, default=20000, help="Number of corrupted matrices for detection curve mode")
    p.add_argument("--max-detection-iterations", type=int, default=50, help="Safety cap on iterations per trial in detection mode")
    p.add_argument("--append", action="store_true", help="Append to existing CSV instead of overwriting")
    return p


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    if args.mode == "fixed_k":
        run_error_benchmark_fixed(args)
    else:
        run_error_benchmark_detection(args)
