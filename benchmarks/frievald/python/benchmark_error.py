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

# Hoist project src path so the script works when launched from repo root.
PROJECT_ROOT = Path(__file__).resolve().parents[3] / "src/frievald/python"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from algorithms import frievald_verify, matmul_numpy, matmul_triple_loop, matmul_strassen
from utils import generate_random_matrix, inject_error

RESULTS_DIR = Path("results/frievald")
RESULTS_PATH = RESULTS_DIR / "error.csv"


def parse_int_list(csv: str):
    """Convert a comma-separated list into ints while skipping blanks."""
    return [int(x) for x in csv.split(',') if x.strip()]


def ci_binomial(p_hat: float, n: int, z: float = 1.96):
    """Approximate confidence interval for a Bernoulli proportion."""
    if n == 0:
        return (float('nan'), float('nan'))
    radius = z * sqrt(p_hat * (p_hat - p_hat**2) / n) if p_hat not in (0.0, 1.0) else z * sqrt((p_hat * (1 - p_hat) + 1/(2*n)) / n)
    lower = max(0.0, p_hat - radius)
    upper = min(1.0, p_hat + radius)
    return lower, upper


def resolve_error_modes(args: argparse.Namespace) -> list[str]:
    """Handle legacy --error-mode flag while allowing comma-separated overrides."""
    if args.error_modes:
        modes = [mode.strip() for mode in args.error_modes.split(',') if mode.strip()]
        if not modes:
            raise ValueError("--error-modes must include at least one mode")
        return modes
    return [args.error_mode]


def run_error_benchmark_fixed(args: argparse.Namespace):
    """Measure Frievald false-positive rates for each ``k`` over multiple error modes."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    modes = resolve_error_modes(args)
    # Split seed deterministically so injection and verification streams are reproducible per mode.
    seed_seq = np.random.SeedSequence(args.seed)
    child_seqs = seed_seq.spawn(1 + len(modes) * 2)
    base_rng = np.random.default_rng(child_seqs[0])
    inject_rngs = {mode: np.random.default_rng(child_seqs[1 + idx]) for idx, mode in enumerate(modes)}
    verifier_rngs = {mode: np.random.default_rng(child_seqs[1 + len(modes) + idx]) for idx, mode in enumerate(modes)}
    records = []

    ks = range(args.k_min, args.k_max + 1) if args.k_range is None else parse_int_list(args.k_range)

    for k in ks:
        mode_false_positives = {mode: 0 for mode in modes}
        mode_trials = {mode: 0 for mode in modes}
        for m in range(args.matrices_per_k):
            # Sample a fresh matrix pair for each outer iteration so trials stay independent.
            A = generate_random_matrix(args.matrix_size, rng=base_rng)
            B = generate_random_matrix(args.matrix_size, rng=base_rng)
            C_true = matmul_numpy(A, B)
            for mode in modes:
                inject_rng = inject_rngs[mode]
                verifier_rng = verifier_rngs[mode]
                for _ in range(args.corruptions_per_matrix):
                    # Produce a corrupted product and tally whether Frievald misses it.
                    C_bad = inject_error(C_true, mode=mode, rng=inject_rng)
                    if frievald_verify(A, B, C_bad, k, rng=verifier_rng):
                        mode_false_positives[mode] += 1
                    mode_trials[mode] += 1
        theoretical = 0.5 ** k
        for mode in modes:
            total_trials = mode_trials[mode]
            false_positives = mode_false_positives[mode]
            observed = false_positives / total_trials if total_trials else float('nan')
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
                "error_mode": mode,
                "seed": args.seed,
            })
            print(
                f"mode={mode:<11} k={k:2d}: observed={observed:.6f} theoretical={theoretical:.6f} "
                f"CI=[{ci_low:.6f}, {ci_high:.6f}]"
            )
    
    df = pd.DataFrame(records)
    if args.append and RESULTS_PATH.exists():
        # Preserve existing CSV rows so repeated runs extend the dataset.
        df.to_csv(RESULTS_PATH, mode='a', header=False, index=False)
        print(f"Appended fixed-k benchmark to {RESULTS_PATH.resolve()}")
    else:
        # Start fresh when append not requested or file absent.
        df.to_csv(RESULTS_PATH, index=False)
        print(f"Saved fixed-k benchmark to {RESULTS_PATH.resolve()}")


def run_error_benchmark_detection(args: argparse.Namespace):
    """Record iteration counts required for Frievald to detect injected errors."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    modes = resolve_error_modes(args)
    # Matching seed strategy keeps detection-curve trials comparable to fixed-k runs.
    seed_seq = np.random.SeedSequence(args.seed)
    child_seqs = seed_seq.spawn(1 + len(modes) * 2)
    base_rng = np.random.default_rng(child_seqs[0])
    inject_rngs = {mode: np.random.default_rng(child_seqs[1 + idx]) for idx, mode in enumerate(modes)}
    verifier_rngs = {mode: np.random.default_rng(child_seqs[1 + len(modes) + idx]) for idx, mode in enumerate(modes)}
    trials = args.detection_trials
    n = args.matrix_size
    max_k = min(args.max_detection_iterations, args.k_max)
    records = []

    for mode in modes:
        detection_counts = []
        inject_rng = inject_rngs[mode]
        verifier_rng = verifier_rngs[mode]
        for _ in range(trials):
            # One corrupted matrix per trial, then measure how many iterations until rejection.
            A = generate_random_matrix(n, rng=base_rng)
            B = generate_random_matrix(n, rng=base_rng)
            C_true = matmul_numpy(A, B)
            C_bad = inject_error(C_true, mode=mode, rng=inject_rng)
            iterations = 0
            while True:
                iterations += 1
                # Probe with a single iteration each loop so we record the exact rejection iteration.
                if not frievald_verify(A, B, C_bad, 1, rng=verifier_rng):
                    break
                if iterations >= args.max_detection_iterations:
                    break
            detection_counts.append(iterations)
        for k in range(1, max_k + 1):
            # Convert raw detection counts into empirical CDF / survival curve values.
            detected_by_k = sum(1 for d in detection_counts if d <= k)
            cumulative_detection_prob = detected_by_k / trials
            survival_prob = 1.0 - cumulative_detection_prob
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
                "error_mode": mode,
                "seed": args.seed,
            })
            if k <= 10 or k % 5 == 0:
                print(
                    f"mode={mode:<11} k={k:2d}: cum_detect={cumulative_detection_prob:.6e} "
                    f"survival={survival_prob:.6e} theory_survival={theoretical_survival:.6e}"
                )
    df = pd.DataFrame(records)
    if args.append and RESULTS_PATH.exists():
        # Append detection curve results to the consolidated CSV.
        df.to_csv(RESULTS_PATH, mode='a', header=False, index=False)
        print(f"Appended detection curve benchmark to {RESULTS_PATH.resolve()}")
    else:
        # Overwrite by default so mode switches replace stale data.
        df.to_csv(RESULTS_PATH, index=False)
        print(f"Saved detection curve benchmark to {RESULTS_PATH.resolve()}")


def build_parser():
    """Construct the CLI argument parser shared by all error benchmarks."""
    p = argparse.ArgumentParser(description="Empirical failure probability benchmark for Frievald's algorithm")
    p.add_argument("--matrix-size", type=int, default=32, help="Matrix dimension n")
    p.add_argument("--matrices-per-k", type=int, default=25, help="Distinct (A,B) pairs per k")
    p.add_argument("--corruptions-per-matrix", type=int, default=32, help="Corrupted versions per matrix pair (tuned for n=32)")
    p.add_argument("--k-min", type=int, default=1, help="Minimum k (inclusive) if --k-range not supplied")
    p.add_argument("--k-max", type=int, default=20, help="Maximum k (inclusive) if --k-range not supplied")
    p.add_argument("--k-range", type=str, default=None, help="Explicit comma-separated list of k values (overrides k-min/max)")
    p.add_argument("--error-mode", type=str, default="sparse", choices=["sparse", "dense", "worst_case"], help="Error injection mode")
    p.add_argument("--error-modes", type=str, default=None, help="Comma-separated list of error modes to benchmark (overrides --error-mode)")
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
