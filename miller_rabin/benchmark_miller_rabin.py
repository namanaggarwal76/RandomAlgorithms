"""Benchmark Miller-Rabin primality test across bit sizes, rounds, and seeds.

Metrics collected:
- accuracy vs sympy.isprime
- runtime per test (avg, median, p95)
- stability: variance of correctness across seeds
- false positive / false negative counts & rates
- theoretical error probability 4^{-k} (upper bound for composite misclassification)
- base collision fraction (randomness diagnostic)
- deterministic comparison timings (sympy.isprime, deterministic_mr when applicable)
- Carmichael detection (optional injected set)

Outputs:
- CSV file in results directory
Usage (small quick run):
    python benchmark_miller_rabin.py --bits 32 40 48 --rounds 5 10 15 --seeds 1 2 3 --samples 20
For larger dataset increase bits up to e.g. 128, 256, 512 (sympy handles these sizes reasonably).
"""
from __future__ import annotations
import argparse
import os
import statistics
import time
import math
import numpy as np
import pandas as pd
from sympy import isprime as sympy_isprime
from sympy import primerange
from src.miller_rabin import (
    miller_rabin,
    miller_rabin_with_bases,
    generate_random_number,
    deterministic_mr,
)
from src.miller_rabin import CARMICHAEL_NUMBERS_SMALL


def _fixed_small_bases(k: int) -> list[int]:
    # Take first primes starting from 2, at least k entries
    primes = list(primerange(2, 1000))
    if len(primes) < k:
        # extend conservatively
        primes = primes * ((k // len(primes)) + 1)
    return primes[:k]


def _choose_strategy(bits: int, k: int, strategy: str, n: int):
    strategy = strategy.lower()
    if strategy == 'random':
        return ('random', None)
    if strategy == 'fixed_small':
        return ('fixed_small', _fixed_small_bases(k))
    if strategy == 'det64':
        # ignore k; deterministic fixed bases for <=64-bit using deterministic_mr semantics
        return ('det64', None)
    # fallback to random
    return ('random', None)


def run_mr_with_strategy(n: int, k: int, strategy: str):
    if strategy == 'random':
        res = miller_rabin(n, k)
    elif strategy == 'fixed_small':
        bases = _fixed_small_bases(k)
        res = miller_rabin_with_bases(n, bases, stop_early=True)
    elif strategy == 'det64':
        # mimic MRResult for deterministic_mr decision
        start = time.perf_counter()
        ok = deterministic_mr(n)
        res = type('X', (), {})()
        res.is_probable_prime = ok
        res.time_seconds = time.perf_counter() - start
        res.witnesses = []
        res.bases_used = []
        res.base_collision_fraction = 0.0
    else:
        res = miller_rabin(n, k)
    return res


def run_benchmark(bit_sizes, rounds, seeds, samples, strategies, include_carmichael=False):
    records = []
    for bits in bit_sizes:
        for seed in seeds:
            for i in range(samples):
                n_seed = seed * 10_000 + i
                n = generate_random_number(bits, n_seed)
                truth = sympy_isprime(n)
                for k in rounds:
                    for strat in strategies:
                        res = run_mr_with_strategy(n, k, strat)
                        # Deterministic baseline timings
                        t0 = time.perf_counter()
                        truth_sym = sympy_isprime(n)
                        sym_t = time.perf_counter() - t0
                        det_t = None
                        det_ok = None
                        if bits <= 64:
                            t1 = time.perf_counter()
                            det_ok = deterministic_mr(n)
                            det_t = time.perf_counter() - t1
                        records.append({
                            "bits": bits,
                            "seed": seed,
                            "sample_index": i,
                            "rounds": k,
                            "strategy": strat,
                            "n": n,
                            "probable_prime": res.is_probable_prime,
                            "ground_truth": truth,
                            "correct": res.is_probable_prime == truth,
                            "time_core": res.time_seconds,
                            "witness_count": len(res.witnesses) if hasattr(res, 'witnesses') else 0,
                            "base_collision_fraction": getattr(res, 'base_collision_fraction', None),
                            "impl_sympy_time": sym_t,
                            "impl_det64_time": det_t,
                            "impl_det64_result": det_ok,
                            "is_carmichael": False,
                            "pseudoprime_flag": (res.is_probable_prime and (not truth)),
                        })
        # Optional: inject Carmichael tests
        if include_carmichael:
            for c in CARMICHAEL_NUMBERS_SMALL:
                if c.bit_count() > bits or c.bit_count() < bits - 2:
                    continue
                truth = sympy_isprime(c)
                for k in rounds:
                    for strat in strategies:
                        res = run_mr_with_strategy(c, k, strat)
                        t0 = time.perf_counter()
                        truth_sym = sympy_isprime(c)
                        sym_t = time.perf_counter() - t0
                        det_t = None
                        det_ok = None
                        if bits <= 64:
                            t1 = time.perf_counter()
                            det_ok = deterministic_mr(c)
                            det_t = time.perf_counter() - t1
                        records.append({
                            "bits": bits,
                            "seed": -1,
                            "sample_index": -1,
                            "rounds": k,
                            "strategy": strat,
                            "n": c,
                            "probable_prime": res.is_probable_prime,
                            "ground_truth": truth,
                            "correct": res.is_probable_prime == truth,
                            "time_core": res.time_seconds,
                            "witness_count": len(res.witnesses) if hasattr(res, 'witnesses') else 0,
                            "base_collision_fraction": getattr(res, 'base_collision_fraction', None),
                            "impl_sympy_time": sym_t,
                            "impl_det64_time": det_t,
                            "impl_det64_result": det_ok,
                            "is_carmichael": True,
                            "pseudoprime_flag": (res.is_probable_prime and (not truth)),
                        })
    return pd.DataFrame.from_records(records)


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    agg_rows = []
    for (bits, rounds, strategy), group in df.groupby(["bits", "rounds", "strategy"]):
        seeds = group["seed"].unique()
        seed_acc = []
        for s in seeds:
            sub = group[group.seed == s]
            if len(sub) == 0 or s == -1:
                continue
            seed_acc.append(sub.correct.mean())
        variance = statistics.pvariance(seed_acc) if len(seed_acc) > 1 else 0.0
        false_pos = ((group.probable_prime) & (~group.ground_truth)).sum()
        false_neg = ((~group.probable_prime) & (group.ground_truth)).sum()
        total = len(group)
        comp = group[~group.ground_truth]
        comp_total = len(comp)
        comp_fp = ((comp.probable_prime) & (~comp.ground_truth)).sum()
        # time distributions
        core_times = group.time_core.dropna().tolist()
        def pctl(vals, q):
            if not vals:
                return 0.0
            s = sorted(vals)
            idx = int((q/100.0)*(len(s)-1))
            return s[idx]
        # deterministic averages
        sym_t = group.impl_sympy_time.dropna().mean() if 'impl_sympy_time' in group else None
        det_t = group.impl_det64_time.dropna().mean() if 'impl_det64_time' in group else None
        agg_rows.append({
            "bits": bits,
            "rounds": rounds,
            "strategy": strategy,
            "samples": total,
            "accuracy": group.correct.mean(),
            "false_positives": false_pos,
            "false_negatives": false_neg,
            "false_positive_rate": false_pos / total if total else 0.0,
            "false_negative_rate": false_neg / total if total else 0.0,
            "composite_only_fp_rate": (comp_fp / comp_total) if comp_total else 0.0,
            "avg_time_core": group.time_core.mean(),
            "median_time_core": statistics.median(core_times) if core_times else 0.0,
            "p95_time_core": pctl(core_times, 95),
            "stability_variance": variance,
            "theoretical_error_prob": math.pow(4.0, -rounds),
            "avg_base_collision_fraction": group.base_collision_fraction.dropna().mean() if 'base_collision_fraction' in group else None,
            "impl_sympy_time": sym_t,
            "impl_det64_time": det_t,
            "carmichael_count": int(group.is_carmichael.sum()),
            "carmichael_pseudoprimes": int(((group.is_carmichael) & (group.probable_prime) & (~group.ground_truth)).sum()),
            "carmichael_detection_rate": 1.0 - (int(((group.is_carmichael) & (group.probable_prime) & (~group.ground_truth)).sum()) / int(group.is_carmichael.sum()) if int(group.is_carmichael.sum()) else 0.0),
        })
    return pd.DataFrame.from_records(agg_rows)


def recommend_k(summary_df: pd.DataFrame, fp_threshold=1e-6, acc_threshold=0.999) -> pd.DataFrame:
    rows = []
    for (bits, strategy), group in summary_df.groupby(["bits", "strategy"]):
        g = group.sort_values("rounds")
        best = g[(g.composite_only_fp_rate <= fp_threshold) & (g.accuracy >= acc_threshold)]
        k_rec = int(best.rounds.iloc[0]) if len(best) else None
        rows.append({"bits": bits, "strategy": strategy, "recommended_k": k_rec, "fp_threshold": fp_threshold, "acc_threshold": acc_threshold})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bits", nargs="*", type=int, default=[32, 40, 48, 56, 64])
    parser.add_argument("--rounds", nargs="*", type=int, default=[5, 10, 15])
    parser.add_argument("--seeds", nargs="*", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--samples", type=int, default=30, help="Samples per seed per bit size")
    parser.add_argument("--strategies", nargs="*", type=str, default=["random", "fixed_small"], help="Strategies: random, fixed_small, det64")
    parser.add_argument("--include-carmichael", action="store_true", help="Inject Carmichael numbers near bit sizes")
    parser.add_argument("--outdir", type=str, default="miller_rabin/results", help="Output directory")
    args = parser.parse_args()

    df = run_benchmark(args.bits, args.rounds, args.seeds, args.samples, args.strategies, include_carmichael=args.include_carmichael)
    summary = aggregate(df)
    rec = recommend_k(summary)

    os.makedirs(args.outdir, exist_ok=True)
    raw_path = os.path.join(args.outdir, "miller_rabin_raw.csv")
    sum_path = os.path.join(args.outdir, "miller_rabin_summary.csv")
    rec_path = os.path.join(args.outdir, "miller_rabin_recommended_k.csv")
    df.to_csv(raw_path, index=False)
    summary.to_csv(sum_path, index=False)
    rec.to_csv(rec_path, index=False)
    print(f"Wrote raw results to {raw_path}")
    print(f"Wrote summary to {sum_path}")
    print(f"Wrote recommendations to {rec_path}")

if __name__ == "__main__":
    main()
