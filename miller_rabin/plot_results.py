"""Plot results from Miller-Rabin benchmark CSVs.

Usage:
    python plot_results.py --summary miller_rabin/results/miller_rabin_summary.csv --outdir miller_rabin/results/plots
Generates per bit size:
    accuracy_vs_rounds_<bits>.png
    runtime_vs_rounds_<bits>.png
    stability_vs_rounds_<bits>.png
    fp_rate_vs_rounds_<bits>.png
And overall:
    accuracy_vs_error_bound.png (observed accuracy vs theoretical bound)
"""
from __future__ import annotations
import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt


def plot_for_bits(df, bits, outdir):
    sub = df[df.bits == bits].sort_values("rounds")
    if sub.empty:
        return
    # Accuracy plot
    plt.figure(figsize=(6,4))
    plt.plot(sub.rounds, sub.accuracy, marker='o')
    plt.xlabel("Rounds (k)")
    plt.ylabel("Accuracy")
    plt.title(f"Miller-Rabin Accuracy (bits={bits})")
    plt.grid(alpha=0.3)
    plt.ylim(0.0, 1.05)
    plt.savefig(os.path.join(outdir, f"accuracy_vs_rounds_{bits}.png"), dpi=150)
    plt.close()

    # Runtime plot (avg & p95 core times if available)
    plt.figure(figsize=(6,4))
    if 'avg_time_core' in sub.columns:
        plt.plot(sub.rounds, sub.avg_time_core, marker='o', label='avg core')
    if 'p95_time_core' in sub.columns:
        plt.plot(sub.rounds, sub.p95_time_core, marker='s', label='p95 core')
    plt.xlabel("Rounds (k)")
    plt.ylabel("Time (s)")
    plt.title(f"Miller-Rabin Core Runtime (bits={bits})")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.savefig(os.path.join(outdir, f"runtime_vs_rounds_{bits}.png"), dpi=150)
    plt.close()

    # Stability plot
    if 'stability_variance' in sub.columns:
        plt.figure(figsize=(6,4))
        plt.plot(sub.rounds, sub.stability_variance, marker='o', color='purple')
        plt.xlabel("Rounds (k)")
        plt.ylabel("Variance of per-seed accuracy")
        plt.title(f"Miller-Rabin Stability (bits={bits})")
        plt.grid(alpha=0.3)
        plt.savefig(os.path.join(outdir, f"stability_vs_rounds_{bits}.png"), dpi=150)
        plt.close()

    # False positive rate plot
    if 'false_positive_rate' in sub.columns:
        plt.figure(figsize=(6,4))
        plt.plot(sub.rounds, sub.false_positive_rate, marker='o', color='red')
        plt.xlabel("Rounds (k)")
        plt.ylabel("False Positive Rate")
        plt.title(f"False Positive Rate (bits={bits})")
        plt.grid(alpha=0.3)
        plt.yscale('log')
        plt.savefig(os.path.join(outdir, f"fp_rate_vs_rounds_{bits}.png"), dpi=150)
        plt.close()


def overall_accuracy_vs_bound(df, outdir):
    if 'theoretical_error_prob' not in df.columns:
        return
    plt.figure(figsize=(6,4))
    for bits in sorted(df.bits.unique()):
        sub = df[df.bits == bits]
        plt.plot(sub.theoretical_error_prob, sub.accuracy, marker='o', label=f'bits={bits}')
    plt.xscale('log')
    plt.xlabel('Theoretical error bound (4^{-k})')
    plt.ylabel('Observed Accuracy')
    plt.title('Accuracy vs Theoretical Error Bound')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'accuracy_vs_error_bound.png'), dpi=160)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--summary', required=True, help='Summary CSV path')
    ap.add_argument('--outdir', default='miller_rabin/results/plots')
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    df = pd.read_csv(args.summary)
    for bits in sorted(df.bits.unique()):
        plot_for_bits(df, bits, args.outdir)
    overall_accuracy_vs_bound(df, args.outdir)
    print(f"Plots written to {args.outdir}")


if __name__ == '__main__':
    main()
