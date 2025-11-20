"""Visualization helpers for benchmark outputs."""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator
import pandas as pd
import seaborn as sns
import numpy as np

# ensure project root on sys.path for direct script execution
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

sns.set_theme(style="whitegrid", palette="colorblind", context="talk")

RESULTS_DIR = Path("experiments/results")
RUNTIME_CSV = RESULTS_DIR / "runtime.csv"
ERROR_CSV = RESULTS_DIR / "error.csv"


def plot_runtime():
    if not RUNTIME_CSV.exists():
        raise FileNotFoundError(f"Missing runtime CSV: {RUNTIME_CSV}")
    df = pd.read_csv(RUNTIME_CSV)
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    # Scatter each raw sample with discrete colors for algorithms and frievald k values.
    # Use Set1 which is very distinctive (Red, Blue, Green, Purple, Orange, Yellow, Brown, Pink, Gray)
    full_palette = sns.color_palette("Set1", n_colors=9)
    
    algorithms = [alg for alg in sorted(df.algorithm.unique()) if alg != "frievald"]
    
    # Map specific algorithms to specific colors for consistency
    known_mapping = {
        "triple_loop": full_palette[0],  # Red
        "strassen": full_palette[1],     # Blue
        "numpy_matmul": full_palette[2], # Green
        "numpy": full_palette[2]
    }
    
    alg_color_map = {}
    idx = 3 # Start assigning unknown algorithms from Purple
    for alg in algorithms:
        if alg in known_mapping:
            alg_color_map[alg] = known_mapping[alg]
        else:
            alg_color_map[alg] = full_palette[idx % len(full_palette)]
            idx += 1

    # Plot non-Frievald algorithms
    for alg in algorithms:
        sub = df[df.algorithm == alg]
        if sub.empty:
            continue
        ax.scatter(sub.n, sub.seconds * 1e6, s=12, alpha=0.7, label=alg, color=alg_color_map[alg])
    # Frievald discrete colors per k
    fr = df[df.algorithm == "frievald"]
    if not fr.empty:
        k_values = sorted(fr.k.unique())
        # Continue using Set1 from index 3 onwards (Purple, Orange, Yellow...)
        # If we have many k values, we might cycle, but usually it's just 1, 10, 20
        k_colors = {}
        k_idx_start = 3
        for i, k_val in enumerate(k_values):
             color_idx = k_idx_start + i
             k_colors[k_val] = full_palette[color_idx % len(full_palette)]

        for idx, k_val in enumerate(k_values):
            sub = fr[fr.k == k_val]
            ax.scatter(sub.n, sub.seconds * 1e6, s=14, marker="o", edgecolors="black", linewidths=0.3, alpha=0.75,
                       label=f"frievald k={k_val}", color=k_colors[k_val])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Matrix size n")
    ax.set_ylabel("Runtime (microseconds)")
    ax.set_title("Runtime Scatter: Raw Samples")
    ax.xaxis.set_major_locator(LogLocator(base=10, numticks=7))
    ax.yaxis.set_major_locator(LogLocator(base=10, numticks=7))
    fig.subplots_adjust(right=0.78)
    legend = ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=True, fontsize=9, title="Algorithms", title_fontsize=10, ncol=1)
    legend._legend_box.align = "left"
    output_path = RESULTS_DIR / "runtime_plot.png"
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"Saved runtime plot to {output_path.resolve()}")


def plot_runtime_lines():
    if not RUNTIME_CSV.exists():
        print(f"Missing runtime CSV: {RUNTIME_CSV}")
        return
    df = pd.read_csv(RUNTIME_CSV)
    
    # Create a label column to distinguish frievald k values
    def get_label(row):
        if row['algorithm'] == 'frievald':
            return f"frievald k={int(row['k'])}"
        return row['algorithm']
    
    df['label'] = df.apply(get_label, axis=1)
    
    # Convert seconds to microseconds
    df['microseconds'] = df['seconds'] * 1e6
    df = df[df['microseconds'] > 0]

    # --- Robust Trend Estimation (Binned Medians) ---
    # This method follows the trend while removing outliers by taking the median in log-spaced bins.
    
    # 1. Define Logarithmic Bins
    n_min = max(1, df['n'].min())
    n_max = df['n'].max()
    num_bins = 30 # Sufficient resolution to show curvature/trend
    bins = np.logspace(np.log10(n_min), np.log10(n_max), num_bins)
    
    # 2. Assign Data to Bins
    df['bin'] = pd.cut(df['n'], bins, include_lowest=True)
    
    # 3. Compute Median per Bin (Robust to Outliers)
    # Group by label and bin, then take the median of n and microseconds
    trend_df = df.groupby(['label', 'bin']).agg({
        'n': 'median',
        'microseconds': 'median'
    }).reset_index()
    
    # 4. Clean up (remove empty bins)
    trend_df = trend_df.dropna()
    trend_df = trend_df.sort_values(by=['label', 'n'])

    # --- Plotting ---
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    
    # Plot the trend lines
    sns.lineplot(data=trend_df, x="n", y="microseconds", hue="label", style="label", 
                 markers=True, dashes=False, ax=ax, palette="tab10", linewidth=2.5, markersize=7)
    
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Matrix size n")
    ax.set_ylabel("Runtime (microseconds)")
    ax.set_title("Runtime Trends (Binned Medians)")
    
    ax.xaxis.set_major_locator(LogLocator(base=10, numticks=7))
    ax.yaxis.set_major_locator(LogLocator(base=10, numticks=7))
    
    fig.subplots_adjust(right=0.70)
    legend = ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=True, fontsize=9, title="Algorithms", title_fontsize=10)
    legend._legend_box.align = "left"
    
    output_path = RESULTS_DIR / "runtime_lines.png"
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"Saved runtime lines plot to {output_path.resolve()}")


def plot_error_probability():
    if not ERROR_CSV.exists():
        raise FileNotFoundError(f"Missing error CSV: {ERROR_CSV}")
    df = pd.read_csv(ERROR_CSV)
    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    if "cumulative_detection_prob" in df.columns:  # detection curve mode
        sns.lineplot(data=df, x="k", y="cumulative_detection_prob", marker="o", label="Observed Cumulative Detect", ax=ax)
        sns.lineplot(data=df, x="k", y="theoretical_cumulative", marker="s", label="Theoretical 1-2^-k", ax=ax)
        sns.lineplot(data=df, x="k", y="survival_prob", marker="d", label="Observed Survival", ax=ax)
        sns.lineplot(data=df, x="k", y="theoretical_survival", marker="^", label="Theoretical 2^-k", ax=ax)
        ax.set_ylabel("Probability")
        ax.set_title("Frievald Detection: Cumulative vs Survival")
    else:  # fixed_k mode
        sns.lineplot(data=df, x="k", y="observed_failure", marker="o", label="Observed", ax=ax)
        sns.lineplot(data=df, x="k", y="theoretical_failure", marker="s", label="Theoretical", ax=ax)
        if {"ci_lower", "ci_upper"}.issubset(df.columns):
            ax.fill_between(df["k"], df["ci_lower"], df["ci_upper"], alpha=0.18, color="gray", label="95% CI")
        ax.set_ylabel("Failure probability")
        ax.set_title("Frievald's Failure Probability")
    ax.set_yscale("log")
    ax.set_xlabel("Iterations k")
    ax.yaxis.set_major_locator(LogLocator(base=10, numticks=7))
    fig.subplots_adjust(right=0.78)
    legend = ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=True, fontsize=9, title="Legend", title_fontsize=10)
    legend._legend_box.align = "left"
    output_path = RESULTS_DIR / "error_plot.png"
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"Saved error probability plot to {output_path.resolve()}")


def main():
    plot_runtime()
    plot_runtime_lines()
    plot_error_probability()


if __name__ == "__main__":
    main()
