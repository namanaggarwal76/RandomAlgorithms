#!/usr/bin/env python3
"""
Analyze LogLog/HLL/HLL++ benchmark outputs and produce consolidated visuals.

Artifacts:
* ``relative_error_matrix.png`` - log/log curves showing how error evolves with cardinality.
* ``summary_dashboard.png`` - combined bar charts for final error, throughput, and memory.
* ``relative_error_heatmap.png`` - checkpoint error heatmap per algorithm/dataset.
* ``cardinality_metrics.csv`` - merged metrics table for the written report.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_data(raw_path: Path, summary_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not raw_path.exists() or not summary_path.exists():
        raise FileNotFoundError("Missing raw or summary CSV. Run the benchmarks first.")
    raw = pd.read_csv(raw_path)
    summary = pd.read_csv(summary_path)
    return raw, summary


def plot_metric(summary: pd.DataFrame, metric: str, title: str, filename: Path, logy: bool = False) -> None:
    plt.figure(figsize=(10, 6))
    display_df = summary.copy()
    display_df["dataset_label"] = display_df["distribution"].str.replace("_", " ").str.title()
    sns.barplot(data=display_df, x="dataset_label", y=metric, hue="algorithm")
    if logy:
        plt.yscale("log")
    plt.title(title)
    plt.xlabel("Dataset")
    plt.ylabel(title)
    plt.xticks(rotation=20)
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=170)
    plt.close()


def plot_checkpoint_heatmap(raw: pd.DataFrame, out_dir: Path) -> None:
    heat_df = raw.copy()
    heat_df["dataset_label"] = heat_df["distribution"].str.replace("_", " ").str.title()
    pivot = heat_df.pivot_table(
        index="dataset_label",
        columns="algorithm",
        values="relative_error",
        aggfunc="mean",
    )
    plt.figure(figsize=(8, 6))
    sns.heatmap(pivot, annot=True, fmt=".4f", cmap="viridis")
    plt.title("Average Relative Error Across Checkpoints")
    plt.tight_layout()
    plt.savefig(out_dir / "relative_error_heatmap.png", dpi=170)
    plt.close()


def save_metrics(raw: pd.DataFrame, summary: pd.DataFrame, out_dir: Path) -> None:
    stats = (
        raw.groupby(["dataset", "distribution", "algorithm"])
        .agg(
            avg_relative_error=("relative_error", "mean"),
            max_relative_error=("relative_error", "max"),
            p95_relative_error=("relative_error", lambda s: s.quantile(0.95)),
        )
        .reset_index()
    )
    merged = summary.merge(
        stats,
        on=["dataset", "distribution", "algorithm"],
        how="left",
    )
    merged.to_csv(out_dir / "cardinality_metrics.csv", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze LL/HLL/HLL++ benchmark outputs.")
    parser.add_argument("--raw", default="results/cardinality/cardinality_raw.csv", help="Raw CSV path.")
    parser.add_argument("--summary", default="results/cardinality/cardinality_summary.csv", help="Summary CSV path.")
    parser.add_argument("--out-dir", default="results/cardinality/analysis_plots", help="Plot output directory.")
    return parser.parse_args()


def aggregate_summary(summary: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        summary.groupby(
            ["dataset", "distribution", "dataset_size", "precision", "algorithm"],
            as_index=False,
        )
        .agg(
            stream_length=("stream_length", "mean"),
            true_cardinality=("true_cardinality", "mean"),
            final_estimate=("final_estimate", "mean"),
            relative_error=("relative_error", "mean"),
            memory_bytes=("memory_bytes", "mean"),
            throughput_ops=("throughput_ops", "mean"),
            elapsed_seconds=("elapsed_seconds", "mean"),
            avg_checkpoint_error=("avg_checkpoint_error", "mean"),
            max_checkpoint_error=("max_checkpoint_error", "mean"),
        )
    )
    return grouped


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)
    raw, summary = load_data(Path(args.raw), Path(args.summary))
    summary = aggregate_summary(summary)
    plot_metric(summary, "relative_error", "Final Relative Error", out_dir / "final_relative_error.png")
    plot_metric(summary, "throughput_ops", "Throughput (ops/sec)", out_dir / "throughput_comparison.png", logy=True)
    plot_metric(summary, "memory_bytes", "Memory Footprint (bytes)", out_dir / "memory_comparison.png", logy=True)
    plot_metric(summary, "avg_checkpoint_error", "Average Checkpoint Error", out_dir / "avg_checkpoint_error.png", logy=True)
    plot_metric(summary, "max_checkpoint_error", "Maximum Checkpoint Error", out_dir / "max_checkpoint_error.png", logy=True)
    plot_checkpoint_heatmap(raw, out_dir)
    summary.to_csv(out_dir / "cardinality_summary.csv", index=False)
    save_metrics(raw, summary, out_dir)
    print(f"[analysis] Plots saved to {out_dir}")


if __name__ == "__main__":
    main()
