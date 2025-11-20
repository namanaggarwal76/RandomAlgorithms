#!/usr/bin/env python3
import argparse
import os
import math
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HEADER = [
    'timestamp_utc_iso','category','input_file','n','seed','rep_id',
    'elapsed_ms','cpu_ms','comparisons','swaps','correct'
]


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def load_results(csv_path: str) -> pd.DataFrame:
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Results CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    missing = [c for c in HEADER if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns in results: {missing}")

    # Type conversions
    int_cols = ['n', 'seed', 'rep_id', 'comparisons', 'swaps', 'correct']
    float_cols = ['elapsed_ms', 'cpu_ms']
    for c in int_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce').astype('Int64')
    for c in float_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    # Ensure categories and filenames are strings
    df['category'] = df['category'].astype(str)
    df['filename'] = df['input_file'].apply(lambda p: os.path.basename(str(p)))
    df = df.dropna(subset=['n','elapsed_ms'])
    df['n'] = df['n'].astype(int)
    df['correct'] = df['correct'].fillna(0).astype(int)
    return df


def plot_boxplot_per_category(df_cat: pd.DataFrame, out_dir: str, category: str):
    # Group by filename
    groups = [g['elapsed_ms'].values for fname, g in df_cat.groupby('filename')]
    labels = [fname for fname, _ in df_cat.groupby('filename')]
    if not groups:
        return

    plt.figure(figsize=(max(6, len(labels) * 0.6), 5))
    plt.boxplot(groups, labels=labels, showfliers=False)
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('elapsed_ms')
    plt.title(f'QuickSort Runtime Distribution per File — {category}')
    plt.tight_layout()
    out_path = os.path.join(out_dir, f"{category}_boxplot.png")
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_n_vs_runtime(df_cat: pd.DataFrame, out_dir: str, category: str):
    # Median runtime per file
    agg = df_cat.groupby('filename').agg(n=('n', 'median'), time_ms=('elapsed_ms', 'median')).reset_index()
    if agg.empty:
        return
    agg['n'] = agg['n'].astype(int)

    x = agg['n'].values.astype(float)
    x2 = np.where(x > 1, x * np.log2(x), x)  # n * log2(n), safe at n<=1
    y = agg['time_ms'].values.astype(float)

    # Fit k in y ~ k * (n log n)
    denom = np.dot(x2, x2)
    k = float(np.dot(x2, y) / denom) if denom > 0 else 0.0

    # Create smooth line over sorted n
    order = np.argsort(x)
    xs = x[order]
    xs2 = np.where(xs > 1, xs * np.log2(xs), xs)
    ys = k * xs2

    plt.figure(figsize=(6, 5))
    plt.scatter(x, y, c='tab:blue', label='median per file')
    plt.plot(xs, ys, c='tab:orange', label=f'fit: k*n*log2(n), k={k:.6g}')
    plt.xlabel('n')
    plt.ylabel('median elapsed_ms')
    plt.title(f'Median Runtime vs n — {category}')
    plt.legend()
    plt.tight_layout()
    out_path = os.path.join(out_dir, f"{category}_n_vs_runtime.png")
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_comp_vs_time(df_cat: pd.DataFrame, out_dir: str, category: str):
    if df_cat.empty:
        return
    plt.figure(figsize=(6, 5))
    plt.scatter(df_cat['comparisons'], df_cat['elapsed_ms'], s=10, alpha=0.7)
    plt.xlabel('comparisons')
    plt.ylabel('elapsed_ms')
    plt.title(f'Comparisons vs Runtime — {category}')
    plt.tight_layout()
    out_path = os.path.join(out_dir, f"{category}_comp_vs_time.png")
    plt.savefig(out_path, dpi=150)
    plt.close()


def save_summaries(df: pd.DataFrame, out_dir: str):
    ensure_dir(out_dir)
    all_rows = []
    for category, df_cat in df.groupby('category'):
        g = df_cat.groupby('filename').agg(
            n=('n', 'median'),
            median_time=('elapsed_ms', 'median'),
            mean_time=('elapsed_ms', 'mean'),
            std_time=('elapsed_ms', 'std'),
            median_comparisons=('comparisons', 'median'),
            median_swaps=('swaps', 'median'),
            correctness_rate=('correct', 'mean'),
        ).reset_index()
        g['category'] = category
        # Order columns
        g = g[['category','filename','n','median_time','mean_time','std_time','median_comparisons','median_swaps','correctness_rate']]
        out_csv = os.path.join(out_dir, f"{category}_summary.csv")
        g.to_csv(out_csv, index=False)
        all_rows.append(g)

    if all_rows:
        all_df = pd.concat(all_rows, ignore_index=True)
        all_df.to_csv(os.path.join(out_dir, 'summary_all_categories.csv'), index=False)


def run():
    parser = argparse.ArgumentParser(description='Analyze randomized quicksort results and produce plots.')
    parser.add_argument('--results', required=True, help='Path to master CSV (e.g., ./results/qsort/qsort_master.csv)')
    parser.add_argument('--out-dir', required=True, help='Directory to save plots and summaries')
    args = parser.parse_args()

    ensure_dir(args.out_dir)
    df = load_results(args.results)

    for category, df_cat in df.groupby('category'):
        plot_boxplot_per_category(df_cat, args.out_dir, category)
        plot_n_vs_runtime(df_cat, args.out_dir, category)
        # Optional plot
        plot_comp_vs_time(df_cat, args.out_dir, category)

    save_summaries(df, args.out_dir)
    print(f"Analysis complete. Plots and summaries saved to {args.out_dir}")


if __name__ == '__main__':
    run()
