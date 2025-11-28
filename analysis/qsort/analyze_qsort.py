#!/usr/bin/env python3
import argparse
import os
import math
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

HEADER = [
    'timestamp_utc_iso','category','input_file','n','seed','rep_id',
    'elapsed_ms','comparisons','swaps','correct','std_sort_ms',
    'recursion_depth','bad_split_count'
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
    int_cols = ['n', 'seed', 'rep_id', 'comparisons', 'swaps', 'correct', 'recursion_depth', 'bad_split_count']
    float_cols = ['elapsed_ms', 'std_sort_ms']
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


def plot_overall_runtime_scaling(df: pd.DataFrame, out_dir: str):
    # Q1: General time complexity graph irrespective of data type
    plt.figure(figsize=(10, 6))
    
    # Plot aggregate mean with error bands
    sns.lineplot(data=df, x='n', y='elapsed_ms', label='Average Performance (All Types)', marker='o', color='b')
    
    # Fit c * n log n to the overall mean of the largest n
    max_n = df['n'].max()
    if max_n > 0:
        avg_time_max_n = df[df['n'] == max_n]['elapsed_ms'].mean()
        c = avg_time_max_n / (max_n * np.log2(max_n))
        print(f"Overall fitted constant c: {c:.2e}")
        
        x_ref = np.linspace(df['n'].min(), df['n'].max(), 100)
        y_ref = c * x_ref * np.log2(x_ref)
        plt.plot(x_ref, y_ref, 'k--', alpha=0.8, label=f'Reference ~ {c:.1e} n log n')
        
        # Optional: O(n^2) reference anchored at min_n
        min_n = df['n'].min()
        if min_n > 1:
            start_y = c * min_n * np.log2(min_n)
            c2 = start_y / (min_n * min_n)
            y_ref2 = c2 * x_ref * x_ref
            plt.plot(x_ref, y_ref2, 'r:', alpha=0.5, label=f'Reference ~ {c2:.1e} n^2')

    plt.title('Overall Runtime Scaling (All Categories Combined)')
    plt.xlabel('Input Size (n)')
    plt.ylabel('Time (ms)')
    
    # Limit Y-axis
    if not df.empty:
        plt.ylim(0, df['elapsed_ms'].max() * 1.2)
        
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.savefig(os.path.join(out_dir, 'overall_runtime_scaling.png'), dpi=150)
    plt.close()


def plot_runtime_scaling(df_cat: pd.DataFrame, out_dir: str):
    # Q2: Runtime scaling by category
    # Plot: Line plot – average time_ms vs n
    # Series: one line per input_type (category)
    
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df_cat, x='n', y='elapsed_ms', hue='category', marker='o')
    
    # Overlay c * n log n reference
    # Fit c to the largest n of random category
    random_data = df_cat[df_cat['category'] == 'random']
    if not random_data.empty:
        max_n = random_data['n'].max()
        avg_time = random_data[random_data['n'] == max_n]['elapsed_ms'].mean()
        if max_n > 0:
            # Calculate c (constant factor) from the largest random input
            # c = time / (n * log2(n))
            c = avg_time / (max_n * np.log2(max_n))
            print(f"Fitted constant c for O(n log n): {c:.2e} (based on random input size {max_n})")

            x_ref = np.linspace(df_cat['n'].min(), df_cat['n'].max(), 100)
            y_ref = c * x_ref * np.log2(x_ref)
            plt.plot(x_ref, y_ref, 'k--', alpha=0.5, label=f'Reference ~ {c:.1e} n log n')

            # Add O(n^2) reference
            # Anchor to the first point of n log n reference to show divergence
            if not df_cat.empty:
                min_n = df_cat['n'].min()
                # Calculate c2 such that c2 * min_n^2 = c * min_n * log2(min_n)
                # This makes them start at the same point, allowing us to compare growth rates
                if min_n > 1:
                    start_y = c * min_n * np.log2(min_n)
                    c2 = start_y / (min_n * min_n)
                    print(f"Derived constant c2 for O(n^2): {c2:.2e} (anchored at n={min_n})")
                    y_ref2 = c2 * x_ref * x_ref
                    plt.plot(x_ref, y_ref2, 'r:', alpha=0.5, label=f'Reference ~ {c2:.1e} n^2')
    
    plt.title('Runtime Scaling by Category (Average Time vs n)')
    plt.xlabel('Input Size (n)')
    plt.ylabel('Time (ms)')
    
    # Limit Y-axis to keep data visible (n^2 curve will shoot off)
    if not df_cat.empty:
        plt.ylim(0, df_cat['elapsed_ms'].max() * 1.5)

    # plt.xscale('log')
    # plt.yscale('log')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.savefig(os.path.join(out_dir, 'runtime_scaling_by_category.png'), dpi=150)
    plt.close()


def plot_comparisons_scaling(df_cat: pd.DataFrame, out_dir: str):
    # Q3: Comparisons scaling
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df_cat, x='n', y='comparisons', hue='category', marker='o')
    plt.title('Comparisons Scaling (Average Comparisons vs n)')
    plt.xlabel('Input Size (n)')
    plt.ylabel('Comparisons')
    # plt.xscale('log')
    # plt.yscale('log')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.savefig(os.path.join(out_dir, 'comparisons_scaling.png'), dpi=150)
    plt.close()


def plot_swaps_scaling(df_cat: pd.DataFrame, out_dir: str):
    # Q4: Swaps scaling
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df_cat, x='n', y='swaps', hue='category', marker='o')
    plt.title('Swaps Scaling (Average Swaps vs n)')
    plt.xlabel('Input Size (n)')
    plt.ylabel('Swaps')
    # plt.xscale('log')
    # plt.yscale('log')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.savefig(os.path.join(out_dir, 'swaps_scaling.png'), dpi=150)
    plt.close()


def plot_seed_stability(df, out_dir):
    # Q5: Seed stability (fixed n)
    # Find the n with the most samples (likely the stability target n=50000)
    n_counts = df['n'].value_counts()
    if n_counts.empty: return
    target_n = n_counts.idxmax()
    
    sub = df[df['n'] == target_n]
    
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=sub, x='category', y='elapsed_ms')
    plt.title(f'Seed Stability (n={target_n})')
    plt.xlabel('Input Type')
    plt.ylabel('Time (ms)')
    plt.savefig(os.path.join(out_dir, 'seed_stability_time.png'), dpi=150)
    plt.close()
    
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=sub, x='category', y='comparisons')
    plt.title(f'Seed Stability - Comparisons (n={target_n})')
    plt.xlabel('Input Type')
    plt.ylabel('Comparisons')
    plt.savefig(os.path.join(out_dir, 'seed_stability_comparisons.png'), dpi=150)
    plt.close()


def plot_recursion_depth(df, out_dir):
    # Q6: Recursion depth vs n
    if 'recursion_depth' not in df.columns: return
    
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x='n', y='recursion_depth', hue='category', marker='o')
    plt.title('Recursion Depth vs n')
    plt.xlabel('Input Size (n)')
    plt.ylabel('Max Recursion Depth')
    # plt.xscale('log')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.savefig(os.path.join(out_dir, 'recursion_depth.png'), dpi=150)
    plt.close()


def plot_bad_splits(df, out_dir):
    # Q7: Pivot split quality
    if 'bad_split_count' not in df.columns: return
    
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x='n', y='bad_split_count', hue='category', marker='o')
    plt.title('Bad Split Count (>90%) vs n')
    plt.xlabel('Input Size (n)')
    plt.ylabel('Average Bad Split Count')
    # plt.xscale('log')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.savefig(os.path.join(out_dir, 'bad_splits.png'), dpi=150)
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
    parser.add_argument('--results', default='results/qsort/qsort_master.csv', help='Path to master CSV')
    parser.add_argument('--out-dir', default='results/qsort/analysis_plots', help='Directory to save plots and summaries')
    args = parser.parse_args()

    if not os.path.exists(args.results):
        print(f"Error: Results file not found: {args.results}\nPlease run benchmarks first.")
        return

    ensure_dir(args.out_dir)
    df = load_results(args.results)
    
    # Run new plots
    plot_overall_runtime_scaling(df, args.out_dir)
    plot_runtime_scaling(df, args.out_dir)
    plot_comparisons_scaling(df, args.out_dir)
    plot_swaps_scaling(df, args.out_dir)
    plot_seed_stability(df, args.out_dir)
    plot_recursion_depth(df, args.out_dir)
    plot_bad_splits(df, args.out_dir)

    save_summaries(df, args.out_dir)
    print(f"Analysis complete. Plots and summaries saved to {args.out_dir}")


if __name__ == '__main__':
    run()
