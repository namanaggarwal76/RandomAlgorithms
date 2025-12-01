#!/usr/bin/env python3
"""
File: analyze_qsort.py
Description: Comprehensive analysis and visualization for randomized quicksort benchmarks.
             Generates plots for runtime, comparisons, swaps, recursion depth, and space complexity.
"""

import argparse      # Used for parsing command-line arguments
import os            # Used for operating system dependent functionality
import math          # Used for mathematical operations
import pandas as pd  # Used for data manipulation and analysis
import numpy as np   # Used for numerical operations
import matplotlib    # Used for plotting backend configuration
matplotlib.use('Agg')  # Use non-interactive backend for server environments
import matplotlib.pyplot as plt  # Used for creating visualizations
import seaborn as sns  # Used for statistical data visualization

# Expected CSV header columns from benchmark results
HEADER = [
    'timestamp_utc_iso','category','input_file','n','seed','rep_id',
    'elapsed_ms','comparisons','swaps','correct','std_sort_ms',
    'recursion_depth','bad_split_count','max_stack_depth','estimated_stack_bytes'
]


def ensure_dir(path: str):
    """
    Ensures that a directory exists, creating it if necessary.
    
    Args:
        path (str): Directory path to create.
    """
    os.makedirs(path, exist_ok=True)


def load_results(csv_path: str) -> pd.DataFrame:
    """
    Loads and preprocesses benchmark results from CSV file.
    
    Args:
        csv_path (str): Path to the results CSV file.
        
    Returns:
        pd.DataFrame: Processed dataframe with correct types and cleaned data.
        
    Raises:
        FileNotFoundError: If CSV file doesn't exist.
        ValueError: If required columns are missing.
    """
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Results CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Validate required columns
    missing = [c for c in HEADER if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns in results: {missing}")

    # Type conversions for proper data handling
    int_cols = ['n', 'seed', 'rep_id', 'comparisons', 'swaps', 'correct', 'recursion_depth', 'bad_split_count', 'max_stack_depth', 'estimated_stack_bytes']
    float_cols = ['elapsed_ms', 'std_sort_ms']
    for c in int_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce').astype('Int64')
    for c in float_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    # Data cleaning and derived columns
    df['category'] = df['category'].astype(str)
    df['filename'] = df['input_file'].apply(lambda p: os.path.basename(str(p)))
    df = df.dropna(subset=['n','elapsed_ms'])  # Remove incomplete records
    df['n'] = df['n'].astype(int)
    df['correct'] = df['correct'].fillna(0).astype(int)
    return df


def plot_overall_runtime_scaling(df: pd.DataFrame, out_dir: str):
    """
    Plots overall runtime scaling across all data categories.
    Shows average performance with O(n log n) reference curve.
    
    Args:
        df (pd.DataFrame): Benchmark results dataframe.
        out_dir (str): Output directory for saving the plot.
    """
    # Q1: General time complexity graph irrespective of data type
    plt.figure(figsize=(10, 6))
    
    # Plot aggregate mean with error bands
    sns.lineplot(data=df, x='n', y='elapsed_ms', label='Average Performance (All Types)', marker='o', color='b')
    
    # Fit O(n log n) reference curve to empirical data
    # Calculate constant factor from largest dataset
    max_n = df['n'].max()
    if max_n > 0:
        avg_time_max_n = df[df['n'] == max_n]['elapsed_ms'].mean()
        c = avg_time_max_n / (max_n * np.log2(max_n))
        print(f"Overall fitted constant c: {c:.2e}")
        
        # Generate reference curves
        x_ref = np.linspace(df['n'].min(), df['n'].max(), 100)
        y_ref = c * x_ref * np.log2(x_ref)
        plt.plot(x_ref, y_ref, 'k--', alpha=0.8, label=f'Reference ~ {c:.1e} n log n')
        
        # Add O(n²) reference for comparison (worst case)
        min_n = df['n'].min()
        if min_n > 1:
            # Anchor O(n²) curve to start at same point as O(n log n)
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
    """
    Plots runtime scaling comparison across different data categories.
    Shows how performance varies with input type (random, sorted, duplicates, etc.).
    
    Args:
        df_cat (pd.DataFrame): Benchmark results dataframe.
        out_dir (str): Output directory for saving the plot.
    """
    # Q2: Runtime scaling by category
    # Plot: Line plot – average time_ms vs n
    # Series: one line per input_type (category)
    
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df_cat, x='n', y='elapsed_ms', hue='category', marker='o')
    
    # Overlay theoretical complexity references
    # Fit constant to random data (expected average case)
    random_data = df_cat[df_cat['category'] == 'random']
    if not random_data.empty:
        max_n = random_data['n'].max()
        avg_time = random_data[random_data['n'] == max_n]['elapsed_ms'].mean()
        if max_n > 0:
            # Calculate constant factor: c = time / (n * log₂(n))
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
    """
    Plots the number of comparisons vs input size across categories.
    Helps understand algorithmic complexity in terms of comparison operations.
    
    Args:
        df_cat (pd.DataFrame): Benchmark results dataframe.
        out_dir (str): Output directory for saving the plot.
    """
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
    """
    Plots the number of swaps vs input size across categories.
    Swaps indicate data movement, useful for cache performance analysis.
    
    Args:
        df_cat (pd.DataFrame): Benchmark results dataframe.
        out_dir (str): Output directory for saving the plot.
    """
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
    """
    Plots variance in performance across different random seeds (fixed input size).
    Assesses algorithmic stability and randomness impact.
    
    Args:
        df (pd.DataFrame): Benchmark results dataframe.
        out_dir (str): Output directory for saving plots.
    """
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
    """
    Plots maximum recursion depth vs input size.
    Expected to follow O(log n) for balanced partitions.
    
    Args:
        df (pd.DataFrame): Benchmark results dataframe.
        out_dir (str): Output directory for saving the plot.
    """
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
    """
    Plots the frequency of unbalanced partitions (>90% on one side).
    High counts indicate poor pivot selection or adversarial inputs.
    
    Args:
        df (pd.DataFrame): Benchmark results dataframe.
        out_dir (str): Output directory for saving the plot.
    """
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


def plot_stack_depth(df, out_dir):
    """
    Plots maximum call stack depth vs input size (space complexity).
    Expected O(log n) average case, O(n) worst case.
    
    Args:
        df (pd.DataFrame): Benchmark results dataframe.
        out_dir (str): Output directory for saving the plot.
    """
    # Q8: Stack depth (space complexity indicator)
    if 'max_stack_depth' not in df.columns: return
    
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x='n', y='max_stack_depth', hue='category', marker='o')
    
    # Overlay theoretical complexity references
    if not df.empty:
        max_n = df['n'].max()
        min_n = df['n'].min()
        if max_n > 0:
            # Fit O(log n) constant to random data (expected average case)
            random_data = df[df['category'] == 'random']
            if not random_data.empty:
                avg_depth = random_data[random_data['n'] == max_n]['max_stack_depth'].mean()
                c = avg_depth / np.log2(max_n)
                x_ref = np.linspace(min_n, max_n, 100)
                y_ref = c * np.log2(x_ref)
                plt.plot(x_ref, y_ref, 'k--', alpha=0.8, label=f'Expected O(log n) ~ {c:.1f} log₂(n)')
            
            # Show worst case O(n) for comparison
            if min_n > 1:
                y_worst = x_ref  # Linear: depth = n
                plt.plot(x_ref, y_worst, 'r:', alpha=0.5, label='Worst case O(n)')
    
    plt.title('Maximum Stack Depth vs Input Size (Space Complexity)')
    plt.xlabel('Input Size (n)')
    plt.ylabel('Max Stack Depth (recursive calls)')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.savefig(os.path.join(out_dir, 'stack_depth.png'), dpi=150)
    plt.close()


def plot_stack_memory(df, out_dir):
    """
    Plots estimated stack memory usage vs input size.
    Converts bytes to KB for readability. Expected O(log n) growth.
    
    Args:
        df (pd.DataFrame): Benchmark results dataframe.
        out_dir (str): Output directory for saving the plot.
    """
    # Q9: Estimated stack memory usage
    if 'estimated_stack_bytes' not in df.columns: return
    
    plt.figure(figsize=(10, 6))
    # Convert bytes to KB for better readability
    df_copy = df.copy()
    df_copy['stack_kb'] = df_copy['estimated_stack_bytes'] / 1024.0
    
    sns.lineplot(data=df_copy, x='n', y='stack_kb', hue='category', marker='o')
    
    # Overlay theoretical O(log n) reference for memory
    if not df_copy.empty:
        max_n = df_copy['n'].max()
        min_n = df_copy['n'].min()
        if max_n > 0:
            random_data = df_copy[df_copy['category'] == 'random']
            if not random_data.empty:
                avg_mem = random_data[random_data['n'] == max_n]['stack_kb'].mean()
                c = avg_mem / np.log2(max_n)
                x_ref = np.linspace(min_n, max_n, 100)
                y_ref = c * np.log2(x_ref)
                plt.plot(x_ref, y_ref, 'k--', alpha=0.8, label=f'Expected O(log n) ~ {c:.2f} log₂(n) KB')
    
    plt.title('Estimated Stack Memory Usage vs Input Size')
    plt.xlabel('Input Size (n)')
    plt.ylabel('Stack Memory (KB)')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.savefig(os.path.join(out_dir, 'stack_memory.png'), dpi=150)
    plt.close()


def plot_space_complexity_comparison(df, out_dir):
    """
    Generates comprehensive space complexity comparison plots.
    Two subplots: stack depth and memory usage with log-scale x-axis.
    
    Args:
        df (pd.DataFrame): Benchmark results dataframe.
        out_dir (str): Output directory for saving the plot.
    """
    # Q10: Compare space complexity across categories with log scale
    if 'max_stack_depth' not in df.columns: return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Stack depth with log scale on x-axis
    for category in df['category'].unique():
        cat_data = df[df['category'] == category]
        agg = cat_data.groupby('n')['max_stack_depth'].mean().reset_index()
        ax1.plot(agg['n'], agg['max_stack_depth'], marker='o', label=category)
    
    if not df.empty:
        max_n = df['n'].max()
        min_n = df['n'].min()
        x_ref = np.linspace(min_n, max_n, 100)
        
        # Log n reference
        random_data = df[df['category'] == 'random']
        if not random_data.empty and max_n > 0:
            avg_depth = random_data[random_data['n'] == max_n]['max_stack_depth'].mean()
            c = avg_depth / np.log2(max_n)
            y_log = c * np.log2(x_ref)
            ax1.plot(x_ref, y_log, 'k--', alpha=0.6, label=f'O(log n)')
        
        # Linear reference for worst case
        ax1.plot(x_ref, x_ref / 1000, 'r:', alpha=0.4, label='O(n)/1000')
    
    ax1.set_xlabel('Input Size (n)')
    ax1.set_ylabel('Max Stack Depth')
    ax1.set_title('Stack Depth: Average Case vs Worst Case')
    ax1.legend()
    ax1.grid(True, which="both", ls="-", alpha=0.2)
    ax1.set_xscale('log')
    
    # Plot 2: Memory usage comparison
    if 'estimated_stack_bytes' in df.columns:
        for category in df['category'].unique():
            cat_data = df[df['category'] == category]
            cat_data = cat_data.copy()
            cat_data['stack_kb'] = cat_data['estimated_stack_bytes'] / 1024.0
            agg = cat_data.groupby('n')['stack_kb'].mean().reset_index()
            ax2.plot(agg['n'], agg['stack_kb'], marker='o', label=category)
        
        if not df.empty:
            random_data = df[df['category'] == 'random']
            if not random_data.empty and max_n > 0:
                random_data = random_data.copy()
                random_data['stack_kb'] = random_data['estimated_stack_bytes'] / 1024.0
                avg_mem = random_data[random_data['n'] == max_n]['stack_kb'].mean()
                c = avg_mem / np.log2(max_n)
                y_log = c * np.log2(x_ref)
                ax2.plot(x_ref, y_log, 'k--', alpha=0.6, label=f'O(log n)')
        
        ax2.set_xlabel('Input Size (n)')
        ax2.set_ylabel('Stack Memory (KB)')
        ax2.set_title('Stack Memory Usage Comparison')
        ax2.legend()
        ax2.grid(True, which="both", ls="-", alpha=0.2)
        ax2.set_xscale('log')
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'space_complexity_comparison.png'), dpi=150)
    plt.close()


def save_summaries(df: pd.DataFrame, out_dir: str):
    """
    Generates and saves statistical summaries for each category.
    Creates both per-category and aggregate summary CSV files.
    
    Args:
        df (pd.DataFrame): Benchmark results dataframe.
        out_dir (str): Output directory for saving summaries.
    """
    ensure_dir(out_dir)
    all_rows = []
    for category, df_cat in df.groupby('category'):
        # Define aggregation metrics
        agg_dict = {
            'n': ('n', 'median'),
            'median_time': ('elapsed_ms', 'median'),
            'mean_time': ('elapsed_ms', 'mean'),
            'std_time': ('elapsed_ms', 'std'),
            'median_comparisons': ('comparisons', 'median'),
            'median_swaps': ('swaps', 'median'),
            'correctness_rate': ('correct', 'mean'),
        }
        
        # Add space complexity metrics if available
        if 'max_stack_depth' in df_cat.columns:
            agg_dict['median_stack_depth'] = ('max_stack_depth', 'median')
            agg_dict['max_stack_depth'] = ('max_stack_depth', 'max')
        if 'estimated_stack_bytes' in df_cat.columns:
            agg_dict['median_stack_kb'] = ('estimated_stack_bytes', lambda x: x.median() / 1024.0)
            agg_dict['max_stack_kb'] = ('estimated_stack_bytes', lambda x: x.max() / 1024.0)
        
        g = df_cat.groupby('filename').agg(**agg_dict).reset_index()
        g['category'] = category
        
        # Order columns
        base_cols = ['category','filename','n','median_time','mean_time','std_time','median_comparisons','median_swaps','correctness_rate']
        space_cols = [c for c in g.columns if 'stack' in c]
        ordered_cols = base_cols + space_cols
        ordered_cols = [c for c in ordered_cols if c in g.columns]
        g = g[ordered_cols]
        
        out_csv = os.path.join(out_dir, f"{category}_summary.csv")
        g.to_csv(out_csv, index=False)
        all_rows.append(g)

    if all_rows:
        all_df = pd.concat(all_rows, ignore_index=True)
        all_df.to_csv(os.path.join(out_dir, 'summary_all_categories.csv'), index=False)


def run():
    """
    Main analysis execution function.
    
    Workflow:
      1. Load benchmark results from CSV
      2. Generate all visualization plots:
         - Runtime scaling (overall and by category)
         - Comparisons and swaps analysis
         - Seed stability assessment
         - Recursion depth and pivot quality
         - Space complexity metrics
      3. Save statistical summaries
    """
    parser = argparse.ArgumentParser(description='Analyze randomized quicksort results and produce plots.')
    parser.add_argument('--results', default='results/qsort/qsort_master.csv', help='Path to master CSV')
    parser.add_argument('--out-dir', default='results/qsort/analysis_plots', help='Directory to save plots and summaries')
    args = parser.parse_args()

    if not os.path.exists(args.results):
        print(f"Error: Results file not found: {args.results}\nPlease run benchmarks first.")
        return

    # Prepare output directory and load data
    ensure_dir(args.out_dir)
    df = load_results(args.results)
    
    # Generate all visualization plots
    # Time complexity analysis
    plot_overall_runtime_scaling(df, args.out_dir)
    plot_runtime_scaling(df, args.out_dir)
    
    # Operation count analysis
    plot_comparisons_scaling(df, args.out_dir)
    plot_swaps_scaling(df, args.out_dir)
    
    # Stability and quality metrics
    plot_seed_stability(df, args.out_dir)
    plot_recursion_depth(df, args.out_dir)
    plot_bad_splits(df, args.out_dir)
    
    # Space complexity analysis
    plot_stack_depth(df, args.out_dir)
    plot_stack_memory(df, args.out_dir)
    plot_space_complexity_comparison(df, args.out_dir)

    # Generate statistical summaries
    save_summaries(df, args.out_dir)
    print(f"Analysis complete. Plots and summaries saved to {args.out_dir}")


if __name__ == '__main__':
    run()
