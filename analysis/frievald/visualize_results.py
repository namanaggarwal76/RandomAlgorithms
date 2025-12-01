#!/usr/bin/env python3
import argparse  # Used for parsing command line arguments
import os  # Used for operating system dependent functionality
import pandas as pd  # Used for data manipulation and analysis
import seaborn as sns  # Used for statistical data visualization
import matplotlib.pyplot as plt  # Used for creating static, animated, and interactive visualizations
import numpy as np  # Used for numerical operations

def ensure_dir(path):
    """
    Ensures that a directory exists.
    
    Args:
        path (str): Path to the directory.
    """
    os.makedirs(path, exist_ok=True)

def load_runtime_results(csv_path):
    """
    Loads runtime benchmark results from a CSV file.
    
    Args:
        csv_path (str): Path to the CSV file.
        
    Returns:
        pd.DataFrame: DataFrame containing the results.
    """
    if not os.path.exists(csv_path):
        print(f"Warning: Runtime results not found at {csv_path}")
        return pd.DataFrame()
    df = pd.read_csv(csv_path)
    # Convert seconds to ms for easier reading
    df['time_ms'] = df['seconds'] * 1000
    return df

def load_error_results(csv_path):
    """
    Loads error benchmark results from a CSV file.
    
    Args:
        csv_path (str): Path to the CSV file.
        
    Returns:
        pd.DataFrame: DataFrame containing the results.
    """
    if not os.path.exists(csv_path):
        print(f"Warning: Error results not found at {csv_path}")
        return pd.DataFrame()
    return pd.read_csv(csv_path)

def plot_runtime_scaling(df, out_dir):
    """
    Plots runtime scaling (Time vs n).
    
    Args:
        df (pd.DataFrame): DataFrame containing the results.
        out_dir (str): Output directory for the plot.
    """
    # F1: Time vs n (Frievald vs Naive)
    if df.empty: return
    
    plt.figure(figsize=(10, 6))
    
    # Filter for relevant algorithms
    # We might have 'naive' and 'frievald'
    # For frievald, we might have multiple k, let's pick one representative k (e.g., k=5) or average over k if k is not the focus here.
    # Actually, the prompt says "Scaling vs Naive".
    
    # Let's plot all available algorithms.
    # For frievald, if there are multiple k, we can plot them as separate lines or just one.
    # Let's assume we want to compare the main Frievald performance (e.g. k=1 or k=5) vs Naive.
    
    sns.lineplot(data=df, x='n', y='time_ms', hue='algorithm', style='algorithm', markers=True, dashes=False)
    
    plt.title('F1: Runtime Scaling (Frievald vs Naive)')
    plt.xlabel('Matrix Size (n)')
    plt.ylabel('Time (ms)')
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.savefig(os.path.join(out_dir, 'F1_runtime_scaling.png'), dpi=150)
    plt.close()

def plot_error_scaling(df, out_dir):
    # F2: Error probability vs k
    if df.empty: return
    
    plt.figure(figsize=(10, 6))
    
    # Plot Observed Error Rate
    sns.lineplot(data=df, x='k', y='false_accept_rate', marker='o', label='Observed Error')
    
    # Plot Theoretical Error Rate (2^-k)
    k_values = np.sort(df['k'].unique())
    theoretical_error = 2.0 ** (-k_values)
    plt.plot(k_values, theoretical_error, 'r--', label='Theoretical Bound ($2^{-k}$)')
    
    plt.title('F2: Error Probability vs k')
    plt.xlabel('Iterations (k)')
    plt.ylabel('Error Probability')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.savefig(os.path.join(out_dir, 'F2_error_scaling.png'), dpi=150)
    plt.close()

def plot_time_vs_k(df, out_dir):
    # F3: Time vs k (for fixed n)
    if df.empty: return
    
    # Filter for Frievald only
    frievald_df = df[df['algorithm'] == 'frievald']
    if frievald_df.empty: return
    
    # Find n with most k values or largest n
    n_counts = frievald_df['n'].value_counts()
    if n_counts.empty: return
    target_n = n_counts.idxmax()
    
    sub = frievald_df[frievald_df['n'] == target_n]
    
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=sub, x='k', y='time_ms', marker='o')
    
    plt.title(f'F3: Time vs k (n={target_n})')
    plt.xlabel('Iterations (k)')
    plt.ylabel('Time (ms)')
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, 'F3_time_vs_k.png'), dpi=150)
    plt.close()

def plot_stability(df, out_dir):
    # F4: Stability (Boxplot of time for fixed n)
    if df.empty: return
    
    # Find n with most samples
    n_counts = df['n'].value_counts()
    if n_counts.empty: return
    target_n = n_counts.idxmax()
    
    sub = df[df['n'] == target_n]
    
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=sub, x='algorithm', y='time_ms')
    
    plt.title(f'F4: Runtime Stability (n={target_n})')
    plt.xlabel('Algorithm')
    plt.ylabel('Time (ms)')
    plt.savefig(os.path.join(out_dir, 'F4_stability.png'), dpi=150)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Analyze Frievald benchmark results.')
    parser.add_argument('--runtime-csv', default='results/frievald/runtime.csv', help='Path to runtime CSV')
    parser.add_argument('--error-csv', default='results/frievald/error.csv', help='Path to error CSV')
    parser.add_argument('--out-dir', default='results/frievald/analysis_plots', help='Output directory for plots')
    args = parser.parse_args()

    ensure_dir(args.out_dir)

    runtime_df = load_runtime_results(args.runtime_csv)
    error_df = load_error_results(args.error_csv)

    plot_runtime_scaling(runtime_df, args.out_dir)
    plot_error_scaling(error_df, args.out_dir)
    plot_time_vs_k(runtime_df, args.out_dir)
    plot_stability(runtime_df, args.out_dir)

    print(f"Analysis complete. Plots saved to {args.out_dir}")

if __name__ == '__main__':
    main()
