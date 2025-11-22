#!/usr/bin/env python3
import argparse
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def load_results(summary_csv, raw_csv):
    if not os.path.exists(summary_csv) or not os.path.exists(raw_csv):
        print(f"Warning: Results not found.")
        return pd.DataFrame(), pd.DataFrame()
    return pd.read_csv(summary_csv), pd.read_csv(raw_csv)

def plot_time_vs_bits(summary_df, outdir):
    # M1: Time vs bits (Scaling)
    if summary_df.empty: return
    
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=summary_df, x='bits', y='avg_time_ms', hue='rounds', marker='o', palette='viridis')
    
    plt.title('M1: Runtime Scaling (Time vs Bits)')
    plt.xlabel('Bit Length')
    plt.ylabel('Average Time (ms)')
    plt.yscale('log')
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.legend(title='Rounds (k)')
    plt.savefig(os.path.join(outdir, 'M1_time_vs_bits.png'), dpi=150)
    plt.close()

def plot_modexp_vs_bits(summary_df, outdir):
    # M2: Modexp count vs bits
    if summary_df.empty: return
    if 'avg_modexp' not in summary_df.columns: return
    
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=summary_df, x='bits', y='avg_modexp', hue='rounds', marker='o', palette='viridis')
    
    plt.title('M2: Modexp Count vs Bits')
    plt.xlabel('Bit Length')
    plt.ylabel('Average Modular Exponentiations')
    plt.grid(True)
    plt.legend(title='Rounds (k)')
    plt.savefig(os.path.join(outdir, 'M2_modexp_vs_bits.png'), dpi=150)
    plt.close()

def plot_error_vs_k(summary_df, outdir):
    # M3: Error probability vs k
    if summary_df.empty: return
    
    # Filter for composites only error rate
    # We want to see how error drops with k
    
    plt.figure(figsize=(10, 6))
    
    # Plot Observed Error Rate (Composites Only)
    sns.lineplot(data=summary_df, x='rounds', y='composite_only_fp_rate', hue='bits', marker='o', palette='flare')
    
    # Plot Theoretical Bound (4^-k)
    k_values = np.sort(summary_df['rounds'].unique())
    theoretical_error = 4.0 ** (-k_values)
    plt.plot(k_values, theoretical_error, 'k--', label='Theoretical Bound ($4^{-k}$)')
    
    plt.title('M3: Error Probability vs k (Composites Only)')
    plt.xlabel('Rounds (k)')
    plt.ylabel('False Positive Rate')
    plt.yscale('log')
    plt.legend(title='Bits')
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.savefig(os.path.join(outdir, 'M3_error_vs_k.png'), dpi=150)
    plt.close()

def plot_stability(raw_df, outdir):
    # M4: Stability (Boxplot of time for fixed bits)
    if raw_df.empty: return
    
    # Find bit length with most samples
    bits_counts = raw_df['bits'].value_counts()
    if bits_counts.empty: return
    target_bits = bits_counts.idxmax()
    
    sub = raw_df[raw_df['bits'] == target_bits]
    
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=sub, x='rounds', y='time_ms')
    
    plt.title(f'M4: Runtime Stability (Bits={target_bits})')
    plt.xlabel('Rounds (k)')
    plt.ylabel('Time (ms)')
    plt.savefig(os.path.join(outdir, 'M4_stability.png'), dpi=150)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Analyze Miller-Rabin benchmark results.')
    parser.add_argument('--summary', default='results/miller_rabin/miller_rabin_summary.csv', help='Path to summary CSV')
    parser.add_argument('--raw', default='results/miller_rabin/miller_rabin_raw.csv', help='Path to raw CSV')
    parser.add_argument('--outdir', default='results/miller_rabin/analysis_plots', help='Output directory for plots')
    args = parser.parse_args()

    ensure_dir(args.outdir)

    summary_df, raw_df = load_results(args.summary, args.raw)

    plot_time_vs_bits(summary_df, args.outdir)
    plot_modexp_vs_bits(summary_df, args.outdir)
    plot_error_vs_k(summary_df, args.outdir)
    plot_stability(raw_df, args.outdir)

    print(f"Analysis complete. Plots saved to {args.outdir}")

if __name__ == '__main__':
    main()
