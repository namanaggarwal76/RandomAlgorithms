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

def load_results(summary_csv, raw_csv):
    """
    Loads benchmark results from CSV files.
    
    Args:
        summary_csv (str): Path to the summary CSV file.
        raw_csv (str): Path to the raw CSV file.
        
    Returns:
        tuple: A tuple containing the summary DataFrame and the raw DataFrame.
    """
    if not os.path.exists(summary_csv) or not os.path.exists(raw_csv):
        print(f"Warning: Results not found.")
        return pd.DataFrame(), pd.DataFrame()
    return pd.read_csv(summary_csv), pd.read_csv(raw_csv)

def plot_time_vs_bits(summary_df, outdir):
    """
    Plots execution time vs bit length.
    
    Args:
        summary_df (pd.DataFrame): Summary DataFrame.
        outdir (str): Output directory for the plot.
    """
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
    """
    Plots modular exponentiation count vs bit length.
    
    Args:
        summary_df (pd.DataFrame): Summary DataFrame.
        outdir (str): Output directory for the plot.
    """
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

def plot_error_vs_k(error_csv, outdir):
    # M3: Error probability vs k
    if not os.path.exists(error_csv):
        print("Error CSV not found.")
        return
        
    df = pd.read_csv(error_csv)
    if df.empty: return
    
    # Calculate False Positive Rate for each k
    # Since input is all composites, any 'is_probable_prime' == 1 is a False Positive
    summary = df.groupby('k')['is_probable_prime'].mean().reset_index()
    summary.rename(columns={'is_probable_prime': 'fp_rate'}, inplace=True)
    
    plt.figure(figsize=(10, 6))
    
    # Plot Observed Error Rate
    sns.lineplot(data=summary, x='k', y='fp_rate', marker='o', label='Observed Error Rate')
    
    # Plot Theoretical Bound (4^-k)
    k_values = np.sort(summary['k'].unique())
    theoretical_error = 4.0 ** (-k_values)
    plt.plot(k_values, theoretical_error, 'k--', label='Theoretical Bound ($4^{-k}$)')
    
    plt.title('M3: Error Probability vs k (Composites Only)')
    plt.xlabel('Rounds (k)')
    plt.ylabel('False Positive Rate')
    plt.yscale('log')
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.legend()
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
    sns.boxplot(data=sub, x='k', y='time_ms')
    
    plt.title(f'M4: Runtime Stability (Bits={target_bits})')
    plt.xlabel('Rounds (k)')
    plt.ylabel('Time (ms)')
    plt.savefig(os.path.join(outdir, 'M4_stability.png'), dpi=150)
    plt.close()

def plot_witness_analysis(witness_csv, outdir):
    if not os.path.exists(witness_csv):
        print("Witness CSV not found.")
        return
        
    df = pd.read_csv(witness_csv)
    if df.empty: return
    
    # Calculate witness ratio
    df['witness_ratio'] = df['witnesses'] / df['total_bases']
    
    # Plot 1: Distribution of Witness Ratios (Carmichael vs Composite)
    # Filter out primes as they have 0 witnesses (or check if we want to show them)
    sub = df[df['category'].isin(['carmichael', 'composite'])]
    
    if not sub.empty:
        plt.figure(figsize=(10, 6))
        sns.histplot(data=sub, x='witness_ratio', hue='category', element="step", stat="density", common_norm=False, bins=50)
        plt.title('M5: Distribution of Witness Ratios (Carmichael vs Composite)')
        plt.xlabel('Ratio of Witnesses (Witnesses / Total Bases)')
        plt.ylabel('Density')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(outdir, 'M5_witness_distribution.png'), dpi=150)
        plt.close()
    
    # Plot 2: Scatter plot Witness Ratio vs n
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='n', y='witness_ratio', hue='category', alpha=0.6, s=15)
    plt.title('M6: Witness Ratio vs Number Size')
    plt.xlabel('n')
    plt.ylabel('Witness Ratio')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(outdir, 'M6_witness_vs_n.png'), dpi=150)
    plt.close()

    # Plot 3: Strong Liar Ratio Distribution
    df['liar_ratio'] = df['liars'] / df['total_bases']
    sub = df[df['category'].isin(['carmichael', 'composite'])]
    
    if not sub.empty:
        plt.figure(figsize=(10, 6))
        sns.histplot(data=sub, x='liar_ratio', hue='category', element="step", stat="density", common_norm=False, bins=50)
        plt.title('M7: Distribution of Strong Liar Ratios')
        plt.xlabel('Ratio of Strong Liars (Liars / Total Bases)')
        plt.ylabel('Density')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(outdir, 'M7_liar_distribution.png'), dpi=150)
        plt.close()

    # Plot 4: Theoretical Error Rate vs Trials (k) for Carmichael Numbers
    # Error probability for one number n with k trials is (liar_ratio)^k
    # We average this over all numbers in the category
    
    k_values = range(1, 21)
    carmichael_df = df[df['category'] == 'carmichael']
    composite_df = df[df['category'] == 'composite']
    
    if not carmichael_df.empty:
        avg_errors_carm = []
        avg_errors_comp = []
        
        for k in k_values:
            # Calculate error prob for each number, then mean
            avg_errors_carm.append( (carmichael_df['liar_ratio'] ** k).mean() )
            if not composite_df.empty:
                avg_errors_comp.append( (composite_df['liar_ratio'] ** k).mean() )
        
        plt.figure(figsize=(10, 6))
        plt.plot(k_values, avg_errors_carm, 'o-', label='Carmichael Numbers')
        if avg_errors_comp:
            plt.plot(k_values, avg_errors_comp, 's-', label='Composite Numbers')
            
        # Theoretical bound 4^-k
        theoretical = [4.0**(-k) for k in k_values]
        plt.plot(k_values, theoretical, 'k--', label='Theoretical Bound ($4^{-k}$)')
        
        plt.title('M8: Average Error Probability vs Trials (k)')
        plt.xlabel('Number of Trials (k)')
        plt.ylabel('Average Error Probability')
        plt.yscale('log')
        plt.grid(True, which="both", ls="-", alpha=0.2)
        plt.legend()
        plt.savefig(os.path.join(outdir, 'M8_error_vs_k_theoretical.png'), dpi=150)
        plt.close()

def main():
    parser = argparse.ArgumentParser(description='Analyze Miller-Rabin benchmark results.')
    parser.add_argument('--summary', default='results/miller_rabin/miller_rabin_summary.csv', help='Path to summary CSV')
    parser.add_argument('--raw', default='results/miller_rabin/miller_rabin_raw.csv', help='Path to raw CSV')
    parser.add_argument('--error', default='results/miller_rabin/miller_rabin_error.csv', help='Path to error CSV')
    parser.add_argument('--witnesses', default='results/miller_rabin/miller_rabin_witnesses.csv', help='Path to witness CSV')
    parser.add_argument('--outdir', default='results/miller_rabin/analysis_plots', help='Output directory for plots')
    args = parser.parse_args()

    ensure_dir(args.outdir)

    summary_df, raw_df = load_results(args.summary, args.raw)

    plot_time_vs_bits(summary_df, args.outdir)
    plot_modexp_vs_bits(summary_df, args.outdir)
    plot_error_vs_k(args.error, args.outdir)
    plot_stability(raw_df, args.outdir)
    
    plot_witness_analysis(args.witnesses, args.outdir)

    print(f"Analysis complete. Plots saved to {args.outdir}")

if __name__ == '__main__':
    main()
