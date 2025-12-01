import pandas as pd  # Used for data manipulation and analysis
import matplotlib.pyplot as plt  # Used for creating static, animated, and interactive visualizations
import seaborn as sns  # Used for statistical data visualization
from pathlib import Path  # Used for object-oriented filesystem paths
import sys  # Used for system-specific parameters and functions

def analyze_results(results_file: str, output_dir: str):
    """
    Analyzes mincut benchmark results and generates plots.
    
    Args:
        results_file (str): Path to the results CSV file.
        output_dir (str): Directory to save the plots.
    """
    df = pd.read_csv(results_file)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Convert seconds to milliseconds
    df['duration_ms'] = df['duration_sec'] * 1000

    # 1. Runtime Comparison
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x='vertices', y='duration_ms', hue='algorithm', marker='o')
    plt.title('Runtime: Karger vs Karger-Stein')
    plt.xlabel('Number of Vertices')
    plt.ylabel('Time (ms)')
    plt.grid(True)
    plt.savefig(output_path / 'runtime_comparison.png')
    plt.close()

    # Calculate metrics
    df['min_cut_for_dataset'] = df.groupby('dataset')['cut_size'].transform('min')
    df['relative_error'] = (df['cut_size'] - df['min_cut_for_dataset']) / df['min_cut_for_dataset']

    # 2. Relative Error Analysis (Boxplot)
    # We use Relative Error because raw cut sizes vary wildly between graph types (Cycle=2 vs Random=50).
    # Relative Error = (Found - Best) / Best. 0 means perfect.
    
    plt.figure(figsize=(14, 8))
    sns.boxplot(data=df, x='dataset', y='relative_error', hue='algorithm')
    plt.title('Relative Error Distribution (Lower is Better)')
    plt.xticks(rotation=90) # Rotate labels to fit them all
    plt.ylabel('Relative Error (0 = Optimal)')
    plt.tight_layout()
    plt.savefig(output_path / 'relative_error_distribution.png')
    plt.close()

    # 3. Success Rate
    # We define "success" as finding the min_cut_for_dataset
    df['is_optimal'] = df['cut_size'] == df['min_cut_for_dataset']
    success_rates = df.groupby(['algorithm', 'dataset'])['is_optimal'].mean().reset_index()
    
    plt.figure(figsize=(14, 8))
    sns.barplot(data=success_rates, x='dataset', y='is_optimal', hue='algorithm')
    plt.title('Success Rate (Frequency of finding best known cut)')
    plt.xticks(rotation=90)
    plt.ylabel('Success Rate')
    plt.tight_layout()
    plt.savefig(output_path / 'success_rate.png')
    plt.close()
    
    plt.figure(figsize=(12, 6))
    sns.barplot(data=success_rates, x='dataset', y='is_optimal', hue='algorithm')
    plt.title('Success Rate (Frequency of finding best known cut)')
    plt.xticks(rotation=45)
    plt.ylabel('Success Rate')
    plt.tight_layout()
    plt.savefig(output_path / 'success_rate.png')
    plt.close()

    print(f"Analysis plots saved to {output_dir}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_mincut.py <results_csv>")
        sys.exit(1)
    
    results_csv = sys.argv[1]
    out_dir = Path(results_csv).parent / "analysis_plots"
    analyze_results(results_csv, str(out_dir))
