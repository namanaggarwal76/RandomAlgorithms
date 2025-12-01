import argparse  # Used for parsing command line arguments
import subprocess  # Used for running external commands (Python script, C++ binary)
import os  # Used for operating system dependent functionality
from pathlib import Path  # Used for object-oriented filesystem paths
import pandas as pd  # Used for data manipulation and analysis
from sympy import isprime  # Used for checking primality (ground truth)
import sys  # Used for system-specific parameters and functions

def run_benchmark(dataset_dir, out_dir):
    """
    Runs the Miller-Rabin benchmark suite.
    
    Args:
        dataset_dir (str): Directory containing dataset files.
        out_dir (str): Directory to save benchmark results.
    """
    dataset_dir = Path(dataset_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    python_script = Path("src/miller_rabin/python/miller_rabin.py").resolve()
    cpp_binary = Path("bin/miller_rabin").resolve()
    
    if not python_script.exists():
        print(f"Error: Python script not found at {python_script}")
        return
        
    # --- 1. Performance Benchmark (Python, Large Numbers) ---
    print("Running Performance Benchmark (Python)...")
    perf_csv_path = out_dir / "miller_rabin_raw.csv"
    temp_perf_csv = out_dir / "temp_perf.csv"
    if temp_perf_csv.exists(): temp_perf_csv.unlink()
    
    large_files = list(dataset_dir.glob("*_large.txt"))
    for f in large_files:
        print(f"  Processing {f.name}...")
        for k in [5, 10]:
            subprocess.run([sys.executable, str(python_script), "--input-file", str(f), "--out-csv", str(temp_perf_csv), "--k", str(k)], check=True)
            
    if temp_perf_csv.exists():
        df = pd.read_csv(temp_perf_csv)
        # Ensure n is treated as python int (arbitrary precision)
        df['n'] = df['n'].astype(str).apply(int)
        df['bits'] = df['n'].apply(lambda x: x.bit_length())
        df['time_ms'] = df['time_ns'] / 1e6
        df.to_csv(perf_csv_path, index=False)
        
        # Summary
        summary_csv = out_dir / "miller_rabin_summary.csv"
        summary = df.groupby(['bits', 'k'])[['time_ms', 'modexp_count']].mean().reset_index()
        summary.rename(columns={'time_ms': 'avg_time_ms', 'modexp_count': 'avg_modexp', 'k': 'rounds'}, inplace=True)
        summary.to_csv(summary_csv, index=False)
        print(f"  Saved performance results to {perf_csv_path}")
        temp_perf_csv.unlink()

    # --- 2. Error Rate Analysis (Python, Composites) ---
    print("Running Error Rate Analysis (Python)...")
    error_csv_path = out_dir / "miller_rabin_error.csv"
    if error_csv_path.exists(): error_csv_path.unlink()
    
    comp_file = dataset_dir / "composites_small.txt"
    if comp_file.exists():
        print(f"  Processing {comp_file.name} for error rates...")
        for k in [1, 2, 3, 4, 5]:
            subprocess.run([sys.executable, str(python_script), "--input-file", str(comp_file), "--out-csv", str(error_csv_path), "--k", str(k)], check=True)
    
    # --- 3. Witness Analysis (C++, Small Numbers) ---
    print("Running Witness Analysis (C++)...")
    witness_csv_path = out_dir / "miller_rabin_witnesses.csv"
    if witness_csv_path.exists(): witness_csv_path.unlink()
    
    if not cpp_binary.exists():
        print(f"Error: C++ binary not found at {cpp_binary}. Did you run 'make'?")
    else:
        small_files = list(dataset_dir.glob("*_small.txt"))
        for f in small_files:
            print(f"  Analyzing {f.name}...")
            subprocess.run([str(cpp_binary), "--input-file", str(f), "--out-csv", str(witness_csv_path), "--mode", "analysis"], check=True)
            
        # Post-process witness data to add categories
        if witness_csv_path.exists():
            df = pd.read_csv(witness_csv_path)
            
            # Create map
            cat_map = {}
            
            # Load Composites first (default)
            comp_file = dataset_dir / "composites_small.txt"
            if comp_file.exists():
                with open(comp_file, 'r') as f:
                    for line in f:
                        if line.strip(): cat_map[int(line.strip())] = 'composite'
            
            # Load Primes
            prime_file = dataset_dir / "primes_small.txt"
            if prime_file.exists():
                with open(prime_file, 'r') as f:
                    for line in f:
                        if line.strip(): cat_map[int(line.strip())] = 'prime'
                        
            # Load Carmichael (overwrite composite)
            carm_file = dataset_dir / "carmichael_small.txt"
            if carm_file.exists():
                with open(carm_file, 'r') as f:
                    for line in f:
                        if line.strip(): cat_map[int(line.strip())] = 'carmichael'
            
            df['category'] = df['n'].map(cat_map)
            df.to_csv(witness_csv_path, index=False)
            print(f"  Saved witness analysis to {witness_csv_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default="datasets/miller_rabin")
    parser.add_argument("--out-dir", default="results/miller_rabin")
    args = parser.parse_args()
    run_benchmark(args.dataset_dir, args.out_dir)
