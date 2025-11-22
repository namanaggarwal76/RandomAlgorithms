import argparse
import subprocess
import os
from pathlib import Path
import pandas as pd
from sympy import isprime
import math
import sys

def run_benchmark(dataset_dir, raw_csv_path):
    dataset_dir = Path(dataset_dir)
    raw_csv_path = Path(raw_csv_path)
    raw_csv_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Temporary raw output
    temp_csv = raw_csv_path.with_name("temp_out.csv")
    if temp_csv.exists():
        temp_csv.unlink()
        
    files = sorted(list(dataset_dir.glob("*.txt")))
    
    # Path to Python implementation
    script_path = Path("src/miller_rabin/python/miller_rabin.py").resolve()
    if not script_path.exists():
        print(f"Error: Python implementation not found at {script_path}")
        sys.exit(1)

    for f in files:
        print(f"Processing {f}...")
        # Experiment 1: Scaling with bit-length (fixed k=5, 10)
        for k in [5, 10]:
            cmd = [
                sys.executable, str(script_path),
                "--input-file", str(f),
                "--out-csv", str(temp_csv),
                "--k", str(k),
                "--seed", "42"
            ]
            subprocess.run(cmd, check=True)
            
        # Experiment 2: Error rate vs k (composites only)
        if "composites" in f.name or "carmichael" in f.name:
             for k in [1, 2, 3, 5, 10, 15, 20]:
                # Skip 5 and 10 as they are covered above, unless we want separate runs? 
                # The script appends, so we might get duplicates if we are not careful.
                # But the analysis can handle it or we can just run all k here.
                if k in [5, 10]: continue 
                
                cmd = [
                    sys.executable, str(script_path),
                    "--input-file", str(f),
                    "--out-csv", str(temp_csv),
                    "--k", str(k),
                    "--seed", "42"
                ]
                subprocess.run(cmd, check=True)

    return temp_csv

def post_process(temp_csv, final_raw_csv, summary_csv, dataset_dir):
    print("Post-processing results...")
    if not temp_csv.exists():
        print("No results generated.")
        return

    df = pd.read_csv(temp_csv)
    
    # Load Carmichael numbers
    carmichael_file = Path(dataset_dir) / "carmichael.txt"
    carmichaels = set()
    if carmichael_file.exists():
        with open(carmichael_file, 'r') as f:
            carmichaels = set(int(line.strip()) for line in f if line.strip())
            
    df['is_carmichael'] = df['n'].isin(carmichaels)
    
    # Add derived columns
    df['bits'] = df['n'].apply(lambda x: x.bit_length())
    df['rounds'] = df['k']
    df['time_ms'] = df['time_ns'] / 1e6
    
    # Ground truth
    print("Calculating ground truth...")
    unique_n = df['n'].unique()
    truth_map = {n: isprime(int(n)) for n in unique_n}
    df['ground_truth'] = df['n'].map(truth_map)
    
    df['probable_prime'] = df['is_probable_prime'].astype(bool)
    df['correct'] = df['probable_prime'] == df['ground_truth']
    
    # False positive: Composite but said probable prime
    df['false_positive'] = (~df['ground_truth']) & (df['probable_prime'])
    
    # Save raw
    df.to_csv(final_raw_csv, index=False)
    print(f"Saved raw results to {final_raw_csv}")
    
    # Aggregate for summary
    agg_rows = []
    for (bits, rounds), group in df.groupby(['bits', 'rounds']):
        total = len(group)
        false_pos = group['false_positive'].sum()
        
        comp_group = group[~group['ground_truth']]
        comp_total = len(comp_group)
        comp_fp = comp_group['false_positive'].sum()
        
        agg_rows.append({
            "bits": bits,
            "rounds": rounds,
            "samples": total,
            "accuracy": group.correct.mean(),
            "false_positives": false_pos,
            "false_positive_rate": false_pos / total if total else 0.0,
            "composite_only_fp_rate": (comp_fp / comp_total) if comp_total else 0.0,
            "avg_time_ms": group.time_ms.mean(),
            "avg_modexp": group.modexp_count.mean() if 'modexp_count' in group.columns else 0
        })
        
    summary_df = pd.DataFrame(agg_rows)
    summary_df.to_csv(summary_csv, index=False)
    print(f"Saved summary to {summary_csv}")
    
    # Cleanup
    if temp_csv.exists():
        temp_csv.unlink()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default="datasets/miller_rabin", help="Dataset directory")
    parser.add_argument("--out-dir", default="results/miller_rabin", help="Output directory")
    args = parser.parse_args()
    
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    raw_csv = out_dir / "miller_rabin_raw.csv"
    summary_csv = out_dir / "miller_rabin_summary.csv"
    
    temp_csv = run_benchmark(args.dataset_dir, raw_csv)
    post_process(temp_csv, raw_csv, summary_csv, args.dataset_dir)

if __name__ == "__main__":
    main()
