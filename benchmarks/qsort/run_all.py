#!/usr/bin/env python3
"""
File: run_all.py
Description: Comprehensive benchmark runner for randomized quicksort algorithm.
             Executes sorting benchmarks across multiple data categories and sizes.
"""

import argparse     # Used for parsing command-line arguments
import os           # Used for operating system dependent functionality
import sys          # Used for system-specific parameters and functions
import shutil       # Used for high-level file operations (finding executables)
import subprocess   # Used for spawning new processes (running C++ binary)
from datetime import datetime  # Used for timestamp generation

# CSV header definition for results output
HEADER = "timestamp_utc_iso,category,input_file,n,seed,rep_id,elapsed_ms,comparisons,swaps,correct,std_sort_ms,recursion_depth,bad_split_count,max_stack_depth,estimated_stack_bytes"


def find_executable(project_root: str) -> str:
    """
    Locates the random_qsort executable in common build locations.
    
    Args:
        project_root (str): Root directory of the project.
        
    Returns:
        str: Path to executable if found, empty string otherwise.
    """
    # Check common build directories
    candidates = [
        os.path.join(project_root, "bin", "random_qsort"),
        os.path.join(project_root, "build", "random_qsort"),
        os.path.join(project_root, "random_qsort"),
    ]
    for c in candidates:
        if os.path.isfile(c) and os.access(c, os.X_OK):
            return c
    
    # Fallback: check system PATH
    which = shutil.which("random_qsort")
    if which:
        return which
    return ""


def list_categories(dataset_root: str):
    """
    Lists all category subdirectories in the dataset root.
    Categories represent different input types (random, sorted, duplicates, etc.).
    
    Args:
        dataset_root (str): Root directory containing category folders.
        
    Returns:
        list: Sorted list of category names.
    """
    cats = []
    if not os.path.isdir(dataset_root):
        return cats
    for entry in os.scandir(dataset_root):
        if entry.is_dir():
            cats.append(entry.name)
    cats.sort()
    return cats


def list_txt_files_sorted_by_size(folder: str):
    """
    Lists all .txt files in a folder, sorted by file size (smallest first).
    This ensures benchmarks run from smallest to largest datasets.
    
    Args:
        folder (str): Directory to scan for text files.
        
    Returns:
        list: List of file paths sorted by size.
    """
    files = []
    if not os.path.isdir(folder):
        return files
    for entry in os.scandir(folder):
        if entry.is_file() and entry.name.endswith('.txt'):
            files.append(entry.path)
    files.sort(key=lambda p: os.path.getsize(p))  # Sort by file size
    return files


def ensure_dir(path: str):
    """
    Ensures that a directory exists, creating it if necessary.
    
    Args:
        path (str): Directory path to create.
    """
    os.makedirs(path, exist_ok=True)


def now_iso():
    """
    Returns current UTC time in ISO 8601 format.
    
    Returns:
        str: Timestamp string in format "YYYY-MM-DDTHH:MM:SSZ".
    """
    return datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')


def run():
    """
    Main benchmark execution function.
    
    Workflow:
      1. Parse command-line arguments
      2. Locate random_qsort executable
      3. Iterate through all categories and datasets
      4. Execute benchmarks with varying repetitions:
         - Scaling experiments: 20 repetitions per dataset
         - Stability experiments: 100 repetitions for n=50000
      5. Aggregate results into master CSV
    """
    parser = argparse.ArgumentParser(description="Run random_qsort over all datasets and aggregate results.")
    parser.add_argument("--dataset-root", default="./datasets/qsort", help="Root folder containing category subfolders")
    parser.add_argument("--out-root", default="results/qsort", help="Base output folder")
    parser.add_argument("--seed-base", type=int, default=42, help="Base seed (default 42)")
    args = parser.parse_args()

    # Locate project root and find executable
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
    exe = find_executable(project_root)
    if not exe:
        print("Error: Could not find 'random_qsort' executable. Build it first (see README).", file=sys.stderr)
        sys.exit(2)

    # Prepare output directories and files
    ensure_dir(args.out_root)

    master_csv = os.path.join(args.out_root, "qsort_master.csv")
    failures_log = os.path.join(args.out_root, "failures.log")
    
    # Remove existing master CSV to ensure clean start with correct schema
    # This prevents header mismatches if schema was updated
    if os.path.exists(master_csv):
        os.remove(master_csv)

    # Discover all dataset categories
    categories = list_categories(args.dataset_root)
            
    if not categories:
        print(f"No categories found under {args.dataset_root}. Please run datasets/generate_all.py first.")
        sys.exit(0)

    # Initialize benchmark statistics
    total_runs = 0
    successes = 0
    failures = 0

    # Process each category (random, sorted, duplicates, etc.)
    for cat in categories:
        cat_folder = os.path.join(args.dataset_root, cat)
        files = list_txt_files_sorted_by_size(cat_folder)
        if not files:
            continue

        # Prepare category-specific CSV (remove if exists for clean run)
        cat_csv = os.path.join(args.out_root, f"{cat}.csv")
        if os.path.exists(cat_csv): os.remove(cat_csv)

        print(f"Processing category '{cat}'...")

        for fpath in files:
            # Determine experiment type:
            # - Scaling experiments: 20 repetitions (varying n)
            # - Stability experiments: 100 repetitions (fixed n=50000, testing seed variance)
            
            filename = os.path.basename(fpath)
            is_stability_target = "50000" in filename and (cat == "random" or cat == "sorted")
            
            reps = 100 if is_stability_target else 20
            
            print(f"  File: {filename}, Reps: {reps}")

            # Execute multiple runs with different seeds
            for rep in range(reps):
                seed = int(args.seed_base + rep)  # Increment seed for each repetition
                
                # Build command to execute C++ binary
                cmd = [
                    exe,
                    "--input-file", fpath,
                    "--seed", str(seed),
                    "--rep", str(rep),
                    "--out-csv", cat_csv,
                ]
                total_runs += 1
                
                # Execute benchmark
                try:
                    res = subprocess.run(cmd, capture_output=True, text=True)
                except Exception as e:
                    failures += 1
                    continue

                # Check execution status
                if res.returncode != 0:
                    failures += 1
                    continue

                # Parse result from stdout (last non-header line)
                out_lines = [ln.strip() for ln in res.stdout.splitlines() if ln.strip()]
                row_line = ""
                for ln in reversed(out_lines):
                    if not ln.startswith("timestamp_utc_iso"):
                        row_line = ln
                        break
                
                if row_line:
                    # Initialize master CSV with header if needed
                    if not os.path.exists(master_csv):
                         # Extract header from stdout
                         for ln in out_lines:
                             if ln.startswith("timestamp_utc_iso"):
                                 with open(master_csv, 'w') as mf:
                                     mf.write(ln + "\n")
                                 break
                    
                    # Append result to master CSV
                    with open(master_csv, 'a') as mf:
                        mf.write(row_line + "\n")
                    successes += 1
                else:
                    failures += 1

    # Print benchmark execution summary
    print("\nSummary:")
    print(f"  total runs   : {total_runs}")
    print(f"  successes    : {successes}")
    print(f"  failures     : {failures}")


if __name__ == "__main__":
    run()
