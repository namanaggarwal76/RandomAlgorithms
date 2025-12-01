#!/usr/bin/env python3
import argparse
import os
import sys
import shutil
import subprocess
from datetime import datetime

HEADER = "timestamp_utc_iso,category,input_file,n,seed,rep_id,elapsed_ms,comparisons,swaps,correct,std_sort_ms,recursion_depth,bad_split_count,max_stack_depth,estimated_stack_bytes"


def find_executable(project_root: str) -> str:
    candidates = [
        os.path.join(project_root, "bin", "random_qsort"),
        os.path.join(project_root, "build", "random_qsort"),
        os.path.join(project_root, "random_qsort"),
    ]
    for c in candidates:
        if os.path.isfile(c) and os.access(c, os.X_OK):
            return c
    which = shutil.which("random_qsort")
    if which:
        return which
    return ""


def list_categories(dataset_root: str):
    cats = []
    if not os.path.isdir(dataset_root):
        return cats
    for entry in os.scandir(dataset_root):
        if entry.is_dir():
            cats.append(entry.name)
    cats.sort()
    return cats


def list_txt_files_sorted_by_size(folder: str):
    files = []
    if not os.path.isdir(folder):
        return files
    for entry in os.scandir(folder):
        if entry.is_file() and entry.name.endswith('.txt'):
            files.append(entry.path)
    files.sort(key=lambda p: os.path.getsize(p))
    return files


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def now_iso():
    return datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')


def run():
    parser = argparse.ArgumentParser(description="Run random_qsort over all datasets and aggregate results.")
    parser.add_argument("--dataset-root", default="./datasets/qsort", help="Root folder containing category subfolders")
    parser.add_argument("--out-root", default="results/qsort", help="Base output folder")
    parser.add_argument("--seed-base", type=int, default=42, help="Base seed (default 42)")
    args = parser.parse_args()

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
    exe = find_executable(project_root)
    if not exe:
        print("Error: Could not find 'random_qsort' executable. Build it first (see README).", file=sys.stderr)
        sys.exit(2)

    ensure_dir(args.out_root)

    master_csv = os.path.join(args.out_root, "qsort_master.csv")
    failures_log = os.path.join(args.out_root, "failures.log")
    
    # Remove master csv if exists to start fresh or append? 
    # The script appends, but we want to ensure headers are correct if we changed schema.
    # Let's remove it if it exists to be safe with new schema.
    if os.path.exists(master_csv):
        os.remove(master_csv)

    categories = list_categories(args.dataset_root)
            
    if not categories:
        print(f"No categories found under {args.dataset_root}. Please run datasets/generate_all.py first.")
        sys.exit(0)

    total_runs = 0
    successes = 0
    failures = 0

    for cat in categories:
        cat_folder = os.path.join(args.dataset_root, cat)
        files = list_txt_files_sorted_by_size(cat_folder)
        if not files:
            continue

        cat_csv = os.path.join(args.out_root, f"{cat}.csv")
        if os.path.exists(cat_csv): os.remove(cat_csv)

        print(f"Processing category '{cat}'...")

        for fpath in files:
            # Determine experiment type based on file size or name?
            # Experiment 1: Scaling with n (20-30 seeds)
            # Experiment 2: Stability (fixed n=50000, 100-200 seeds)
            
            filename = os.path.basename(fpath)
            is_stability_target = "50000" in filename and (cat == "random" or cat == "sorted")
            
            reps = 100 if is_stability_target else 20
            
            print(f"  File: {filename}, Reps: {reps}")

            for rep in range(reps):
                seed = int(args.seed_base + rep) # Simple seed increment
                cmd = [
                    exe,
                    "--input-file", fpath,
                    "--seed", str(seed),
                    "--rep", str(rep),
                    "--out-csv", cat_csv,
                ]
                total_runs += 1
                try:
                    res = subprocess.run(cmd, capture_output=True, text=True)
                except Exception as e:
                    failures += 1
                    continue

                if res.returncode != 0:
                    failures += 1
                    continue

                # On success, pick the last non-empty stdout line as the row
                out_lines = [ln.strip() for ln in res.stdout.splitlines() if ln.strip()]
                row_line = ""
                for ln in reversed(out_lines):
                    if not ln.startswith("timestamp_utc_iso"):
                        row_line = ln
                        break
                
                if row_line:
                    # Write header to master if needed
                    if not os.path.exists(master_csv):
                         # Extract header from stdout (it's printed if file was empty)
                         # We need to find the header line
                         for ln in out_lines:
                             if ln.startswith("timestamp_utc_iso"):
                                 with open(master_csv, 'w') as mf:
                                     mf.write(ln + "\n")
                                 break
                    
                    with open(master_csv, 'a') as mf:
                        mf.write(row_line + "\n")
                    successes += 1
                else:
                    failures += 1

    print("\nSummary:")
    print(f"  total runs   : {total_runs}")
    print(f"  successes    : {successes}")
    print(f"  failures     : {failures}")


if __name__ == "__main__":
    run()
