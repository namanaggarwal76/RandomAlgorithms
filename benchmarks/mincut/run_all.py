#!/usr/bin/env python3
"""
Wrapper for the C++ mincut benchmark.
"""
import argparse
import subprocess
from pathlib import Path
import sys

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Wrapper for the C++ mincut benchmark.")
    parser.add_argument("--dataset-dir", default="datasets/mincut", help="Folder with *.txt graph files.")
    parser.add_argument("--out-dir", default="results/mincut", help="Directory for CSV outputs.")
    parser.add_argument("--reps", type=int, default=10, help="Number of repetitions per dataset.")
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[2]
    binary = project_root / "bin" / "mincut_benchmark"
    
    if not binary.exists():
        print(f"Binary not found: {binary}")
        print("Attempting to compile...")
        try:
            subprocess.run(["make"], cwd=project_root, check=True)
        except subprocess.CalledProcessError:
            print("Compilation failed.")
            sys.exit(1)

    cmd = [
        str(binary),
        "--dataset-dir",
        str(args.dataset_dir),
        "--out-dir",
        str(args.out_dir),
        "--reps",
        str(args.reps),
    ]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()
