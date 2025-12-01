#!/usr/bin/env python3
"""
Compatibility wrapper that simply invokes the C++ cardinality benchmark binary.
"""
import argparse  # Used for parsing command line arguments
import subprocess  # Used for running external commands (C++ binary)
from pathlib import Path  # Used for object-oriented filesystem paths


def parse_args() -> argparse.Namespace:
    """
    Parses command line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Wrapper for the C++ cardinality benchmark.")
    parser.add_argument("--dataset-dir", default="datasets/cardinality", help="Folder with *.txt streams.")
    parser.add_argument("--out-dir", default="results/cardinality", help="Directory for CSV outputs.")
    parser.add_argument("--precision", type=int, default=14, help="Number of index bits (default: 14).")
    parser.add_argument("--reps", type=int, default=3, help="Number of repetitions per dataset.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[2]
    binary = project_root / "bin" / "cardinality_benchmark"
    if not binary.exists():
        raise SystemExit(f"Binary not found: {binary}. Run `make` first.")

    cmd = [
        str(binary),
        "--dataset-dir",
        str(args.dataset_dir),
        "--out-dir",
        str(args.out_dir),
        "--precision",
        str(args.precision),
        "--reps",
        str(args.reps),
    ]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
