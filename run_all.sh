#!/bin/bash
set -e

echo "=== Building C++ Binaries ==="
make

echo "=== Generating Datasets ==="
python3 datasets/generate_all.py

echo "=== Running Quicksort Benchmarks ==="
python3 benchmarks/qsort/run_all.py

echo "=== Running Frievald Benchmarks ==="
# Runtime benchmark
./bin/frievald_benchmark_runtime --dataset-dir datasets/frievald --out-csv results/frievald/runtime.csv
# Error benchmark
./bin/frievald_benchmark_error --out-csv results/frievald/error.csv

echo "=== Running Miller-Rabin Benchmarks ==="
python3 benchmarks/miller_rabin/run_all.py --dataset-dir datasets/miller_rabin --out-dir results/miller_rabin

echo "=== Running Analysis ==="
# Quicksort
python3 analysis/qsort/analyze_qsort.py --results results/qsort/qsort_master.csv --out-dir results/qsort/analysis_plots

# Frievald
python3 analysis/frievald/visualize_results.py --runtime-csv results/frievald/runtime.csv --error-csv results/frievald/error.csv --out-dir results/frievald/analysis_plots

# Miller-Rabin
python3 analysis/miller_rabin/analyze_miller_rabin.py --summary results/miller_rabin/miller_rabin_summary.csv --raw results/miller_rabin/miller_rabin_raw.csv --outdir results/miller_rabin/analysis_plots

echo "=== All Done! Results are in results/ ==="
