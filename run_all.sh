#!/bin/bash
set -e

PYTHON=python3
if [ -x "venv/bin/python" ]; then
    PYTHON="venv/bin/python"
fi

echo "=== Building C++ Binaries ==="
make



echo "=== Generating Datasets ==="
$PYTHON datasets/generate_all.py

echo "=== Generating Miller-Rabin Datasets ==="
$PYTHON datasets/miller_rabin/generate_data.py --out-dir datasets/miller_rabin

echo "=== Generating Cardinality Datasets ==="
$PYTHON datasets/cardinality/generate_streams.py --out-dir datasets/cardinality

echo "=== Generating MinCut Datasets ==="
$PYTHON datasets/mincut/generate_graphs.py --out-dir datasets/mincut



echo "=== Running Quicksort Benchmarks ==="
$PYTHON benchmarks/qsort/run_all.py

echo "=== Running Frievald Benchmarks ==="
# Runtime benchmark
./bin/frievald_benchmark_runtime --dataset-dir datasets/frievald --out-csv results/frievald/runtime.csv
# Error benchmark
./bin/frievald_benchmark_error --out-csv results/frievald/error.csv

echo "=== Running Miller-Rabin Benchmarks ==="
$PYTHON benchmarks/miller_rabin/run_all.py --dataset-dir datasets/miller_rabin --out-dir results/miller_rabin

echo "=== Running Cardinality Benchmarks ==="
./bin/cardinality_benchmark --dataset-dir datasets/cardinality --out-dir results/cardinality

echo "=== Running MinCut Benchmarks ==="
$PYTHON benchmarks/mincut/run_all.py --dataset-dir datasets/mincut --out-dir results/mincut --reps 5



echo "=== Running Analysis ==="
# Quicksort
$PYTHON analysis/qsort/analyze_qsort.py --results results/qsort/qsort_master.csv --out-dir results/qsort/analysis_plots

# Frievald
$PYTHON analysis/frievald/visualize_results.py --runtime-csv results/frievald/runtime.csv --error-csv results/frievald/error.csv --out-dir results/frievald/analysis_plots

# Miller-Rabin
$PYTHON analysis/miller_rabin/analyze_miller_rabin.py --summary results/miller_rabin/miller_rabin_summary.csv --raw results/miller_rabin/miller_rabin_raw.csv --error results/miller_rabin/miller_rabin_error.csv --witnesses results/miller_rabin/miller_rabin_witnesses.csv --outdir results/miller_rabin/analysis_plots

# Cardinality
$PYTHON analysis/cardinality/analyze_cardinality.py --raw results/cardinality/cardinality_raw.csv --summary results/cardinality/cardinality_summary.csv --out-dir results/cardinality/analysis_plots

# MinCut
$PYTHON analysis/mincut/analyze_mincut.py results/mincut/mincut_results.csv

echo "=== All Done! Results are in results/ ==="
