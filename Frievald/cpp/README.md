# C++ Implementation of Frievald's Algorithm

This directory contains a high-performance C++ implementation of the matrix multiplication algorithms and benchmarks.

## Structure

- `src/`: Source code for algorithms and matrix class.
- `benchmarks/`: Benchmark runners.
- `bin/`: Compiled executables (created after build).

## Building

Run `make` in this directory:

```bash
cd cpp
make
```

## Running Benchmarks

To run all benchmarks and process the results for visualization:

```bash
make run_benchmarks
```

This will:
1. Run `benchmark_runtime` -> `../experiments/results/runtime.csv`
2. Run `benchmark_error` -> `../experiments/results/cpp_error_raw.csv`
3. Run `process_results.py` -> `../experiments/results/error.csv`

After running this, you can use the existing visualization script in the root directory:

```bash
cd ..
python3 visualize_results.py
```
