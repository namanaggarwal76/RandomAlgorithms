# Analysis of Random Algorithms

This project contains implementations and benchmarks for five randomized algorithms:
1.  **Frievald's Algorithm**: Probabilistic matrix multiplication verification.
2.  **Miller-Rabin Primality Test**: Probabilistic primality testing.
3.  **Randomized Quicksort**: Sorting algorithm with random pivots.
4.  **LogLog / HyperLogLog / HyperLogLog++**: Cardinality estimation algorithms (benchmark and analysis under the `cardinality` folder).
5.  **Karger's & Karger-Stein Algorithm**: Randomized Minimum Cut algorithms.

## Structure

-   `src/`: Source code for algorithms (C++ and Python).
-   `datasets/`: Scripts to generate test data.
-   `benchmarks/`: Scripts to run benchmarks.
-   `analysis/`: Scripts to analyze and visualize results.
-   `results/`: Output directory for benchmark results and plots.
-   `bin/`: Compiled C++ binaries.

## Usage

To reproduce all results (generate datasets, build binaries, run benchmarks, and generate plots), simply follow these steps:

### 1. Install Dependencies

Install the required Python packages:
```bash
pip install -r requirements.txt
```

### 2. Run the Pipeline

Execute the master script:
```bash
./run_all.sh
```

This script handles the entire pipeline:
1.  Generates necessary datasets.
2.  Compiles C++ implementations.
3.  Runs benchmarks for all algorithms.
4.  Analyzes results and generates plots in the `results/` directory.

## Cleaning Up

To remove compiled binaries:
```bash
make clean
```
