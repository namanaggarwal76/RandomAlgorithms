# Analysis of Random Algorithms

This project contains implementations and benchmarks for three randomized algorithms:
1.  **Frievald's Algorithm**: Probabilistic matrix multiplication verification.
2.  **Miller-Rabin Primality Test**: Probabilistic primality testing.
3.  **Randomized Quicksort**: Sorting algorithm with random pivots.

## Structure

-   `src/`: Source code for algorithms (C++ and Python).
-   `datasets/`: Scripts to generate test data.
-   `benchmarks/`: Scripts to run benchmarks.
-   `analysis/`: Scripts to analyze and visualize results.
-   `results/`: Output directory for benchmark results and plots.
-   `bin/`: Compiled C++ binaries.

## User Flow

just do 
```bash
./run_all.sh
```

Follow these steps to reproduce the results.

### 1. Setup & Dependencies

Install the required Python packages:
```bash
pip install -r requirements.txt
```

### 2. Generate Datasets

Generate the datasets required for all benchmarks:
```bash
python datasets/generate_all.py
```
This will populate the `datasets/` directory with test files.

### 3. Build C++ Binaries

Compile the C++ implementations:
```bash
make
```
This creates the executables in the `bin/` directory.

### 4. Run Benchmarks

Run the benchmarks for each algorithm.

**Frievald's Algorithm:**
```bash
# Runtime benchmark
./bin/frievald_benchmark_runtime

# Error probability benchmark
./bin/frievald_benchmark_error
```

**Miller-Rabin:**
```bash
python benchmarks/miller_rabin/run_all.py
```

**Randomized Quicksort:**
```bash
python benchmarks/qsort/run_all.py
```

### 5. Analyze Results

Generate plots and analysis from the benchmark results.

**Frievald:**
```bash
python analysis/frievald/visualize_results.py
```

**Miller-Rabin:**
```bash
python analysis/miller_rabin/analyze_miller_rabin.py
```

**Randomized Quicksort:**
```bash
python analysis/qsort/analyze_qsort.py
```

## Cleaning Up

To remove compiled binaries:
```bash
make clean
```
