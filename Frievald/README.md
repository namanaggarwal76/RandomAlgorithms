# Randomized Algorithms and Complexity: Frievald's Algorithm

This project implements and benchmarks Frievald's randomized matrix multiplication verification alongside deterministic matrix multiplication baselines.

## Project Layout
- `src/algorithms.py` – Deterministic multiplication (NumPy, naive triple loop, Strassen) and Frievald's verifier implementations.
- `src/utils.py` – Matrix generation helpers and error injection utilities.
- `experiments/benchmark_runtime.py` – Runtime benchmarking across matrix sizes.
- `experiments/benchmark_error.py` – Empirical failure-rate study across iteration counts.
- `visualize_results.py` – Plotting scripts for benchmark CSV outputs.

## Quick Start
1. Install dependencies:
   ```bash
   python3 -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. Run the runtime benchmark (random sampled sizes):
    ```bash
    python experiments/benchmark_runtime.py \
       --samples 1000 \
       --n-min 1 --n-max 3000 \
       --frievald-iters 1,2,5,10,15,20 \
       --skip-triple-loop-above 500 \
       --strassen-threshold 64
    ```
    Produces raw sample rows in `runtime.csv` (columns: n, algorithm, k, sample, seconds).

3. Run the error benchmark (two modes):
    a) Fixed-k failure probability (binomial CI):
   ```bash
   python experiments/benchmark_error.py \
     --matrix-size 500 \
     --matrices-per-k 25 \
     --corruptions-per-matrix 20 \
     --k-min 1 --k-max 20 \
       --error-mode sparse \
       --mode fixed_k
   ```
    b) Detection curve (20000 sparse corruptions; survival prob vs k):
    ```bash
    python experiments/benchmark_error.py \
       --matrix-size 300 \
       --mode detection_curve \
       --detection-trials 20000 \
       --max-detection-iterations 50 \
       --error-mode sparse
    ```
   Output includes confidence intervals (columns ci_lower/ci_upper).

4. Visualize results after benchmarks finish:
   ```bash
   python visualize_results.py
   ```

Benchmark CSVs are written to `experiments/results/`. In detection mode columns `cumulative_detection_prob`, `survival_prob`, and theoretical counterparts approximate the 2^-k bound.
