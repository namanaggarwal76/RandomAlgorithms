# Miller-Rabin Primality Test

Implementation and benchmarking utilities for the probabilistic Miller-Rabin primality test on large random integers of various bit sizes.

## Contents
- `src/miller_rabin.py` core implementation
- `benchmark_miller_rabin.py` benchmarking & metrics collection
- `plot_results.py` basic per-bit plots and error bound comparison
- `analyze_miller_rabin.py` advanced analysis (heatmaps, distributions)
- `requirements.txt` dependencies
- `miller_rabin/results/` consolidated output directory (raw CSVs & plots)

## Metrics
For each (bit_size, rounds k):
- accuracy: fraction of tests where Miller-Rabin result matches `sympy.isprime`
- false_positives / false_negatives and their rates
- avg_time_core / avg_time_total
- median_time_core / median_time_total
- p95_time_core / p95_time_total (tail latency)
- stability_variance: variance of per-seed accuracy (lower is more stable)
- theoretical_error_prob: upper bound ≤ 4^{-k}

## Plots
Basic (`plot_results.py`):
- Accuracy vs rounds (per bit size)
- Runtime vs rounds (avg & p95 core time)
- Stability variance vs rounds
- False positive rate (log scale) vs rounds
- Observed accuracy vs theoretical error bound (log x-axis)

Advanced (`analyze_miller_rabin.py`):
- Heatmap: accuracy bits × rounds
- Heatmap: false positive rate bits × rounds
- Heatmap: stability variance bits × rounds
- Line plots: average & p95 core time across rounds colored by bit size
- Distribution histogram + KDE: core time for selected (bits, rounds)
- Scatter: accuracy vs theoretical error probability

## Usage
Install dependencies:
```bash
pip install -r miller_rabin/requirements.txt
```
Run a quick benchmark:
```bash
python miller_rabin/benchmark_miller_rabin.py --bits 32 40 48 --rounds 5 10 15 --seeds 1 2 3 --samples 20
```
Generate basic plots:
```bash
python miller_rabin/plot_results.py --summary miller_rabin/results/miller_rabin_summary.csv --outdir miller_rabin/results/plots
```
Run advanced analysis:
```bash
python miller_rabin/analyze_miller_rabin.py --summary miller_rabin/results/miller_rabin_summary.csv --raw miller_rabin/results/miller_rabin_raw.csv --outdir miller_rabin/results/analysis_plots --dist-bits 32 --dist-rounds 10
```

## Larger Datasets
Increase `--bits` to include sizes like 64 96 128 192 256 384 512. For very large bit lengths (>1024) sympy slows; reduce samples or use fewer rounds.

## Recommendations
- For 64-bit integers, deterministic bases `[2,3,5,7,11,13]` suffice.
- Choose rounds k where false positive rate plateaus near theoretical bound and stability variance ~0.
- p95 time helps size trade-offs: higher bits increase cost roughly O(k * log^3 n) for modular exponentiations.

## Theory (Brief)
Given odd n > 2, write n-1 = 2^r * d with d odd. Random bases a in [2, n-2] are tested; if any produces a nontrivial square root of 1 modulo n, n is composite. If none do for k rounds, n is declared probable prime. Error probability ≤ 4^{-k} for composite n.

## Next Steps
- Parallel benchmarking with multiprocessing.
- Cache prime ground truth results to speed larger runs.
- Add deterministic base sets for 128/256-bit ranges.
- Add confidence interval estimation for accuracy.
