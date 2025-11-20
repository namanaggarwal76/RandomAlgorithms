# Randomized QuickSort Benchmark (minimal)

This repository provides a minimal, reproducible setup to run a randomized in-place QuickSort on existing integer datasets and collect analysis-ready CSVs and plots. It does not generate datasets.

- Language: C++17 (binary: `random_qsort`)
- Orchestration: Python 3 (pandas, numpy, matplotlib)
- Datasets: expected at `./datasets/<category>/*.txt` (one integer per line)

## Build

```
mkdir build
cd build
cmake ..
make
```

This produces the executable `random_qsort` in the `build/` directory.

## Usage

- Run all datasets (processes all `.txt` files in each category):

```
python3 tools/run_all.py --dataset-root ./datasets --out-root ./results/qsort --seed-base 42 --reps 5
```

- Analyze master results and generate plots + summaries:

```
python3 analysis/analyze_qsort.py --results ./results/qsort/qsort_master.csv --out-dir ./results/qsort/plots
```

## Binary: `random_qsort`

CLI:
```
./random_qsort --input-file <path> --seed <int> --rep <rep_id> --out-csv <path>
```
- Required flags: `--input-file`, `--seed`, `--rep`, `--out-csv`.
- Reads integers (one per line) from the given input file; does not modify dataset files.
- Implements in-place randomized QuickSort using a single `std::mt19937` seeded with `--seed`.
- Pivot selection: `uniform_int_distribution<int> dist(L,R); pivot_index = dist(rng);`
- Counts comparisons and swaps, measures wall time (ms) and CPU time (ms).
- Verification: after QuickSort, it verifies correctness by comparing to `std::sort` on a copy of the input.
- Output: appends one CSV row to `--out-csv`; writes header if the file is empty. Also prints the header (if newly created) and the row to stdout.

CSV header (exact):
```
timestamp_utc_iso,category,input_file,n,seed,rep_id,elapsed_ms,cpu_ms,comparisons,swaps,correct
```
See `data/schema.md` for details.

## Runner: `tools/run_all.py`
- Enumerates immediate subfolders in `--dataset-root` as categories.
- For each category, processes all `*.txt` files (sorted by size ascending).
- Deterministic seeding per run: `seed = seed_base + file_index*1000 + rep_id`.
- Invokes the `random_qsort` binary for each file and repetition, writing per-category CSVs at `<out-root>/<category>.csv` and a master CSV at `<out-root>/qsort_master.csv`.
- Logs failures to `<out-root>/failures.log` and continues.

## Analysis: `analysis/analyze_qsort.py`
- Loads the master CSV and produces per-category plots in `<out-dir>`:
  - Boxplot of `elapsed_ms` grouped by input file.
  - Median runtime vs `n` scatter with fitted `k * n * log2(n)` curve.
  - Optional comparisons vs time scatter.
- Exports per-category summaries as `<category>_summary.csv` and a consolidated `summary_all_categories.csv`.
