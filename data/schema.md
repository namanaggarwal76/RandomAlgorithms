# Data Schema: random_qsort outputs

CSV header (exact order):

```
timestamp_utc_iso,category,input_file,n,seed,rep_id,elapsed_ms,cpu_ms,comparisons,swaps,correct
```

Field descriptions:
- timestamp_utc_iso: UTC timestamp in ISO 8601 format (e.g., 2025-11-16T12:34:56Z) at the end of the run.
- category: Category name inferred from the parent directory of `input_file` (e.g., `./datasets/uniform/arr1.txt` -> `uniform`).
- input_file: Path to the dataset file used for this run.
- n: Number of integers read from `input_file`.
- seed: The per-run RNG seed used by the C++ binary.
- rep_id: Repetition index for this input file (0-based).
- elapsed_ms: Wall-clock elapsed time in milliseconds measured with `std::chrono::high_resolution_clock`.
- cpu_ms: CPU time in milliseconds measured via `std::clock()` (if available on the platform).
- comparisons: Number of key comparisons counted inside QuickSort.
- swaps: Number of swaps performed by QuickSort (does not count trivial self-swaps).
- correct: `1` if QuickSort output exactly matches `std::sort`, else `0`.

Seeding policy:
- The top-level runner sets a base seed `seed_base` (default 42).
- Deterministic per-run seed is computed as:
  - `seed = seed_base + file_index * 1000 + rep_id`
  - `file_index` is the index of the file within its category after sorting files by size (ascending), starting at 0.
  - `rep_id` ranges from 0 to `reps - 1`.
- All random decisions in QuickSort use a single `std::mt19937` instance seeded with this per-run seed.
