# Cardinality Estimation: LogLog, HyperLogLog, HyperLogLog++

This note explains how the three estimators implemented in `src/cardinality/cpp` turn a data stream into a near-linear cardinality estimate. It ties each step back to the concrete code so you can reason about runtime, accuracy, and implementation trade-offs.

## Shared Pipeline

All three estimators share the same hashing and leading-zero counting primitives:

- [`hash64`](src/cardinality/cpp/cardinality_estimators.hpp#L22) implements SplitMix64 to turn each element into a 64-bit value with good bit-mixing.
- [`rho`](src/cardinality/cpp/cardinality_estimators.hpp#L39) counts the position of the first `1` bit in the hashed suffix (the “rank”).

ASCII flow:

```
input x
   │
   ▼       64-bit hash
hash64(x) ──────────────► [idx bits | suffix bits]
             │                    │
             │                    └─► rho() ⇒ rank
             └─► bucket index          (leading zeros)
                            │
                            ▼
                      registers[idx] ← max(registers[idx], rank)
```

Regardless of the estimator, `add()` performs O(1) work per element (one hash, simple bit arithmetic, vector or map write), while `estimate()` scans all registers, making it O(m) where `m = 2^precision`.

## Summary Snapshot

| Algorithm | Implementation refs | `add` cost | `estimate` cost | Memory footprint | Relative std. error | Notable corrections |
|-----------|--------------------|------------|-----------------|------------------|---------------------|---------------------|
| LogLog | [`LogLog::add`](src/cardinality/cpp/cardinality_estimators.cpp#L31), [`LogLog::estimate`](src/cardinality/cpp/cardinality_estimators.cpp#L49) | O(1) | O(m) | `m` bytes of 8-bit registers | ≈ `1.30/√m` | Small-range linear-counting fallback |
| HyperLogLog | [`HyperLogLog::add`](src/cardinality/cpp/cardinality_estimators.cpp#L87), [`HyperLogLog::estimate`](src/cardinality/cpp/cardinality_estimators.cpp#L130) | O(1) | O(m) | `m` bytes | ≈ `1.04/√m` | Harmonic mean + small/large-range correction |
| HyperLogLog++ | [`HyperLogLogPlusPlus::add`](src/cardinality/cpp/cardinality_estimators.cpp#L158), [`estimate`](src/cardinality/cpp/cardinality_estimators.cpp#L215) | O(1) (with sparse map until threshold) | O(m) once dense, otherwise O(sparse size) | Sparse map (`unordered_map`) until densified, then `m` bytes | ≈ `1.04/√m` with lower bias | Sparse linear counting, empirically derived bias table, 64-bit large-range correction |

## LogLog (Durand–Flajolet)

**Implementation hooks.** The constructor configures `m = 2^precision` buckets and a mask for the lower `(64 - precision)` bits ([`cardinality_estimators.cpp#L17`](src/cardinality/cpp/cardinality_estimators.cpp#L17)). [`LogLog::add`](src/cardinality/cpp/cardinality_estimators.cpp#L31) hashes each value, picks the bucket by the top bits, and stores the max observed rank per bucket. [`LogLog::estimate`](src/cardinality/cpp/cardinality_estimators.cpp#L49) averages all register ranks, exponentiates (`2^{avg}`), and multiplies by the constant `alpha = 0.39701`. A “small-range” correction reverts to linear counting when many registers remain zero.

**Pseudo-code (matches the implementation).**

```text
initialize registers[0..m-1] = 0

procedure add(x):
    h = hash64(x)
    idx = h >> (64 - precision)
    suffix = h & ((1 << (64 - precision)) - 1)
    rank = rho(suffix, 64 - precision)
    registers[idx] = max(registers[idx], rank)

function estimate():
    avg = average(registers)
    est = 0.39701 * m * 2^avg
    zeros = count(registers == 0)
    if zeros > 0 and est < 2.5 * m:
        return m * ln(m / zeros)   // linear counting
    return est
```

**Example (p = 3, m = 8).** Suppose the hashed stream produces the ranks shown below. Buckets without observations stay at 0.

```
bucket:   0 1 2 3 4 5 6 7
rank:     4 1 0 3 0 2 0 0

avg rank = (4+1+0+3+0+2+0+0) / 8 = 1.25
raw est = 0.39701 * 8 * 2^1.25 ≈ 7.5
zeros = 4 ⇒ linear counting est = 8 * ln(8/4) ≈ 5.5
```

**Analysis.**

- *Runtime:* `add()` is constant-time hashing and vector writes, while `estimate()` scans `m` registers (see the single loop at [`LogLog::estimate`](src/cardinality/cpp/cardinality_estimators.cpp#L50)).
- *Memory:* `std::vector<uint8_t> registers_` uses one byte per bucket ([`cardinality_estimators.cpp#L17`](src/cardinality/cpp/cardinality_estimators.cpp#L17) plus header fields).
- *Error:* The relative standard deviation is about `1.30/√m`. The implementation mirrors the original paper: averaging ranks amplifies variance relative to HyperLogLog. The linear-counting fallback reduces bias for very small true cardinalities because zero registers contain more information than the averaged rank.

## HyperLogLog (Flajolet et al. 2007)

**Implementation hooks.** [`HyperLogLog::add`](src/cardinality/cpp/cardinality_estimators.cpp#L87) is identical to LogLog’s update path (still using `hash64` and `rho`). Accuracy improves inside [`HyperLogLog::raw_estimate`](src/cardinality/cpp/cardinality_estimators.cpp#L115): instead of averaging ranks, it computes the harmonic mean of `2^{-register}` values and multiplies by `alpha(m) * m^2`. [`HyperLogLog::estimate`](src/cardinality/cpp/cardinality_estimators.cpp#L130) applies two well-known corrections: linear counting when many registers are zero and a large-range correction that uses the 64-bit hash space as an upper bound.

**Pseudo-code.**

```text
procedure add(x):
    same as LogLog

function raw_estimate():
    denom = Σ 2^{-registers[i]}
    return alpha(m) * m^2 / denom

function estimate():
    raw = raw_estimate()
    zeros = count(registers == 0)
    if raw <= 2.5m and zeros > 0:
        return m * ln(m / zeros)   // small-range
    if raw > 2^32 / 30:
        return -2^32 * ln(1 - raw / 2^32)   // large-range
    return raw
```

**Diagram: correcting extremes.**

```
raw harmonic mean
        │
        ├── small-range? ──► linear counting (uses zero buckets)
        │
        └── else large-range? ──► saturation correction
```

**Example scenario.** With `p = 14 (m = 16384)` and a stream of ~10,000 unique items, a few thousand registers stay zero. The raw harmonic mean slightly overestimates because noisy registers dominate. The small-range branch activates because `raw <= 2.5m` and `zeros > 0`, yielding a value close to the ground truth using the more precise occupancy signal.

**Analysis.**

- *Runtime/memory:* Identical to LogLog (vector of 8-bit registers; loops over `m` registers for estimation).
- *Error:* HyperLogLog’s harmonic mean shrinks the estimator variance to ~`1.04/√m`. The adaptive corrections coded in [`HyperLogLog::estimate`](src/cardinality/cpp/cardinality_estimators.cpp#L130) preserve unbiasedness both when the stream is tiny (linear counting) and when it approaches the hash-space size (saturation correction).
- *Reasoning about `alpha(m)`:* [`HyperLogLog::alpha`](src/cardinality/cpp/cardinality_estimators.cpp#L101) matches the empirically derived constants from the paper, with special cases for small `m` to prevent bias spikes.

## HyperLogLog++ (Google’s refinement)

**Implementation hooks.** The constructor configures a sparse-to-dense transition threshold (`threshold_ = sparse_factor * m`, [`cardinality_estimators.cpp#L151`](src/cardinality/cpp/cardinality_estimators.cpp#L151)). Two member fields keep track of the current mode:

- `dense_` (bool) starts `false` and flips to `true` once the sketch is densified.
- `sparse_` (`unordered_map<uint32_t, uint8_t>`) holds the bucket→rank map whenever `dense_ == false`.

[`HyperLogLogPlusPlus::add`](src/cardinality/cpp/cardinality_estimators.cpp#L158) writes into `sparse_` until it accumulates `threshold_` distinct bucket indices; at that moment `convert_to_dense()` merges the sparse entries into the dense `registers_` array and clears the map (`cardinality_estimators.cpp#L179-L189`). From then on `dense_` stays `true` and updates land directly in the vector, so the estimator behaves like a bias-corrected HyperLogLog.

[`HyperLogLogPlusPlus::estimate`](src/cardinality/cpp/cardinality_estimators.cpp#L215) contains an additional guard: if the sketch is still sparse, it computes `v = m_ - sparse_.size()`. When `v > 0` it uses the sparse linear-counting shortcut; if `v <= 0` (which can only happen if all buckets have been seen, meaning sparse mode is no longer useful) it forces a conversion by calling `convert_to_dense()` before proceeding.

[`HyperLogLogPlusPlus::estimate`](src/cardinality/cpp/cardinality_estimators.cpp#L215) keeps three tiers:

1. **Sparse linear counting** — when still sparse it estimates cardinality directly from the number of missing buckets (`m * ln(m / v)` where `v` is the number of empty buckets).
2. **Bias-corrected dense mode** — uses HyperLogLog’s `raw_estimate()` and subtracts a bias interpolated from [`bias()`](src/cardinality/cpp/cardinality_estimators.cpp#L191) when `raw ≤ 5m`.
3. **Small/large range corrections** — same logic as HyperLogLog for zero buckets and near-saturation.

**Pseudo-code.**

```text
procedure add(x):
    if not dense:
        sparse[idx] = max(sparse[idx], rank)
        if |sparse| >= threshold: convert_to_dense()
    else:
        registers[idx] = max(registers[idx], rank)

function estimate():
    if not dense:
        v = m - |sparse|
        if v > 0: return m * ln(m / v)
        convert_to_dense()
    raw = raw_estimate()
    if raw <= 5m: est = raw - bias(raw) else est = raw
    zeros = count(registers == 0)
    if est <= 5m and zeros > 0: est = m * ln(m / zeros)
    if est > 2^32 / 30: est = -2^32 * ln(1 - est / 2^32)
    return est
```

**Example (p = 12, sparse factor = 0.25).** With up to `0.25 * m = 1024` populated buckets, the sketch stores only observed buckets inside the `sparse_` map. Estimating 500 distinct elements touches just the `unordered_map`, so memory usage stays proportional to the observed support and `estimate()` reduces to `m * ln(m / v)` without scanning all `m = 4096` registers. Once the stream grows (say 20k elements) the map crosses the threshold, automatically densifies, and the estimator behaves like an improved HyperLogLog with bias subtraction.

**Analysis.**

- *Sparse-mode benefits:* Small cardinalities avoid scanning or storing the full register array. Memory roughly equals `(precision + 6)` bits per stored bucket plus overhead, matching [`memory_usage_bytes`](src/cardinality/cpp/cardinality_estimators.cpp#L244).
- *Bias handling:* The bias-correction table in [`bias()`](src/cardinality/cpp/cardinality_estimators.cpp#L191) stores `(raw_estimate, measured_bias)` samples gathered offline (similar to Google’s published tables). During estimation the code linearly interpolates between the two nearest entries to approximate the expected bias for the current raw value, then subtracts it (`estimator = raw - bias(raw)` at [`cardinality_estimators.cpp#L225-L229`](src/cardinality/cpp/cardinality_estimators.cpp#L225-L229)). This explicit subtraction removes the systematic upward drift that plain HyperLogLog exhibits when `raw ≤ 5m`, while leaving large-cardinality behavior unchanged.
- *Error bounds:* Once dense, HyperLogLog++ inherits the `1.04/√m` variance but with lower bias, especially for `n < 6m` thanks to the combination of sparse counting and bias subtraction. The 64-bit hash space also ensures the large-range correction is safe until astronomical cardinalities.

## Putting It All Together

- Choose **LogLog** only if you need the absolute simplest estimator; it uses the least logic but also carries the highest variance.
- **HyperLogLog** is the general-purpose choice: fixed memory, predictable error, and robust across ranges.
- **HyperLogLog++** shines when your data sets swing between tiny and huge; sparse mode keeps memory small early on, and the bias table plus 64-bit corrections provide production-grade accuracy.

Across the codebase, the uniform `CardinalityEstimator` interface ([`cardinality_estimators.hpp#L51`](src/cardinality/cpp/cardinality_estimators.hpp#L51)) means you can swap estimators without changing calling code—`add()` and `estimate()` always have the same semantics, but the internal logic above determines the space/accuracy/runtime trade-offs.
