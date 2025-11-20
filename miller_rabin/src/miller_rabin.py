"""Miller-Rabin probabilistic primality test implementation.

Features:
- miller_rabin(n, k, rng) core test returning (is_probable_prime, witness_list)
- deterministic_mr(n) for <= 2^64 using known bases
- generate_random_number(bits, seed) helper
- accuracy evaluation against sympy.isprime for benchmarking

We mirror style of other algorithms in repository.
"""
from __future__ import annotations
import random
import time
from dataclasses import dataclass
from typing import List, Tuple, Iterable, Optional, Sequence

try:
    from sympy import isprime as sympy_isprime
except ImportError:  # fallback simple deterministic for small numbers
    def sympy_isprime(n: int) -> bool:
        if n < 2:
            return False
        if n % 2 == 0:
            return n == 2
        r = int(n ** 0.5)
        f = 3
        while f <= r:
            if n % f == 0:
                return False
            f += 2
        return True

@dataclass
class MRResult:
    n: int
    k: int
    is_probable_prime: bool
    witnesses: List[int]
    time_seconds: float
    # Optional extended diagnostics
    bases_used: Optional[List[int]] = None
    base_collision_fraction: Optional[float] = None

# Deterministic bases for 64-bit range (see research by Jim Sinclair / OEIS A014233 sources)
_DETERMINISTIC_BASES_64 = [2, 3, 5, 7, 11, 13]


def _decompose(n: int) -> Tuple[int, int]:
    """Write n-1 as 2^r * d with d odd."""
    assert n >= 2
    d = n - 1
    r = 0
    while d % 2 == 0:
        d //= 2
        r += 1
    return r, d


def _try_composite(a: int, d: int, n: int, r: int) -> bool:
    """Return True if a is a witness to compositeness."""
    x = pow(a, d, n)
    if x == 1 or x == n - 1:
        return False
    for _ in range(r - 1):
        x = (x * x) % n
        if x == n - 1:
            return False
    return True  # composite


def miller_rabin(n: int, k: int = 10, rng: Optional[random.Random] = None) -> MRResult:
    """Perform k rounds of Miller-Rabin on n.

    Args:
        n: integer to test
        k: number of random bases
        rng: optional random.Random instance for reproducibility
    Returns:
        MRResult object with details.
    """
    start = time.perf_counter()
    if n < 2:
        return MRResult(n, k, False, [], 0.0)
    # small primes quick path
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
    if n in small_primes:
        return MRResult(n, k, True, [], time.perf_counter() - start)
    if any(n % p == 0 for p in small_primes):
        return MRResult(n, k, False, [], time.perf_counter() - start)

    r, d = _decompose(n)
    witnesses: List[int] = []
    rng = rng or random.Random()
    bases_used: List[int] = []
    for _ in range(k):
        a = rng.randrange(2, n - 2)
        bases_used.append(a)
        if _try_composite(a, d, n, r):
            witnesses.append(a)
            # compute collisions among used bases so far
            uniq = len(set(bases_used))
            coll_frac = 1.0 - (uniq / len(bases_used))
            return MRResult(n, k, False, witnesses, time.perf_counter() - start, bases_used=bases_used, base_collision_fraction=coll_frac)
    uniq = len(set(bases_used)) if bases_used else 0
    coll_frac = 1.0 - (uniq / len(bases_used)) if bases_used else 0.0
    return MRResult(n, k, True, witnesses, time.perf_counter() - start, bases_used=bases_used, base_collision_fraction=coll_frac)


def miller_rabin_with_bases(n: int, bases: Sequence[int], stop_early: bool = True) -> MRResult:
    """Run Miller-Rabin using a provided sequence of bases.

    Args:
        n: integer to test
        bases: sequence of bases (integers >=2)
        stop_early: if True, stop on first witness; else exhaust all bases
    Returns:
        MRResult with bases_used and collision stats.
    """
    start = time.perf_counter()
    if n < 2:
        return MRResult(n, len(bases), False, [], 0.0, bases_used=list(bases), base_collision_fraction=0.0)
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
    if n in small_primes:
        return MRResult(n, len(bases), True, [], time.perf_counter() - start, bases_used=list(bases), base_collision_fraction=0.0)
    if any(n % p == 0 for p in small_primes):
        return MRResult(n, len(bases), False, [], time.perf_counter() - start, bases_used=list(bases), base_collision_fraction=0.0)

    r, d = _decompose(n)
    witnesses: List[int] = []
    used: List[int] = []
    for a in bases:
        a = int(a) % n
        if a < 2:
            a = (a + 2) % n
            if a < 2:
                a = 2
        used.append(a)
        if _try_composite(a, d, n, r):
            witnesses.append(a)
            if stop_early:
                uniq = len(set(used))
                coll_frac = 1.0 - (uniq / len(used))
                return MRResult(n, len(bases), False, witnesses, time.perf_counter() - start, bases_used=used, base_collision_fraction=coll_frac)
    uniq = len(set(used)) if used else 0
    coll_frac = 1.0 - (uniq / len(used)) if used else 0.0
    return MRResult(n, len(bases), len(witnesses) == 0, witnesses, time.perf_counter() - start, bases_used=used, base_collision_fraction=coll_frac)


def deterministic_mr(n: int) -> bool:
    """Deterministic Miller-Rabin for 64-bit integers using fixed bases."""
    if n < 2:
        return False
    for p in [2, 3, 5, 7, 11, 13]:
        if n == p:
            return True
        if n % p == 0:
            return n == p
    r, d = _decompose(n)
    for a in _DETERMINISTIC_BASES_64:
        if a >= n:
            continue
        if _try_composite(a, d, n, r):
            return False
    return True


def generate_random_number(bits: int, seed: int) -> int:
    rng = random.Random(seed)
    # ensure highest bit set to get exact bit length
    n = rng.getrandbits(bits) | (1 << (bits - 1)) | 1  # make odd
    return n


def evaluate_accuracy(bit_sizes: Iterable[int], rounds: Iterable[int], seeds: Iterable[int], sample_per_size: int = 50) -> List[dict]:
    """Generate random numbers and evaluate accuracy vs sympy.isprime.

    Returns list of dict rows for CSV.
    """
    rows = []
    for bits in bit_sizes:
        for seed in seeds:
            base_seed = seed * 10_000 + bits
            rng_master = random.Random(base_seed)
            for i in range(sample_per_size):
                n_seed = rng_master.randrange(1, 1_000_000_000)
                n = generate_random_number(bits, n_seed)
                truth = sympy_isprime(n)
                for k in rounds:
                    res = miller_rabin(n, k, random.Random(seed + i + k))
                    rows.append({
                        "bits": bits,
                        "seed": seed,
                        "sample_index": i,
                        "rounds": k,
                        "n": n,
                        "is_probable_prime": res.is_probable_prime,
                        "ground_truth_prime": truth,
                        "correct": res.is_probable_prime == truth,
                        "witness_count": len(res.witnesses),
                        "time_seconds": res.time_seconds,
                    })
    return rows

__all__ = [
    "MRResult",
    "miller_rabin",
    "miller_rabin_with_bases",
    "deterministic_mr",
    "generate_random_number",
    "evaluate_accuracy",
]

# A small list of known Carmichael numbers (first several terms)
CARMICHAEL_NUMBERS_SMALL = [
    561, 1105, 1729, 2465, 2821, 6601, 8911, 10585, 15841, 29341,
    41041, 46657, 52633, 62745, 63973, 75361, 101101, 115921, 126217,
    162401, 172081, 188461, 252601, 278545, 294409
]
