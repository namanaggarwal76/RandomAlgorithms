"""Core algorithms for matrix multiplication and verification.

This module provides both deterministic matrix multiplication baselines and
Frievald's randomized verification algorithm. All routines operate on square
matrices represented as NumPy ndarrays. Frievald's algorithm runs in
O(k * n^2) time, providing a fast probabilistic verifier for A * B = C.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

EPSILON = 1e-8  # tolerance for floating point comparisons


def _validate_square_matrices(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> int:
    """Validate shapes and return the matrix dimension."""

    # Guard against nonsensical inputs before any heavy computation runs.
    if a.ndim != 2 or b.ndim != 2 or c.ndim != 2:
        raise ValueError("All inputs must be 2D matrices.")
    if a.shape[1] != b.shape[0] or a.shape[0] != c.shape[0] or b.shape[1] != c.shape[1]:
        raise ValueError("Matrix dimensions are incompatible for multiplication.")
    if a.shape[0] != a.shape[1]:
        raise ValueError("Frievald's algorithm assumes square matrices.")
    return a.shape[0]


def frievald_verify(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    k: int = 1,
    rng: Optional[np.random.Generator] = None,
    epsilon: float = EPSILON,
) -> bool:
    """Probabilistically verify whether A * B == C using Frievald's algorithm.

    Runs in O(k * n^2) time where k is the number of repetitions and n is the
    matrix dimension. If the product is incorrect, the probability of a false
    positive is at most (1/2)^k.
    """

    n = _validate_square_matrices(a, b, c)
    if k <= 0:
        raise ValueError("k must be positive.")
    rng = rng or np.random.default_rng()

    dtype = np.result_type(a.dtype, b.dtype, c.dtype)
    # Ensure contiguous buffers so repeated matmul calls avoid implicit copies.
    a = np.ascontiguousarray(a, dtype=dtype)
    b = np.ascontiguousarray(b, dtype=dtype)
    c = np.ascontiguousarray(c, dtype=dtype)

    # Scratch buffers reused across iterations to limit allocations.
    tmp = np.empty(n, dtype=dtype)
    left = np.empty(n, dtype=dtype)
    right = np.empty(n, dtype=dtype)

    for _ in range(k):
        # Random {0,1} witness vector defining the projection tested this round.
        r = rng.integers(0, 2, size=n, dtype=np.int8)
        np.matmul(b, r, out=tmp)
        np.matmul(a, tmp, out=left)
        np.matmul(c, r, out=right)
        if not np.allclose(left, right, atol=epsilon):
            return False
    return True


def matmul_numpy(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute matrix product using NumPy's optimized matmul (O(n^3))."""

    # np.matmul delegates to BLAS when available, so rely on it for the baseline.
    return np.matmul(a, b)


def matmul_triple_loop(a: np.ndarray, b: np.ndarray, block_size: int = 64) -> np.ndarray:
    """Compute matrix product using a cache-friendly blocked triple-loop implementation."""

    if a.shape[1] != b.shape[0]:
        raise ValueError("Matrix dimensions are incompatible for multiplication.")
    if block_size <= 0:
        raise ValueError("block_size must be positive.")

    a_c = np.ascontiguousarray(a)
    b_c = np.ascontiguousarray(b)
    n, m = a_c.shape
    p = b_c.shape[1]

    result = np.zeros((n, p), dtype=a.dtype)
    b_t = np.ascontiguousarray(b_c.T)

    block = max(1, block_size)

    for i0 in range(0, n, block):
        i_end = min(i0 + block, n)
        a_block = a_c[i0:i_end]
        c_block = result[i0:i_end]
        for j0 in range(0, p, block):
            j_end = min(j0 + block, p)
            for k0 in range(0, m, block):
                k_end = min(k0 + block, m)
                # Blocked multiply reuses tiles in cache and leverages vectorized matmul.
                c_block[:, j0:j_end] += a_block[:, k0:k_end] @ b_t[j0:j_end, k0:k_end].T
                # Tight innermost loop stays in NumPy space, so Python overhead is low.
    return result


def _next_power_of_two(n: int) -> int:
    """Return the smallest power of two greater than or equal to ``n``."""
    return 1 if n == 0 else 2 ** (n - 1).bit_length()


def _pad_to_power_of_two(a: np.ndarray, size: int) -> np.ndarray:
    """Pad matrix ``a`` with zeros to match a square matrix of ``size``."""
    if a.shape[0] == size and a.shape[1] == size:
        return a
    padded = np.zeros((size, size), dtype=a.dtype)
    padded[: a.shape[0], : a.shape[1]] = a
    return padded


def matmul_strassen(a: np.ndarray, b: np.ndarray, threshold: int = 64) -> np.ndarray:
    """Multiply two square matrices using Strassen's algorithm.

    Recursively applies Strassen when size > threshold; below threshold uses
    NumPy's matmul. Pads inputs up to next power-of-two dimension, then trims
    result back to original shape.
    Complexity: O(n^{log_2 7}) ≈ O(n^2.807).
    """

    if a.shape[1] != b.shape[0]:
        raise ValueError("Matrix dimensions are incompatible for multiplication.")
    n = a.shape[0]
    m = _next_power_of_two(n)
    if n != m or a.shape[1] != m or b.shape[1] != m:
        # Pad to power-of-two square before recursing so Strassen splits evenly.
        a_pad = np.ascontiguousarray(_pad_to_power_of_two(a, m))
        b_pad = np.ascontiguousarray(_pad_to_power_of_two(b, m))
    else:
        a_pad = np.ascontiguousarray(a)
        b_pad = np.ascontiguousarray(b)

    def _strassen(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Recursively multiply two equally-sized square matrices via Strassen."""
        size = x.shape[0]
        if size <= threshold:
            # Small blocks fall back to fast dense multiplication.
            return x @ y
        mid = size // 2
        a11 = x[:mid, :mid]
        a12 = x[:mid, mid:]
        a21 = x[mid:, :mid]
        a22 = x[mid:, mid:]
        b11 = y[:mid, :mid]
        b12 = y[:mid, mid:]
        b21 = y[mid:, :mid]
        b22 = y[mid:, mid:]

        # Strassen's seven products (reduced multiplication count vs naive eight).
        p1 = _strassen(a11 + a22, b11 + b22)
        p2 = _strassen(a21 + a22, b11)
        p3 = _strassen(a11, b12 - b22)
        p4 = _strassen(a22, b21 - b11)
        p5 = _strassen(a11 + a12, b22)
        p6 = _strassen(a21 - a11, b11 + b12)
        p7 = _strassen(a12 - a22, b21 + b22)

        c11 = p1 + p4 - p5 + p7
        c12 = p3 + p5
        c21 = p2 + p4
        c22 = p1 - p2 + p3 + p6

        c = np.empty((size, size), dtype=x.dtype)
        # Reassemble the four quadrants into a single result matrix.
        c[:mid, :mid] = c11
        c[:mid, mid:] = c12
        c[mid:, :mid] = c21
        c[mid:, mid:] = c22
        return c

    c_pad = _strassen(a_pad, b_pad)
    # Remove padding so the returned matrix matches original dimensions.
    return c_pad[: a.shape[0], : b.shape[1]]


def matrices_equal(a: np.ndarray, b: np.ndarray, epsilon: float = EPSILON) -> bool:
    """Return True if matrices are equal within the given tolerance."""

    if a.shape != b.shape:
        return False
    return np.allclose(a, b, atol=epsilon)
