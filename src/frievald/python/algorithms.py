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
    a = np.ascontiguousarray(a, dtype=dtype)
    b = np.ascontiguousarray(b, dtype=dtype)
    c = np.ascontiguousarray(c, dtype=dtype)

    tmp = np.empty(n, dtype=dtype)
    left = np.empty(n, dtype=dtype)
    right = np.empty(n, dtype=dtype)

    for _ in range(k):
        r = rng.integers(0, 2, size=n, dtype=np.int8)  # random {0,1} vector
        np.matmul(b, r, out=tmp)
        np.matmul(a, tmp, out=left)
        np.matmul(c, r, out=right)
        if not np.allclose(left, right, atol=epsilon):
            return False
    return True


def matmul_numpy(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute matrix product using NumPy's optimized matmul (O(n^3))."""

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
                # Blocked multiplication keeps sub-blocks in cache and leverages BLAS kernels.
                c_block[:, j0:j_end] += a_block[:, k0:k_end] @ b_t[j0:j_end, k0:k_end].T
    return result


def _next_power_of_two(n: int) -> int:
    return 1 if n == 0 else 2 ** (n - 1).bit_length()


def _pad_to_power_of_two(a: np.ndarray, size: int) -> np.ndarray:
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
        # Pad to power-of-two square
        a_pad = np.ascontiguousarray(_pad_to_power_of_two(a, m))
        b_pad = np.ascontiguousarray(_pad_to_power_of_two(b, m))
    else:
        a_pad = np.ascontiguousarray(a)
        b_pad = np.ascontiguousarray(b)

    def _strassen(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        size = x.shape[0]
        if size <= threshold:
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
        c[:mid, :mid] = c11
        c[:mid, mid:] = c12
        c[mid:, :mid] = c21
        c[mid:, mid:] = c22
        return c

    c_pad = _strassen(a_pad, b_pad)
    return c_pad[: a.shape[0], : b.shape[1]]


def matrices_equal(a: np.ndarray, b: np.ndarray, epsilon: float = EPSILON) -> bool:
    """Return True if matrices are equal within the given tolerance."""

    if a.shape != b.shape:
        return False
    return np.allclose(a, b, atol=epsilon)
