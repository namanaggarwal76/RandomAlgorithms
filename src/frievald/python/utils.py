"""Utility helpers for experiments."""

from __future__ import annotations

from typing import Literal, Optional

import numpy as np

Matrix = np.ndarray


def generate_random_matrix(
    n: int,
    low: float = -1.0,
    high: float = 1.0,
    rng: Optional[np.random.Generator] = None,
    dtype: type = np.float64,
) -> Matrix:
    """Return an n x n matrix with entries sampled uniformly from [low, high)."""

    if n <= 0:
        raise ValueError("n must be positive")
    rng = rng or np.random.default_rng()
    return rng.uniform(low, high, size=(n, n)).astype(dtype)


def inject_error(
    matrix: Matrix,
    mode: Literal["sparse", "dense", "worst_case"] = "sparse",
    magnitude: float = 1.0,
    rng: Optional[np.random.Generator] = None,
) -> Matrix:
    """Return a copy of `matrix` with injected error according to the mode."""

    if mode not in {"sparse", "dense", "worst_case"}:
        raise ValueError("mode must be 'sparse', 'dense', or 'worst_case'")
    rng = rng or np.random.default_rng()
    corrupted = matrix.copy()
    n = matrix.shape[0]

    if mode == "sparse":
        i = rng.integers(0, n)
        j = rng.integers(0, n)
        corrupted[i, j] += magnitude * rng.choice([-1.0, 1.0])
    elif mode == "dense":
        row = rng.integers(0, n)
        noise = magnitude * rng.uniform(-1.0, 1.0, size=n)
        corrupted[row, :] += noise
    else:
        if n < 2:
            raise ValueError("worst_case mode requires matrices with at least two columns")
        row = rng.integers(0, n)
        col1 = rng.integers(0, n)
        col2 = rng.integers(0, n - 1)
        if col2 >= col1:
            col2 += 1  # ensure two distinct columns
        corrupted[row, col1] -= magnitude
        corrupted[row, col2] += magnitude

    return corrupted
