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
    mode: Literal["sparse", "dense"] = "sparse",
    magnitude: float = 1.0,
    rng: Optional[np.random.Generator] = None,
) -> Matrix:
    """Return a copy of `matrix` with injected error according to the mode."""

    if mode not in {"sparse", "dense"}:
        raise ValueError("mode must be 'sparse' or 'dense'")
    rng = rng or np.random.default_rng()
    corrupted = matrix.copy()
    n = matrix.shape[0]

    if mode == "sparse":
        i = rng.integers(0, n)
        j = rng.integers(0, n)
        corrupted[i, j] += magnitude * rng.choice([-1.0, 1.0])
    else:
        row = rng.integers(0, n)
        noise = magnitude * rng.uniform(-1.0, 1.0, size=n)
        corrupted[row, :] += noise

    return corrupted
