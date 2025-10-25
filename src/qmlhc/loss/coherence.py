#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inter-branch Coherence Loss
---------------------------
Controls dispersion among K projected futures to promote stable projections.

Two modes are supported:

- ``"variance"`` (default): mean squared deviation from the per-dimension mean.
- ``"mad"``: mean absolute deviation from the per-dimension mean.


"""

from __future__ import annotations

import numpy as np

from ..core.types import Array, LossFn


class CoherenceLoss(LossFn):
    """
    Penalize dispersion across candidate future branches.

    Parameters
    ----------
    mode : str, optional
        Dispersion metric to use. Options:
        - ``"variance"`` (scale-sensitive, smooth)
        - ``"mad"`` (mean absolute deviation)
        Default is ``"variance"``.
    """

    def __init__(self, mode: str = "variance"):
        # Mode selects the dispersion metric; stored in lowercase for fast branching.
        self._mode = str(mode).lower()

    def __call__(self, futures: Array) -> float:
        """
        Compute the coherence penalty for a set of candidate futures.

        Parameters
        ----------
        futures : Array
            Matrix of candidate futures with shape ``(K, D)``.

        Returns
        -------
        float
            Dispersion penalty (lower is better).

        Raises
        ------
        ValueError
            If ``futures`` is not a 2-D array, or if ``mode`` is unsupported.
        """
        fut = np.asarray(futures, dtype=float)
        if fut.ndim != 2:
            raise ValueError("futures must have shape (K, D)")

        # Per-dimension mean, kept 2-D for broadcasting (K, D) - (1, D)
        mu = fut.mean(axis=0, keepdims=True)

        if self._mode == "variance":
            # Mean of squared deviations (scale-sensitive and differentiable).
            var = ((fut - mu) ** 2).mean()
            return float(var)

        if self._mode == "mad":
            # Mean absolute deviation from the mean (more robust to outliers).
            mad = np.abs(fut - mu).mean()
            return float(mad)

        raise ValueError(f"unsupported coherence mode: {self._mode}")
