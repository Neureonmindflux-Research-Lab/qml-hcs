#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Triadic Consistency Loss
------------------------
Enforces coherence among past, present, and projected future states
(`S_{t-1}`, `S_t`, `Ŝ_{t+1}`).

This loss penalizes deviations between the current state and both its
immediate predecessor and projected successor, promoting smooth temporal
evolution in hypercausal models.
"""

from __future__ import annotations

import numpy as np

from ..core.types import Array, LossFn, TensorLike


class ConsistencyLoss(LossFn):
    """
    Compute triadic consistency loss.

    Penalizes incoherence between the previous, current, and projected
    future states. The overall loss is a weighted sum of the mean squared
    deviations between consecutive temporal states.

    Parameters
    ----------
    alpha : float, optional
        Weight for the present–past term, by default ``1.0``.
    beta : float, optional
        Weight for the present–future term, by default ``1.0``.

    Notes
    -----
    This formulation stabilizes dynamic state transitions by encouraging
    local temporal consistency:
    ``L = α‖S_t - S_{t-1}‖² + β‖S_t - Ŝ_{t+1}‖²``.
    """

    def __init__(self, alpha: float = 1.0, beta: float = 1.0):
        # Alpha weights present–past coherence; beta weights present–future coherence.
        self._a = float(alpha)
        self._b = float(beta)

    def __call__(self, s_tm1: TensorLike, s_t: TensorLike, s_tp1_hat: TensorLike) -> float:
        """
        Evaluate the triadic consistency loss.

        Parameters
        ----------
        s_tm1 : TensorLike
            Previous state vector ``S_{t-1}``.
        s_t : TensorLike
            Current state vector ``S_t``.
        s_tp1_hat : TensorLike
            Projected future state vector ``Ŝ_{t+1}``.

        Returns
        -------
        float
            Weighted triadic consistency loss.

        Raises
        ------
        ValueError
            If input vectors differ in dimensionality.
        """
        s1 = np.asarray(s_tm1, dtype=float).reshape(-1)
        s2 = np.asarray(s_t, dtype=float).reshape(-1)
        s3 = np.asarray(s_tp1_hat, dtype=float).reshape(-1)

        if not (s1.size == s2.size == s3.size):
            raise ValueError("All state vectors must have the same dimension.")

        # Mean squared deviations: present–past and present–future
        d_prev = np.mean((s2 - s1) ** 2)
        d_fut = np.mean((s2 - s3) ** 2)

        loss = self._a * d_prev + self._b * d_fut
        return float(loss)
