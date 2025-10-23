#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Control Metrics
---------------
Stability and responsiveness metrics for dynamic systems.

Provided metrics
----------------
- ``overshoot``: maximum relative overshoot of the response over the target.
- ``settling_time``: samples required to remain within a tolerance band.
- ``robustness``: inverse sensitivity to disturbances (MSE-based).
"""

from __future__ import annotations

import numpy as np

from ..core.types import TensorLike


def overshoot(y_true: TensorLike, y_pred: TensorLike) -> float:
    """
    Maximum relative overshoot of the response over the reference.

    The metric is computed as:
    ``max(0, max(y_pred) - max(y_true)) / |max(y_true)|``.
    If the reference maximum is zero, returns ``0.0``.

    Parameters
    ----------
    y_true : TensorLike
        Reference/target signal, shape ``(T,)``.
    y_pred : TensorLike
        Response/prediction signal, shape ``(T,)``.

    Returns
    -------
    float
        Relative overshoot (unitless, ``>= 0``).
    """
    t = np.asarray(y_true, dtype=float).reshape(-1)
    p = np.asarray(y_pred, dtype=float).reshape(-1)
    max_ref = np.max(t)
    if max_ref == 0:
        return 0.0
    max_err = np.max(np.maximum(0.0, p - max_ref))
    return float(max_err / abs(max_ref))


def settling_time(y_true: TensorLike, y_pred: TensorLike, tol: float = 0.05) -> int:
    """
    Samples until the response stays within a tolerance band around the target.

    The tolerance band is defined around the final target value:
    ``[ref*(1 - tol), ref*(1 + tol)]``, where ``ref = y_true[-1]``.
    The function scans backward and returns the last index violating the band,
    plus one. If the entire sequence is within band, returns ``0``.

    Parameters
    ----------
    y_true : TensorLike
        Reference/target signal, shape ``(T,)``.
    y_pred : TensorLike
        Response/prediction signal, shape ``(T,)``.
    tol : float, optional
        Relative tolerance (e.g., ``0.05`` for ±5%), by default ``0.05``.

    Returns
    -------
    int
        Settling time in samples.
    """
    t = np.asarray(y_true, dtype=float).reshape(-1)
    p = np.asarray(y_pred, dtype=float).reshape(-1)
    ref = t[-1]
    lower = ref * (1 - tol)
    upper = ref * (1 + tol)
    for i in range(len(p) - 1, -1, -1):
        if not (lower <= p[i] <= upper):
            return i + 1
    return 0


def robustness(y_true: TensorLike, y_pred: TensorLike) -> float:
    """
    Inverse sensitivity to disturbances based on mean squared error.

    Defined as ``1 / (1 + MSE(y_true, y_pred))``. Values lie in ``(0, 1]``,
    where larger is better (more robust).

    Parameters
    ----------
    y_true : TensorLike
        Reference/target signal, shape ``(T,)``.
    y_pred : TensorLike
        Response/prediction signal, shape ``(T,)``.

    Returns
    -------
    float
        Robustness score in ``(0, 1]``.
    """
    t = np.asarray(y_true, dtype=float).reshape(-1)
    p = np.asarray(y_pred, dtype=float).reshape(-1)
    mse = np.mean((t - p) ** 2)
    return float(1.0 / (1.0 + mse))
