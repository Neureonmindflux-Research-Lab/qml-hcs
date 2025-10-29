#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Forecasting Metrics
-------------------
Evaluation metrics for temporal and predictive performance analysis.

This module provides the following functions:
- ``mape``: Mean Absolute Percentage Error
- ``mase``: Mean Absolute Scaled Error
- ``delta_lag``: Lag anticipation (ΔLag) metric for directional alignment.
"""

from __future__ import annotations

import numpy as np

from ..core.types import Array, TensorLike


def mape(y_true: TensorLike, y_pred: TensorLike) -> float:
    """
    Mean Absolute Percentage Error (MAPE).

    Measures the average absolute percentage deviation between the predicted and
    true values.

    Parameters
    ----------
    y_true : TensorLike
        Ground-truth time series or target values.
    y_pred : TensorLike
        Predicted time series or model outputs.

    Returns
    -------
    float
        Mean absolute percentage error (in percent).

    Notes
    -----
    A small epsilon (1e-12) is added to the denominator to prevent division by zero.
    """
    t = np.asarray(y_true, dtype=float).reshape(-1)
    p = np.asarray(y_pred, dtype=float).reshape(-1)
    eps = 1e-12
    return float(np.mean(np.abs((t - p) / (t + eps))) * 100.0)


def mase(y_true: TensorLike, y_pred: TensorLike, y_naive: TensorLike) -> float:
    """
    Mean Absolute Scaled Error (MASE).

    Scales the absolute forecast error by the mean absolute difference of a naive
    forecast model, providing a scale-free error metric that can be compared across
    datasets.

    Parameters
    ----------
    y_true : TensorLike
        Ground-truth time series or target values.
    y_pred : TensorLike
        Predicted time series or model outputs.
    y_naive : TensorLike
        Naive baseline time series (e.g., lag-1 shifted version of y_true).

    Returns
    -------
    float
        Mean absolute scaled error.

    Notes
    -----
    Values below 1.0 indicate that the model outperforms the naive baseline.
    """
    t = np.asarray(y_true, dtype=float).reshape(-1)
    p = np.asarray(y_pred, dtype=float).reshape(-1)
    n = np.asarray(y_naive, dtype=float).reshape(-1)
    denom = np.mean(np.abs(t - n)) + 1e-12
    return float(np.mean(np.abs(t - p)) / denom)


def delta_lag(y_true_seq: Array, y_pred_seq: Array) -> float:
    """
    Lag anticipation metric (ΔLag).

    Measures alignment between the predicted and actual change directions in a
    temporal sequence. It compares the sign of consecutive differences and computes
    their mean product.

    Parameters
    ----------
    y_true_seq : Array
        Ground-truth sequence of values.
    y_pred_seq : Array
        Predicted sequence of values.

    Returns
    -------
    float
        Mean signed alignment between predicted and true change directions,
        ranging from ``-1`` (complete anti-alignment) to ``1`` (perfect alignment).

    Notes
    -----
    A value near zero implies random or lagged responses.
    """
    t = np.asarray(y_true_seq, dtype=float).reshape(-1)
    p = np.asarray(y_pred_seq, dtype=float).reshape(-1)
    dt_true = np.diff(t)
    dt_pred = np.diff(p)
    sign_true = np.sign(dt_true)
    sign_pred = np.sign(dt_pred)
    alignment = (sign_true * sign_pred).mean()
    return float(alignment)

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error (RMSE)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
