#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Anomaly Detection Metrics
-------------------------
Early-warning and evaluation utilities for anomaly detection on time series.

Provided metrics
----------------
- ``early_roc_auc``: ROC-AUC restricted to an early-detection horizon.
- ``recall_at_lag``: Recall counting predictions made within a lag window.

"""

from __future__ import annotations

import numpy as np

from ..core.types import TensorLike


def early_roc_auc(y_true: TensorLike, scores: TensorLike, horizon: int = 1) -> float:
    """
    Compute ROC-AUC restricted to an early-detection horizon.

    Positive events occurring within the next ``horizon`` steps are labeled as
    positives for the current time index. The metric then estimates the ROC-AUC
    by pairwise comparisons between positive and negative score sets.

    Parameters
    ----------
    y_true : TensorLike
        Ground-truth anomaly indicator (1 for anomaly, 0 otherwise), shape ``(T,)``.
    scores : TensorLike
        Continuous anomaly scores aligned with ``y_true``, shape ``(T,)``.
    horizon : int, optional
        Early-detection lookahead window, by default ``1``.

    Returns
    -------
    float
        Early-window ROC-AUC in ``[0, 1]``. Returns ``0.5`` if positives or
        negatives are absent (uninformative baseline).
    """
    y = np.asarray(y_true, dtype=float).reshape(-1)
    s = np.asarray(scores, dtype=float).reshape(-1)

    n = len(y)
    # Shifted labels: mark current index positive if an event occurs within horizon.
    y_shift = np.zeros_like(y)
    for i in range(n - horizon):
        if y[i + horizon] > 0:
            y_shift[i] = 1.0

    pos = s[y_shift == 1]
    neg = s[y_shift == 0]

    if len(pos) == 0 or len(neg) == 0:
        return 0.5

    # Empirical AUC via pairwise comparisons
    greater = sum(p > n_ for p in pos for n_ in neg)
    auc = greater / (len(pos) * len(neg))
    return float(auc)


def recall_at_lag(y_true: TensorLike, y_pred: TensorLike, lag: int = 1) -> float:
    """
    Fraction of anomalies recalled within a backward lag window.

    For each true anomaly at index ``i``, counts a hit if any predicted positive
    occurs in ``[i - lag, i]``. Suitable for evaluating early alarms that may
    precede the labeled event by up to ``lag`` steps.

    Parameters
    ----------
    y_true : TensorLike
        Ground-truth anomaly indicator (1 for anomaly, 0 otherwise), shape ``(T,)``.
    y_pred : TensorLike
        Binary predictions aligned with ``y_true``, shape ``(T,)``.
    lag : int, optional
        Backward window size for crediting early detections, by default ``1``.

    Returns
    -------
    float
        Recall in ``[0, 1]`` computed with lag tolerance.
    """
    y = np.asarray(y_true, dtype=float).reshape(-1)
    p = np.asarray(y_pred, dtype=float).reshape(-1)

    hits = 0
    total = 0
    for i in range(len(y)):
        if y[i] > 0:
            total += 1
            # Credit any positive prediction within [i - lag, i]
            for j in range(max(0, i - lag), i + 1):
                if j < len(p) and p[j] > 0:
                    hits += 1
                    break
    return float(hits / (total + 1e-12))
