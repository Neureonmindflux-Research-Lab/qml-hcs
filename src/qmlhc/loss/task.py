#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Task-Level Losses
-----------------
Standard objectives used for training quantum–hypercausal models.

"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from ..core.types import Array, LossFn, TensorLike


class TaskLoss(LossFn, Protocol):
    """
    Base protocol for any task-level loss function.

    Methods
    -------
    __call__(pred, target)
        Compute a scalar loss given prediction and target tensors.
    """

    def __call__(self, pred: TensorLike, target: TensorLike) -> float:
        ...


class MSELoss:
    """
    Mean Squared Error (MSE) between predicted and target tensors.

    Notes
    -----
    Both inputs are flattened to 1-D before comparison.
    """

    def __call__(self, pred: TensorLike, target: TensorLike) -> float:
        """
        Compute MSE.

        Parameters
        ----------
        pred : TensorLike
            Prediction tensor.
        target : TensorLike
            Target tensor.

        Returns
        -------
        float
            Mean squared error.

        Raises
        ------
        ValueError
            If ``pred`` and ``target`` differ in size.
        """
        p = np.asarray(pred, dtype=float).reshape(-1)
        t = np.asarray(target, dtype=float).reshape(-1)
        if p.size != t.size:
            raise ValueError("pred and target must have the same size")
        return float(np.mean((p - t) ** 2))


class MAELoss:
    """
    Mean Absolute Error (MAE) between predicted and target tensors.

    Notes
    -----
    Both inputs are flattened to 1-D before comparison.
    """

    def __call__(self, pred: TensorLike, target: TensorLike) -> float:
        """
        Compute MAE.

        Parameters
        ----------
        pred : TensorLike
            Prediction tensor.
        target : TensorLike
            Target tensor.

        Returns
        -------
        float
            Mean absolute error.

        Raises
        ------
        ValueError
            If ``pred`` and ``target`` differ in size.
        """
        p = np.asarray(pred, dtype=float).reshape(-1)
        t = np.asarray(target, dtype=float).reshape(-1)
        if p.size != t.size:
            raise ValueError("pred and target must have the same size")
        return float(np.mean(np.abs(p - t)))


class CrossEntropyLoss:
    """
    Cross-entropy for probabilistic outputs normalized over classes.

    Notes
    -----
    Inputs are clipped to ``[1e-12, 1.0]`` and normalized to sum to 1.0
    prior to computing the negative log-likelihood.
    """

    def __call__(self, pred: TensorLike, target: TensorLike) -> float:
        """
        Compute cross-entropy between two discrete distributions.

        Parameters
        ----------
        pred : TensorLike
            Predicted (unnormalized or probabilistic) scores per class.
        target : TensorLike
            Target distribution (unnormalized or probabilistic).

        Returns
        -------
        float
            Cross-entropy value.

        Raises
        ------
        ValueError
            If ``pred`` and ``target`` differ in size.
        """
        p = np.asarray(pred, dtype=float).reshape(-1)
        t = np.asarray(target, dtype=float).reshape(-1)
        if p.size != t.size:
            raise ValueError("pred and target must have the same size")

        # Normalize to probabilities (prevent log(0) with clipping).
        p = np.clip(p, 1e-12, 1.0)
        p /= np.sum(p)
        t = np.clip(t, 1e-12, 1.0)
        t /= np.sum(t)

        return float(-np.sum(t * np.log(p)))
