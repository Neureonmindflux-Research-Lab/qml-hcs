#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Projection Policies
-------------------
Strategies to aggregate K candidate futures into a single representative vector.

This module provides three policies:

- ``MeanPolicy``: element-wise arithmetic mean across branches.
- ``MedianPolicy``: element-wise median across branches.
- ``MinRiskPolicy``: selects the single branch that minimizes a user-provided
  risk functional.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from ..core.types import Array, ProjectionPolicy, RiskFunctional, SelectionPolicy


class MeanPolicy(ProjectionPolicy):
    """
    Element-wise arithmetic mean across branches.

    Notes
    -----
    The representative vector is computed as ``mean(futures, axis=0)``.
    Diagnostic info includes the policy name and number of branches.
    """

    def select(self, futures: Array) -> tuple[Array, Mapping[str, Any]]:
        """
        Select a representative by averaging all candidate branches.

        Parameters
        ----------
        futures : Array
            Matrix of candidate futures with shape ``(K, D)``.

        Returns
        -------
        tuple
            ``(rep, diag)`` where:
            - ``rep`` is the element-wise mean with shape ``(D,)``.
            - ``diag`` contains ``{"policy": "mean", "branches": K}``.

        Raises
        ------
        ValueError
            If ``futures`` is not a 2-D array.
        """
        fut = np.asarray(futures, dtype=float)
        if fut.ndim != 2:
            raise ValueError("futures must have shape (K, D)")
        rep = fut.mean(axis=0)
        diag = {
            "policy": SelectionPolicy.MEAN.value,
            "branches": fut.shape[0],
        }
        return rep, diag


class MedianPolicy(ProjectionPolicy):
    """
    Element-wise median across branches.

    Notes
    -----
    The representative vector is computed as ``median(futures, axis=0)``.
    Diagnostic info includes the policy name and number of branches.
    """

    def select(self, futures: Array) -> tuple[Array, Mapping[str, Any]]:
        """
        Select a representative by taking the median per dimension.

        Parameters
        ----------
        futures : Array
            Matrix of candidate futures with shape ``(K, D)``.

        Returns
        -------
        tuple
            ``(rep, diag)`` where:
            - ``rep`` is the element-wise median with shape ``(D,)``.
            - ``diag`` contains ``{"policy": "median", "branches": K}``.

        Raises
        ------
        ValueError
            If ``futures`` is not a 2-D array.
        """
        fut = np.asarray(futures, dtype=float)
        if fut.ndim != 2:
            raise ValueError("futures must have shape (K, D)")
        rep = np.median(fut, axis=0)
        diag = {
            "policy": SelectionPolicy.MEDIAN.value,
            "branches": fut.shape[0],
        }
        return rep, diag


class MinRiskPolicy(ProjectionPolicy):
    """
    Select the single branch minimizing a provided risk functional.

    Parameters
    ----------
    risk : RiskFunctional
        Callable that scores a branch ``(D,)`` → float, where lower is better.

    Notes
    -----
    The chosen representative is ``futures[argmin(scores)]``. Diagnostics
    include the policy name, number of branches, the chosen index, and the
    minimal score value.
    """

    def __init__(self, risk: RiskFunctional):
        self._risk = risk

    def select(self, futures: Array) -> tuple[Array, Mapping[str, Any]]:
        """
        Select a representative by minimizing the risk score across branches.

        Parameters
        ----------
        futures : Array
            Matrix of candidate futures with shape ``(K, D)``.

        Returns
        -------
        tuple
            ``(rep, diag)`` where:
            - ``rep`` is the selected branch with shape ``(D,)``.
            - ``diag`` contains
            - ``{"policy": "min-risk", "branches": K, "chosen_index": i, "min_score": s}``.

        Raises
        ------
        ValueError
            If ``futures`` is not a 2-D array.
        """
        fut = np.asarray(futures, dtype=float)
        if fut.ndim != 2:
            raise ValueError("futures must have shape (K, D)")
        scores = np.asarray([self._risk(b) for b in fut], dtype=float)
        idx = int(np.argmin(scores))
        rep = fut[idx]
        diag = {
            "policy": SelectionPolicy.MIN_RISK.value,
            "branches": fut.shape[0],
            "chosen_index": idx,
            "min_score": float(scores[idx]),
        }
        return rep, diag
