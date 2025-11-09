#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Deterministic Projectors
------------------------
Mappings from a compact present state ``S_t`` to a set of K candidate futures
``{S_{t+1}^{(k)}}``.

This module defines:

- ``Projector``: protocol/contract for deterministic future projection.
- ``LinearProjector``: affine projector that generates evenly spaced
  perturbations around a base prediction, followed by ``tanh`` stabilization.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from ..core.types import Array, TensorLike


class Projector(Protocol):
    """
    Contract for deterministic future projection from a present compact state.
    """

    def project(self, s_t: TensorLike, branches: int = 2) -> Array:
        """
        Produce K candidate futures from the current state.

        Parameters
        ----------
        s_t : TensorLike
            Current compact state vector ``(D,)``.
        branches : int, optional
            Number of candidate futures (K), by default ``2``.

        Returns
        -------
        Array
            Matrix of candidate futures with shape ``(K, D)``.
        """
        ...


class LinearProjector:
    """
    Affine projector with evenly spaced perturbations around the base prediction.

    The projector computes a base prediction ``base = w * x + b`` and then
    generates ``K`` values by adding linearly spaced deltas in
    ``[-span, span]``; each result is passed through ``tanh`` for numerical
    stability.

    Parameters
    ----------
    weight : float, optional
        Scalar weight ``w`` used in the affine base prediction, by default ``1.0``.
    bias : float, optional
        Scalar bias ``b`` used in the affine base prediction, by default ``0.0``.
    span : float, optional
        Half-width of the linear delta range, by default ``0.2``.
    """

    def __init__(self, weight: float = 1.0, bias: float = 0.0, span: float = 0.2):
        self._w = float(weight)
        self._b = float(bias)
        self._span = float(span)

    def project(self, s_t: TensorLike, branches: int = 2) -> Array:
        """
        Generate K candidate futures via affine transform + evenly spaced deltas.

        Parameters
        ----------
        s_t : TensorLike
            Current compact state vector ``(D,)``.
        branches : int, optional
            Number of candidate futures (K), by default ``2`` (clamped to ``>= 2``).

        Returns
        -------
        Array
            Matrix of candidate futures with shape ``(K, D)``.

        Notes
        -----
        The final futures are computed as:
        ``tanh(w * s_t + b + delta_k)`` for ``delta_k`` in evenly spaced values
        across ``[-span, span]``.
        """
        x = np.asarray(s_t, dtype=float).reshape(-1)
        k = max(2, int(branches))

        base = self._w * x + self._b
        deltas = np.linspace(-self._span, self._span, k, dtype=float)
        fut = np.stack([np.tanh(base + d) for d in deltas], axis=0)
        return fut
