#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optimizer API
-------------
Lightweight adapter layer for external optimizers (e.g., Torch, JAX, or NumPy).

This module defines a minimal backend-agnostic interface for optimizers,
supporting step and initialization hooks for flexible integration into
training loops.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Mapping, Tuple

import numpy as np


class OptimizerAPI:
    """
    Minimal interface wrapper for optimizer backends.

    Parameters
    ----------
    step_fn : Callable[[Mapping[str, Any], Mapping[str, Any]], Mapping[str, Any]]
        Function that performs one parameter update given ``params`` and ``grads``.
    init_fn : Callable[[Mapping[str, Any]], Mapping[str, Any]] or None, optional
        Optional function that initializes optimizer state. If ``None``,
        defaults to an empty dictionary.
    """

    def __init__(
        self,
        step_fn: Callable[[Mapping[str, Any], Mapping[str, Any]], Mapping[str, Any]],
        init_fn: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None = None,
    ) -> None:
        # step_fn: performs parameter update given params and grads.
        # init_fn: optional hook for initializing optimizer state.
        self._step = step_fn
        self._init = init_fn or (lambda p: {})

    # ----------------------------------------------------------------------
    def initialize(self, params: Mapping[str, Any]) -> Mapping[str, Any]:
        """
        Initialize the optimizer state.

        Parameters
        ----------
        params : Mapping[str, Any]
            Model parameters to be used for optimizer state initialization.

        Returns
        -------
        Mapping[str, Any]
            Initialized optimizer state (can be empty).
        """
        return self._init(params)

    def step(
        self,
        params: Mapping[str, Any],
        grads: Mapping[str, Any],
        state: Mapping[str, Any],
    ) -> Tuple[Mapping[str, Any], Mapping[str, Any]]:
        """
        Apply one optimization step.

        Parameters
        ----------
        params : Mapping[str, Any]
            Current model parameters.
        grads : Mapping[str, Any]
            Corresponding gradients for each parameter.
        state : Mapping[str, Any]
            Optimizer state (if used).

        Returns
        -------
        tuple
            ``(new_params, new_state)``, where:
            - ``new_params`` : Mapping[str, Any]
            Updated parameter dictionary.
            - ``new_state`` : Mapping[str, Any]
            Updated optimizer state.
        """
        new_params = self._step(params, grads)
        new_state = dict(state)
        return new_params, new_state


# ----------------------------------------------------------------------
# Built-in fallback: simple gradient descent
# ----------------------------------------------------------------------
def _gd_step(
    params: Mapping[str, Any],
    grads: Mapping[str, Any],
    lr: float = 1e-2,
) -> Dict[str, Any]:
    """
    Pure NumPy gradient descent update rule.

    Parameters
    ----------
    params : Mapping[str, Any]
        Parameter dictionary to be updated.
    grads : Mapping[str, Any]
        Gradient dictionary corresponding to ``params``.
    lr : float, optional
        Learning rate, by default ``1e-2``.

    Returns
    -------
    dict
        Updated parameter dictionary after gradient descent step.
    """
    new_params = {
        k: np.asarray(v, dtype=float) - lr * np.asarray(grads[k], dtype=float)
        for k, v in params.items()
    }
    return new_params


def make_gradient_descent(lr: float = 1e-2) -> OptimizerAPI:
    """
    Factory for a standalone NumPy-based gradient descent optimizer.

    Parameters
    ----------
    lr : float, optional
        Learning rate, by default ``1e-2``.

    Returns
    -------
    OptimizerAPI
        Configured gradient descent optimizer instance.
    """

    def step_fn(params: Mapping[str, Any], grads: Mapping[str, Any]) -> Mapping[str, Any]:
        return _gd_step(params, grads, lr=lr)

    return OptimizerAPI(step_fn=step_fn)
