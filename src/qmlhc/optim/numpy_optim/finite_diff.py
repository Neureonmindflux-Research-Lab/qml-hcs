# -*- coding: utf-8 -*-
"""
Finite-Difference Optimizer (Central Difference)
------------------------------------------------
Derivative-free gradient estimation by central differences. Suitable for
low- to medium-dimensional parameter vectors and backends without analytic
gradients. Cost: 2 evaluations per parameter per step.

Interface:
    - initialize(params) -> state
    - step_params(model, params, context) -> (new_params, state)
"""

from __future__ import annotations
from typing import Any, Dict, Mapping, Tuple
import numpy as np
from .utils import flatten_params, deflatten_params, total_loss_for


class HCFiniteDiffOptimizer:
    """Central-difference gradient descent with optional clipping."""

    def __init__(self, lr: float = 1e-2, eps: float = 1e-3, clip: float | None = None):
        self.lr = float(lr)
        self.eps = float(eps)
        self.clip = clip
        self._state: Dict[str, Any] = {}

    def initialize(self, params: Mapping[str, Any]) -> Dict[str, Any]:
        """Optionally initialize optimizer state (none needed)."""
        self._state = {"steps": 0}
        return dict(self._state)

    def step_params(
        self, model: Any, params: Mapping[str, Any], context: Mapping[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        theta, layout = flatten_params(params)
        grad = np.zeros_like(theta)

        # central finite differences
        for i in range(theta.size):
            e = np.zeros_like(theta); e[i] = self.eps
            lp = total_loss_for(model, theta + e, context)
            lm = total_loss_for(model, theta - e, context)
            grad[i] = (lp - lm) / (2.0 * self.eps)

        theta_new = theta - self.lr * grad
        if self.clip is not None:
            theta_new = np.clip(theta_new, -self.clip, self.clip)

        new_params = deflatten_params(theta_new, layout, params)
        self._state = {"steps": self._state.get("steps", 0) + 1, "grad_norm": float(np.linalg.norm(grad))}
        return new_params, dict(self._state)
