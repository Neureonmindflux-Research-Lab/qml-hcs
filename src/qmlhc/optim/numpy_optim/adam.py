# -*- coding: utf-8 -*-
"""
Adam on Estimated Gradients
---------------------------
Momentum-based optimizer (Adam) applied to *estimated* gradients produced by
an external estimator (e.g., FD or SPSA) or provided directly in context.

Interface:
    - initialize(params) -> state
    - step_params(model, params, context) -> (new_params, state)

Context options:
    - context["grads"]: dict aligned with `params` (if provided, used directly)
    - or pass a `grad_estimator(model, params, context) -> np.ndarray` in ctor
"""

from __future__ import annotations
from typing import Any, Callable, Dict, Mapping, Tuple
import numpy as np
from .utils import flatten_params, deflatten_params


class HCAdam:
    """Adam optimizer over externally estimated gradients."""

    def __init__(
        self,
        lr: float = 1e-2,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        clip: float | None = None,
        grad_estimator: Callable[[Any, Mapping[str, Any], Mapping[str, Any]], np.ndarray] | None = None,
    ):
        self.lr = float(lr)
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.eps = float(eps)
        self.clip = clip
        self.grad_estimator = grad_estimator
        self._t = 0
        self._m: np.ndarray | None = None
        self._v: np.ndarray | None = None
        self._state: Dict[str, Any] = {}

    def initialize(self, params: Mapping[str, Any]) -> Dict[str, Any]:
        theta, _ = flatten_params(params)
        self._t = 0
        self._m = np.zeros_like(theta)
        self._v = np.zeros_like(theta)
        self._state = {"steps": 0}
        return dict(self._state)

    def step_params(
        self, model: Any, params: Mapping[str, Any], context: Mapping[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        keys_theta, layout = flatten_params(params)

        # get gradient
        if self.grad_estimator is not None:
            g = np.asarray(self.grad_estimator(model, params, context), dtype=float).reshape(keys_theta.shape)
        else:
            grads_dict = context.get("grads", {})
            if not grads_dict:
                raise ValueError("HCAdam expects either grad_estimator or context['grads']")
            # flatten grads in the same key order
            g_chunks = []
            for k, n in layout:
                v = np.atleast_1d(np.asarray(grads_dict[k], dtype=float)).reshape(-1)
                if v.size != n:
                    raise ValueError(f"Gradient size mismatch for key '{k}'")
                g_chunks.append(v)
            g = np.concatenate(g_chunks)

        # adam moments
        self._t += 1
        self._m = self.beta1 * self._m + (1.0 - self.beta1) * g
        self._v = self.beta2 * self._v + (1.0 - self.beta2) * (g * g)
        m_hat = self._m / (1.0 - self.beta1 ** self._t)
        v_hat = self._v / (1.0 - self.beta2 ** self._t)

        theta_new = keys_theta - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
        if self.clip is not None:
            theta_new = np.clip(theta_new, -self.clip, self.clip)

        new_params = deflatten_params(theta_new, layout, params)
        self._state = {"steps": self._state.get("steps", 0) + 1, "t": self._t}
        return new_params, dict(self._state)
