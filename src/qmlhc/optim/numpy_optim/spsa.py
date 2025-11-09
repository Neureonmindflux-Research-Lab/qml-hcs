# -*- coding: utf-8 -*-
"""
SPSA Optimizer (Antithetic + Adaptive)
--------------------------------------
Simultaneous Perturbation Stochastic Approximation with antithetic sampling
and simple power-law decays for learning rate and perturbation magnitude.
Cost: ~2 evaluations per step, independent of parameter dimension.

Interface:
    - initialize(params) -> state
    - step_params(model, params, context) -> (new_params, state)
"""

from __future__ import annotations
from typing import Any, Dict, Mapping, Tuple
import numpy as np
from .utils import flatten_params, deflatten_params, total_loss_for


class HCSPSAOptimizer:
    """Robust optimizer for noisy, low-shot regimes."""

    def __init__(
        self,
        lr0: float = 5e-2,
        eps0: float = 1e-1,
        decay_lr: float = 0.101,
        decay_eps: float = 0.102,
        antithetic: bool = True,
        clip: float | None = None,
        seed: int = 12345,
    ):
        self.lr0 = float(lr0)
        self.eps0 = float(eps0)
        self.decay_lr = float(decay_lr)
        self.decay_eps = float(decay_eps)
        self.antithetic = bool(antithetic)
        self.clip = clip
        self._k = 0
        self._state: Dict[str, Any] = {}
        self._rng = np.random.default_rng(seed)

    def initialize(self, params: Mapping[str, Any]) -> Dict[str, Any]:
        self._k = 0
        self._state = {"steps": 0}
        return dict(self._state)

    def step_params(
        self, model: Any, params: Mapping[str, Any], context: Mapping[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        self._k += 1
        theta, layout = flatten_params(params)

        lr = self.lr0 / (self._k ** self.decay_lr)
        eps = self.eps0 / (self._k ** self.decay_eps)

        # Rademacher perturbation (+1/-1)
        delta = self._rng.choice([-1.0, 1.0], size=theta.size)

        lp = total_loss_for(model, theta + eps * delta, context)
        lm = total_loss_for(model, theta - eps * delta, context)
        g_hat = (lp - lm) / (2.0 * eps) * delta  # SPSA gradient estimate

        theta_new = theta - lr * g_hat
        if self.clip is not None:
            theta_new = np.clip(theta_new, -self.clip, self.clip)

        new_params = deflatten_params(theta_new, layout, params)
        self._state = {
            "steps": self._state.get("steps", 0) + 1,
            "lr": lr,
            "eps": eps,
            "lp": float(lp),
            "lm": float(lm),
            "grad_norm": float(np.linalg.norm(g_hat)),
        }
        return new_params, dict(self._state)
