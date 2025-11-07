# -*- coding: utf-8 -*-
"""
Short-Horizon MPC Optimizer
---------------------------
Model-Predictive Control (MPC) over a short horizon. Optimizes control-like
parameters (e.g., alpha) by rolling out a few steps ahead, minimizing the
cumulative cost with a small action penalty. Uses simple projected gradient
descent over the horizon (can be swapped for a QP solver later).

Interface:
    - initialize(params) -> state
    - step_params(model, params, context) -> (new_params, state)

Context:
    - context["rollout_fn"](model, params, horizon, context) -> (traj_info, cum_loss)
      where cum_loss already aggregates Task + Cons + Coh + action_penalty.
    - "horizon": int, number of predictive steps (default 3)
    - Optional: "project_fn"(params) to project back to feasible set.
"""

from __future__ import annotations
from typing import Any, Dict, Mapping, Tuple
import numpy as np


class HCMPCShortHorizon:
    """Short-horizon MPC with simple gradient descent over cumulative loss."""

    def __init__(self, lr: float = 1e-2, horizon: int = 3, clip: float | None = None):
        self.lr = float(lr)
        self.horizon_default = int(horizon)
        self.clip = clip
        self._state: Dict[str, Any] = {}

    def initialize(self, params: Mapping[str, Any]) -> Dict[str, Any]:
        self._state = {"steps": 0}
        return dict(self._state)

    def step_params(
        self, model: Any, params: Mapping[str, Any], context: Mapping[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        horizon = int(context.get("horizon", self.horizon_default))
        rollout_fn = context["rollout_fn"]
        project_fn = context.get("project_fn", None)

        # finite-difference on params w.r.t cumulative loss over the horizon
        keys = sorted(params.keys())
        theta = np.concatenate([np.atleast_1d(np.asarray(params[k], dtype=float)) for k in keys])
        eps = 1e-3
        grad = np.zeros_like(theta)

        def loss_at(vec):
            p = {}
            idx = 0
            for k in keys:
                v_like = np.atleast_1d(np.asarray(params[k], dtype=float))
                n = v_like.size
                p[k] = np.squeeze(vec[idx: idx + n]) if n == 1 else vec[idx: idx + n].reshape(v_like.shape)
                idx += n
            _, cum_loss = rollout_fn(model, p, horizon, context)
            return float(cum_loss)

        for i in range(theta.size):
            e = np.zeros_like(theta); e[i] = eps
            lp = loss_at(theta + e)
            lm = loss_at(theta - e)
            grad[i] = (lp - lm) / (2.0 * eps)

        theta_new = theta - self.lr * grad
        if self.clip is not None:
            theta_new = np.clip(theta_new, -self.clip, self.clip)

        # rebuild params
        new_params: Dict[str, Any] = {}
        idx = 0
        for k in keys:
            v_like = np.atleast_1d(np.asarray(params[k], dtype=float))
            n = v_like.size
            new_params[k] = np.squeeze(theta_new[idx: idx + n]) if n == 1 else theta_new[idx: idx + n].reshape(v_like.shape)
            idx += n

        # optional projection
        if project_fn is not None:
            new_params = project_fn(new_params)

        self._state = {
            "steps": self._state.get("steps", 0) + 1,
            "horizon": horizon,
            "grad_norm": float(np.linalg.norm(grad)),
        }
        return new_params, dict(self._state)
