# -*- coding: utf-8 -*-
"""
HyperCausal Natural Gradient (State-Space)
------------------------------------------
Precondition estimated gradients using an empirical Fisher-like metric derived
from the covariance of state branches. Operates in state geometry and maps
back to parameter space via random projection (simple, effective proxy).

Interface:
    - initialize(params) -> state
    - step_params(model, params, context) -> (new_params, state)

Context:
    - context["info"]["branches"]: (K x D) states from model.forward(...)
    - gradient estimated via context['grads'] or a grad_estimator
"""

from __future__ import annotations
from typing import Any, Callable, Dict, Mapping, Tuple
import numpy as np
from .utils import flatten_params, deflatten_params, cov_empirical, cg_solve


class HCNaturalGrad:
    """Natural-gradient preconditioning using state covariance."""

    def __init__(
        self,
        lr: float = 5e-3,
        fisher_damp: float = 1e-3,
        cg_iters: int = 8,
        clip: float | None = None,
        grad_estimator: Callable[[Any, Mapping[str, Any], Mapping[str, Any]], np.ndarray] | None = None,
        seed: int = 12345,
    ):
        self.lr = float(lr)
        self.fisher_damp = float(fisher_damp)
        self.cg_iters = int(cg_iters)
        self.clip = clip
        self.grad_estimator = grad_estimator
        self._rng = np.random.default_rng(seed)
        self._state: Dict[str, Any] = {}

    def initialize(self, params: Mapping[str, Any]) -> Dict[str, Any]:
        self._state = {"steps": 0}
        return dict(self._state)

    def step_params(
        self, model: Any, params: Mapping[str, Any], context: Mapping[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        theta, layout = flatten_params(params)

        # get gradient vector g
        if self.grad_estimator is not None:
            g = np.asarray(self.grad_estimator(model, params, context), dtype=float).reshape(theta.shape)
        else:
            grads_dict = context.get("grads", {})
            if not grads_dict:
                raise ValueError("HCNaturalGrad expects either grad_estimator or context['grads']")
            g_chunks = []
            for k, n in layout:
                v = np.atleast_1d(np.asarray(grads_dict[k], dtype=float)).reshape(-1)
                if v.size != n:
                    raise ValueError(f"Gradient size mismatch for key '{k}'")
                g_chunks.append(v)
            g = np.concatenate(g_chunks)

        # state covariance metric
        info = context.get("info", {})
        B = info.get("branches", None)
        if B is None or np.asarray(B).ndim != 2:
            # fallback: no preconditioning
            precond = g
        else:
            B = np.asarray(B, dtype=float)
            C = cov_empirical(B)  # D x D
            D = C.shape[0]
            # random projection param->state
            P = self._rng.normal(size=(theta.size, D)) / np.sqrt(D)
            g_state = (P.T @ g)

            def A_mul(v: np.ndarray) -> np.ndarray:
                return C @ v + self.fisher_damp * v

            v_star = cg_solve(A_mul, g_state, iters=self.cg_iters, tol=1e-6)  # F^{-1} g_state
            precond = P @ v_star  # back to param space

        theta_new = theta - self.lr * precond
        if self.clip is not None:
            theta_new = np.clip(theta_new, -self.clip, self.clip)

        new_params = deflatten_params(theta_new, layout, params)
        self._state = {"steps": self._state.get("steps", 0) + 1, "precond_norm": float(np.linalg.norm(precond))}
        return new_params, dict(self._state)
