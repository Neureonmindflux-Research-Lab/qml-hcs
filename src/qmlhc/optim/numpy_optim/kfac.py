# -*- coding: utf-8 -*-
"""
K-FAC-like Preconditioner (Branch-Factored)
-------------------------------------------
Kronecker-Factored Approximate Curvature inspired preconditioning using
branch-wise covariance factors as a low-cost second-order proxy.

This is a simplified K-FAC-like optimizer tailored to hypercausal state
branches: assumes block-diagonal structure across branch groups.

Interface:
    - initialize(params) -> state
    - step_params(model, params, context) -> (new_params, state)

Context:
    - context["info"]["branches"]: (K x D) matrix of state samples
    - context['grads'] or a grad_estimator for parameter gradients
"""

from __future__ import annotations
from typing import Any, Callable, Dict, Mapping, Tuple
import numpy as np
from .utils import flatten_params, deflatten_params, cov_empirical, cg_solve


class HCKFACLike:
    """Branch-factored curvature preconditioning, simplified K-FAC variant."""

    def __init__(
        self,
        lr: float = 5e-3,
        damp: float = 1e-3,
        blocks: int = 4,
        grad_estimator: Callable[[Any, Mapping[str, Any], Mapping[str, Any]], np.ndarray] | None = None,
        clip: float | None = None,
        seed: int = 2027,
    ):
        self.lr = float(lr)
        self.damp = float(damp)
        self.blocks = int(blocks)
        self.grad_estimator = grad_estimator
        self.clip = clip
        self._rng = np.random.default_rng(seed)
        self._state: Dict[str, Any] = {}

    def initialize(self, params: Mapping[str, Any]) -> Dict[str, Any]:
        self._state = {"steps": 0}
        return dict(self._state)

    def step_params(
        self, model: Any, params: Mapping[str, Any], context: Mapping[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        theta, layout = flatten_params(params)

        # gradient
        if self.grad_estimator is not None:
            g = np.asarray(self.grad_estimator(model, params, context), dtype=float).reshape(theta.shape)
        else:
            grads_dict = context.get("grads", {})
            if not grads_dict:
                raise ValueError("HCKFACLike expects either grad_estimator or context['grads']")
            g_chunks = []
            for k, n in layout:
                v = np.atleast_1d(np.asarray(grads_dict[k], dtype=float)).reshape(-1)
                if v.size != n:
                    raise ValueError(f"Gradient size mismatch for key '{k}'")
                g_chunks.append(v)
            g = np.concatenate(g_chunks)

        # state branches covariance and factorization into blocks
        info = context.get("info", {})
        B = info.get("branches", None)
        if B is None or np.asarray(B).ndim != 2:
            precond = g
        else:
            B = np.asarray(B, dtype=float)
            D = B.shape[1]
            C = cov_empirical(B) + self.damp * np.eye(D)

            # block partition of state-dim
            blocks = max(1, min(self.blocks, D))
            sizes = np.full(blocks, D // blocks, dtype=int)
            sizes[: D % blocks] += 1
            cuts = np.cumsum(sizes)

            # random linear map param->state (thin)
            P = self._rng.normal(size=(theta.size, D)) / np.sqrt(D)
            g_state = P.T @ g

            v_star = np.zeros_like(g_state)
            start = 0
            for c in cuts:
                sl = slice(start, c)
                Cb = C[sl, sl]
                gb = g_state[sl]
                # solve (Cb) v = gb
                v_star[sl] = np.linalg.solve(Cb, gb)
                start = c

            precond = P @ v_star

        theta_new = theta - self.lr * precond
        if self.clip is not None:
            theta_new = np.clip(theta_new, -self.clip, self.clip)

        new_params = deflatten_params(theta_new, layout, params)
        self._state = {"steps": self._state.get("steps", 0) + 1, "precond_norm": float(np.linalg.norm(precond))}
        return new_params, dict(self._state)
