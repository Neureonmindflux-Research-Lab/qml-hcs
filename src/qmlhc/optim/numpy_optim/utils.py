# -*- coding: utf-8 -*-
"""
Utility functions for NumPy-based optimizers.

Includes:
- Parameter flatten/deflatten helpers
- Total loss wrapper (Task + 0.5*(Consistency + Coherence))
- Empirical covariance, conjugate gradient solver
- A simple symmetric KL proxy over state statistics
"""

from __future__ import annotations
from typing import Any, Dict, Iterable, Mapping, Tuple
import numpy as np


def flatten_params(params: Mapping[str, Any]) -> Tuple[np.ndarray, list[tuple[str, int]]]:
    """Flatten a dict of parameters into a 1D vector with a layout spec."""
    keys = sorted(params.keys())
    vecs = []
    layout = []
    for k in keys:
        v = np.atleast_1d(np.asarray(params[k], dtype=float))
        vecs.append(v.ravel())
        layout.append((k, v.size))
    theta = np.concatenate(vecs) if vecs else np.zeros(0, dtype=float)
    return theta, layout


def deflatten_params(vec: np.ndarray, layout: list[tuple[str, int]], like: Mapping[str, Any]) -> Dict[str, Any]:
    """Rebuild a parameter dict from a flat vector and a layout spec."""
    new_params: Dict[str, Any] = {}
    idx = 0
    for k, n in layout:
        chunk = vec[idx: idx + n]
        v_like = np.atleast_1d(np.asarray(like[k], dtype=float))
        new_params[k] = np.squeeze(chunk) if v_like.size == 1 else chunk.reshape(v_like.shape)
        idx += n
    return new_params


def total_loss_for(model: Any, theta: np.ndarray, context: Mapping[str, Any]) -> float:
    """
    Compute total loss:
        L = Task(s_t, target) + 0.5*(Consistency + Coherence)

    Expected context fields:
        - x0, drift, target
        - losses: (task_loss, cons_loss, coh_loss)
        - branches: int
    """
    x0 = np.asarray(context["x0"], dtype=float).reshape(-1)
    drift = np.asarray(context["drift"], dtype=float).reshape(-1)
    target = np.asarray(context["target"], dtype=float).reshape(-1)
    task_loss, cons_loss, coh_loss = context["losses"]
    branches = int(context["branches"])

    # broadcast scalar-like parameter vector onto x0 (one control applied to all dims)
    alpha = float(theta.reshape(-1).sum() / max(1, theta.size))
    x = alpha * x0
    s_tm1 = np.zeros_like(x)
    s_t, s_hat, info = model.forward(x + drift, s_tm1, branches)

    lt = float(task_loss(s_t, target))
    lc = float(cons_loss(s_tm1, s_t, s_hat))
    # Coherence expects branches matrix if available; fallback uses a proxy
    branches_arr = info.get("branches", None)
    if branches_arr is None:
        branches_arr = np.vstack([s_t, s_hat])
    lq = float(coh_loss(branches_arr))
    return lt + 0.5 * (lc + lq)


def cov_empirical(X: np.ndarray) -> np.ndarray:
    """Empirical covariance (unbiased) of samples X (N x D)."""
    X = np.asarray(X, dtype=float)
    Xc = X - X.mean(axis=0, keepdims=True)
    denom = max(1, X.shape[0] - 1)
    return (Xc.T @ Xc) / denom


def cg_solve(A_mul, b: np.ndarray, iters: int = 10, tol: float = 1e-6) -> np.ndarray:
    """Conjugate Gradient solver on an implicit SPD operator A_mul."""
    x = np.zeros_like(b)
    r = b.copy()
    p = r.copy()
    rs_old = r @ r
    for _ in range(iters):
        Ap = A_mul(p)
        denom = max(1e-12, p @ Ap)
        alpha = rs_old / denom
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = r @ r
        if np.sqrt(rs_new) < tol:
            break
        p = r + (rs_new / max(1e-12, rs_old)) * p
        rs_old = rs_new
    return x


def kl_proxy(old_info: Mapping[str, Any], new_info: Mapping[str, Any], eps: float = 1e-8) -> float:
    """
    Symmetric KL-like proxy using Gaussian approximations over state branches.
    Not a true KL unless states are Gaussian; intended as a safe divergence proxy.

    KL_sym ≈ 0.5 * [ tr(S1^{-1} S0 + S0^{-1} S1) + (m1-m0)^T (S^{-1}_avg) (m1-m0) - D ]
    Where S0,S1 are covariances and m0,m1 means. We avoid explicit inverses via
    CG on S_avg.

    Falls back to ||mean diff||^2 if covariance is unavailable.
    """
    B0 = old_info.get("branches", None)
    B1 = new_info.get("branches", None)
    if B0 is None or B1 is None:
        m0 = np.asarray(old_info.get("state", np.zeros(1)))
        m1 = np.asarray(new_info.get("state", np.zeros(1)))
        return float(np.sum((m1 - m0) ** 2))
    m0 = np.mean(B0, axis=0)
    m1 = np.mean(B1, axis=0)
    S0 = cov_empirical(B0) + eps * np.eye(B0.shape[1])
    S1 = cov_empirical(B1) + eps * np.eye(B1.shape[1])
    S_avg = 0.5 * (S0 + S1)

    def A_mul(v):
        return S_avg @ v + eps * v

    term_m = (m1 - m0)
    w = cg_solve(A_mul, term_m, iters=8, tol=1e-6)
    # Trace terms via Hutchinson's trick:
    D = S0.shape[0]
    h = 8  # probe vectors
    tr01 = 0.0
    tr10 = 0.0
    rng = np.random.default_rng(1234)
    for _ in range(h):
        z = rng.normal(size=D)
        tr01 += z @ (np.linalg.solve(S1, S0 @ z))
        tr10 += z @ (np.linalg.solve(S0, S1 @ z))
    tr01 /= h
    tr10 /= h
    kl_sym = 0.5 * (tr01 + tr10 + term_m @ w - D)
    return float(max(0.0, kl_sym))
