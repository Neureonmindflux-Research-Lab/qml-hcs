from __future__ import annotations
import numpy as np
import pytest
from qmlhc.optim.numpy_optim import utils as U


# =============================================================================
# flatten_params / deflatten_params: sorting, shapes, and round-trip
# =============================================================================
def test_flatten_deflatten_roundtrip_and_sorting():
    params = {"c": np.array([[1.0, -1.0]]), "a": 0.5, "b": np.array([2.0])}
    theta, layout = U.flatten_params(params)
    assert theta.ndim == 1 and theta.size == 1 + 1 + 2
    assert [k for k, _ in layout] == ["a", "b", "c"]
    rebuilt = U.deflatten_params(theta, layout, params)
    assert np.asarray(rebuilt["a"]).shape == ()
    assert np.asarray(rebuilt["b"]).shape == ()
    assert np.asarray(rebuilt["c"]).shape == (1, 2)
    assert np.allclose(theta, U.flatten_params(rebuilt)[0])  # :contentReference[oaicite:0]{index=0}


# =============================================================================
# flatten_params: empty input produces zero-length vector
# =============================================================================
def test_flatten_empty_params():
    theta, layout = U.flatten_params({})
    assert theta.size == 0 and layout == []  # :contentReference[oaicite:1]{index=1}


# =============================================================================
# total_loss_for: path without branches (fallback) and with branches
# =============================================================================
def test_total_loss_for_without_and_with_branches():
    class Model:
        def __init__(self, with_branches=False):
            self.with_branches = with_branches
        def forward(self, x, s_tm1, branches: int):
            s_t = x * 0.5
            s_hat = x * 0.25
            info = {}
            if self.with_branches:
                info["branches"] = np.vstack([s_t, s_hat])
            return s_t, s_hat, info

    def task_loss(s_t, target): return np.sum((s_t - target) ** 2)
    def cons_loss(s_tm1, s_t, s_hat): return np.sum((s_t - s_hat) ** 2)
    def coh_loss(B): return float(np.var(np.asarray(B)))

    ctx = {
        "x0": np.array([1.0, -2.0]),
        "drift": np.array([0.3, -0.1]),
        "target": np.array([0.0, 0.0]),
        "losses": (task_loss, cons_loss, coh_loss),
        "branches": 2,
    }

    theta = np.array([0.2, -0.4])
    no_br = U.total_loss_for(Model(False), theta, ctx)
    yes_br = U.total_loss_for(Model(True), theta, ctx)
    assert isinstance(no_br, float) and isinstance(yes_br, float)
    assert yes_br >= 0.0 and no_br >= 0.0  # :contentReference[oaicite:2]{index=2}


# =============================================================================
# cov_empirical: unbiased covariance and N=1 corner case
# =============================================================================
def test_cov_empirical_unbiased_and_single_row():
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    C = U.cov_empirical(X)
    Xc = X - X.mean(axis=0, keepdims=True)
    C_ref = (Xc.T @ Xc) / (X.shape[0] - 1)
    assert np.allclose(C, C_ref)
    C1 = U.cov_empirical(np.array([[7.0, -3.0]]))
    assert np.allclose(C1, np.zeros((2, 2)))  # :contentReference[oaicite:3]{index=3}


# =============================================================================
# cg_solve: solves simple SPD system A x = b (A = 2I)
# =============================================================================
def test_cg_solve_identity_like():
    def A_mul(v): return 2.0 * v
    b = np.array([1.0, -2.0, 3.0])
    x = U.cg_solve(A_mul, b, iters=20, tol=1e-10)
    assert np.allclose(x, b / 2.0, atol=1e-6)  # :contentReference[oaicite:4]{index=4}


# =============================================================================
# kl_proxy: fallback to mean-diff when branches are missing
# =============================================================================
def test_kl_proxy_fallback_means():
    old_info = {"state": np.array([1.0, 0.0])}
    new_info = {"state": np.array([2.0, 2.0])}
    val = U.kl_proxy(old_info, new_info)
    assert np.isclose(val, np.sum((new_info["state"] - old_info["state"]) ** 2))  # :contentReference[oaicite:5]{index=5}


# =============================================================================
# kl_proxy: with branches, positive and finite, touches CG path
# =============================================================================
def test_kl_proxy_with_branches_positive_and_finite():
    rng = np.random.default_rng(123)
    B0 = rng.normal(size=(16, 3))
    B1 = rng.normal(loc=0.1, scale=1.1, size=(16, 3))
    old_info = {"branches": B0}
    new_info = {"branches": B1}
    val = U.kl_proxy(old_info, new_info, eps=1e-6)
    assert np.isfinite(val) and val >= 0.0  # :contentReference[oaicite:6]{index=6}


# =============================================================================
# cg_solve: no-break path (tol = 0) forces full iteration loop
# =============================================================================
def test_cg_solve_no_break_runs_full_loop():
    def A_mul(v): return 2.0 * v
    b = np.array([1.0, -2.0, 3.0])
    x = U.cg_solve(A_mul, b, iters=3, tol=0.0)
    assert np.allclose(x, b / 2.0, atol=1e-12)


# =============================================================================
# cg_solve: clamps on denom and rs_old via very small operator and residual
# =============================================================================
def test_cg_solve_clamps_denom_and_rs_old():
    def A_mul(v): return 1e-20 * v  # p@Ap ~ 1e-20 * (p@p) -> triggers denom clamp
    b = np.array([1e-12, -1e-12])   # rs_old = 1e-24 -> triggers rs_old clamp
    x = U.cg_solve(A_mul, b, iters=2, tol=1e-15)  # avoid break: sqrt(rs_new) >= tol
    assert np.isfinite(x).all()
