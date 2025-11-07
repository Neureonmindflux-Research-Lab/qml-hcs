from __future__ import annotations
import numpy as np
import pytest

from qmlhc.optim.numpy_optim import kfac as kfac


# =============================================================================
# Test: HCKFACLike initialization and grad_estimator path (no branches)
# =============================================================================
def test_kfac_init_and_step_with_grad_estimator_no_branches(monkeypatch):
    def _flat(params):
        keys = list(params.keys())
        vects = [np.asarray(params[k], dtype=float).reshape(-1) for k in keys]
        layout = [(k, v.size) for k, v in zip(keys, vects)]
        return np.concatenate(vects), layout

    def _deflat(theta, layout, params):
        out, i = {}, 0
        for k, n in layout:
            shape = np.asarray(params[k]).shape
            out[k] = theta[i:i+n].reshape(shape)
            i += n
        return out

    def _cov_emp(B):
        X = np.asarray(B, dtype=float)
        Xc = X - X.mean(axis=0, keepdims=True)
        n = X.shape[0]
        return (Xc.T @ Xc) / max(n - 1, 1)

    monkeypatch.setattr(kfac, "flatten_params", _flat, raising=True)
    monkeypatch.setattr(kfac, "deflatten_params", _deflat, raising=True)
    monkeypatch.setattr(kfac, "cov_empirical", _cov_emp, raising=True)

    opt = kfac.HCKFACLike(lr=0.1, damp=1e-3, blocks=4, grad_estimator=lambda m, p, c: np.array([0.5, -0.5]))
    params = {"w": np.array([1.0, -2.0])}
    state0 = opt.initialize(params)
    assert state0 == {"steps": 0}

    new_params, state1 = opt.step_params(model=None, params=params, context={})
    assert np.allclose(new_params["w"], np.array([0.95, -1.95]))
    assert state1["steps"] == 1
    assert state1["precond_norm"] == pytest.approx(np.linalg.norm([0.5, -0.5]), rel=1e-8)


# =============================================================================
# Test: HCKFACLike grads-dict path with branches, block solve, and clip
# =============================================================================
def test_kfac_with_grads_and_branches_blocks_and_clip(monkeypatch):
    def _flat(params):
        keys = list(params.keys())
        vects = [np.asarray(params[k], dtype=float).reshape(-1) for k in keys]
        layout = [(k, v.size) for k, v in zip(keys, vects)]
        return np.concatenate(vects), layout

    def _deflat(theta, layout, params):
        out, i = {}, 0
        for k, n in layout:
            shape = np.asarray(params[k]).shape
            out[k] = theta[i:i+n].reshape(shape)
            i += n
        return out

    def _cov_emp(B):
        X = np.asarray(B, dtype=float)
        Xc = X - X.mean(axis=0, keepdims=True)
        n = X.shape[0]
        return (Xc.T @ Xc) / max(n - 1, 1)

    monkeypatch.setattr(kfac, "flatten_params", _flat, raising=True)
    monkeypatch.setattr(kfac, "deflatten_params", _deflat, raising=True)
    monkeypatch.setattr(kfac, "cov_empirical", _cov_emp, raising=True)

    opt = kfac.HCKFACLike(lr=0.2, damp=1e-2, blocks=2, grad_estimator=None, clip=0.5, seed=7)

    params = {"a": np.array([1.2, -0.9]), "b": np.array([0.8])}
    grads = {"a": np.array([0.1, -0.2]), "b": np.array([0.3])}

    B = np.array([
        [0.2, 1.0, -0.3],
        [0.0, 0.5,  0.7],
        [1.1, 0.2, -0.1],
        [0.9, 1.2,  0.4],
    ])

    new_params, state = opt.step_params(model=None, params=params, context={"grads": grads, "info": {"branches": B}})
    flat_new, _ = _flat(new_params)
    assert np.all(np.abs(flat_new) <= 0.5 + 1e-12)
    assert state["steps"] >= 1
    assert state["precond_norm"] >= 0.0


# =============================================================================
# Test: HCKFACLike error on missing grads
# =============================================================================
def test_kfac_raises_when_no_grads_and_no_estimator(monkeypatch):
    def _flat(params):
        keys = list(params.keys())
        vects = [np.asarray(params[k], dtype=float).reshape(-1) for k in keys]
        layout = [(k, v.size) for k, v in zip(keys, vects)]
        return np.concatenate(vects), layout

    def _deflat(theta, layout, params):
        out, i = {}, 0
        for k, n in layout:
            shape = np.asarray(params[k]).shape
            out[k] = theta[i:i+n].reshape(shape)
            i += n
        return out

    monkeypatch.setattr(kfac, "flatten_params", _flat, raising=True)
    monkeypatch.setattr(kfac, "deflatten_params", _deflat, raising=True)

    opt = kfac.HCKFACLike(grad_estimator=None)
    params = {"w": np.array([1.0, 2.0])}

    with pytest.raises(ValueError):
        _ = opt.step_params(model=None, params=params, context={})


# =============================================================================
# Test: HCKFACLike error on gradient size mismatch
# =============================================================================
def test_kfac_raises_on_gradient_size_mismatch(monkeypatch):
    def _flat(params):
        keys = list(params.keys())
        vects = [np.asarray(params[k], dtype=float).reshape(-1) for k in keys]
        layout = [(k, v.size) for k, v in zip(keys, vects)]
        return np.concatenate(vects), layout

    def _deflat(theta, layout, params):
        out, i = {}, 0
        for k, n in layout:
            shape = np.asarray(params[k]).shape
            out[k] = theta[i:i+n].reshape(shape)
            i += n
        return out

    monkeypatch.setattr(kfac, "flatten_params", _flat, raising=True)
    monkeypatch.setattr(kfac, "deflatten_params", _deflat, raising=True)

    opt = kfac.HCKFACLike(grad_estimator=None)
    params = {"a": np.array([1.0, 2.0])}
    grads = {"a": np.array([0.1])}

    with pytest.raises(ValueError) as exc:
        _ = opt.step_params(model=None, params=params, context={"grads": grads})
    assert "Gradient size mismatch for key 'a'" in str(exc.value)
