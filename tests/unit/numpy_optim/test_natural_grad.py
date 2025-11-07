from __future__ import annotations
import numpy as np
import pytest
from qmlhc.optim.numpy_optim import natural_grad as ng


# =============================================================================
# Initialize: state shape
# =============================================================================
def test_naturalgrad_initialize_state():
    opt = ng.HCNaturalGrad(lr=0.1)
    params = {"a": np.array([1.0, -2.0])}
    state = opt.initialize(params)
    assert isinstance(state, dict)
    assert state == {"steps": 0}


# =============================================================================
# Grad path: uses grad_estimator and no-branches fallback (no preconditioning)
# =============================================================================
def test_naturalgrad_with_grad_estimator_no_branches(monkeypatch):
    def _flat(params):
        keys = list(params.keys())
        vects = [np.asarray(params[k], float).reshape(-1) for k in keys]
        layout = [(k, v.size) for k, v in zip(keys, vects)]
        return np.concatenate(vects), layout
    def _deflat(theta, layout, params):
        out, i = {}, 0
        for k, n in layout:
            out[k] = theta[i:i+n].reshape(np.asarray(params[k]).shape)
            i += n
        return out

    monkeypatch.setattr(ng, "flatten_params", _flat, raising=True)
    monkeypatch.setattr(ng, "deflatten_params", _deflat, raising=True)

    opt = ng.HCNaturalGrad(lr=0.1, grad_estimator=lambda m, p, c: np.array([0.5, -0.5, 1.0]))
    params = {"w": np.array([1.0, -2.0, 0.5])}
    state0 = opt.initialize(params)
    new_params, state1 = opt.step_params(model=None, params=params, context={"info": {}})
    assert state0 == {"steps": 0}
    assert state1["steps"] == 1
    assert state1["precond_norm"] == pytest.approx(np.linalg.norm([0.5, -0.5, 1.0]), rel=1e-8)
    assert np.allclose(new_params["w"], np.array([0.95, -1.95, 0.4]))


# =============================================================================
# Grad path: uses context['grads'] with 2D branches and Fisher preconditioning
# =============================================================================
def test_naturalgrad_with_grads_and_branches_preconditioning(monkeypatch):
    def _flat(params):
        keys = list(params.keys())
        vects = [np.asarray(params[k], float).reshape(-1) for k in keys]
        layout = [(k, v.size) for k, v in zip(keys, vects)]
        return np.concatenate(vects), layout

    def _deflat(theta, layout, params):
        out, i = {}, 0
        for k, n in layout:
            out[k] = theta[i:i+n].reshape(np.asarray(params[k]).shape)
            i += n
        return out

    def _cov_emp(B):
        D = np.asarray(B).shape[1]
        return np.eye(D)

    def _cg_solve(A_mul, b, iters=8, tol=1e-6):
        _ = A_mul(b)          
        return np.asarray(b)  

    monkeypatch.setattr(ng, "flatten_params", _flat, raising=True)
    monkeypatch.setattr(ng, "deflatten_params", _deflat, raising=True)
    monkeypatch.setattr(ng, "cov_empirical", _cov_emp, raising=True)
    monkeypatch.setattr(ng, "cg_solve", _cg_solve, raising=True)

    opt = ng.HCNaturalGrad(lr=0.2, fisher_damp=1e-3, cg_iters=4, seed=123)
    params = {"a": np.array([1.0, -1.0]), "b": np.array([0.5])}
    grads = {"a": np.array([0.1, -0.2]), "b": np.array([0.3])}
    B = np.random.RandomState(0).randn(5, 4)

    state0 = opt.initialize(params)
    new_params, state1 = opt.step_params(model=None, params=params, context={"grads": grads, "info": {"branches": B}})
    assert state0 == {"steps": 0}
    assert state1["steps"] == 1
    assert state1["precond_norm"] >= 0.0
    assert set(new_params.keys()) == {"a", "b"}


# =============================================================================
# Error path: missing grads and no estimator
# =============================================================================
def test_naturalgrad_raises_without_grads_and_no_estimator(monkeypatch):
    def _flat(params):
        keys = list(params.keys())
        vects = [np.asarray(params[k], float).reshape(-1) for k in keys]
        layout = [(k, v.size) for k, v in zip(keys, vects)]
        return np.concatenate(vects), layout
    def _deflat(theta, layout, params):
        return params

    monkeypatch.setattr(ng, "flatten_params", _flat, raising=True)
    monkeypatch.setattr(ng, "deflatten_params", _deflat, raising=True)

    opt = ng.HCNaturalGrad(lr=0.05, grad_estimator=None)
    params = {"w": np.array([1.0])}
    opt.initialize(params)
    with pytest.raises(ValueError):
        _ = opt.step_params(model=None, params=params, context={"info": {}})


# =============================================================================
# Error path: gradient size mismatch for a key
# =============================================================================
def test_naturalgrad_raises_on_size_mismatch(monkeypatch):
    def _flat(params):
        keys = list(params.keys())
        vects = [np.asarray(params[k], float).reshape(-1) for k in keys]
        layout = [(k, v.size) for k, v in zip(keys, vects)]
        return np.concatenate(vects), layout
    def _deflat(theta, layout, params):
        return params

    monkeypatch.setattr(ng, "flatten_params", _flat, raising=True)
    monkeypatch.setattr(ng, "deflatten_params", _deflat, raising=True)

    opt = ng.HCNaturalGrad()
    params = {"a": np.array([1.0, 2.0])}
    grads = {"a": np.array([0.1])}
    opt.initialize(params)
    with pytest.raises(ValueError) as exc:
        _ = opt.step_params(model=None, params=params, context={"grads": grads, "info": {"branches": np.ones((3, 2))}})
    assert "Gradient size mismatch for key 'a'" in str(exc.value)


# =============================================================================
# Clip: caps updated theta within [-clip, clip]
# =============================================================================
def test_naturalgrad_clip_caps_updates(monkeypatch):
    def _flat(params):
        keys = list(params.keys())
        vects = [np.asarray(params[k], float).reshape(-1) for k in keys]
        layout = [(k, v.size) for k, v in zip(keys, vects)]
        return np.concatenate(vects), layout
    def _deflat(theta, layout, params):
        out, i = {}, 0
        for k, n in layout:
            out[k] = theta[i:i+n].reshape(np.asarray(params[k]).shape)
            i += n
        return out

    monkeypatch.setattr(ng, "flatten_params", _flat, raising=True)
    monkeypatch.setattr(ng, "deflatten_params", _deflat, raising=True)

    opt = ng.HCNaturalGrad(lr=10.0, clip=0.25, grad_estimator=lambda m, p, c: np.array([1.0, -2.0]))
    params = {"w": np.array([0.1, -0.1])}
    opt.initialize(params)
    new_params, _ = opt.step_params(model=None, params=params, context={"info": {}})
    flat = np.asarray(new_params["w"], float).ravel()
    assert np.all(flat <= 0.25 + 1e-12)
    assert np.all(flat >= -0.25 - 1e-12)
