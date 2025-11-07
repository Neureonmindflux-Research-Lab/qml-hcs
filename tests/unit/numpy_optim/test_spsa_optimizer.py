from __future__ import annotations
import numpy as np
import pytest
from qmlhc.optim.numpy_optim import spsa as spsa


# =============================================================================
# Initialize: state reset
# =============================================================================
def test_spsa_initialize_state():
    opt = spsa.HCSPSAOptimizer(lr0=0.05, eps0=0.1, seed=7)
    params = {"w": np.array([1.0, -2.0, 0.5])}
    state = opt.initialize(params)
    assert isinstance(state, dict) and state == {"steps": 0}


# =============================================================================
# Step: zero-gradient case (target == params) yields lp==lm, grad_norm==0, no change
# =============================================================================
def test_spsa_step_zero_gradient_no_change(monkeypatch):
    def _flat(params):
        keys = list(params.keys())
        vecs = [np.asarray(params[k], float).ravel() for k in keys]
        layout = [(k, v.size) for k, v in zip(keys, vecs)]
        return np.concatenate(vecs), layout

    def _deflat(theta, layout, params):
        out, i = {}, 0
        for k, n in layout:
            out[k] = theta[i:i+n].reshape(np.asarray(params[k]).shape)
            i += n
        return out

    def _total_loss_for(_model, theta, context):
        tgt = np.asarray(context["target"], float)
        return float(np.sum((theta - tgt) ** 2))

    monkeypatch.setattr(spsa, "flatten_params", _flat, raising=True)
    monkeypatch.setattr(spsa, "deflatten_params", _deflat, raising=True)
    monkeypatch.setattr(spsa, "total_loss_for", _total_loss_for, raising=True)

    params = {"w": np.array([1.0, -2.0, 0.5])}
    opt = spsa.HCSPSAOptimizer(lr0=0.05, eps0=0.1, seed=123)
    opt.initialize(params)
    new_params, st = opt.step_params(
        model=None,
        params=params,
        context={"target": np.concatenate([np.asarray(v, float).ravel() for v in params.values()])},
    )

    assert np.allclose(new_params["w"], params["w"])
    assert st["steps"] == 1
    assert pytest.approx(st["lp"], rel=0, abs=1e-12) == st["lm"]
    assert st["grad_norm"] == pytest.approx(0.0, abs=1e-12)
    assert "lr" in st and "eps" in st


# =============================================================================
# Step: clipping caps updated theta to [-clip, clip]
# =============================================================================
def test_spsa_clip_caps_updates(monkeypatch):
    def _flat(params):
        keys = list(params.keys())
        vecs = [np.asarray(params[k], float).ravel() for k in keys]
        layout = [(k, v.size) for k, v in zip(keys, vecs)]
        return np.concatenate(vecs), layout

    def _deflat(theta, layout, params):
        out, i = {}, 0
        for k, n in layout:
            out[k] = theta[i:i+n].reshape(np.asarray(params[k]).shape)
            i += n
        return out

    def _total_loss_for(_model, theta, context):
        tgt = np.asarray(context["target"], float)
        return float(np.sum((theta - tgt) ** 2))

    monkeypatch.setattr(spsa, "flatten_params", _flat, raising=True)
    monkeypatch.setattr(spsa, "deflatten_params", _deflat, raising=True)
    monkeypatch.setattr(spsa, "total_loss_for", _total_loss_for, raising=True)

    params = {"w": np.array([0.9, -0.9])}
    opt = spsa.HCSPSAOptimizer(lr0=10.0, eps0=0.05, clip=0.25, seed=1)
    opt.initialize(params)
    new_params, _ = opt.step_params(model=None, params=params, context={"target": np.zeros(2)})
    flat = np.asarray(new_params["w"], float).ravel()
    assert np.all(flat <= 0.25 + 1e-12)
    assert np.all(flat >= -0.25 - 1e-12)


# =============================================================================
# Step: lr/eps decay across steps and state bookkeeping
# =============================================================================
def test_spsa_decay_and_state_bookkeeping(monkeypatch):
    def _flat(params):
        keys = list(params.keys())
        vecs = [np.asarray(params[k], float).ravel() for k in keys]
        layout = [(k, v.size) for k, v in zip(keys, vecs)]
        return np.concatenate(vecs), layout

    def _deflat(theta, layout, params):
        out, i = {}, 0
        for k, n in layout:
            out[k] = theta[i:i+n].reshape(np.asarray(params[k]).shape)
            i += n
        return out

    def _total_loss_for(_model, theta, context):
        tgt = np.asarray(context["target"], float)
        return float(np.sum((theta - tgt) ** 2))

    monkeypatch.setattr(spsa, "flatten_params", _flat, raising=True)
    monkeypatch.setattr(spsa, "deflatten_params", _deflat, raising=True)
    monkeypatch.setattr(spsa, "total_loss_for", _total_loss_for, raising=True)

    params = {"w": np.array([0.3, -0.7, 0.2])}
    opt = spsa.HCSPSAOptimizer(lr0=0.5, eps0=0.2, decay_lr=0.2, decay_eps=0.3, seed=99)
    opt.initialize(params)

    _, st1 = opt.step_params(model=None, params=params, context={"target": np.zeros(3)})
    _, st2 = opt.step_params(model=None, params=params, context={"target": np.zeros(3)})

    assert st1["steps"] == 1 and st2["steps"] == 2
    assert st2["lr"] < st1["lr"]
    assert st2["eps"] < st1["eps"]
