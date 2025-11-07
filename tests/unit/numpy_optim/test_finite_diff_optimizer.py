from __future__ import annotations
import numpy as np
import pytest
from qmlhc.optim.numpy_optim import finite_diff as fd


# =============================================================================
# Initialize: state reset
# =============================================================================
def test_finite_diff_initialize_state():
    opt = fd.HCFiniteDiffOptimizer(lr=0.1, eps=1e-3)
    state = opt.initialize({"w": np.array([1.0])})
    assert isinstance(state, dict) and state == {"steps": 0}


# =============================================================================
# Step: central differences on quadratic loss produce exact gradient 2*theta
# =============================================================================
def test_finite_diff_central_difference_exact_grad(monkeypatch):
    def _flat(params):
        vec = np.concatenate([np.atleast_1d(np.asarray(v, float)).ravel() for v in params.values()])
        layout = [(k, np.atleast_1d(np.asarray(params[k], float)).size) for k in params]
        return vec, layout

    def _deflat(theta, layout, params_ref):
        out, i = {}, 0
        for k, n in layout:
            out[k] = theta[i:i+n].reshape(np.asarray(params_ref[k]).shape)
            i += n
        return out

    def _total_loss_for(_m, theta, _ctx):
        return float(np.sum(theta**2))

    monkeypatch.setattr(fd, "flatten_params", _flat, raising=True)
    monkeypatch.setattr(fd, "deflatten_params", _deflat, raising=True)
    monkeypatch.setattr(fd, "total_loss_for", _total_loss_for, raising=True)

    opt = fd.HCFiniteDiffOptimizer(lr=0.25, eps=1e-4)
    params = {"w": np.array([1.0, -2.0, 0.5])}
    opt.initialize(params)
    new_params, st = opt.step_params(model=None, params=params, context={})

    theta0, _ = _flat(params)
    theta1, _ = _flat(new_params)
    expected = theta0 - 0.25 * (2.0 * theta0)
    assert np.allclose(theta1, expected, atol=1e-6)
    assert st["steps"] == 1 and st["grad_norm"] == pytest.approx(np.linalg.norm(2.0 * theta0), rel=1e-6)  # :contentReference[oaicite:0]{index=0}


# =============================================================================
# Step: clipping caps updated theta within [-clip, clip]
# =============================================================================
def test_finite_diff_clip_caps_updates(monkeypatch):
    def _flat(params):
        vec = np.concatenate([np.atleast_1d(np.asarray(v, float)).ravel() for v in params.values()])
        layout = [(k, np.atleast_1d(np.asarray(params[k], float)).size) for k in params]
        return vec, layout

    def _deflat(theta, layout, params_ref):
        out, i = {}, 0
        for k, n in layout:
            out[k] = theta[i:i+n].reshape(np.asarray(params_ref[k]).shape)
            i += n
        return out

    def _total_loss_for(_m, theta, _ctx):
        return float(np.sum(theta**2))

    monkeypatch.setattr(fd, "flatten_params", _flat, raising=True)
    monkeypatch.setattr(fd, "deflatten_params", _deflat, raising=True)
    monkeypatch.setattr(fd, "total_loss_for", _total_loss_for, raising=True)

    opt = fd.HCFiniteDiffOptimizer(lr=10.0, eps=1e-3, clip=0.25)
    params = {"w": np.array([0.8, -0.7])}
    opt.initialize(params)
    new_params, _ = opt.step_params(model=None, params=params, context={})
    flat = np.asarray(new_params["w"], float).ravel()
    assert np.all(flat <= 0.25 + 1e-12) and np.all(flat >= -0.25 - 1e-12)  # :contentReference[oaicite:1]{index=1}


# =============================================================================
# Step: mixed scalar/array params preserve shapes and decrease quadratic loss
# =============================================================================
def test_finite_diff_shapes_preserved_and_loss_decreases(monkeypatch):
    def _flat(params):
        vec = np.concatenate([
            np.atleast_1d(np.asarray(params["a"], float)).ravel(),
            np.atleast_1d(np.asarray(params["b"], float)).ravel(),
            np.atleast_1d(np.asarray(params["c"], float)).ravel(),
        ])
        layout = [
            ("a", np.atleast_1d(np.asarray(params["a"], float)).size),
            ("b", np.atleast_1d(np.asarray(params["b"], float)).size),
            ("c", np.atleast_1d(np.asarray(params["c"], float)).size),
        ]
        return vec, layout

    def _deflat(theta, layout, params_ref):
        out, i = {}, 0
        for k, n in layout:
            out[k] = theta[i:i+n].reshape(np.asarray(params_ref[k]).shape)
            i += n
        return out

    def _total_loss_for(_m, theta, _ctx):
        return float(np.sum(theta**2))

    def _loss_params(p):
        return float(np.sum(np.concatenate([
            np.atleast_1d(np.asarray(p["a"], float)).ravel(),
            np.atleast_1d(np.asarray(p["b"], float)).ravel(),
            np.atleast_1d(np.asarray(p["c"], float)).ravel(),
        ]) ** 2))

    monkeypatch.setattr(fd, "flatten_params", _flat, raising=True)
    monkeypatch.setattr(fd, "deflatten_params", _deflat, raising=True)
    monkeypatch.setattr(fd, "total_loss_for", _total_loss_for, raising=True)

    opt = fd.HCFiniteDiffOptimizer(lr=0.1, eps=1e-4)
    params = {"a": 0.6, "b": np.array([-0.4, 0.2]), "c": np.array([[0.1, -0.3]])}
    opt.initialize(params)
    before = _loss_params(params)
    new_params, st = opt.step_params(model=None, params=params, context={})
    after = _loss_params(new_params)

    assert after < before
    assert np.asarray(new_params["a"]).shape == ()
    assert np.asarray(new_params["b"]).shape == (2,)
    assert np.asarray(new_params["c"]).shape == (1, 2)
    assert st["steps"] == 1 and st["grad_norm"] >= 0.0  # :contentReference[oaicite:2]{index=2}
