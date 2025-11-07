from __future__ import annotations
import numpy as np
import pytest
from qmlhc.optim.numpy_optim import adam as adam


# =============================================================================
# Initialize: returns base state and shapes are set through flatten
# =============================================================================
def test_adam_initialize_state(monkeypatch):
    def _flat(params):
        vec = np.concatenate([np.atleast_1d(np.asarray(v, float)).ravel() for v in params.values()])
        layout = [(k, np.atleast_1d(np.asarray(params[k], float)).size) for k in params]
        return vec, layout

    monkeypatch.setattr(adam, "flatten_params", _flat, raising=True)

    opt = adam.HCAdam(lr=0.1)
    state = opt.initialize({"w": np.array([1.0, -2.0])})
    assert state == {"steps": 0}


# =============================================================================
# Step: uses grad_estimator path and performs a correct Adam update
# =============================================================================
def test_adam_step_with_grad_estimator(monkeypatch):
    def _flat(params):
        vec = np.concatenate([np.atleast_1d(np.asarray(v, float)).ravel() for v in params.values()])
        layout = [(k, np.atleast_1d(np.asarray(params[k], float)).size) for k in params]
        return vec, layout

    def _deflat(theta, layout, params):
        out, i = {}, 0
        for k, n in layout:
            out[k] = theta[i:i+n].reshape(np.asarray(params[k]).shape)
            i += n
        return out

    monkeypatch.setattr(adam, "flatten_params", _flat, raising=True)
    monkeypatch.setattr(adam, "deflatten_params", _deflat, raising=True)

    g = np.array([0.5, -1.5, 2.0])
    opt = adam.HCAdam(lr=0.1, beta1=0.9, beta2=0.999, eps=1e-8, grad_estimator=lambda m, p, c: g)
    params = {"w": np.array([1.0, -2.0, 0.0])}

    opt.initialize(params)
    new_params, st = opt.step_params(model=None, params=params, context={})

    expected = np.asarray(params["w"]) - 0.1 * np.sign(g)
    assert np.allclose(new_params["w"], expected, atol=1e-7)
    assert st["steps"] == 1 and st["t"] == 1


# =============================================================================
# Step: uses context['grads'] path and respects key/layout ordering
# =============================================================================
def test_adam_step_with_context_grads(monkeypatch):
    def _flat(params):
        # enforce order a,b for layout
        vec = np.concatenate([np.atleast_1d(np.asarray(params["a"], float)).ravel(),
                              np.atleast_1d(np.asarray(params["b"], float)).ravel()])
        layout = [("a", np.atleast_1d(np.asarray(params["a"], float)).size),
                  ("b", np.atleast_1d(np.asarray(params["b"], float)).size)]
        return vec, layout

    def _deflat(theta, layout, params):
        out, i = {}, 0
        for k, n in layout:
            out[k] = theta[i:i+n].reshape(np.asarray(params[k]).shape)
            i += n
        return out

    monkeypatch.setattr(adam, "flatten_params", _flat, raising=True)
    monkeypatch.setattr(adam, "deflatten_params", _deflat, raising=True)

    opt = adam.HCAdam(lr=0.05)
    params = {"a": np.array([1.0, -1.0]), "b": 0.5}
    grads = {"a": np.array([0.2, -0.4]), "b": 0.1}

    opt.initialize(params)
    new_params, st = opt.step_params(model=None, params=params, context={"grads": grads})

    assert st["steps"] == 1 and st["t"] == 1
    assert isinstance(new_params, dict) and set(new_params.keys()) == {"a", "b"}


# =============================================================================
# Error: missing grads and no estimator
# =============================================================================
def test_adam_raises_without_grads_and_no_estimator(monkeypatch):
    def _flat(params):
        vec = np.concatenate([np.atleast_1d(np.asarray(v, float)).ravel() for v in params.values()])
        layout = [(k, np.atleast_1d(np.asarray(params[k], float)).size) for k in params]
        return vec, layout

    monkeypatch.setattr(adam, "flatten_params", _flat, raising=True)

    opt = adam.HCAdam(lr=0.01, grad_estimator=None)
    params = {"w": np.array([1.0])}
    opt.initialize(params)
    with pytest.raises(ValueError):
        _ = opt.step_params(model=None, params=params, context={})


# =============================================================================
# Error: gradient size mismatch for a key
# =============================================================================
def test_adam_raises_on_size_mismatch(monkeypatch):
    def _flat(params):
        vec = np.concatenate([np.atleast_1d(np.asarray(v, float)).ravel() for v in params.values()])
        layout = [(k, np.atleast_1d(np.asarray(params[k], float)).size) for k in params]
        return vec, layout

    monkeypatch.setattr(adam, "flatten_params", _flat, raising=True)

    opt = adam.HCAdam()
    params = {"a": np.array([1.0, 2.0])}
    grads = {"a": np.array([0.5])}

    opt.initialize(params)
    with pytest.raises(ValueError) as exc:
        _ = opt.step_params(model=None, params=params, context={"grads": grads})
    assert "Gradient size mismatch for key 'a'" in str(exc.value)


# =============================================================================
# Clip: caps updated theta to [-clip, clip]
# =============================================================================
def test_adam_clip_caps_updates(monkeypatch):
    def _flat(params):
        vec = np.concatenate([np.atleast_1d(np.asarray(v, float)).ravel() for v in params.values()])
        layout = [(k, np.atleast_1d(np.asarray(params[k], float)).size) for k in params]
        return vec, layout

    def _deflat(theta, layout, params):
        out, i = {}, 0
        for k, n in layout:
            out[k] = theta[i:i+n].reshape(np.asarray(params[k]).shape)
            i += n
        return out

    monkeypatch.setattr(adam, "flatten_params", _flat, raising=True)
    monkeypatch.setattr(adam, "deflatten_params", _deflat, raising=True)

    opt = adam.HCAdam(lr=10.0, clip=0.25, grad_estimator=lambda m, p, c: np.array([1.0, -2.0]))
    params = {"w": np.array([0.1, -0.1])}
    opt.initialize(params)
    new_params, _ = opt.step_params(model=None, params=params, context={})

    flat = np.asarray(new_params["w"], float).ravel()
    assert np.all(flat <= 0.25 + 1e-12)
    assert np.all(flat >= -0.25 - 1e-12)
