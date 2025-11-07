from __future__ import annotations
import numpy as np
import pytest

import qmlhc.optim as O
from qmlhc.optim import registry_numpy


# =============================================================================
# Public API re-exports
# =============================================================================
def test_optim_public_reexports_exist():
    assert hasattr(O, "OptimizerAPI")
    assert hasattr(O, "make_gradient_descent")


# =============================================================================
# OptimizerAPI: initialize + step contract
# =============================================================================
def test_optimizerapi_initialize_and_step_contract():
    def init_fn(params):
        return {"steps": 0}

    def step_fn(params, grads):
        out = {}
        for k, v in params.items():
            out[k] = np.asarray(v, dtype=float) - 0.1 * np.asarray(grads[k], dtype=float)
        return out

    opt = O.OptimizerAPI(step_fn=step_fn, init_fn=init_fn)
    params = {"w": np.array([1.0, -2.0])}
    grads = {"w": np.array([0.5, -0.5])}

    params_before = {k: v.copy() for k, v in params.items()}
    grads_before = {k: v.copy() for k, v in grads.items()}

    state = opt.initialize(params)
    assert isinstance(state, dict)
    assert state == {"steps": 0}

    new_params, new_state = opt.step(params, grads, state)
    assert new_state == {"steps": 0}
    assert np.allclose(new_params["w"], np.array([0.95, -1.95]))

    newer_params, newer_state = opt.step(new_params, grads, new_state)
    assert newer_state == {"steps": 0}
    assert np.allclose(newer_params["w"], np.array([0.90, -1.90]))

    assert np.allclose(params["w"], params_before["w"])
    assert np.allclose(grads["w"], grads_before["w"])


# =============================================================================
# Built-in gradient descent (NumPy)
# =============================================================================
def test_make_gradient_descent_updates_twice_and_is_dtype_safe():
    opt = O.make_gradient_descent(lr=0.05)
    params = {"b": [0.0, 1.0, 2.0]}
    grads = {"b": [1.0, -1.0, 0.5]}

    state = opt.initialize(params)
    updated1, state1 = opt.step(params, grads, state)
    assert isinstance(updated1["b"], np.ndarray)
    assert updated1["b"].dtype == float
    assert np.allclose(updated1["b"], np.array([-0.05, 1.05, 1.975]))

    updated2, _ = opt.step(updated1, grads, state1)
    assert np.allclose(updated2["b"], np.array([-0.10, 1.10, 1.95]))


def test_make_gradient_descent_raises_on_missing_grad_key():
    opt = O.make_gradient_descent(lr=0.1)
    params = {"w": np.array([1.0])}
    grads = {}
    with pytest.raises(KeyError) as exc:
        _ = opt.step(params, grads, state={})
    assert "w" in str(exc.value)


# =============================================================================
# NumPy optimizer registry coverage
# =============================================================================
def test_create_optimizer_numpy_full(monkeypatch):
    fake_creators = {k: (lambda _k=k, **kw: f"created:{_k}") for k in registry_numpy._CREATORS}
    monkeypatch.setattr(registry_numpy, "_CREATORS", fake_creators, raising=True)

    for name in fake_creators:
        result = registry_numpy.create_optimizer_numpy(name, lr0=0.01)
        assert result == f"created:{name}"

    assert registry_numpy.create_optimizer_numpy("  SPSA  ", lr0=0.01) == "created:spsa"

    with pytest.raises(KeyError) as exc:
        registry_numpy.create_optimizer_numpy("nonexistent", lr=0.1)
    msg = str(exc.value)
    assert "Unknown optimizer 'nonexistent'" in msg
    assert any(k in msg for k in fake_creators.keys())
