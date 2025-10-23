
# Comprehensive tests for src/qmlhc/optim: __init__.py, api.py


from __future__ import annotations

import numpy as np
import pytest

import qmlhc.optim as O


# ---------- Public API re-exports --------------------------------------------

def test_optim_public_reexports_exist():
    assert hasattr(O, "OptimizerAPI")
    assert hasattr(O, "make_gradient_descent")


# ---------- OptimizerAPI: initialize + step ----------------------------------

def test_optimizerapi_initialize_and_step_contract():
    # custom init to verify initialize() is used
    def init_fn(params):
        return {"steps": 0}

    # simple step: subtract grads * 0.1
    def step_fn(params, grads):
        out = {}
        for k, v in params.items():
            out[k] = np.asarray(v, dtype=float) - 0.1 * np.asarray(grads[k], dtype=float)
        return out

    opt = O.OptimizerAPI(step_fn=step_fn, init_fn=init_fn)

    params = {"w": np.array([1.0, -2.0])}
    grads = {"w": np.array([0.5, -0.5])}

    state = opt.initialize(params)
    assert state == {"steps": 0}

    new_params, new_state = opt.step(params, grads, state)
    # step() returns a copy of state; content preserved
    assert new_state == {"steps": 0}
    assert np.allclose(new_params["w"], np.array([0.95, -1.95]))

    # second step: use previous output as input
    newer_params, newer_state = opt.step(new_params, grads, new_state)
    assert newer_state == {"steps": 0}
    assert np.allclose(newer_params["w"], np.array([0.90, -1.90]))


# ---------- Built-in gradient descent (NumPy) --------------------------------

def test_make_gradient_descent_updates_twice_and_is_dtype_safe():
    opt = O.make_gradient_descent(lr=0.05)

    # accept list-like values; ensure float dtype after update
    params = {"b": [0.0, 1.0, 2.0]}
    grads = {"b": [1.0, -1.0, 0.5]}

    state = opt.initialize(params)
    p1, s1 = opt.step(params, grads, state)
    assert isinstance(p1["b"], np.ndarray)
    assert p1["b"].dtype == float
    assert np.allclose(p1["b"], np.array([-0.05, 1.05, 1.975]))

    p2, s2 = opt.step(p1, grads, s1)
    assert np.allclose(p2["b"], np.array([-0.10, 1.10, 1.95]))


def test_make_gradient_descent_raises_on_missing_grad_key():
    opt = O.make_gradient_descent(lr=0.1)
    params = {"w": np.array([1.0])}
    grads = {}  # missing "w"
    with pytest.raises(KeyError):
        _ = opt.step(params, grads, state={})
