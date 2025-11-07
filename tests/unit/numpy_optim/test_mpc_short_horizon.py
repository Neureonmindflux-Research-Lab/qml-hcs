from __future__ import annotations
import numpy as np
import pytest

from qmlhc.optim.numpy_optim import mpc as mpc


# =============================================================================
# Initialize: state shape and immutability
# =============================================================================
def test_mpc_initialize_state():
    opt = mpc.HCMPCShortHorizon(lr=0.1, horizon=3)
    params = {"a": 1.0, "b": np.array([0.5, -0.5])}
    state = opt.initialize(params)
    assert isinstance(state, dict)
    assert state == {"steps": 0}


# =============================================================================
# Step: finite-diff descent decreases loss and updates state (default horizon)
# =============================================================================
def test_mpc_step_decreases_loss_and_updates_state_default_horizon():
    def rollout_fn(_model, p, _h, _ctx):
        val = 0.0
        for v in p.values():
            arr = np.atleast_1d(np.asarray(v, dtype=float))
            val += float(np.sum(arr**2))
        return None, val

    opt = mpc.HCMPCShortHorizon(lr=0.2, horizon=3)
    params = {"a": 1.0, "b": np.array([1.0, -2.0])}
    state0 = opt.initialize(params)

    def total_loss(pp):
        s = 0.0
        for v in pp.values():
            arr = np.atleast_1d(np.asarray(v, dtype=float))
            s += float(np.sum(arr**2))
        return s

    new_params, state1 = opt.step_params(model=None, params=params, context={"rollout_fn": rollout_fn})
    assert total_loss(new_params) < total_loss(params)
    assert state1["steps"] == 1
    assert state1["horizon"] == 3
    assert state1["grad_norm"] >= 0.0


# =============================================================================
# Step: explicit horizon override in context
# =============================================================================
def test_mpc_step_uses_context_horizon_override():
    def rollout_fn(_model, _p, h, _ctx):
        return None, float(h)

    opt = mpc.HCMPCShortHorizon(lr=0.01, horizon=2)
    params = {"x": np.array([0.0, 0.0])}
    opt.initialize(params)
    _, st = opt.step_params(model=None, params=params, context={"rollout_fn": rollout_fn, "horizon": 7})
    assert st["horizon"] == 7


# =============================================================================
# Step: projection function is applied to rebuilt params
# =============================================================================
def test_mpc_project_fn_is_applied_once():
    def rollout_fn(_m, p, _h, _c):
        s = 0.0
        for v in p.values():
            arr = np.atleast_1d(np.asarray(v, dtype=float))
            s += float(np.sum(arr**2))
        return None, s

    def project_fn(p):
        q = {k: np.asarray(v, dtype=float) * 0.0 for k, v in p.items()}
        return {k: (float(v) if np.asarray(v).size == 1 else v) for k, v in q.items()}

    opt = mpc.HCMPCShortHorizon(lr=0.5, horizon=1)
    params = {"s": 0.9, "v": np.array([0.4, -0.3])}
    opt.initialize(params)
    new_params, st = opt.step_params(model=None, params=params, context={"rollout_fn": rollout_fn, "project_fn": project_fn})
    assert float(np.sum(np.atleast_1d(new_params["s"]))) == 0.0
    assert np.allclose(new_params["v"], np.array([0.0, 0.0]))
    assert st["steps"] == 1


# =============================================================================
# Step: clipping caps parameter updates to [-clip, clip]
# =============================================================================
def test_mpc_clip_caps_updates():
    def rollout_fn(_m, p, _h, _c):
        s = 0.0
        for v in p.values():
            arr = np.atleast_1d(np.asarray(v, dtype=float))
            s += float(np.sum(arr**2))
        return None, s

    opt = mpc.HCMPCShortHorizon(lr=10.0, horizon=1, clip=0.25)
    params = {"a": 1.0, "b": np.array([0.1, -0.1])}
    opt.initialize(params)
    new_params, _ = opt.step_params(model=None, params=params, context={"rollout_fn": rollout_fn})
    flat = np.concatenate([np.atleast_1d(new_params["a"]), new_params["b"].reshape(-1)])
    assert np.all(flat <= 0.25 + 1e-12)
    assert np.all(flat >= -0.25 - 1e-12)


# =============================================================================
# Step: preserves scalar vs array shapes across rebuild
# =============================================================================
def test_mpc_shape_preservation_and_key_order_insensitivity():
    def rollout_fn(_m, p, _h, _c):
        s = 0.0
        for v in p.values():
            arr = np.atleast_1d(np.asarray(v, dtype=float))
            s += float(np.sum(arr**2))
        return None, s

    opt = mpc.HCMPCShortHorizon(lr=0.01, horizon=2)
    params = {"z": np.array([1.0, -1.0]), "a": 0.5, "b": 0.2} 
    opt.initialize(params)
    new_params, _ = opt.step_params(model=None, params=params, context={"rollout_fn": rollout_fn})

    assert np.asarray(new_params["z"]).shape == (2,)
    assert np.asarray(new_params["a"]).shape == ()
    assert np.asarray(new_params["b"]).shape == ()

# =============================================================================
# Error: missing rollout_fn in context
# =============================================================================
def test_mpc_raises_when_rollout_fn_missing():
    opt = mpc.HCMPCShortHorizon(lr=0.1, horizon=3)
    params = {"a": 1.0}
    opt.initialize(params)
    with pytest.raises(KeyError):
        _ = opt.step_params(model=None, params=params, context={})
