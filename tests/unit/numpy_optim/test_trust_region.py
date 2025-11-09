from __future__ import annotations
import numpy as np
import pytest
from qmlhc.optim.numpy_optim import trust_region as tr


# =============================================================================
# Initialize: with and without base.initialize
# =============================================================================
def test_trustregion_initialize_with_and_without_base_init():
    class BaseWithInit:
        def __init__(self): self.called = False
        def initialize(self, params): self.called = True; return {"ok": True}
        def step_params(self, model, params, context): return params, {}

    class BaseNoInit:
        def step_params(self, model, params, context): return params, {}

    opt1 = tr.HCTrustRegion(base_opt=BaseWithInit())
    s1 = opt1.initialize({"w": np.array([1.0])})
    assert isinstance(s1, dict) and s1.get("steps", 0) == 0

    opt2 = tr.HCTrustRegion(base_opt=BaseNoInit())
    s2 = opt2.initialize({"w": np.array([1.0])})
    assert isinstance(s2, dict) and s2.get("steps", 0) == 0


# =============================================================================
# Step: accepts proposal without backtracking 
# =============================================================================
def test_trustregion_accepts_without_backtracking():
    class Base:
        def step_params(self, model, params, context):
            return {k: np.asarray(v, float) + 0.05 for k, v in params.items()}, {"ok": True}

    def refresh_info(_m, p, _c):
        vec = np.concatenate([np.atleast_1d(np.asarray(v, float)).ravel() for v in p.values()])
        return {"vec": vec}

    def kl_fn(old, new):
        return float(np.linalg.norm(new["vec"] - old["vec"]))

    params = {"a": np.array([1.0, -1.0])}
    old_info = refresh_info(None, params, None)
    opt = tr.HCTrustRegion(base_opt=Base(), delta_kl=1.0, backtrack=0.5, max_backtracks=3)
    _ = opt.initialize(params)

    new_params, state = opt.step_params(
        model=None,
        params=params,
        context={"kl_fn": kl_fn, "refresh_info": refresh_info, "info": old_info},
    )

    assert not np.allclose(new_params["a"], params["a"])
    assert state.get("steps") == 1
    assert state.get("alpha_bt", 0.0) > 0.0
    assert state.get("kl", 0.0) <= 1.0 + 1e-9


# =============================================================================
# Step: backtracks then accepts under KL bound (large initial step)
# =============================================================================
def test_trustregion_backtracks_then_accepts():
    class Base:
        def step_params(self, model, params, context):
            return {k: np.asarray(v, float) + 5.0 for k, v in params.items()}, {}

    def refresh_info(_m, p, _c):
        vec = np.concatenate([np.atleast_1d(np.asarray(v, float)).ravel() for v in p.values()])
        return {"vec": vec}

    def kl_fn(old, new):
        return float(np.linalg.norm(new["vec"] - old["vec"]))

    params = {"x": np.array([0.0, 0.0])}
    old_info = refresh_info(None, params, None)
    opt = tr.HCTrustRegion(base_opt=Base(), delta_kl=1.0, backtrack=0.5, max_backtracks=8)
    _ = opt.initialize(params)

    new_params, state = opt.step_params(
        model=None,
        params=params,
        context={"kl_fn": kl_fn, "refresh_info": refresh_info, "info": old_info},
    )

    assert state.get("steps") == 1
    assert 0.0 < state.get("alpha_bt", 0.0) < 1.0
    assert state.get("kl", 0.0) <= 1.0 + 1e-9
    assert np.linalg.norm(new_params["x"] - params["x"]) > 0.0


# =============================================================================
# Step: exhausts backtracking and keeps original params (impossible to satisfy KL)
# =============================================================================
def test_trustregion_exhausts_and_returns_original():
    class Base:
        def step_params(self, model, params, context):
            return {k: np.asarray(v, float) + 10.0 for k, v in params.items()}, {}

    def refresh_info(_m, p, _c):
        vec = np.concatenate([np.atleast_1d(np.asarray(v, float)).ravel() for v in p.values()])
        return {"vec": vec}

    def kl_fn(_old, _new):
        return 1e9

    params = {"w": np.array([1.0])}
    old_info = refresh_info(None, params, None)
    opt = tr.HCTrustRegion(base_opt=Base(), delta_kl=0.1, backtrack=0.5, max_backtracks=1)
    _ = opt.initialize(params)

    new_params, state = opt.step_params(
        model=None,
        params=params,
        context={"kl_fn": kl_fn, "refresh_info": refresh_info, "info": old_info},
    )

    assert np.allclose(new_params["w"], params["w"])
    assert state.get("steps") == 1
    assert state.get("alpha_bt", 1.0) == 0.0
