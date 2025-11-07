from __future__ import annotations
import numpy as np
import pytest
from qmlhc.optim.numpy_optim import dual_ascent as da


# =============================================================================
# Initialization: state and lambda reset
# =============================================================================
def test_dual_ascent_initialize_resets_lambdas_and_state():
    class DummyBase:
        def __init__(self):
            self.init_called = False
        def initialize(self, params):
            self.init_called = True
            return {"ok": True}

    base = DummyBase()
    opt = da.HCDualAscent(base_opt=base, dual_lr=0.05, cons_bound=0.3, coh_bound=0.2, clip_lambda=5.0)
    params = {"x": np.array([1.0, 2.0])}
    state = opt.initialize(params)
    assert base.init_called
    assert state == {"steps": 0, "lambda_cons": 0.0, "lambda_coh": 0.0}
    assert opt.lmb_cons == 0.0
    assert opt.lmb_coh == 0.0


# =============================================================================
# Step: lambda update with clipping and proper context passing
# =============================================================================
def test_dual_ascent_step_updates_lambdas_and_passes_context():
    called = {"context": None}

    def fake_eval(_m, _p, _c):
        return {"task": 1.0, "cons": 0.8, "coh": 0.5, "total": 1.0, "info": {}}

    class Base:
        def initialize(self, _p):
            return None
        def step_params(self, model, params, context):
            called["context"] = context
            return {"x": 42}, {"base_step": True}

    base = Base()
    opt = da.HCDualAscent(base_opt=base, dual_lr=0.5, cons_bound=0.5, coh_bound=0.1, clip_lambda=1.0)
    params = {"x": np.array([0.0])}
    opt.initialize(params)
    new_params, state = opt.step_params(model=None, params=params, context={"evaluate": fake_eval})
    assert new_params == {"x": 42}
    assert 0.0 <= state["lambda_cons"] <= 1.0
    assert 0.0 <= state["lambda_coh"] <= 1.0
    assert "dual" in called["context"]
    dual = called["context"]["dual"]
    assert np.isclose(dual["lambda_cons"], opt.lmb_cons)
    assert np.isclose(dual["lambda_coh"], opt.lmb_coh)
    assert state["steps"] == 1


# =============================================================================
# Step: no clipping and repeated accumulation across multiple steps
# =============================================================================
def test_dual_ascent_accumulates_and_no_clip():
    def fake_eval(_m, _p, _c):
        return {"task": 0.0, "cons": 2.0, "coh": 3.0, "total": 0.0, "info": {}}

    class Base:
        def initialize(self, _p):
            return None
        def step_params(self, model, params, context):
            return {"y": np.array([1.0])}, {"ok": True}

    base = Base()
    opt = da.HCDualAscent(base_opt=base, dual_lr=0.2, clip_lambda=None)
    params = {"y": np.array([1.0])}
    opt.initialize(params)
    for _ in range(3):
        _, state = opt.step_params(model=None, params=params, context={"evaluate": fake_eval})
    assert state["steps"] == 3
    assert state["lambda_cons"] > 0.0
    assert state["lambda_coh"] > 0.0
    assert state["lambda_cons"] == opt.lmb_cons
    assert state["lambda_coh"] == opt.lmb_coh


# =============================================================================
# Step: zero or negative violations keep lambdas non-negative
# =============================================================================
def test_dual_ascent_lambdas_non_negative():
    def fake_eval(_m, _p, _c):
        return {"task": 0.0, "cons": 0.0, "coh": 0.0, "total": 0.0, "info": {}}

    class Base:
        def initialize(self, _p):
            return None
        def step_params(self, model, params, context):
            return {"p": 1.0}, {}

    base = Base()
    opt = da.HCDualAscent(base_opt=base, dual_lr=0.5)
    params = {"p": 1.0}
    opt.initialize(params)
    _, state = opt.step_params(model=None, params=params, context={"evaluate": fake_eval})
    assert state["lambda_cons"] >= 0.0
    assert state["lambda_coh"] >= 0.0
    assert state["cons_violation"] <= 0.0
    assert state["coh_violation"] <= 0.0
    
# =============================================================================
# Initialization: base without initialize method (False branch)
# =============================================================================
def test_dual_ascent_initialize_without_base_init():
    class BaseNoInit:
        def step_params(self, model, params, context):
            return params, {}

    base = BaseNoInit()
    opt = da.HCDualAscent(base_opt=base, dual_lr=0.1)
    params = {"w": np.array([1.0])}
    state = opt.initialize(params)
    assert state == {"steps": 0, "lambda_cons": 0.0, "lambda_coh": 0.0}
