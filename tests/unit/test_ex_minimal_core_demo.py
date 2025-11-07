from __future__ import annotations
import numpy as np
import pytest

# Import the example module
from qmlhc.examples import ex_minimal_core_demo as ex


# =============================================================================
# ToyBackend: run/project_future shapes, clamping, and K>=2 invariant
# =============================================================================
def test_toybackend_run_and_project_future_shapes():
    cfg = ex.BackendConfig(output_dim=3, seed=0)
    be = ex.ToyBackend(cfg)

    # encode + run → state has expected dimension
    x = np.array([0.2, -0.1, 0.4], dtype=float)
    be.encode(x)
    s_t = be.run()
    assert isinstance(s_t, np.ndarray) and s_t.shape == (3,)

    # project_future with K=1 should be clamped to at least 2
    fut2 = be.project_future(s_t, branches=1)
    assert fut2.shape == (2, 3)

    # project_future with K=5 produces (5, D)
    fut5 = be.project_future(s_t, branches=5)
    assert fut5.shape == (5, 3)
    # values are bounded by tanh
    assert np.all(np.abs(fut5) <= 1.0 + 1e-12)


# =============================================================================
# Demo function: prints summary and validates HCModel equivalence
# =============================================================================
def test_minimal_core_demo_prints_and_matches_model(capsys: pytest.CaptureFixture[str]):
    # Run demo and capture stdout
    ex.minimal_core_demo()
    out = capsys.readouterr().out

    # Expected markers and summary lines
    assert "=== Minimal Core Demo ===" in out
    assert "HCModel.forward() matches single-node result" in out
    assert "branches shape:" in out
    assert "loss =" in out

def test_entrypoint_executes_demo_runpy(capsys):
    import runpy
    import qmlhc.examples.ex_minimal_core_demo as ex
    runpy.run_module(ex.__name__, run_name="__main__")
    out = capsys.readouterr().out
    assert "=== Minimal Core Demo ===" in out
    assert "HCModel.forward() matches single-node result" in out

# =============================================================================