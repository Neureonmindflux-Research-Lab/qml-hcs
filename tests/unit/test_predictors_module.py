
# Comprehensive tests for src/qmlhc/predictors: __init__.py, projector.py, anticipator.py


from __future__ import annotations

import numpy as np
import pytest

import qmlhc.predictors as P


# ---------- Public API re-exports --------------------------------------------

def test_predictors_public_reexports_exist():
    expected = {"Projector", "LinearProjector", "ContrafactualAnticipator"}
    for name in expected:
        assert hasattr(P, name), f"{name} missing from qmlhc.predictors"


# ---------- LinearProjector ---------------------------------------------------

def test_linear_projector_basic_and_branches():
    lp = P.LinearProjector(weight=0.9, bias=0.1, span=0.3)
    s = np.array([0.2, -0.1])
    fut = lp.project(s, branches=5)
    assert fut.shape == (5, 2)
    # monotone span around affine base after tanh squashing
    base = 0.9 * s + 0.1
    assert np.allclose(fut[2], np.tanh(base))  # middle delta = 0 when k=5

    # branches < 2 should be coerced to 2
    fut2 = lp.project(s, branches=1)
    assert fut2.shape == (2, 2)


def test_linear_projector_is_deterministic_and_type_safe():
    lp = P.LinearProjector()
    s = [0.0, 0.5, 1.0]  # list input accepted (TensorLike)
    a = lp.project(s, branches=4)
    b = lp.project(np.asarray(s), branches=4)
    assert np.allclose(a, b)
    assert a.dtype == float


# ---------- ContrafactualAnticipator -----------------------------------------

def test_anticipator_uses_projector_and_concatenates_variants():
    lp = P.LinearProjector(weight=1.0, bias=0.0, span=0.2)

    def bump(v: np.ndarray) -> np.ndarray:
        # simple structured perturbation on the center
        return v + 0.1

    ant = P.ContrafactualAnticipator(projector=lp, perturbations=[bump])
    s = np.array([0.3, -0.2])
    fut = ant.generate(s)

    # base set size is K; with one perturbation and symmetric=True we add +2 rows
    base = lp.project(s, branches=3)
    assert fut.shape[0] == base.shape[0] + 2
    assert fut.shape[1] == base.shape[1]

    center = base.mean(axis=0)
    assert any(np.allclose(row, center + 0.1) for row in fut)
    # symmetric counterpart around center must exist as well
    assert any(np.allclose(row, 2 * center - (center + 0.1)) for row in fut)


def test_anticipator_without_perturbations_returns_base_only():
    lp = P.LinearProjector(weight=1.0, bias=0.0, span=0.2)
    ant = P.ContrafactualAnticipator(projector=lp, perturbations=None)
    s = np.array([0.0, 0.5])
    fut = ant.generate(s)
    base = lp.project(s, branches=fut.shape[0])  # usa K real generado
    assert np.allclose(fut, base)




def test_anticipator_asymmetric_mode():
    lp = P.LinearProjector()
    # symmetric=False -> only adds the direct perturbation, not its mirror
    from qmlhc.predictors.anticipator import AnticipatorConfig
    cfg = AnticipatorConfig(branches=2, symmetric=False)

    def tilt(v: np.ndarray) -> np.ndarray:
        return v - 0.05

    ant = P.ContrafactualAnticipator(projector=lp, perturbations=[tilt], config=cfg)
    s = np.array([0.1, -0.1])
    fut = ant.generate(s)
    base = lp.project(s, branches=2)
    # base (K rows) + one extra row for the perturbation (no mirror)
    assert fut.shape == (base.shape[0] + 1, base.shape[1])
