
# Comprehensive tests for src/qmlhc/loss: __init__.py, task.py, consistency.py, coherence.py


from __future__ import annotations

import numpy as np
import pytest
import qmlhc.loss as L


# ---------- Public API re-exports --------------------------------------------

def test_loss_public_reexports_exist():
    expected = {
        "TaskLoss", "MSELoss", "MAELoss", "CrossEntropyLoss",
        "ConsistencyLoss", "CoherenceLoss"
    }
    for name in expected:
        assert hasattr(L, name), f"{name} missing from qmlhc.loss"


# ---------- Task-level losses ------------------------------------------------

def test_mse_loss_values_and_shape_check():
    pred = np.array([0.0, 1.0, 2.0])
    target = np.array([0.0, 2.0, 4.0])
    mse = L.MSELoss()
    val = mse(pred, target)
    assert np.isclose(val, np.mean((pred - target) ** 2))
    with pytest.raises(ValueError):
        _ = mse(np.array([1.0, 2.0]), np.array([1.0]))


def test_mae_loss_values_and_shape_check():
    pred = np.array([1.0, -1.0])
    target = np.array([0.0, 0.0])
    mae = L.MAELoss()
    val = mae(pred, target)
    assert np.isclose(val, np.mean(np.abs(pred - target)))
    with pytest.raises(ValueError):
        _ = mae(np.array([1.0, 2.0]), np.array([1.0]))


def test_crossentropy_loss_valid_and_shape_check():
    pred = np.array([0.2, 0.5, 0.3])
    target = np.array([0.0, 1.0, 0.0])
    ce = L.CrossEntropyLoss()
    val = ce(pred, target)
    assert isinstance(val, float)
    assert val > 0.0
    with pytest.raises(ValueError):
        _ = ce(np.array([0.5, 0.5]), np.array([0.2]))


# ---------- Consistency loss -------------------------------------------------

def test_consistency_loss_correctness_and_dimension_checks():
    s_tm1 = np.array([0.0, 0.5, 1.0])
    s_t = np.array([0.1, 0.6, 1.1])
    s_tp1_hat = np.array([0.2, 0.7, 1.2])
    loss = L.ConsistencyLoss(alpha=0.5, beta=2.0)
    val = loss(s_tm1, s_t, s_tp1_hat)
    d_prev = np.mean((s_t - s_tm1) ** 2)
    d_fut = np.mean((s_t - s_tp1_hat) ** 2)
    expected = 0.5 * d_prev + 2.0 * d_fut
    assert np.isclose(val, expected)
    # Dimension mismatch
    with pytest.raises(ValueError):
        _ = loss(np.array([1.0, 2.0]), np.array([1.0]), np.array([1.0, 2.0]))


# ---------- Coherence loss ---------------------------------------------------

def test_coherence_loss_variance_mode():
    fut = np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]], dtype=float)
    loss_fn = L.CoherenceLoss(mode="variance")
    val = loss_fn(fut)
    assert np.isclose(val, np.var(fut, axis=0).mean())
    assert isinstance(val, float)


def test_coherence_loss_mad_mode():
    fut = np.array([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]], dtype=float)
    loss_fn = L.CoherenceLoss(mode="mad")
    val = loss_fn(fut)
    mu = fut.mean(axis=0, keepdims=True)
    expected = float(np.abs(fut - mu).mean())
    assert np.isclose(val, expected)
    assert isinstance(val, float)


def test_coherence_loss_invalid_inputs_and_mode():
    fut = np.array([1.0, 2.0])
    with pytest.raises(ValueError):
        _ = L.CoherenceLoss(mode="variance")(fut)
    with pytest.raises(ValueError):
        _ = L.CoherenceLoss(mode="unknown")(np.zeros((3, 2)))
