
# Comprehensive tests for src/qmlhc/metrics: __init__.py, forecasting.py, control.py, anomalies.py


from __future__ import annotations

import numpy as np
import pytest
import qmlhc.metrics as M


# ---------- Public API re-exports --------------------------------------------

def test_metrics_public_reexports_exist():
    expected = {
        "mape", "mase", "delta_lag",
        "overshoot", "settling_time", "robustness",
        "early_roc_auc", "recall_at_lag",
    }
    for name in expected:
        assert hasattr(M, name), f"{name} missing from qmlhc.metrics"


# ---------- Forecasting metrics ----------------------------------------------

def test_mape_and_mase_values_and_shapes():
    y_true = np.linspace(0.1, 1.0, 10)
    y_pred = y_true * 0.9
    y_naive = np.roll(y_true, 1)

    mape_val = M.mape(y_true, y_pred)
    mase_val = M.mase(y_true, y_pred, y_naive)

    assert mape_val >= 0.0
    assert mase_val >= 0.0
    # sanity checks: perfect prediction -> near 0
    assert np.isclose(M.mape(y_true, y_true), 0.0, atol=1e-9)
    assert np.isclose(M.mase(y_true, y_true, y_naive), 0.0, atol=1e-9)


def test_delta_lag_alignment_bounds_and_signals():
    # identical sequences -> perfect alignment (1.0)
    y = np.linspace(0.0, 1.0, 8)
    assert np.isclose(M.delta_lag(y, y), 1.0)
    # opposite trend -> -1.0
    y_rev = y[::-1]
    val = M.delta_lag(y, y_rev)
    assert -1.0 <= val <= 1.0
    assert val <= -0.5  # strongly anti-aligned


# ---------- Control metrics ---------------------------------------------------

def test_overshoot_zero_reference_and_positive_case():
    # zero reference -> by definition 0.0 overshoot
    y_true = np.zeros(10)
    y_pred = np.zeros(10)
    assert M.overshoot(y_true, y_pred) == 0.0

    # positive reference with small overshoot
    y_true = np.ones(10)
    y_pred = np.ones(10)
    y_pred[7:] = 1.05
    ov = M.overshoot(y_true, y_pred)
    assert 0.0 <= ov <= 0.1


def test_settling_time_band_and_robustness_bounds():
    y_true = np.linspace(0.0, 1.0, 20)
    y_pred = y_true.copy()
    y_pred[10:] = 1.04  # within 4% band around final ref=1.0
    st = M.settling_time(y_true, y_pred, tol=0.05)
    assert st >= 0

    rb = M.robustness(y_true, y_pred)
    assert 0.0 < rb <= 1.0
    # perfect prediction -> robustness = 1.0
    assert np.isclose(M.robustness(y_true, y_true), 1.0)


# ---------- Anomaly metrics ---------------------------------------------------

def test_early_roc_auc_regular_and_no_pos_neg_cases():
    y = np.array([0, 0, 1, 0, 1, 0], dtype=float)
    s = np.array([0.1, 0.2, 0.9, 0.3, 0.8, 0.2], dtype=float)
    auc = M.early_roc_auc(y, s, horizon=1)
    assert 0.0 <= auc <= 1.0

    # no positives in window -> defined as 0.5
    y_none = np.zeros_like(y)
    assert M.early_roc_auc(y_none, s, horizon=1) == 0.5

    # no negatives (all ones) -> any valid value in [0,1]
    y_all = np.ones_like(y)
    auc_all = M.early_roc_auc(y_all, s, horizon=1)
    assert 0.0 <= auc_all <= 1.0



def test_recall_at_lag_normal_and_zero_anomalies():
    y = np.array([0, 0, 1, 0, 1, 0], dtype=float)
    p = np.array([0, 1, 1, 0, 0, 0], dtype=float)
    r = M.recall_at_lag(y, p, lag=1)
    assert 0.0 <= r <= 1.0

    # no anomalies -> 0.0 by definition (hits / (total+eps))
    y0 = np.zeros_like(y)
    assert M.recall_at_lag(y0, p, lag=1) == 0.0
