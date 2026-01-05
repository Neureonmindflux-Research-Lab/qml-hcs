from __future__ import annotations
import numpy as np
import pytest
import qmlhc.metrics as M


# =============================================================================
# Public API re-exports
# =============================================================================
def test_metrics_public_reexports_exist():
    expected = {
        "mape", "mase", "delta_lag",
        "overshoot", "settling_time", "robustness",
        "early_roc_auc", "recall_at_lag",
        # NEW causal-indefiniteness metric
        "lambda_w_trace", "trace_distance",
    }
    for name in expected:
        assert hasattr(M, name), f"{name} missing from qmlhc.metrics"


# =============================================================================
# Forecasting metrics
# =============================================================================
def test_mape_and_mase_values_and_shapes():
    y_true = np.linspace(0.1, 1.0, 10)
    y_pred = y_true * 0.9
    y_naive = np.roll(y_true, 1)

    mape_val = M.mape(y_true, y_pred)
    mase_val = M.mase(y_true, y_pred, y_naive)

    assert mape_val >= 0.0
    assert mase_val >= 0.0
    assert np.isclose(M.mape(y_true, y_true), 0.0, atol=1e-9)
    assert np.isclose(M.mase(y_true, y_true, y_naive), 0.0, atol=1e-9)


def test_delta_lag_alignment_bounds_and_signals():
    y = np.linspace(0.0, 1.0, 8)
    assert np.isclose(M.delta_lag(y, y), 1.0)

    y_rev = y[::-1]
    val = M.delta_lag(y, y_rev)
    assert -1.0 <= val <= 1.0
    assert val <= -0.5


# =============================================================================
# Control metrics
# =============================================================================
def test_overshoot_zero_reference_and_positive_case():
    y_true = np.zeros(10)
    y_pred = np.zeros(10)
    assert M.overshoot(y_true, y_pred) == 0.0

    y_true = np.ones(10)
    y_pred = np.ones(10)
    y_pred[7:] = 1.05
    ov = M.overshoot(y_true, y_pred)
    assert 0.0 <= ov <= 0.1


def test_settling_time_band_and_robustness_bounds():
    y_true = np.linspace(0.0, 1.0, 20)
    y_pred = y_true.copy()
    y_pred[10:] = 1.04
    st = M.settling_time(y_true, y_pred, tol=0.05)
    assert st >= 0

    rb = M.robustness(y_true, y_pred)
    assert 0.0 < rb <= 1.0
    assert np.isclose(M.robustness(y_true, y_true), 1.0)


# =============================================================================
# Anomaly metrics
# =============================================================================
def test_early_roc_auc_regular_and_no_pos_neg_cases():
    y = np.array([0, 0, 1, 0, 1, 0], dtype=float)
    s = np.array([0.1, 0.2, 0.9, 0.3, 0.8, 0.2], dtype=float)
    auc = M.early_roc_auc(y, s, horizon=1)
    assert 0.0 <= auc <= 1.0

    y_none = np.zeros_like(y)
    assert M.early_roc_auc(y_none, s, horizon=1) == 0.5

    y_all = np.ones_like(y)
    auc_all = M.early_roc_auc(y_all, s, horizon=1)
    assert 0.0 <= auc_all <= 1.0


def test_recall_at_lag_normal_and_zero_anomalies():
    y = np.array([0, 0, 1, 0, 1, 0], dtype=float)
    p = np.array([0, 1, 1, 0, 0, 0], dtype=float)
    r = M.recall_at_lag(y, p, lag=1)
    assert 0.0 <= r <= 1.0

    y0 = np.zeros_like(y)
    assert M.recall_at_lag(y0, p, lag=1) == 0.0

def test_rmse_basic_and_zero_case():
    y_true = np.array([0.0, 1.0, 2.0, 3.0])
    y_pred = np.array([0.0, 1.5, 1.0, 2.0])
    expected = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
    assert np.isclose(M.rmse(y_true, y_pred), expected)
    assert np.isclose(M.rmse(y_true, y_true), 0.0)


def test_rmse_accepts_list_inputs():
    y_true = [1, 2, 3]
    y_pred = [1, 2, 4]
    expected = float(np.sqrt(np.mean((np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)) ** 2)))
    assert np.isclose(M.rmse(y_true, y_pred), expected)

def test_settling_time_and_robustness_cover_all_branches():
    import numpy as np, qmlhc.metrics as M
    y_true = np.linspace(0, 1, 10)
    y_pred = y_true.copy()
    y_pred[-2:] = 2.0
    st = M.settling_time(y_true, y_pred, tol=0.05)
    assert st > 0
    st2 = M.settling_time(y_true, y_true, tol=1.1)
    assert st2 == 0
    rb = M.robustness(y_true, y_pred)
    assert 0 < rb <= 1
    assert np.isclose(M.robustness(y_true, y_true), 1.0)


# =============================================================================
# Causal-indefiniteness metrics (lambda)
# =============================================================================

@pytest.fixture
def Wrefs():
    W_AB = np.array([[1.0, 0.0],
                     [0.0, 0.0]], dtype=complex)
    W_BA = np.array([[0.0, 0.0],
                     [0.0, 1.0]], dtype=complex)
    return W_AB, W_BA


def test_trace_distance_properties_and_symmetrize_false():
    A = np.array([[1.0, 0.0],
                  [0.0, 0.0]], dtype=complex)
    B = np.array([[0.0, 0.0],
                  [0.0, 1.0]], dtype=complex)

    d = M.trace_distance(A, B)
    assert d >= 0.0
    assert np.isclose(d, M.trace_distance(B, A))
    assert np.isclose(M.trace_distance(A, A), 0.0, atol=1e-12)

    # symmetrize=False branch
    X = np.array([[1.0, 0.0],
                  [0.0, -1.0]], dtype=complex)
    assert np.isclose(M.trace_distance(X, np.zeros_like(X), symmetrize=False), 1.0)


def test_lambda_w_trace_zero_case_and_no_improvement_arc(Wrefs):
    W_AB, W_BA = Wrefs

    # Exact reference (q=1): V = 0.5 * W_AB  -> λ=0
    assert np.isclose(M.lambda_w_trace(0.5 * W_AB, W_AB, W_BA, q_grid=81, half_factor=True),
                      0.0, atol=1e-12)

    # No-improvement arc (first grid point already optimal)
    assert np.isclose(M.lambda_w_trace(0.5 * W_BA, W_AB, W_BA, q_grid=2, half_factor=True),
                      0.0, atol=1e-12)


def test_validation_and_reference_process_branches(Wrefs):
    from qmlhc.metrics.causal_indefiniteness import reference_process, _validate_square_same_shape

    W_AB, W_BA = Wrefs
    W = np.eye(2, dtype=complex)

    # lambda input validation
    with pytest.raises(ValueError):
        M.lambda_w_trace(W, W_AB, W_BA, q_grid=1)
    with pytest.raises(ValueError):
        M.lambda_w_trace(W, W_AB, np.eye(3, dtype=complex))

    # internal shape validator branches
    with pytest.raises(ValueError):
        _validate_square_same_shape()
    with pytest.raises(ValueError):
        _validate_square_same_shape(np.zeros((2, 3), dtype=complex))

    # reference_process: q bounds + half_factor=False branch
    with pytest.raises(ValueError):
        reference_process(-0.1, W_AB, W_BA)
    with pytest.raises(ValueError):
        reference_process(1.1, W_AB, W_BA)
    assert np.allclose(reference_process(1.0, W_AB, W_BA, half_factor=False), W_AB)
