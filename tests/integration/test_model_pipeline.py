#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Integration Test: Model Pipeline
--------------------------------
End-to-end verification for the quantum–hypercausal model pipeline.

Covers:
- Backend → HCNode → HCModel execution chain.
- Validation of task, consistency, and coherence losses.
- Forecasting metrics (MAPE, MASE, ΔLag).


"""

from __future__ import annotations

import numpy as np

from qmlhc.core.backend import BackendConfig, QuantumBackend
from qmlhc.core.model import HCModel, ModelConfig
from qmlhc.hc.node import HCNode, NodeConfig
from qmlhc.hc.policy import MeanPolicy
from qmlhc.loss.task import MSELoss
from qmlhc.loss.consistency import ConsistencyLoss
from qmlhc.loss.coherence import CoherenceLoss
from qmlhc.metrics.forecasting import mape, mase, delta_lag


class DummyBackend(QuantumBackend):
    """
    Deterministic backend used for end-to-end integration testing.

    Implements simple nonlinear transformations using ``tanh`` to generate
    reproducible compact states and candidate futures.
    """

    def run(self, params=None):
        """
        Compute the compact state from the current encoded input.

        Returns
        -------
        numpy.ndarray
            Current compact state after transformation.
        """
        x = self._require_input()
        return np.tanh(x + 0.07)

    def project_future(self, s_t, branches: int = 2):
        """
        Generate multiple candidate futures with small symmetric perturbations.

        Parameters
        ----------
        s_t : numpy.ndarray
            Current state vector.
        branches : int, optional
            Number of candidate futures, by default ``2``.

        Returns
        -------
        numpy.ndarray
            Matrix of projected candidate futures with shape ``(K, D)``.
        """
        s = self._validate_state(s_t)
        k = max(2, int(branches))
        deltas = np.linspace(-0.08, 0.08, k, dtype=float)
        fut = np.stack([np.tanh(s + d) for d in deltas], axis=0)
        return self._validate_branches(fut)


def test_integration_single_step_and_losses():
    """
    Validate single-step forward pass and core loss components.

    Ensures numerical stability and non-negativity for:
    - Task loss (MSE)
    - Consistency loss
    - Coherence loss
    """
    cfg = BackendConfig(output_dim=3)
    backend = DummyBackend(cfg)
    node = HCNode(backend=backend, policy=MeanPolicy(), config=NodeConfig(branches=4))
    model = HCModel(nodes=[node], config=ModelConfig(default_branches=4))

    x_t = np.array([0.0, 0.5, 1.0], dtype=float)
    s_tm1 = np.array([-0.1, 0.4, 0.9], dtype=float)
    target = np.array([0.05, 0.55, 0.95], dtype=float)

    s_t, s_tp1_hat, info = model.forward(x_t, s_tm1=s_tm1)

    task = MSELoss()(s_t, target)
    cons = ConsistencyLoss(alpha=1.0, beta=1.0)(s_tm1, s_t, s_tp1_hat)
    coh = CoherenceLoss(mode="variance")(info["branches"])

    assert np.isfinite(task) and task >= 0.0
    assert np.isfinite(cons) and cons >= 0.0
    assert np.isfinite(coh) and coh >= 0.0


def test_sequence_metrics_forecasting_like():
    """
    Validate sequential model execution and forecasting metrics.

    Simulates a simple monotonic increasing sequence and computes:
    - MAPE (Mean Absolute Percentage Error)
    - MASE (Mean Absolute Scaled Error)
    - ΔLag (directional alignment metric)
    """
    cfg = BackendConfig(output_dim=1)
    backend = DummyBackend(cfg)
    node = HCNode(backend=backend, policy=MeanPolicy(), config=NodeConfig(branches=3))
    model = HCModel(nodes=[node])

    # Synthetic increasing target and input sequences
    x_seq = [np.array([v], dtype=float) for v in np.linspace(0.0, 1.0, 10)]
    y_true = np.array([v * 1.1 for v in np.linspace(0.0, 1.0, 10)], dtype=float)

    states, futures, infos = model.predict_sequence(x_seq, s0=None, branches=3, use_chain=False)
    y_pred = np.array([s[0] for s in states], dtype=float)

    val_mape = mape(y_true, y_pred)
    val_mase = mase(y_true, y_pred, y_naive=np.roll(y_true, 1))
    val_dlag = delta_lag(y_true, y_pred)

    assert np.isfinite(val_mape)
    assert np.isfinite(val_mase)
    assert -1.0 <= val_dlag <= 1.0
