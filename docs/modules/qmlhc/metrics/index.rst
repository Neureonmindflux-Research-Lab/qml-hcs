Metrics Module
==============

The **qmlhc.metrics** package provides advanced temporal and statistical evaluation
utilities for anomaly detection, control stability, and forecast accuracy.  
It extends the core loss functions by quantifying *how early*, *how stable*, and
*how precise* a system response or prediction is.

This module can be applied directly to states or predictions produced by the
:ref:`Core Module <core-example>` (via backend outputs) or the
:ref:`Hypercausal Layer <hc-node-graph-example>` (graph steps), and may be combined
with the results from :ref:`Loss Evaluation <loss-evaluation-example>`.

Architecture Overview
---------------------

::

   +----------------------+
   |    qmlhc.metrics     |
   +-----------+----------+
               |
       +-------v--------+     +--------------------+     +----------------------+
       |   anomalies    |     |      control       |     |     forecasting      |
       | (early alarms) |     | (stability/robust) |     | (error & Δ-lag)      |
       +----------------+     +--------------------+     +----------------------+

- ``anomalies`` - metrics for **early detection** and **recall-at-lag** within sequential data.
- ``control`` - indicators of **overshoot**, **settling time**, and **robustness** for response stability.
- ``forecasting`` - statistical errors and temporal alignment metrics (MAPE, MASE, ΔLag, RMSE).

Core Contracts
--------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Component
     - Summary
   * - ``early_roc_auc(y_true, scores, horizon)`` (anomalies)
     - Measures how early anomalies are detected within a future horizon.  
       **Returns 0.5 if either positive or negative labels are absent** (uninformative baseline).
   * - ``recall_at_lag(y_true, y_pred, lag)`` (anomalies)
     - Computes recall while tolerating detection delays of up to ``lag`` timesteps.
   * - ``overshoot(y_true, y_pred)`` (control)
     - Relative excess over the reference maximum.
   * - ``settling_time(y_true, y_pred, tol)`` (control)
     - Number of samples required for stable convergence within a tolerance band.  
       **Returns 0 if the entire sequence is already within tolerance.**
   * - ``robustness(y_true, y_pred)`` (control)
     - Stability index in ``(0, 1]`` inversely proportional to MSE.
   * - ``mape``, ``mase``, ``delta_lag``, ``rmse`` (forecasting)
     - Standard predictive accuracy and alignment metrics.

Integrated Evaluation Example
-----------------------------

The following example extends the :ref:`Loss Evaluation Example <loss-evaluation-example>`
by adding **metric evaluation** from the :mod:`qmlhc.metrics` package.  
This provides a unified diagnostic combining backend state evolution, loss analysis,
and higher-level metrics.

.. code-block:: python

   import numpy as np

   # --- Backend (reused from core) ---------------------------------------------
   # Base configuration identical to previous examples.
   from qmlhc.core.backend import BackendConfig, QuantumBackend
   from qmlhc.core.registry import register_backend, create_backend

   class MyBackend(QuantumBackend):
       """Normalized input and stochastic projection backend."""
       def run(self, params=None):
           x = self._require_input()
           return self._validate_state(x / np.linalg.norm(x))

       def project_future(self, s_t, branches=2):
           s_t = self._validate_state(s_t)
           noise = np.random.normal(scale=0.05, size=(branches, self.output_dim))
           fut = s_t + noise
           return self._validate_branches(fut)

   capabilities = {
       "backend_name": "MyBackend",
       "backend_version": "1.0",
       "max_qubits": 0,
       "output_dim": 3,
       "supports_shots": False,
       "min_shots": 0,
       "max_shots": 0,
       "supports_noise": True,
       "supports_batch": False,
       "gradient": "none",
       "notes": "demo backend",
   }
   register_backend("my-backend", lambda cfg: MyBackend(cfg), capabilities)
   backend = create_backend("my-backend", BackendConfig(output_dim=3))

   # --- Hypercausal layer ------------------------------------------------------
   from qmlhc.hc.node import HCNode
   from qmlhc.hc.graph import HCGraph
   from qmlhc.hc.policy import MeanPolicy

   node = HCNode(backend=backend, policy=MeanPolicy())
   graph = HCGraph.chain(names=["N1"], nodes=[node])

   # --- Single causal step -----------------------------------------------------
   x_t = np.array([1.0, 2.0, 3.0])
   s_map, s_hat_map, info_map = graph.step({"N1": x_t}, branches=3)

   # --- Loss module integration -------------------------------------------------
   from qmlhc.loss.task import MSELoss
   from qmlhc.loss.coherence import CoherenceLoss
   from qmlhc.loss.consistency import ConsistencyLoss

   target = np.array([0.3, 0.5, 0.8])
   task_mse = MSELoss()(s_hat_map["N1"], target)
   K_branches = info_map["N1"]["branches"]
   coh_var = CoherenceLoss(mode="variance")(K_branches)
   coh_mad = CoherenceLoss(mode="mad")(K_branches)
   prev_like = s_map["N1"]
   cons_tri = ConsistencyLoss(alpha=1.0, beta=1.0)(prev_like, s_map["N1"], s_hat_map["N1"])

   # --- Metrics Evaluation ------------------------------------------------------
   #  NEW SECTION: This extends the previous example by adding advanced metrics
   # from qmlhc.metrics. Each function here is designed to complement the loss
   # measures with interpretability (timeliness, stability, accuracy).

   from qmlhc.metrics.anomalies import early_roc_auc, recall_at_lag
   from qmlhc.metrics.control import overshoot, settling_time, robustness
   from qmlhc.metrics.forecasting import mape, mase, delta_lag, rmse

   # --- (1) Anomaly metrics (NEW) ----------------------------------------------
   # Simulate binary ground truth and scores to test timeliness evaluation
   y_true_anom = np.array([0, 0, 1, 0, 0, 1, 0], dtype=float)
   scores = np.array([0.1, 0.2, 0.9, 0.3, 0.2, 0.8, 0.2], dtype=float)
   y_pred_bin = (scores > 0.5).astype(float)

   auc_early = early_roc_auc(y_true_anom, scores, horizon=1)
   rec_lag = recall_at_lag(y_true_anom, y_pred_bin, lag=1)

   # --- (2) Control metrics (NEW) ----------------------------------------------
   # Example reference and response sequences to assess stability
   ref = np.array([0.0, 0.6, 1.0, 1.0, 1.0], dtype=float)
   resp = np.array([0.0, 0.7, 1.1, 0.98, 1.01], dtype=float)

   os_rel = overshoot(ref, resp)
   t_set = settling_time(ref, resp, tol=0.05)
   rob = robustness(ref, resp)

   # --- (3) Forecast metrics (NEW) ---------------------------------------------
   # Example real and predicted sequences for accuracy and alignment
   y_true_seq = np.array([1.0, 1.1, 1.05, 1.2, 1.25], dtype=float)
   y_pred_seq = np.array([0.95, 1.08, 1.10, 1.18, 1.23], dtype=float)
   y_naive = np.roll(y_true_seq, 1); y_naive[0] = y_true_seq[0]

   mape_v = mape(y_true_seq, y_pred_seq)
   mase_v = mase(y_true_seq, y_pred_seq, y_naive)
   dlag_v = delta_lag(y_true_seq, y_pred_seq)
   rmse_v = rmse(y_true_seq, y_pred_seq)

   # --- Output Summary ----------------------------------------------------------
   print("Input:", x_t)
   print("S_t:", s_map["N1"])
   print("Ŝ_{t+1} (mean policy):", s_hat_map["N1"])
   print(f"Task MSE: {task_mse:.6f}")
   print(f"Coherence Var: {coh_var:.6f} | MAD: {coh_mad:.6f}")
   print(f"Triadic Consistency: {cons_tri:.6f}")
   print("---- Metrics Evaluation ----")
   print(f"Early ROC-AUC: {auc_early:.2f} | Recall@Lag: {rec_lag:.2f}")
   print(f"Overshoot: {os_rel:.2f} | Settling Time: {t_set} | Robustness: {rob:.2f}")
   print(f"MAPE: {mape_v:.2f}% | MASE: {mase_v:.2f} | ΔLag: {dlag_v:.2f} | RMSE: {rmse_v:.4f}")

Expected Output (Variable)
--------------------------

.. code-block:: text

   Input: [1. 2. 3.]
   S_t: [0.26726124 0.53452248 0.80178373]
   Ŝ_{t+1} (mean policy): [0.29789606 0.57910593 0.79579622]
   Task MSE: 0.002093
   Coherence Var: 0.001400 | MAD: 0.031374
   Triadic Consistency: 0.000987
   ---- Metrics Evaluation ----
   Early ROC-AUC: 0.75 | Recall@Lag: 1.00
   Overshoot: 0.10 | Settling Time: 3 | Robustness: 0.97
   MAPE: 3.21% | MASE: 0.64 | ΔLag: 0.50 | RMSE: 0.04

Invariants and Conventions
--------------------------

- All metrics accept NumPy arrays and return scalar floats.
- Anomaly metrics quantify *timeliness* (not accuracy) of detection.
- Control metrics quantify *stability and convergence* behavior.
- Forecast metrics quantify *magnitude and directional alignment*.
- Designed to interoperate seamlessly with :mod:`qmlhc.loss` outputs.

Module References
-----------------

.. toctree::
   :maxdepth: 1
   :titlesonly:

   Anomaly Metrics <qmlhc.metrics.anomalies>
   Control Metrics <qmlhc.metrics.control>
   Forecasting Metrics <qmlhc.metrics.forecasting>
