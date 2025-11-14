Predictors Module
=================

The **qmlhc.predictors** package provides deterministic projectors and
counterfactual anticipators that map a compact present state ``S_t`` to a set of
candidate futures ``{S_{t+1}^{(k)}}``. It offers a clean separation between
**how futures are generated** (projection mechanisms) and **how they are later
selected** (policies in :mod:`qmlhc.hc.policy`), enabling reproducible and
inspectable multi-branch predictions.

This module can be combined with states produced by either the
:ref:`Core Module <core-example>` (via a backend’s state) or the
:ref:`Hypercausal Layer <hc-node-graph-example>` (graph step outputs),
and its outputs can be evaluated with :ref:`Loss Evaluation <loss-evaluation-example>`.

Architecture Overview
---------------------

::

   +-------------------------+
   |    qmlhc.predictors     |
   +-----------+-------------+
               |
       +-------v--------+         +----------------------------+
       |   projector    |  --->   |       anticipator          |
       | (protocol +    |         | (counterfactual wrapper    |
       |  LinearProjector)        |  over a base projector)    |
       +-----------------+        +----------------------------+

- ``projector`` - defines the **Projector** protocol and the deterministic
  **LinearProjector** (affine base + evenly spaced deltas + ``tanh`` for stability).
- ``anticipator`` - wraps a base projector to append **structured counterfactuals**
  via user-specified perturbations (and optional symmetric mirrors) controlled by
  **AnticipatorConfig**.

Core Contracts
--------------

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Component
     - Contract Summary
   * - ``Projector.project(s_t, branches)``
     - Deterministic mapping ``(D,) → (K, D)`` with ``K ≥ 2``; produces K futures from the current state.
   * - ``LinearProjector(weight, bias, span)``
     - Computes ``base = w * s_t + b``; adds linearly spaced deltas in ``[-span, span]`` and applies ``tanh``.
   * - ``ContrafactualAnticipator.generate(s_t)``
     - Concatenates the base future set with one row per perturbation (and, if enabled, its symmetric mirror).
   * - ``AnticipatorConfig(branches=3, symmetric=True)``
     - Controls base branch count for the inner projector and whether mirrored counterfactuals are included.

Minimal Execution Flow
----------------------

The following example combines the predictors module with the same end-to-end
pipeline used in the other modules: a minimal backend (from the :ref:`Core Module <core-example>`),
a one-node hypercausal graph (from the :ref:`Hypercausal Layer <hc-node-graph-example>`),
and loss evaluation (from :ref:`Loss Evaluation <loss-evaluation-example>`).  
On top of that baseline, we **add** a deterministic projector and a counterfactual
anticipator over the current state ``S_t`` to demonstrate seamless integration.

.. code-block:: python

   import numpy as np

   # --- Backend (reused from core) ---------------------------------------------
   # We reuse the minimal backend from the core module to focus this example on the
   # hypercausal layer behavior (nodes, graph, policy), not on backend details.
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

   # --- Hypercausal layer (HCNode/HCGraph instead of HCModel) ------------------
   # Rationale: HCModel is a high-level wrapper useful for training/sequence APIs.
   # Here we highlight the operational layer, so we use HCNode + HCGraph explicitly.
   from qmlhc.hc.node import HCNode
   from qmlhc.hc.graph import HCGraph
   from qmlhc.hc.policy import MeanPolicy

   node = HCNode(backend=backend, policy=MeanPolicy())   # minimal node (encode/run/project/select)
   graph = HCGraph.chain(names=["N1"], nodes=[node])     # 1-node linear graph (operational orchestration)

   # --- Single causal step ------------------------------------------------------
   x_t = np.array([1.0, 2.0, 3.0])
   s_map, s_hat_map, info_map = graph.step({"N1": x_t}, branches=3)

   # --- Loss module integration -------------------------------------------------
   # We now evaluate three complementary aspects:
   # (1) task accuracy w.r.t. a target, (2) inter-branch dispersion, (3) triadic stability.
   from qmlhc.loss.task import MSELoss          # external predictive error (pred vs target)
   from qmlhc.loss.coherence import CoherenceLoss   # dispersion across K futures (variance/mad)
   from qmlhc.loss.consistency import ConsistencyLoss  # coherence across (S_{t-1}, S_t, Ŝ_{t+1})

   target = np.array([0.3, 0.5, 0.8])

   task_mse = MSELoss()(s_hat_map["N1"], target)
   K_branches = info_map["N1"]["branches"]      # shape: (K, D)
   coh_var = CoherenceLoss(mode="variance")(K_branches)
   coh_mad = CoherenceLoss(mode="mad")(K_branches)
   prev_like = s_map["N1"]
   cons_tri = ConsistencyLoss(alpha=1.0, beta=1.0)(prev_like, s_map["N1"], s_hat_map["N1"])

   # --- NEW: Predictors integration (deterministic projector + counterfactuals) -
   from qmlhc.predictors.projector import LinearProjector
   from qmlhc.predictors.anticipator import ContrafactualAnticipator, AnticipatorConfig

   S_t = s_map["N1"]                                  # (D,) from the HC graph step
   proj = LinearProjector(weight=1.0, bias=0.0, span=0.2)
   proj_futures = proj.project(S_t, branches=4)       # (K, D), deterministic

   def small_push(center: np.ndarray) -> np.ndarray:
       return center + np.array([0.05, 0.00, -0.03], dtype=float)

   def rotate_like(center: np.ndarray) -> np.ndarray:
       return center[[1, 2, 0]]

   anticip = ContrafactualAnticipator(
       projector=proj,
       perturbations=[small_push, rotate_like],
       config=AnticipatorConfig(branches=4, symmetric=True),
   )
   fut_cf = anticip.generate(S_t)                      # (K', D), base + CF [+ mirrors]

   # (Optional) evaluate predictor outputs with the same losses
   s_hat_proj = proj_futures.mean(axis=0)
   s_hat_cf = fut_cf.mean(axis=0)

   task_mse_proj = MSELoss()(s_hat_proj, target)
   coh_var_proj = CoherenceLoss(mode="variance")(proj_futures)
   coh_mad_proj = CoherenceLoss(mode="mad")(proj_futures)

   task_mse_cf = MSELoss()(s_hat_cf, target)
   coh_var_cf = CoherenceLoss(mode="variance")(fut_cf)
   coh_mad_cf = CoherenceLoss(mode="mad")(fut_cf)

Expected Output (Variable)
--------------------------

.. code-block:: text

   Input: [1. 2. 3.]
   S_t: [0.26726124 0.53452248 0.80178373]
   Ŝ_{t+1} (mean policy, backend branches): [0.29789606 0.57910593 0.79579622]
   Backend branches shape: (3, 3) | Policy: MeanPolicy
   [Backend]  Task MSE: 0.002093 | Coherence Var: 0.001400 | MAD: 0.031374
   [Predictor|Linear]  MSE: 0.007606 | Var: 0.012926 | MAD: 0.099669
   [Predictor|CF]      MSE: 0.007606 | Var: 0.026953 | MAD: 0.123347
   Triadic Consistency (backend): 0.000987
   
Invariants and Numeric Conventions
----------------------------------

- ``branches`` is clamped/expected to satisfy ``K ≥ 2`` in projection flows.  
- Anticipators compute a **center** as the mean of base futures and append
  one row per perturbation (plus its mirror if ``symmetric=True``).
- Outputs are NumPy arrays suitable for downstream policies and loss evaluation.

Module References
-----------------

.. toctree::
   :maxdepth: 1
   :titlesonly:

   Anticipator <qmlhc.predictors.anticipator>
   Projector  <qmlhc.predictors.projector>
