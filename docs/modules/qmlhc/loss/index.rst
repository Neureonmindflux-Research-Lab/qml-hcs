Loss Module
===========

The **qmlhc.loss** module defines the evaluation layer responsible for measuring 
deviations, stability, and coherence within the hypercausal computational process.  
It quantifies both **external discrepancies** (between predictions and targets) and 
**internal consistencies** (across states, projected futures, and temporal evolution).

This layer provides three complementary loss families, each corresponding to a specific 
aspect of hypercausal evaluation:

- **Task losses** - deterministic objectives such as MSE, MAE, or Cross-Entropy 
  for final predictive accuracy.
- **Coherence losses** - control dispersion across ``K`` projected futures using 
  variance or mean absolute deviation to maintain quantum-hypercausal alignment.
- **Consistency losses** - enforce triadic stability between past, present, and 
  projected future states, preserving temporal smoothness.

Architecture Overview
---------------------

::

   +---------------------------+
   |       qmlhc.loss          |
   +------------+--------------+
                |
        +-------v--------+        +-------------------+        +--------------------+
        |     task       |  --->  |    coherence      |  --->  |    consistency     |
        | (external loss |        | (K-branch         |        | (triadic temporal  |
        |  metrics)      |        |  dispersion)      |        |  stability)        |
        +----------------+        +-------------------+        +--------------------+

Cross-Module Flow
-----------------

::

     +---------------------------+
     |       qmlhc.core          |
     | (formal contracts & I/O)  |
     +-------------+-------------+
                   |
                   v
     +---------------------------+
     |        qmlhc.hc           |
     | (graph, nodes, policies)  |
     +-------------+-------------+
                   |
                   v
     +---------------------------+
     |       qmlhc.loss          |
     | (evaluation & metrics)    |
     +-------------+-------------+

Execution Summary:
- **core** defines validated data exchange and backend interfaces.
- **hc** performs causal propagation through nodes and graphs.
- **loss** quantifies deviations and stability from the produced states.

Together they form a complete hypercausal learning cycle:  
**generation → propagation → evaluation**

.. _loss-evaluation-example:

Minimal Evaluation Flow
-----------------------

The following example **continues the Hypercausal Layer example** and extends it by 
adding ``qmlhc.loss`` metrics over the same causal step.  
We **reuse** the :ref:`backend <core-example>` defined in the Core Module, along with 
the same node and :ref:`graph <hc-node-graph-example>`, and then evaluate task accuracy, 
inter-branch coherence, and triadic consistency.

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

   # Define a target for evaluation (example purpose)
   target = np.array([0.3, 0.5, 0.8])

   # (1) Task loss on the representative next state
   task_mse = MSELoss()(s_hat_map["N1"], target)

   # (2) Inter-branch coherence (lower is better): controls dispersion across K projected futures
   K_branches = info_map["N1"]["branches"]      # shape: (K, D)
   coh_var = CoherenceLoss(mode="variance")(K_branches)
   coh_mad = CoherenceLoss(mode="mad")(K_branches)

   # (3) Triadic consistency: encourage smooth evolution prev→curr and curr→future
   # For demonstration, use prev = curr (single-step run); in sequences, pass the real S_{t-1}.
   prev_like = s_map["N1"]
   cons_tri = ConsistencyLoss(alpha=1.0, beta=1.0)(prev_like, s_map["N1"], s_hat_map["N1"])

   # --- Output (stochastic; values vary across runs) ---------------------------
   print("Input:", x_t)
   print("S_t:", s_map["N1"])
   print("Ŝ_{t+1} (mean policy):", s_hat_map["N1"])
   print("Branches shape:", K_branches.shape, "| Policy:", info_map["N1"]["policy"])
   print(f"Task MSE: {task_mse:.6f}")
   print(f"Coherence (variance): {coh_var:.6f} | Coherence (MAD): {coh_mad:.6f}")
   print(f"Triadic Consistency: {cons_tri:.6f}")

Expected Output
---------------

.. code-block:: text

   Input: [1. 2. 3.]
   S_t: [0.26726124 0.53452248 0.80178373]
   Ŝ_{t+1} (mean policy): [0.26 ... 0.81 ...]
   Branches shape: (3, 3) | Policy: mean
   Task MSE: 0.00xxx
   Coherence (variance): 0.00xxx | Coherence (MAD): 0.00xxx
   Triadic Consistency: 0.00xxx

The numerical values vary between runs due to stochastic projection noise.  
This demonstrates how the **loss module** quantifies external deviation, 
inter-branch dispersion, and temporal smoothness on top of the same hypercausal step.

Core Components
---------------

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - Component
     - Description
   * - ``task.MSELoss``, ``task.MAELoss``, ``task.CrossEntropyLoss``
     - Deterministic task-level objectives for predictive accuracy. :contentReference[oaicite:6]{index=6}
   * - ``coherence.CoherenceLoss(mode="variance"|"mad")``
     - Inter-branch dispersion penalty over futures ``(K, D)`` (lower is better). :contentReference[oaicite:7]{index=7}
   * - ``consistency.ConsistencyLoss(alpha, beta)``
     - Triadic loss over ``(S_{t-1}, S_t, Ŝ_{t+1})`` encouraging smooth transitions. :contentReference[oaicite:8]{index=8}

Extension Points
----------------

- **Custom task losses** - follow the callable pattern from ``task.py``  
  (``pred, target → scalar``) to define new deterministic objectives.
- **Alternative coherence criteria** - extend ``CoherenceLoss`` with additional 
  dispersion or stability metrics for projected futures.
- **Temporal variants** - adapt ``ConsistencyLoss`` weighting or compose multi-step 
  variants to model extended temporal dependencies.

Invariants and Numeric Conventions
----------------------------------

- Branch matrices must be shaped as ``(K, D)`` with ``K ≥ 2`` for coherence evaluation.  
- Triadic consistency requires equal dimensionality across ``S_{t-1}, S_t, Ŝ_{t+1}``.  
- Task losses flatten and validate all inputs before computing scalar objectives.  
- All losses are computed deterministically using **NumPy** as the reference backend 
  (reproducible given fixed random seeds).

Module References
-----------------

.. toctree::
   :maxdepth: 1
   :titlesonly:

   Task Losses <qmlhc.loss.task>
   Coherence Loss <qmlhc.loss.coherence>
   Consistency Loss <qmlhc.loss.consistency>
