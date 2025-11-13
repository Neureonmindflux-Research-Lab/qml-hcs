Hypercausal Layer
=================

The **qmlhc.hc** package defines the operational layer responsible for representing, 
connecting, and evolving hypercausal structures within the QML-HCS framework. 
It bridges abstract model composition with concrete computational behavior, 
establishing the logic by which states, nodes, and projections interact dynamically 
during inference or simulation.

This layer provides a graph-based abstraction that supports multi-node causal 
propagation, node encapsulation of transformation logic, and flexible selection 
policies for aggregating future projections into a representative outcome.

Architecture Overview
---------------------

::

   +------------------------+
   |      qmlhc.hc          |
   +-----------+------------+
               |
       +-------v--------+        +--------------------+        +-----------------------+
       |     graph      |  --->  |       node         |  --->  |         policy        |
       | (graph of      |        | (hypercausal unit  |        |    (projection logic) |
       |  nodes & edges)|        |  or transformation)|        |(mean / median / risk) |
       +----------------+        +--------------------+        +-----------------------+

Each internal module cooperates to form a coherent causal propagation mechanism:

- ``graph`` - defines the **HCGraph**, a directed or recurrent structure managing 
  node dependencies, state routing, and update propagation.
- ``node`` - implements the **HCNode**, the minimal computational unit that encapsulates 
  forward evolution logic and connects to one or more backends.
- ``policy`` - defines **projection policies** that consolidate multiple candidate 
  futures into a single representative state (mean, median, or minimal-risk selection).

Relation with the Core Module
-----------------------------

The **qmlhc.core** and **qmlhc.hc** modules operate in complementary layers:

- The **core module** defines the *formal contracts and typed abstractions* that 
  specify how inputs, backends, and hypercausal nodes must interact (see 
  :doc:`Core Module <../core/index>` and :ref:`Core Module example <core-example>`).
- The **hypercausal module (hc)** implements the *dynamic execution model* by providing 
  graph structures, node execution, and projection-policy application over futures.

In short: ``qmlhc.core`` defines **what** must be done and **how components connect**;  
``qmlhc.hc`` defines **how those components evolve and interact in time**.

.. _hc-node-graph-example:

Minimal Execution Flow
----------------------

The following example reuses the **minimal backend** introduced in the 
*Core Module* example (now instantiated and wired into ``HCNode``/``HCGraph``) 
to perform a single hypercausal propagation step.

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

  # --- Output (stochastic; values vary across runs) ---------------------------
  print("Input:", x_t)
  print("S_t:", s_map["N1"])
  print("Ŝ_{t+1} (mean policy):", s_hat_map["N1"])
  print("Branches shape:", info_map["N1"]["branches"].shape)
  print("Policy:", info_map["N1"]["policy"])

Expected Output
---------------

.. code-block:: text

   Input: [1. 2. 3.]
   S_t: [0.26726124 0.53452248 0.80178373]
   Ŝ_{t+1} (mean policy): [0.26 ... 0.81 ...]
   Branches shape: (3, 3)
   Policy: mean

The numerical output may vary slightly between runs due to the stochastic projection.

Execution Context and Component Roles
-------------------------------------

This example illustrates the full propagation cycle within the hypercausal layer:

1. **Backend (`MyBackend`)** - Encodes and normalizes input, generates random 
   projections of possible futures.  
   *(Defined in :doc:`qmlhc.core.backend <qmlhc.core.backend>`)*

2. **Node (`HCNode`)** - Acts as the operational unit that executes one 
   causal step using the backend and a projection policy.  

3. **Graph (`HCGraph`)** - Manages causal dependencies and orchestrates 
   node execution through ordered propagation.  

4. **Policy (`MeanPolicy`)** - Aggregates multiple future projections 
   into a single representative state.  

Together, these components simulate a **hypercausal propagation process**, where the 
current state evolves into multiple possible futures that are statistically consolidated 
into one representative prediction.

Core Components
---------------

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - Component
     - Description
   * - ``HCGraph``
     - Manages nodes, causal links, and propagation logic between components; 
       supports directed or recurrent topologies and runtime execution.
   * - ``HCNode``
     - Encapsulates transformation and forward propagation rules within the 
       hypercausal graph by invoking backend execution and projection.
   * - ``ProjectionPolicy``
     - Defines the mechanism by which multiple candidate futures are reduced 
       to a single representative state.
   * - ``MeanPolicy`` / ``MedianPolicy`` / ``MinRiskPolicy``
     - Concrete implementations of projection policies that perform arithmetic, 
       statistical, or risk-based aggregation.

.. _core-hcnode-hcmodel-note:

.. note::
  **Scope and intended use of `HCNode` vs `HCModel`.**

  - **HCNode** (``qmlhc.hc.node.HCNode``) is the minimal operational unit that binds a
    backend and an optional projection policy. Its ``forward`` implements the full triad
    *(input → state → futures → representative)* and returns diagnostics including all
    branches and aggregation metadata. Use it for unit tests of backend+policy and inside
    graph executions (e.g., ``HCGraph.step``).

  - **HCModel** (``qmlhc.core.model.HCModel``) is a high-level orchestrator that composes
    one or multiple hypercausal nodes and exposes a uniform API for single-step, chained,
    and sequence processing (``forward``, ``forward_chain``, ``predict_sequence``).
    Prefer it in training/optimization pipelines, where a single call point simplifies
    loss evaluation, callbacks, and sequence handling.

  In short: **HCNode** focuses on the node-level triad and diagnostic fidelity; **HCModel**
  provides a convenient, sequence-aware interface for experiments and optimizers.

  For an applied illustration, see the :ref:`Full Hypercausal PennyLane Demo <ex-full-hypercausal-pennylane-demo>`,
  where both components operate together within an integrated optimization workflow.


Extension Points
----------------

- **Custom Nodes** - implement subclasses of ``HCNode`` or compatible callables.  
- **Graph Structures** - compose nodes via ``HCGraph`` to model multi-step dependencies.  
- **Custom Policies** - inherit from ``ProjectionPolicy`` to define new selection rules.  
- **Hybrid Execution** - integrate hypercausal nodes with backends defined in 
  ``qmlhc.core`` to enable mixed deterministic–quantum pipelines.

Invariants and Mathematical Coherence
-------------------------------------

- Node states maintain consistent dimensionality across propagation steps.
- Graph evaluation order is preserved via topological traversal or controlled feedback.  
- Projection policies ensure statistical coherence of aggregated futures.
- All computations are NumPy-based, enabling deterministic and reproducible semantics.

Module References
-----------------

.. toctree::
   :maxdepth: 1
   :titlesonly:

   Hypercausal Graph <qmlhc.hc.graph>
   Hypercausal Node <qmlhc.hc.node>
   Projection Policies <qmlhc.hc.policy>
