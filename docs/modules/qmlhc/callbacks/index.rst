Callbacks Module
================

The **qmlhc.callbacks** package defines extensible interfaces and ready-to-use
callback utilities that enable real-time monitoring, control, and telemetry during
training, inference, or simulation.  
It formalizes how the system reacts to state changes and propagates structured
events such as step/epoch transitions or execution errors.

This module ensures modular observability and execution safety across
the entire QML-HCS stack, integrating seamlessly with components from
:ref:`Core Module <core-example>`, :ref:`Hypercausal Layer <hc-node-graph-example>`,
and :ref:`Loss Evaluation <loss-evaluation-example>`.

Architecture Overview
---------------------

::

   +----------------------------------+
   |        qmlhc.callbacks           |
   +----------------+-----------------+
                    |
       +------------v------------+
       |         base             |  -> Abstract callback contract
       +------------+-------------+
                    |
       +------------v------------+        +----------------------+
       |     depth_control       |  --->  |      telemetry       |
       | (execution depth guard) |        | (training logs, JSON)|
       +--------------------------+       +----------------------+

Each internal module contributes a specific behavioral role:

- ``base`` - defines the **Callback** base class that specifies all
  event hooks (`on_step_begin`, `on_step_end`, `on_epoch_end`, etc.).
- ``depth_control`` - implements **DepthScheduler**, a runtime safeguard
  that interpolates a numeric depth attribute (e.g., circuit depth)
  across epochs or steps for progressive control.
- ``telemetry`` - implements **TelemetryLogger** and **MemoryLogger**, which
  collect execution data either persistently (JSONL) or transiently (in memory).

Minimal Integration Example
---------------------------

The following example extends the :ref:`Loss Evaluation Example <loss-evaluation-example>`
by adding callback-based **telemetry** and **depth scheduling** to the same execution flow.  
This minimal snippet illustrates the implementation pattern - for a **fully runnable example**
(including backend, hypercausal graph, loss metrics, and callbacks), click the dropdown below.

.. raw:: html

   <details>
   <summary>🔽 Show full runnable example (backend + graph + loss + callbacks)</summary>

   <pre>
   <code class="language-python">
   import numpy as np

   # --- Backend (reused from core) ---------------------------------------------
   from qmlhc.core.backend import BackendConfig, QuantumBackend
   from qmlhc.core.registry import register_backend, create_backend

   class MyBackend(QuantumBackend):
       """Normalized input and stochastic projection backend."""
       def __init__(self, cfg: BackendConfig):
           super().__init__(cfg)
           self.depth = 1  # depth exposed for scheduling

       def run(self, params=None):
           x = self._require_input()
           return self._validate_state(x / np.linalg.norm(x))

       def project_future(self, s_t, branches=2):
           s_t = self._validate_state(s_t)
           # noise magnitude modulated by current depth
           noise = np.random.normal(scale=0.05 * float(self.depth),
                                    size=(branches, self.output_dim))
           fut = s_t + noise
           return self._validate_branches(fut)

   capabilities = {
       "backend_name": "MyBackend",
       "backend_version": "1.0",
       "max_qubits": 0,
       "output_dim": 3,
       "supports_shots": False,
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

   # --- Loss modules -----------------------------------------------------------
   from qmlhc.loss.task import MSELoss
   from qmlhc.loss.coherence import CoherenceLoss
   from qmlhc.loss.consistency import ConsistencyLoss

   x_t = np.array([1.0, 2.0, 3.0])
   target = np.array([0.3, 0.5, 0.8])

   s_map, s_hat_map, info_map = graph.step({"N1": x_t}, branches=3)
   task_mse = MSELoss()(s_hat_map["N1"], target)
   coh_var = CoherenceLoss(mode="variance")(info_map["N1"]["branches"])
   coh_mad = CoherenceLoss(mode="mad")(info_map["N1"]["branches"])
   prev_like = s_map["N1"]
   cons_tri = ConsistencyLoss()(prev_like, s_map["N1"], s_hat_map["N1"])

   # --- Callbacks integration --------------------------------------------------
   from qmlhc.callbacks.telemetry import TelemetryLogger
   from qmlhc.callbacks.depth_control import DepthScheduler

   telemetry = TelemetryLogger("run_telemetry.jsonl", flush_interval=1)
   depth_sched = DepthScheduler(target_attr="depth", start=1, end=3, epochs=3)

   telemetry.on_epoch_begin(0, {"info": "start"})
   depth_sched.on_epoch_begin(0, {"backend": backend})

   for step in range(3):
       telemetry.on_step_begin(step, {"epoch": 0, "depth": backend.depth})
       s_map, s_hat_map, info_map = graph.step({"N1": x_t}, branches=3)
       task_mse = MSELoss()(s_hat_map["N1"], target)
       coh_var = CoherenceLoss(mode="variance")(info_map["N1"]["branches"])
       telemetry.on_step_end(step, {"task_mse": float(task_mse),
                                   "coh_var": float(coh_var),
                                   "depth": backend.depth})

   telemetry.on_epoch_end(0, {"done": True})

   print("Telemetry stored in:", telemetry._path)
   print(f"Depth now at: {backend.depth}")
   </code>
   </pre>
   </details>

.. code-block:: python

   from qmlhc.callbacks.telemetry import TelemetryLogger
   from qmlhc.callbacks.depth_control import DepthScheduler

   # Instantiate callbacks
   telemetry = TelemetryLogger("run_telemetry.jsonl", flush_interval=1)
   depth_sched = DepthScheduler(target_attr="depth", start=1, end=3, epochs=3)

   # Integrate callbacks into evaluation loop
   for epoch in range(2):
       telemetry.on_epoch_begin(epoch, {"epoch": epoch})
       depth_sched.on_epoch_begin(epoch, {"backend": backend})
       for step in range(3):
           telemetry.on_step_begin(step, {"epoch": epoch, "depth": backend.depth})
           # ... perform computation (graph step, loss evaluation, etc.) ...
           telemetry.on_step_end(step, {"metric": 0.95, "depth": backend.depth})
       telemetry.on_epoch_end(epoch, {"notes": "completed"})

Expected Output
---------------

.. code-block:: text

   Telemetry stored in: run_telemetry.jsonl
   Contents (example):
   {"ts": 1736761273.91, "tag": "step_begin", "step": 0}
   {"ts": 1736761273.93, "tag": "step_end", "step": 0, "context": {"metric": 0.95, "depth": 1}}

Integration Context
-------------------

The callback layer can be used to:
- Monitor real-time evolution of hypercausal state propagation.
- Log losses, coherence, and consistency metrics during experiments.
- Enforce execution safety and dynamic control via ``DepthScheduler``.
- Provide synchronized observability across the full causal pipeline.

Extension Points
----------------

- **Custom Callbacks** - inherit from ``Callback`` and implement any subset
  of event hooks to inject custom monitoring or adaptive control logic.
- **Hybrid Logging** - combine ``TelemetryLogger`` for long runs with
  ``MemoryLogger`` for debugging short experiments.
- **Safety Controls** - integrate ``DepthScheduler`` to limit recursion
  or dynamically modulate depth during causal evolution.
- **Framework Integration** - callbacks can be plugged into any component
  of ``qmlhc.core`` or ``qmlhc.hc`` for synchronized observability.

Module References
-----------------

.. toctree::
   :maxdepth: 1
   :titlesonly:

   Callback Base Classes <qmlhc.callbacks.base>
   Depth Control Callback <qmlhc.callbacks.depth_control>
   Telemetry Callback <qmlhc.callbacks.telemetry>
