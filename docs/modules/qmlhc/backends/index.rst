Backends
========

Overview
--------
The QML-HCS backend adapters form the computational core of the framework.  
They provide a consistent and extensible interface that connects the abstract
:py:class:`~qmlhc.core.backend.QuantumBackend` layer with multiple quantum,
hybrid, and compiled execution environments.

This architecture ensures that every backend—whether analytic, sampled,
or natively compiled—follows a unified API, making it possible to switch
between simulation, hardware, and high-performance native computation
without altering the high-level model definition.

Backend Adapter Family
----------------------
The current backend family integrates three complementary paradigms of execution,
each designed to serve a distinct purpose within the QML-HCS ecosystem.

**PennyLane Backend**  
A deterministic, analytic simulator built on :mod:`pennylane`.  
It supports symbolic gradients, automatic differentiation, and reproducible
expectation-value computations—ideal for theory validation and hybrid modeling.

.. note::
   **Compatibility (PennyLane ≥ 0.39)**
   This backend uses the new ``qml.set_shots(...)`` API for shot configuration.
   Older versions (< 0.37) used ``qml.transforms.set_shots(...)``, which has
   been deprecated in PennyLane 0.39+.

**Qiskit Backend**  
A stochastic, shot-based adapter built on :mod:`qiskit.primitives.Sampler`.  
It reproduces the statistical behavior of real quantum devices by averaging
bitstring counts, offering realism for prototyping and experimentation.

**C++ Backend (pybind11)**  
A high-performance deterministic adapter that bridges native C++ quantum engines
or numerical solvers through :mod:`pybind11`.  
It enforces strict interface validation and offers native-level speed suitable
for production pipelines and custom HPC integrations.

Comparative Overview
--------------------
A concise operational comparison of all backend adapters is presented below.

.. table::
   :widths: 18 20 18 20 30
   :align: center

   +----------------------+-----------------------------+---------------------------+---------------------------+--------------------------------------------------------------+
   | **Backend**          | **Execution Model**         | **Deterministic Mode**    | **Stochastic Mode**       | **Distinctive Features**                                     |
   +======================+=============================+===========================+===========================+==============================================================+
   | **PennyLane**        | Analytic or sampled via     | ✔️ Analytic (shots=None)  | ✔️ Shot-based (shots>0)   | • Linear entanglement (CNOT chain)                           |
   |                      | configurable **QNode** shots|                           |                           | • Symbolic gradient support (parameter-shift)                |
   |                      |                             |                           |                           | • Optional stochastic execution via device configuration     |
   +----------------------+-----------------------------+---------------------------+---------------------------+--------------------------------------------------------------+
   | **Qiskit**           | Shot-based execution using  | ❌ Unsupported            | ✔️ Always shot-based      | • Uses ``Sampler`` primitive for probabilistic results       |
   |                      | the ``Sampler`` primitive   |                           |                           | • Big-endian correction for wire mapping                     |
   |                      |                             |                           |                           | • Expectation-like averages per qubit                        |
   +----------------------+-----------------------------+---------------------------+---------------------------+--------------------------------------------------------------+
   | **C++ (pybind11)**   | Compiled, native execution  | ✔️ Deterministic only     | ❌ Unsupported            | • C++17 native-level speed                                   |
   |                      | (no sampling mechanism)     |                           |                           | • Strict validation and stable numeric determinism           |
   |                      |                             |                           |                           | • Designed for reproducible HPC workflows                    |
   +----------------------+-----------------------------+---------------------------+---------------------------+--------------------------------------------------------------+


Integration Philosophy
----------------------
All backends conform to the same abstract computational contract, ensuring
predictable behavior and portability across analytic, stochastic, and compiled
environments. The interface is minimal, consistent, and stable across releases:

- **`encode(x)`** – prepares the input vector for execution.  
- **`run()`** – performs the circuit evaluation or simulation.  
- **`project_future(s_t)`** – generates predictive or perturbed state branches.  
- **`capabilities()`** – reports backend metadata such as device, gradient mode,
  and resource constraints.

This unified structure guarantees architectural consistency and mathematical
equivalence across diverse backends, enabling cross-platform experiments and
mixed-model workflows.

.. _nonstationary_env_hcs:

Non-Stationary Environments & Hypercausal Adaptation
----------------------------------------------------

Backends in QML-HCS are deliberately deterministic (analytic) or shot-based
(stochastic) at the *physics layer*. Non-stationarity is applied *outside* the
backend by the hypercausal control loop:

- **Drift injection**: an exogenous, time-varying perturbation is added to the
  model input (e.g., ``x + drift(t)``) to emulate environment shifts.
- **Observation**: the backend returns deterministic expectation vectors
  (analytic) or averaged shot-based estimates; coherence/consistency losses
  quantify temporal stability.
- **Adaptation loop**: a hypercausal controller (e.g., a feedback gain ``alpha``)
  updates per-epoch from losses, shaping the model’s response to the drift.

This separation keeps the quantum evolution reproducible and traceable while
enabling true non-stationary behavior at the system level.

Minimal Non-Stationary Loop (Analytic Backend)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   backend = PennyLaneBackend(cfg)          # analytic/deterministic engine
   model = HCModel([HCNode(backend)])
   alpha, lr, damp = 1.0, 0.02, 0.001

   for epoch in range(epochs):
       drift = drift_amp * np.sin(2*np.pi*epoch/epochs)
       s_t, s_hat, info = model.forward(x + drift, s_tm1, branches)
       loss = task_loss(s_t, target) + 0.5*(cons_loss(s_tm1, s_t, s_hat)
                                            + coh_loss(info["branches"]))
       grad  = np.clip(loss / 1e-3, -100.0, 100.0)
       alpha = alpha - lr*grad - damp*alpha

.. note::

   For the complete implementation of this loop, including telemetry,
   depth scheduling, detailed plotting, and CSV logging, see the full
   example page:

   :doc:`/examples/ex_full_hypercausal_Pennylane_demo`

   That page shows the entire workflow as implemented in code, with both
   the execution process and the resulting visual outputs.

   
Submodules
----------
.. toctree::
   :maxdepth: 1
   :titlesonly:

   C++ Backend (pybind11) <qmlhc.backends.cpp_backend>
   PennyLane Backend <qmlhc.backends.pennylane_backend>
   Qiskit Backend <qmlhc.backends.qiskit_backend>

External Documentation
----------------------
For extended specifications and upstream references, see the following resources:

:mod:`pennylane`  
This backend is built on top of PennyLane, which provides the quantum device layer
and the main tools for creating and running parametrized circuits and QNodes.  
See the official documentation or source repository for implementation details.

- Documentation: https://docs.pennylane.ai/en/stable  
- Repository: https://github.com/PennyLaneAI/pennylane

:mod:`qiskit`  
The Qiskit backend integrates IBM’s Sampler API to perform realistic shot-based
executions and expectation estimation.  
Refer to the following resources for API and backend architecture details.

- Documentation: https://qiskit.org/documentation/  
- Repository: https://github.com/Qiskit/qiskit  

:mod:`pybind11`  
Used to expose compiled C++ code to Python, forming the bridge for the C++ backend.  
It provides modern C++17 bindings, reference-counted memory sharing, and
zero-copy NumPy array interoperability.

- Documentation: https://pybind11.readthedocs.io  
- Repository: https://github.com/pybind/pybind11  

:mod:`numpy`  
Employed across all adapters for array validation, shape consistency, and
floating-point exchange between frameworks.  

- Documentation: https://numpy.org/doc/stable/  
- Repository: https://github.com/numpy/numpy  

:mod:`qmlhc`  
The core framework defining the :class:`~qmlhc.core.backend.QuantumBackend` interface,
configuration classes, and unified capability schemas used across all adapters.

- Documentation: https://qml-hcs.readthedocs.io  
- Repository: https://github.com/NeureonMindFlux-Lab/qmlhc
