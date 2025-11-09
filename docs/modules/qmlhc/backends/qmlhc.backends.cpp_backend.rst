C++ Backend (pybind11)
======================

.. currentmodule:: qmlhc.backends.cpp_backend

Context and Purpose
-------------------
This adapter provides a standardized Python-side interface for **compiled C++ backends**
exposed through :mod:`pybind11`. It ensures that any linked extension conforms to the
:py:class:`~qmlhc.core.backend.QuantumBackend` contract and validates its exported API
before execution.

Conceptually, it acts as a **bridge layer** that allows fast, natively compiled quantum
or numerical engines to interoperate seamlessly with the Python-based QML-HCS ecosystem.

Conceptual Foundations
----------------------
1. **Strict interface validation:**  
   The constructor checks that the provided C++ module implements the required
   functions: ``encode``, ``run``, ``project_future``, and ``capabilities``.  
   Missing definitions raise an immediate :class:`AttributeError`.

2. **Dimensional consistency:**  
   If the C++ extension exposes an ``output_dim`` via its reported capabilities,
   the Python configuration is verified to match, preventing shape mismatches.

3. **Deterministic execution:**  
   Unlike shot-based simulators, this backend produces reproducible results for the
   same inputs, unless the native module introduces randomness internally.

4. **Capability merging:**  
   The adapter merges the Python and C++ capability dictionaries, ensuring unified
   metadata reporting (backend name/version, qubit limits, batching, gradients, etc.).

Integration Guidelines
----------------------
* Use this backend when bridging high-performance compiled modules (C/C++) that
  expose their own simulation or inference logic.
* Always verify that the C++ extension was compiled with **pybind11** and exposes
  the required callable symbols.
* The backend assumes **synchronous, deterministic** execution — no sampling, noise,
  or asynchronous computation is handled automatically.
* Capability metadata should be exposed in the C++ module via a ``capabilities()``
  method returning a JSON-like dictionary.

.. note::
   Because execution is fully delegated to a compiled module, performance,
   determinism, and numerical precision depend entirely on the native C++ backend.
   The Python adapter performs only validation, shape checking, and capability merging.

Performance & Scaling Notes
---------------------------
- **Overhead:** minimal (one NumPy serialization per call).  
- **Latency:** dominated by the compiled C++ core.  
- **Parallelism:** depends on the compiled implementation; the adapter itself
  executes sequentially.  
- **Error handling:** Python exceptions (e.g., :class:`ValueError`, :class:`AttributeError`)
  are raised if the bridge violates the expected interface or dimension constraints.

Extension Pathways
------------------
- **Hybrid computation:** combine C++ backends with higher-level Python orchestration
  for training, simulation, or inference pipelines.  
- **Capability enrichment:** extend the native C++ side to report gradient modes,
  qubit limits, or hardware-specific features via ``capabilities()``.  
- **Batch support:** expose batched encodings and runs at the C++ level to enable
  vectorized execution.

Example
-------
>>> import cpp_backend_bridge  # compiled pybind11 module
>>> from qmlhc.core.backend import BackendConfig
>>> from qmlhc.backends.cpp_backend import CppBackend
>>> cfg = BackendConfig(output_dim=4)
>>> backend = CppBackend(cfg, cpp_backend_bridge)
>>> backend.encode(np.array([0.1, 0.2, 0.3, 0.4]))
>>> y = backend.run()
>>> fut = backend.project_future(y, branches=3)
>>> y.shape, fut.shape
((4,), (3, 4))

Notes
-----
This example illustrates the minimal workflow for using a compiled backend
through the unified interface:

1. Instantiate :class:`CppBackend` with the Python configuration and the loaded
   C++ bridge module.
2. Encode an input vector and invoke ``run()``.
3. Optionally, generate projections with ``project_future()``.

As all computation is handled natively, the adapter introduces no stochastic
variance—outputs are deterministic and reproducible, assuming a fixed compiled backend.

External and Internal Dependencies
----------------------------------
**Third-Party Libraries**

:mod:`numpy`  
Used for array validation, reshaping, and numerical consistency.

- Documentation: https://numpy.org/doc/stable/  
- Repository: https://github.com/numpy/numpy

:mod:`pybind11`  
Required on the C++ side to expose the compiled backend functions to Python.

- Documentation: https://pybind11.readthedocs.io  
- Repository: https://github.com/pybind/pybind11

**Python Standard Library**

:py:mod:`typing` — type hints for clarity and contract definition.  
:py:mod:`__future__` — annotation compatibility.

**QML-HCS Internal Modules**

:mod:`qmlhc.core.backend`  
Defines the base :class:`QuantumBackend` contract and :class:`BackendConfig` used
for validation and configuration.

:mod:`qmlhc.core.types`  
Provides :class:`Array`, :class:`Capabilities`, and
:class:`GradientKind` used for interface typing and metadata consistency.

Design Guarantees
-----------------
- Enforces interface presence (`encode`, `run`, `project_future`, `capabilities`).  
- Synchronizes output dimensions between Python and native layers.  
- Validates and reshapes NumPy arrays for safety and determinism.  
- Integrates natively compiled backends while maintaining consistent API semantics.

API Reference
-------------
.. automodule:: qmlhc.backends.cpp_backend
   :members:
   :undoc-members:
   :show-inheritance:
