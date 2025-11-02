PennyLane Backend
=================

.. currentmodule:: qmlhc.backends.pennylane_backend

Context and Purpose
-------------------

The PennyLane backend serves as the canonical bridge between the
:py:class:`~qmlhc.core.backend.QuantumBackend` abstraction and a real
quantum circuit execution layer.  
It was created to ensure *interchangeability* across backends while
remaining minimal, deterministic, and easy to extend.  
Unlike traditional wrappers, this adapter enforces a strict one-to-one
correspondence between logical outputs and quantum wires, keeping the
model mathematically traceable and compatible with symbolic analyses
within :mod:`qmlhc`.

Conceptual Foundations
----------------------

1. **Deterministic state projection:**  
   Each forward evaluation represents a direct expectation mapping
   (deterministic when executed in analytic mode, i.e., shots=None),
   rather than probabilistic post-processing. This allows reproducible
   gradients and deterministic learning behavior in analytic mode.

2. **Linear entanglement philosophy:**  
   The internal circuit uses a nearest-neighbor entanglement pattern
   to maintain predictable causal flow and bounded circuit depth,
   which scales linearly with the qubit count.

3. **Output–wire equivalence:**  
   The enforced equality between ``output_dim`` and ``num_qubits`` is not
   a constraint by accident—it guarantees stable dimensional alignment
   for tensorized downstream processing and symbolic contraction across
   different backends.

4. **Future state projection model:**  
   :py:meth:`project_future` acts as a *temporal continuity operator*,
   designed for smooth evolution of latent quantum states. The
   hyperbolic tangent ensures bounded propagation when generating
   prospective branches.

Integration Guidelines
----------------------

* Use this backend when deterministic expectation vectors are required
  (e.g., during gradient computation or cross-backend comparisons).
* Avoid substituting it for high-depth or hardware-specific circuits;
  it prioritizes stability over expressiveness.
* Configuration inheritance from :py:class:`~qmlhc.core.backend.BackendConfig`
  allows uniform behavior with other backends (e.g., shot handling,
  analytic vs sampled execution).
* All internal validation routines come from
  :py:class:`~qmlhc.core.backend.QuantumBackend`, ensuring consistent
  error handling and output verification.

Performance & Scaling Notes
---------------------------

- Circuit depth grows as :math:`O(N)` with the number of qubits.  
- Memory footprint remains constant since expectation results are
  collapsed to a single vector rather than a full density matrix.  
- When executed with ``shots=None``, the device operates in analytic mode,
  bypassing stochastic noise and returning exact expectation values.

Extension Pathways
------------------

Developers can subclass :py:class:`PennyLaneBackend` to prototype
alternate ansätze, noise models, or topology variants.  
A typical extension might redefine the circuit body or override
:py:meth:`capabilities` to advertise custom device features.  
Subclasses should always preserve the validation flow defined by
:py:meth:`~qmlhc.core.backend.QuantumBackend._validate_state`
and :py:meth:`~qmlhc.core.backend.QuantumBackend._validate_branches`.

Interoperability & Testing
--------------------------

The backend has been validated against multiple PennyLane devices,
including analytic simulators (``default.qubit``) and mixed-state
variants (``default.mixed``).  
Consistency tests confirm that returned vectors stay within the
expected Pauli-Z range :math:`[-1, 1]`. Orthogonality properties
depend on the specific encoding and dataset and should be validated
per application.

Design Guarantees
-----------------

- Deterministic mapping between encoded vector and observable output  
- Predictable gradient availability via the parameter-shift rule  
- Full compatibility with batch and analytic execution modes  
- Stable projection dynamics via smooth nonlinear propagation

API Reference overview
----------------------

.. automodule:: qmlhc.backends.pennylane_backend
   :members:
   :undoc-members:
   :show-inheritance:

See Also
--------

- :py:class:`qmlhc.core.backend.QuantumBackend`
- :py:class:`qmlhc.core.backend.BackendConfig`
- :py:class:`qmlhc.core.types.Capabilities`

External and Internal Dependencies
----------------------------------

**Third-Party Libraries**

:mod:`pennylane`  
This backend is built on top of PennyLane, which provides the quantum device layer and the main tools for creating and running parametrized circuits and QNodes.  
See the official documentation or source repository for implementation details.

- Documentation: https://docs.pennylane.ai/en/stable  
- Repository: https://github.com/PennyLaneAI/pennylane

:mod:`numpy`  
NumPy is used throughout the backend for numerical operations and array management, including utilities such as ``asarray``, ``tanh``, ``linspace``, and ``stack``.  
Refer to the NumPy documentation or repository to explore its numerical capabilities and integration with this backend.

- Documentation: https://numpy.org/doc/stable/  
- Repository: https://github.com/numpy/numpy

**Python Standard Library**

:py:mod:`__future__`  
Used to enable forward-compatible annotations (``from __future__ import annotations``).

:py:mod:`typing`  
Provides standard type hints such as :py:class:`typing.Any`, :py:class:`typing.Mapping`, and :py:class:`typing.Optional`.

**QML-HCS Internal Modules**

:mod:`qmlhc.core.backend`  
Implements the shared backend contract through :class:`qmlhc.core.backend.QuantumBackend` and :class:`qmlhc.core.backend.BackendConfig`, which this adapter extends.

:mod:`qmlhc.core.types`  
Defines internal data structures like :class:`qmlhc.core.types.Array`, :class:`qmlhc.core.types.Capabilities`, and :class:`qmlhc.core.types.GradientKind`.

Together, these components make the PennyLane backend modular, reproducible, and consistent, combining the analytical precision of PennyLane and NumPy with the architectural stability of the QML-HCS core system.

