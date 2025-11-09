Qiskit Backend
==============

.. currentmodule:: qmlhc.backends.qiskit_backend

Context and Purpose
-------------------
This adapter exposes Qiskit’s :class:`qiskit.primitives.Sampler` through the
:py:class:`~qmlhc.core.backend.QuantumBackend` interface. It builds (or accepts)
a parameterized circuit that encodes the last input vector ``x`` with per-qubit
``RY`` rotations and returns one signed average per measured wire (expectation-like
value derived from bitstring counts).

Conceptual Foundations
----------------------
1. **Shape stability (output–wire alignment):**
   The constructor enforces ``num_qubits == output_dim`` and raises ``ValueError``
   otherwise, ensuring each output index corresponds to a wire.

2. **Minimal default ansatz:**
   The fallback circuit applies one ``RY(angle)`` per wire and a final ``barrier``,
   then calls ``measure_all()`` to ensure classical bits are present for Sampler validation.
   Measurement can be implicit in some flows, but explicit ``measure_all()`` keeps versions compatible.

3. **Signed averages from counts:**
   The execution aggregates bitstrings into a per-wire signed mean
   (bit ``'1'`` → +1, bit ``'0'`` → −1). Because Qiskit bitstrings are **big-endian**,
   the code reverses the bitstring to map to wire indices ``0..N-1``.

4. **Bounded future projections:**
   :py:meth:`project_future` generates ``K`` branches by applying smooth additive
   perturbations in a linear range and squashing with ``tanh`` for numeric stability.

Integration Guidelines
----------------------
* Use this backend when you need **shot-based sampling** with a uniform interface
  across simulators and (optionally) providers that expose the Sampler API.
* Provide a custom ``circuit_builder(x) -> QuantumCircuit`` to introduce
  entanglement or domain-specific structure; the default circuit has **no**
  entangling gates.
* Keep ``output_dim`` and ``num_qubits`` equal to preserve shape invariants.
* Set a fixed random seed and an ideal simulator if you need repeatable sampling.

Installation
------------
Qiskit is included as an optional dependency of ``qml-hcs``.  
However, it is **recommended to install it manually** if you wish to test
a specific version or ensure compatibility with your preferred Qiskit release. 

Choose the appropriate installation depending on your environment:

- **Qiskit 1.x (legacy Sampler API):**

.. code-block:: bash

     pip install "qiskit>=1.2,<2.0"

- **Qiskit 2.x (modern StatevectorSampler API):**

.. code-block:: bash

     pip install "qiskit>=2.0"

The backend automatically detects the available version of Qiskit and imports
the correct primitive:

- For Qiskit 1.x → ``Sampler``
- For Qiskit 2.x → ``StatevectorSampler`` (used as a drop-in replacement)

Ensure Python ≥ 3.9 and that your environment includes ``qiskit.primitives``.


Version Compatibility
---------------------
- **Qiskit ≤1.x:** uses the classic :class:`qiskit.primitives.Sampler`.
  Results are typically exposed as ``quasi_dists`` (probabilities).
- **Qiskit ≥2.x:** the adapter falls back to :class:`qiskit.primitives.StatevectorSampler`.
  Circuits are passed as a list (``[qc]``). The default circuit adds ``measure_all()`` to
  provide classical bits required by stricter validators. Results are read from
  ``quasi_dists`` when present or from ``data.meas['counts']`` otherwise.

.. note::
   This backend is **not strictly deterministic**. Results depend on stochastic
   sampling via :class:`~qiskit.primitives.Sampler`. Determinism can only be
   approximated with an ideal simulator, fixed seeds, and sufficiently large
   shot counts. When ``config.shots`` is not provided, execution defaults to
   **1024 shots**.

Performance & Scaling Notes
---------------------------
- **Circuit depth:** linear in qubit count (one ``RY`` per wire + barrier).
- **Sampling cost:** dominated by the number of shots; empirical variance
  decreases roughly as :math:`O(1/\sqrt{N_{shots}})`.
- **Memory footprint:** compact, since post-processing stores wire-wise aggregates,
  not full distributions.

Extension Pathways
------------------
- **Custom ansatz:** pass ``circuit_builder`` to modify the circuit while keeping
  the backend contract unchanged.
- **Sampler/backend choice:** swap the underlying Sampler backend depending on
  your environment (e.g., Aer simulators, hardware providers that support Sampler).
- **Capability reporting:** override :py:meth:`capabilities` to expose provider-
  specific features (the default advertises name/version, qubit count, shots/noise
  flags, batch support, and :py:class:`~qmlhc.core.types.GradientKind.PARAMETER_SHIFT`).

Interoperability & Testing
--------------------------
- **Endianness handling:** the implementation reverses big-endian bitstrings when
  aggregating counts, preserving the mapping between bit position and wire index.
- **No extra branch validation:** :py:meth:`project_future` validates the input state
  and returns stacked branches directly; downstream checks are expected at higher layers.

Design Guarantees
-----------------
- Enforced output–wire alignment (constructor validation).
- Minimal, auditable default circuit (``RY`` per wire + barrier).
- Expectation-like signed averages computed from counts with explicit endianness handling.
- Bounded multi-branch projections via smooth perturbations and ``tanh`` squashing.

API Reference overview
----------------------
.. automodule:: qmlhc.backends.qiskit_backend
   :members:
   :undoc-members:
   :show-inheritance:

External and Internal Dependencies
----------------------------------
**Third-Party Libraries**

:mod:`qiskit`  
Provides :class:`qiskit.QuantumCircuit` and :class:`qiskit.primitives.Sampler`
used to build/execute circuits and retrieve sampled results.

- Documentation: https://qiskit.org/documentation/
- Repository: https://github.com/Qiskit/qiskit  
  

:mod:`numpy`  
Used for vector operations and utilities (e.g., ``linspace``, ``stack``, ``tanh``,
and array handling).

- Documentation: https://numpy.org/doc/stable/  
- Repository: https://github.com/numpy/numpy  
  

**Python Standard Library**

:py:mod:`__future__` – forward references for annotations.  
:py:mod:`typing` – typing hints for API clarity.

**QML-HCS Internal Modules**

:mod:`qmlhc.core.backend`  
Defines :class:`qmlhc.core.backend.QuantumBackend` and
:class:`qmlhc.core.backend.BackendConfig`, which this adapter implements/extends.

:mod:`qmlhc.core.types`  
Provides :class:`qmlhc.core.types.Array`, :class:`qmlhc.core.types.Capabilities`,
and :class:`qmlhc.core.types.GradientKind` used in method signatures, returns,
and capability metadata.
