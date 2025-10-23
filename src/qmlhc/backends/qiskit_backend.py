#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qiskit Backend Adapter
----------------------
Adapter for Qiskit-based quantum execution backends that conforms to the
``QuantumBackend`` contract.

This wrapper uses Qiskit's ``Sampler`` primitive to execute a (possibly
custom) parameterized circuit that encodes the last input ``x`` via RY
rotations. It returns an expectation-like vector per wire derived from
measured bitstring counts.

Example
-------
>>> import numpy as np
>>> from qmlhc.backends.qiskit_backend_adapter import QiskitBackend
>>> cfg = BackendConfig(output_dim=3, shots=1024)
>>> be = QiskitBackend(cfg, num_qubits=3)
>>> be.encode(np.array([0.1, 0.2, 0.3]))
>>> s_t = be.run()
>>> fut = be.project_future(s_t, branches=5)
"""

from __future__ import annotations

from typing import Any, Callable, Mapping, Optional

import numpy as np

from qiskit import QuantumCircuit
from qiskit.primitives import Sampler

from ..core.backend import QuantumBackend, BackendConfig
from ..core.types import Array, Capabilities, GradientKind


class QiskitBackend(QuantumBackend):
    """
    Wrap Qiskit primitives into the ``QuantumBackend`` API.

    Parameters
    ----------
    config : BackendConfig
        Backend configuration (e.g., ``output_dim``, ``shots``).
    num_qubits : int
        Number of qubits / wires. Must equal ``config.output_dim``.
    circuit_builder : Callable[[np.ndarray], QuantumCircuit], optional
        Custom function that builds a circuit from the encoded input vector
        ``x``. If not provided, a default RY + barrier circuit is used.

    Raises
    ------
    ValueError
        If ``output_dim`` does not match ``num_qubits``.
    """

    def __init__(
        self,
        config: BackendConfig,
        num_qubits: int,
        circuit_builder: Optional[Callable[[np.ndarray], QuantumCircuit]] = None,
    ):
        super().__init__(config)
        self._num_qubits = int(num_qubits)
        self._sampler = Sampler()
        self._circuit_builder = circuit_builder or self._default_circuit

        if self.output_dim != self._num_qubits:
            raise ValueError("output_dim must match number of qubits for QiskitBackend")

    def _default_circuit(self, x: np.ndarray) -> QuantumCircuit:
        """
        Build a default encoding circuit using RY rotations followed by a barrier.

        Parameters
        ----------
        x : np.ndarray
            Encoded input vector of shape ``(D,)``, one angle per qubit.

        Returns
        -------
        QuantumCircuit
            The constructed circuit.
        """
        qc = QuantumCircuit(self._num_qubits)
        for i, val in enumerate(x):
            qc.ry(float(val), i)
        qc.barrier()
        # Measurement is handled implicitly by Sampler in modern Qiskit;
        # counts/expectations are inferred from the primitive result.
        return qc

    # ----------------------------------------------------------------------
    # Contract methods
    # ----------------------------------------------------------------------
    def run(self, params: Mapping[str, Any] | None = None) -> Array:
        """
        Execute the circuit via Qiskit's Sampler and return an expectation-like vector.

        Notes
        -----
        The vector is computed from bitstring counts as a signed average per wire:
        for each bitstring, a '1' contributes +1 and a '0' contributes -1.

        Parameters
        ----------
        params : dict or None, optional
            Unused in this minimal adapter; reserved for future extensions.

        Returns
        -------
        Array
            Vector of shape ``(D,)`` containing per-wire signed averages.
        """
        x = self._require_input()
        qc = self._circuit_builder(x)

        # Use provided shots if available; otherwise default to 1024
        result = self._sampler.run(qc, shots=self._cfg.shots or 1024).result()

        # Extract counts from the Sampler result and compute signed averages
        counts = result[0].data.meas.get("counts", {})
        vec = np.zeros(self._num_qubits, dtype=float)
        total = sum(counts.values()) or 1
        for bitstring, freq in counts.items():
            # Qiskit uses big-endian bitstrings; reverse to map to wire index 0..n-1
            for i, bit in enumerate(reversed(bitstring)):
                vec[i] += (1.0 if bit == "1" else -1.0) * freq
        vec /= total
        return vec

    def project_future(self, s_t: np.ndarray, branches: int = 2) -> Array:
        """
        Generate future state projections by applying smooth additive perturbations.

        Parameters
        ----------
        s_t : np.ndarray
            Current state vector.
        branches : int, optional
            Number of projected branches (K), by default 2.

        Returns
        -------
        Array
            Matrix of shape ``(K, D)`` with ``tanh``-stabilized perturbations.
        """
        s_t = self._validate_state(s_t)
        k = max(2, int(branches))
        noise = np.linspace(-0.1, 0.1, k, dtype=float)
        futures = np.stack([np.tanh(s_t + n) for n in noise], axis=0)
        return futures

    def capabilities(self) -> Capabilities:
        """
        Report merged capabilities (base + Qiskit-specific).

        Returns
        -------
        Capabilities
            Capability dictionary including backend name/version, qubit count,
            shot/noise support, batching, and gradient method.
        """
        caps = super().capabilities()
        caps.update(
            {
                "backend_name": "QiskitSampler",
                "backend_version": "1.x",
                "max_qubits": self._num_qubits,
                "supports_shots": True,
                "supports_noise": True,
                "supports_batch": True,
                "gradient": GradientKind.PARAMETER_SHIFT,
            }
        )
        return caps
