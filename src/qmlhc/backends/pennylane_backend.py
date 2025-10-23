#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PennyLane Backend Adapter
-------------------------
Adapter for PennyLane-based quantum execution backends that conforms to the
``QuantumBackend`` contract.

This wrapper creates a PennyLane device, defines a lightweight variational
circuit (RY rotations + nearest-neighbor CNOT entanglement), and exposes
``run`` and ``project_future`` in the expected interface. The circuit returns
per-wire expectation values of Pauli-Z, yielding a vector ``S_t`` with
dimension equal to the number of qubits (and therefore to ``output_dim``).

Examples
--------
>>> import numpy as np
>>> from qmlhc.backends.pennylane_backend_adapter import PennyLaneBackend
>>> cfg = BackendConfig(output_dim=4, shots=None)
>>> be = PennyLaneBackend(cfg, num_qubits=4, device_name="default.qubit")
>>> be.encode(np.array([0.1, 0.2, 0.3, 0.4]))
>>> s_t = be.run()
>>> fut = be.project_future(s_t, branches=5)
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

import numpy as np
import pennylane as qml

from ..core.backend import QuantumBackend, BackendConfig
from ..core.types import Array, Capabilities, GradientKind


class PennyLaneBackend(QuantumBackend):
    """
    Wrap a PennyLane device and variational circuit into the ``QuantumBackend`` API.

    Parameters
    ----------
    config : BackendConfig
        Backend configuration (e.g., ``output_dim``, ``shots``).
    num_qubits : int
        Number of qubits / wires. Must equal ``config.output_dim``.
    device_name : str, optional
        PennyLane device name, by default ``"default.qubit"``.
    shots : int or None, optional
        Number of device shots (``None`` for analytic mode). If ``None``,
        falls back to ``config.shots``.

    Raises
    ------
    ValueError
        If ``output_dim`` does not match ``num_qubits``.
    """

    def __init__(
        self,
        config: BackendConfig,
        num_qubits: int,
        device_name: str = "default.qubit",
        shots: Optional[int] = None,
    ) -> None:
        super().__init__(config)
        self._num_qubits = int(num_qubits)

        dev_shots = shots if shots is not None else self._cfg.shots
        self._dev = qml.device(device_name, wires=self._num_qubits, shots=dev_shots)

        if self.output_dim != self._num_qubits:
            raise ValueError(
                "output_dim must match number of qubits for PennyLaneBackend"
            )

        # Define a compact circuit:
        # - RY rotations parameterized by the encoded input x
        # - Linear entanglement via CNOT gates
        # - Return Pauli-Z expectations per wire
        @qml.qnode(self._dev)
        def circuit(x: Array) -> tuple[float, ...]:
            for i, val in enumerate(x):
                qml.RY(float(val), wires=i)
            for i in range(self._num_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            return tuple(qml.expval(qml.PauliZ(i)) for i in range(self._num_qubits))

        self._circuit = circuit

    # ----------------------------------------------------------------------
    # Contract methods
    # ----------------------------------------------------------------------
    def run(self, params: Mapping[str, Any] | None = None) -> Array:
        """
        Execute the PennyLane circuit on the last encoded input.

        Parameters
        ----------
        params : dict or None, optional
            Unused in this minimal adapter; reserved for future extensions.

        Returns
        -------
        Array
            Validated state vector ``S_t`` of shape ``(D,)``.
        """
        x = self._require_input()
        out = np.asarray(self._circuit(x), dtype=float).reshape(-1)
        return self._validate_state(out)

    def project_future(self, s_t: np.ndarray, branches: int = 2) -> Array:
        """
        Generate future projections around ``s_t`` using smooth additive deltas
        followed by ``tanh`` for numeric stability.

        Parameters
        ----------
        s_t : np.ndarray
            Current state vector.
        branches : int, optional
            Number of future branches (K), by default 2.

        Returns
        -------
        Array
            Future states matrix of shape ``(K, D)``.
        """
        s = self._validate_state(s_t)
        k = max(2, int(branches))
        deltas = np.linspace(-0.12, 0.12, k, dtype=float)
        fut = np.stack([np.tanh(s + d) for d in deltas], axis=0)
        return self._validate_branches(fut)

    def capabilities(self) -> Capabilities:
        """
        Report merged capabilities (base + PennyLane device).

        Returns
        -------
        Capabilities
            Capability dictionary including device version, qubit count,
            shot/noise support, batching, and gradient method.
        """
        caps = super().capabilities()
        caps.update(
            {
                "backend_name": "PennyLaneDevice",
                "backend_version": qml.__version__,
                "max_qubits": self._num_qubits,
                "supports_shots": self._dev.shots is not None,
                "supports_noise": hasattr(self._dev, "noise")
                or "default.mixed" in str(self._dev.name),
                "supports_batch": True,
                "gradient": GradientKind.PARAMETER_SHIFT,
            }
        )
        return caps
