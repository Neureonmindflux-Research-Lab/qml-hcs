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
>>> from qmlhc.core.backend import BackendConfig
>>> from qmlhc.backends.pennylane_backend import PennyLaneBackend
>>> cfg = BackendConfig(output_dim=4, shots=None)
>>> be = PennyLaneBackend(cfg, num_qubits=4, device_name="default.qubit")
>>> be.encode(np.array([0.1, 0.2, 0.3, 0.4]))
>>> s_t = be.run()
>>> fut = be.project_future(s_t, branches=5)
>>> X = np.stack([np.zeros(4), np.ones(4)*0.1], axis=0)      # Example: batch execution
>>> be.run_batch(X).shape  # (B, D)
(2, 4)
>>> caps = be.capabilities()                          # Check device and backend capabilities (values depend on your device/config)
>>> caps["using_shots"], caps["supports_noise"]
(False, False)
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
    supports_noise : bool or None, optional
        Manual override for noise support in ``capabilities()``. If ``None`` (default),
        a heuristic is used based on the device; if ``True``/``False`` the reported
        capability is forced accordingly.  
    Raises
    ------
    ValueError
        If ``output_dim`` does not match ``num_qubits``.

    Notes
    -----
    API usage in brief (details in the RST):
    - Call ``encode(x)`` with shape ``(D,)`` before ``run``.
    - ``D`` must match ``num_qubits`` and ``config.output_dim``.
    - Analytic mode is deterministic (``shots=None``); sampling mode (``shots>0``) introduces variance.
    - ``run_batch(X)`` expects shape ``(B, D)`` and returns ``(B, D)``.
    - ``supports_noise`` (init arg) lets you override noise reporting in ``capabilities()``.

    """

    def __init__(
        self,
        config: BackendConfig,
        num_qubits: int,
        device_name: str = "default.mixed",  # "default.mixed" enables realistic noise (decoherence, damping); use "default.qubit" for ideal or "lightning.qubit" for faster pure-state simulation.
        shots: Optional[int] = None,
        supports_noise: Optional[bool] = None,
    ) -> None:
        super().__init__(config)
        self._num_qubits = int(num_qubits)

        dev_shots = shots if shots is not None else self._cfg.shots
        self._dev = qml.device(device_name, wires=self._num_qubits)
        self._supports_noise_override = supports_noise

        if self.output_dim != self._num_qubits:
            raise ValueError(
                "output_dim must match number of qubits for PennyLaneBackend"
            )

        # Define a compact circuit:
        # - RY rotations parameterized by the encoded input x
        # - Linear entanglement via CNOT gates
        # - Return Pauli-Z expectations per wire
       
        def circuit(x: Array) -> tuple[float, ...]:
            for i, val in enumerate(x):
                qml.RY(float(val), wires=i)
            for i in range(self._num_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            
            # Physical noise model 
            for i in range(self._num_qubits):
                qml.DepolarizingChannel(0.0012, wires=i)   # gate error
                qml.PhaseDamping(0.0020, wires=i)          # phase decoherence (T2)
                qml.AmplitudeDamping(0.0012, wires=i)      # energy relaxation (T1)

            return tuple(qml.expval(qml.PauliZ(i)) for i in range(self._num_qubits))

        qnode = qml.QNode(circuit, self._dev)
        self._shots_current = None

        if dev_shots is not None:
            qnode = qml.set_shots(qnode, dev_shots)
            self._shots_current = dev_shots
            
        self._circuit = qnode

    # ======================================================================
    # Contract methods
    # ======================================================================
    def run(self, params: Mapping[str, Any] | None = None) -> Array:
        """
        Execute the PennyLane circuit on the last encoded input.

        Parameters
        ----------
        params : dict or None, optional
            Unused in this minimal adapter; reserved for future extensions.
            
        Note
        ----
        ``encode(x)`` must be called beforehand. The base class enforces this
        via ``_require_input()`` and raises if the input is missing.

        Returns
        -------
        Array
            Validated state vector ``S_t`` of shape ``(D,)``.
        """
        x = self._require_input()
        out = np.asarray(self._circuit(x), dtype=float).reshape(-1)
        return self._validate_state(out)
    
    def run_batch(self, X: Array) -> Array:
        """
        Execute a batch of inputs with shape ``(B, D)`` and return a matrix of
        shape ``(B, D)``. Requires ``D == num_qubits``. Validation of the batch
        result is delegated to the base class.

        Note
        ----
        The batch must contain at least one row; empty batches are not supported.

        """
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self._num_qubits:
            raise ValueError(f"X must be (B, {self._num_qubits}), got {X.shape}.")
        outs = [np.asarray(self._circuit(x), dtype=float).reshape(-1) for x in X]
        return self._validate_branches(np.stack(outs, axis=0))

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
        Note
        ----
        This is a device-agnostic, low-cost projection utility. It does not
        change circuit parameters nor implement physical time evolution.

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
        Notes
        -----
        - ``max_qubits``: number of qubits configured for this instance.
        - ``supports_shots``: device family accepts finite shots (capability).
        - ``using_shots``: this instance currently samples (``shots`` is not None).
        - ``supports_noise``: noise support (override takes precedence).
        - ``supports_batch``: batch API is available via ``run_batch``.

        """
        caps = super().capabilities()
        caps.update(
            {
                "backend_name": "PennyLaneDevice",
                "backend_version": qml.__version__,
                "max_qubits": self._num_qubits,
                "supports_shots": True,  # PennyLane devices generally accept finite shots
                "using_shots": (getattr(self, "_shots_current", None) is not None),
                "supports_noise": (
                    self._supports_noise_override
                    if self._supports_noise_override is not None
                    else (hasattr(self._dev, "noise") or "default.mixed" in str(self._dev.name))
                ),
                "supports_batch": True,
                "gradient": GradientKind.PARAMETER_SHIFT,
            }
        )
        return caps
