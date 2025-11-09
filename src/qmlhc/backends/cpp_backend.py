#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
C++ Backend Adapter
-------------------
Bridge layer for compiled C++ backends exposed via **pybind11**.

This adapter validates the extension interface and provides a consistent
Python-side abstraction layer compatible with the QuantumBackend contract.

The connected C++ module must implement the following minimal API:

    - ``encode(np.ndarray) -> None``
    - ``run(dict | None) -> np.ndarray`` of shape ``(D,)``
    - ``project_future(np.ndarray, int) -> np.ndarray`` of shape ``(K, D)``
    - ``capabilities() -> dict``

If any of these functions are missing, initialization will raise an
AttributeError. Output dimensions are verified to match the Python configuration.
"""

from __future__ import annotations

from typing import Any, Mapping
import numpy as np

from ..core.backend import QuantumBackend, BackendConfig
from ..core.types import Array, Capabilities, GradientKind


class CppBackend(QuantumBackend):
    """
    Adapter that bridges a compiled C++ backend exposed through **pybind11**.

    Parameters
    ----------
    config : BackendConfig
        Configuration object defining output dimensions and other parameters.
    bridge_module : Any
        Loaded pybind11 module exposing the required C++ backend interface.

    Raises
    ------
    AttributeError
        If the bridge module does not implement the required functions.
    ValueError
        If the output dimension reported by the C++ module does not match
        the expected Python configuration.
    """

    def __init__(self, config: BackendConfig, bridge_module: Any) -> None:
        super().__init__(config)
        self._m = bridge_module

        # Validate the C++ extension interface
        for fn in ("encode", "run", "project_future", "capabilities"):
            if not hasattr(self._m, fn):
                raise AttributeError(f"C++ bridge module missing required function '{fn}'")

        # Synchronize output_dim if exposed by the extension
        caps = self._m.capabilities() if callable(self._m.capabilities) else {}
        ext_dim = int(caps.get("output_dim", self.output_dim))
        if ext_dim != self.output_dim:
            raise ValueError(
                f"Output dimension mismatch: Python={self.output_dim}  C++={ext_dim}"
            )

    # ----------------------------------------------------------------------
    # Core API implementation
    # ----------------------------------------------------------------------
    def encode(self, x: Array) -> None:
        """
        Encode the input vector and forward it to the C++ backend.

        Parameters
        ----------
        x : Array
            Input vector of shape ``(D,)``.

        Raises
        ------
        ValueError
            If the input dimension does not match ``output_dim``.
        """
        arr = np.asarray(x, dtype=float).reshape(-1)
        if arr.size != self.output_dim:
            raise ValueError(f"Input size {arr.size} != output_dim {self.output_dim}")
        self._m.encode(arr)
        self._last_input = arr  # keep local copy for validation

    def run(self, params: Mapping[str, Any] | None = None) -> Array:
        """
        Execute the main backend operation.

        Parameters
        ----------
        params : dict or None, optional
            Optional dictionary of parameters for the C++ backend.

        Returns
        -------
        Array
            Output state vector ``S_t`` as a validated NumPy array.
        """
        out = np.asarray(
            self._m.run(dict(params) if params is not None else None),
            dtype=float,
        ).reshape(-1)
        return self._validate_state(out)

    def project_future(self, s_t: np.ndarray, branches: int = 2) -> Array:
        """
        Generate future projections using the C++ backend.

        Parameters
        ----------
        s_t : np.ndarray
            Current state vector.
        branches : int, optional
            Number of future branches to generate. Default is 2.

        Returns
        -------
        Array
            Matrix of projected future states with shape ``(K, D)``.
        """
        s = self._validate_state(s_t)
        fut = np.asarray(
            self._m.project_future(s, int(max(2, branches))), dtype=float
        )
        return self._validate_branches(fut)

    def capabilities(self) -> Capabilities:
        """
        Merge base and C++-reported capabilities.

        Returns
        -------
        Capabilities
            Dictionary describing backend capabilities such as:
            - backend name and version
            - supported features (batching, noise, etc.)
            - gradient support level
        """
        caps = super().capabilities()
        try:
            ext_caps = self._m.capabilities() or {}
        except Exception:
            ext_caps = {}

        caps.update(
            {
                "backend_name": ext_caps.get("backend_name", "CppBackend"),
                "backend_version": ext_caps.get("backend_version", "0"),
                "max_qubits": int(ext_caps.get("max_qubits", 0)),
                "supports_shots": bool(ext_caps.get("supports_shots", False)),
                "supports_noise": bool(ext_caps.get("supports_noise", False)),
                "supports_batch": bool(ext_caps.get("supports_batch", True)),
                "gradient": GradientKind(
                    ext_caps.get("gradient", GradientKind.NONE.value)
                ),
            }
        )
        return caps
