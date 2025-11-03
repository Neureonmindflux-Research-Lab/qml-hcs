#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Core Backend Base
-----------------
Backend interface implementations wired to the core Protocols.

This module defines:

- ``BackendConfig``: immutable configuration for a backend instance.
- ``QuantumBackend``: abstract base class that enforces the backend contract
  and offers shared validation utilities for inputs, states, and future branches.  
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

import numpy as np

from .types import (
    Array,
    Capabilities,
    GradientKind,
    QuantumBackend as QuantumBackendProtocol,
    RunInfo,        # imported for protocol compatibility
    TensorLike,
)


@dataclass(frozen=True)
class BackendConfig:
    """
    Immutable configuration for a backend instance.

    Parameters
    ----------
    output_dim : int
        Dimensionality of the state vector produced by the backend.
    shots : int or None, optional
        Number of execution shots (if the backend is stochastic). ``None`` means
        analytic mode. Default is ``None``.
    seed : int or None, optional
        Random seed used by stochastic backends. Default is ``None``.
    """
    output_dim: int
    shots: Optional[int] = None
    seed: Optional[int] = None


class QuantumBackend(QuantumBackendProtocol):
    """
    Abstract base class with shared validation utilities.

    Concrete adapters must implement ``run`` and ``project_future`` while using
    the helpers provided here to validate inputs and outputs.
    """

    def __init__(self, config: BackendConfig):
        """
        Initialize the backend with a given configuration.

        Parameters
        ----------
        config : BackendConfig
            Backend configuration (e.g., ``output_dim``, ``shots``).

        Raises
        ------
        ValueError
            If ``output_dim`` is not positive.
        """
        self._cfg = config
        self.output_dim: int = int(config.output_dim)
        if self.output_dim <= 0:
            raise ValueError("output_dim must be positive")

        self._last_input: Optional[Array] = None

    # ------------------------------------------------------------------
    # Contract methods (to be implemented/overridden by subclasses)
    # ------------------------------------------------------------------
    def encode(self, x: TensorLike) -> None:
        """
        Store and validate the last input vector ``x``.

        Parameters
        ----------
        x : TensorLike
            Input vector of shape ``(output_dim,)``.

        Raises
        ------
        ValueError
            If the input dimension does not match ``output_dim``.
        """
        arr = np.asarray(x, dtype=float).reshape(-1)
        if arr.size != self.output_dim:
            raise ValueError(f"input size {arr.size} != output_dim {self.output_dim}")
        self._last_input = arr

    def run(self, params: Mapping[str, Any] | None = None) -> Array:
        """
        Execute the backend on the last encoded input.

        Notes
        -----
        Must be overridden by concrete adapters.

        Parameters
        ----------
        params : Mapping[str, Any] or None, optional
            Optional parameter override for the backend execution.

        Returns
        -------
        Array
            Current state vector ``s_t`` of shape ``(output_dim,)``.
        """
        raise NotImplementedError("run() must be implemented by concrete backend")

    def project_future(self, s_t: TensorLike, branches: int = 2) -> Array:
        """
        Generate future state projections from the current state ``s_t``.

        Notes
        -----
        Must be overridden by concrete adapters.

        Parameters
        ----------
        s_t : TensorLike
            Current state vector.
        branches : int, optional
            Number of future branches (K), by default ``2``.

        Returns
        -------
        Array
            Future state matrix of shape ``(K, output_dim)``.
        """
        raise NotImplementedError("project_future() must be implemented by concrete backend")

    def capabilities(self) -> Capabilities:
        """
        Return a conservative default set of backend capabilities.

        Returns
        -------
        Capabilities
            Dictionary describing the backend's advertised features.
        """
        return {
            "backend_name": self.__class__.__name__,
            "backend_version": "0",
            "max_qubits": 0,
            "output_dim": self.output_dim,
            "supports_shots": self._cfg.shots is not None,
            "min_shots": 0 if self._cfg.shots is None else self._cfg.shots,
            "max_shots": 0 if self._cfg.shots is None else self._cfg.shots,
            "supports_noise": False,
            "supports_batch": False,
            "gradient": GradientKind.NONE,
            "notes": "contract-only base; provide a concrete adapter",
        }

    # ------------------------------------------------------------------
    # Helpers for subclasses
    # ------------------------------------------------------------------
    def _require_input(self) -> Array:
        """
        Return the last encoded input or raise if none is available.

        Returns
        -------
        Array
            Last encoded input vector.

        Raises
        ------
        RuntimeError
            If ``encode(x)`` has not been called before ``run()``.
        """
        if self._last_input is None:
            raise RuntimeError("encode(x) must be called before run()")
        return self._last_input

    def _validate_state(self, s: Array) -> Array:
        """
        Validate and reshape a state vector to ``(output_dim,)``.

        Parameters
        ----------
        s : Array
            Candidate state vector.

        Returns
        -------
        Array
            Validated state vector.

        Raises
        ------
        ValueError
            If the vector size does not match ``output_dim``.
        """
        arr = np.asarray(s, dtype=float).reshape(-1)
        if arr.size != self.output_dim:
            raise ValueError(f"state size {arr.size} != output_dim {self.output_dim}")
        return arr

    def _validate_branches(self, fut: Array) -> Array:
        """
        Validate a future-branches matrix of shape ``(K, output_dim)``.

        Parameters
        ----------
        fut : Array
            Candidate future branches.

        Returns
        -------
        Array
            Validated future branches matrix.

        Raises
        ------
        ValueError
            If the array is not 2D or its second dimension differs from
            ``output_dim``.
        """
        arr = np.asarray(fut, dtype=float)
        if arr.ndim != 2 or arr.shape[1] != self.output_dim:
            raise ValueError("future branches must have shape (K, output_dim)")
        return arr
