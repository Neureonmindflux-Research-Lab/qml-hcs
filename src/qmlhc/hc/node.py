#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hypercausal Node
----------------
Triadic forward propagation linking past, present, and projected future states.

Each node binds a ``QuantumBackend`` (which handles state evolution) and an
optional ``ProjectionPolicy`` (which determines how multiple candidate futures
are aggregated into a single representative future).

Implements the ``HypercausalNode`` protocol.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Tuple

import numpy as np

from ..core.types import (
    Array,
    HypercausalNode as HypercausalNodeProtocol,
    ProjectionPolicy,
    QuantumBackend,
    TensorLike,
)


@dataclass(frozen=True)
class NodeConfig:
    """
    Static configuration for a hypercausal node.

    Parameters
    ----------
    branches : int, optional
        Number of candidate futures (K) to generate from the backend, by default ``2``.
    """
    branches: int = 2


class HCNode(HypercausalNodeProtocol):
    """
    Concrete implementation of a hypercausal node.

    This node combines a ``QuantumBackend``, which generates both the current and
    future states, together with an optional ``ProjectionPolicy`` that aggregates
    multiple candidate futures into a single representative vector.


    Notes
    -----
    The ``forward`` method implements the triadic relation:
    ``(past ↔ present ↔ projected futures)``.
    """

    def __init__(
        self,
        backend: QuantumBackend,
        policy: Optional[ProjectionPolicy] = None,
        config: Optional[NodeConfig] = None,
    ) -> None:
        """
        Initialize a hypercausal node.

        Parameters
        ----------
        backend : QuantumBackend
            Quantum backend responsible for computing state transitions.
        policy : ProjectionPolicy or None, optional
            Strategy for selecting a representative future. If ``None``,
            the mean across branches is used.
        config : NodeConfig or None, optional
            Static configuration object, by default ``NodeConfig()``.
        """
        self._backend = backend
        self._policy = policy
        self._cfg = config or NodeConfig()

    # ------------------------------------------------------------------
    # Core computation
    # ------------------------------------------------------------------
    def forward(
        self,
        x_t: TensorLike,
        s_tm1: TensorLike | None,
        branches: int = 2,
    ) -> Tuple[Array, Array, Mapping[str, Any]]:
        """
        Perform one forward pass through the node.

        Steps
        -----
        1. Encode the current input ``x_t`` into the backend.
        2. Execute the backend to compute the current compact state ``S_t``.
        3. Generate K candidate futures ``F = {S_{t+1}^k}``.
        4. Select a representative future ``Ŝ_{t+1}`` using the policy.

        Parameters
        ----------
        x_t : TensorLike
            Current input at time ``t``.
        s_tm1 : TensorLike or None
            Previous state at time ``t-1`` (may be ``None`` for the first step).
        branches : int, optional
            Number of candidate futures to generate, by default ``2``.

        Returns
        -------
        tuple
            ``(S_t, Ŝ_{t+1}, info)`` where:
            - ``S_t`` : Array
            Current state vector.
            - ``Ŝ_{t+1}`` : Array
            Aggregated representative future.
            - ``info`` : dict
            Additional metadata including:
            -``s_tm1`` : previous state (if provided)
            - ``branches`` : matrix of candidate futures
            - ``policy`` : policy name
            - ``diagnostics`` : policy diagnostics or aggregation info
        """
        # Encode current input into the backend
        self._backend.encode(x_t)

        # Compute present compact state S_t
        s_t: Array = self._backend.run()

        # Determine number of candidate futures
        k = max(branches, 2) if branches is not None else max(self._cfg.branches, 2)

        # Generate K candidate futures
        futures: Array = self._backend.project_future(s_t, branches=k)

        # Apply the provided policy (or default mean aggregation)
        if self._policy is not None:
            s_tp1_hat, diag = self._policy.select(futures)
            policy_name = getattr(self._policy, "__class__", type(self._policy)).__name__
        else:
            s_tp1_hat = futures.mean(axis=0)
            diag = {"aggregation": "mean"}
            policy_name = "Mean"

        # Prepare auxiliary metadata
        info = {
            "s_tm1": None if s_tm1 is None else np.asarray(s_tm1, dtype=float).reshape(-1),
            "branches": futures,
            "policy": policy_name,
            "diagnostics": diag,
        }

        return s_t, s_tp1_hat, info
