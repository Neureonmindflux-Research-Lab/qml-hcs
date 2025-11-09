#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Core Types and Protocols
------------------------
Common type aliases, Protocols, and lightweight data structures shared across
the package. 
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Sequence, Protocol, runtime_checkable, TypedDict, Union


import numpy as np


# ---- Scalar and tensor aliases ------------------------------------------------
Scalar = float  # loss values, metrics, probabilities
Array = np.ndarray  # primary numeric container across the library

# Accepts ndarray or simple Python sequences convertible to ndarray.
TensorLike = Union[Array, Sequence[float]]


# ---- Capability and run metadata ---------------------------------------------
class GradientKind(str, Enum):
    """
    Declare the gradient computation strategy exposed by a backend.

    Members
    -------
    PARAMETER_SHIFT
        Parameter-shift differentiation.
    FINITE_DIFF
        Finite-difference estimation.
    ADJOINT
        Adjoint differentiation.
    NONE
        No gradient support.
    """
    PARAMETER_SHIFT = "parameter-shift"
    FINITE_DIFF = "finite-diff"
    ADJOINT = "adjoint"
    NONE = "none"


class Capabilities(TypedDict, total=False):
    """
    Static features reported by a quantum backend.

    Keys
    ----
    backend_name : str
        Human-readable backend name.
    backend_version : str
        Version string of the backend provider.
    max_qubits : int
        Maximum number of supported qubits (if applicable).
    output_dim : int
        Dimension of the output state vector.
    supports_shots : bool
        Whether the backend uses sampling (shot-based execution).
    min_shots : int
        Minimum number of supported shots.
    max_shots : int
        Maximum number of supported shots.
    supports_noise : bool
        Whether the backend can simulate/handle noise.
    supports_batch : bool
        Whether batched execution is supported.
    gradient : GradientKind
        Gradient strategy provided by the backend.
    notes : str
        Additional free-form information.
    """
    backend_name: str
    backend_version: str
    max_qubits: int
    output_dim: int
    supports_shots: bool
    min_shots: int
    max_shots: int
    supports_noise: bool
    supports_batch: bool
    gradient: GradientKind
    notes: str


class RunInfo(TypedDict, total=False):
    """
    Per-execution metadata for telemetry and auditing.

    Keys
    ----
    shots : int
        Number of execution shots used for this run.
    latency_ms : float
        Latency in milliseconds for the execution.
    seed : int
        Random seed (if relevant to stochastic behavior).
    metadata : Mapping[str, Any]
        Arbitrary provider-specific information.
    """
    shots: int
    latency_ms: float
    seed: int
    metadata: Mapping[str, Any]


# ---- Hypercausal state containers --------------------------------------------
@dataclass(frozen=True)
class State:
    """
    One-step compact representation (e.g., expectation vector).

    Parameters
    ----------
    values : Array
        State values with shape ``(D,)``.
    name : str or None, optional
        Optional label for identification, by default ``None``.
    """
    values: Array  # shape: (D,)
    name: str | None = None  # optional label


@dataclass(frozen=True)
class FutureSet:
    """
    Collection of K candidate future states with an optional chosen index.

    Parameters
    ----------
    branches : Array
        Candidate futures with shape ``(K, D)``.
    chosen_index : int or None, optional
        Index of the selected representative branch, or ``None`` if selection
        is deferred. Default is ``None``.
    """
    branches: Array  # shape: (K, D)
    chosen_index: int | None = None  # None when policy selection is deferred


# ---- Policies and selection semantics ----------------------------------------
class SelectionPolicy(str, Enum):
    """
    Deterministic strategies to aggregate K branches into one representative.

    Members
    -------
    MEAN
        Arithmetic mean across branches.
    MEDIAN
        Element-wise median across branches.
    MIN_RISK
        Select branch minimizing a risk functional.
    """
    MEAN = "mean"          # arithmetic average across branches
    MEDIAN = "median"      # element-wise median across branches
    MIN_RISK = "min-risk"  # selects branch minimizing a risk functional


@runtime_checkable
class RiskFunctional(Protocol):
    """
    Scoring function used by MIN_RISK selection.

    Notes
    -----
    Lower scores indicate lower risk and are preferred.
    """
    def __call__(self, branch: Array) -> float:  # lower is better
        ...


# ---- Public Protocols (contracts) --------------------------------------------
@runtime_checkable
class QuantumBackend(Protocol):
    """
    Minimal contract for any quantum execution provider.

    Attributes
    ----------
    output_dim : int
        Dimension of the produced state vector.

    Methods
    -------
    encode(x)
        Load input into backend registers/state.
    run(params=None)
        Execute and return the current state vector ``S_t`` with shape ``(output_dim,)``.
    project_future(s_t, branches=2)
        Generate K candidate futures with shape ``(branches, output_dim)``.
    capabilities()
        Report static backend capabilities.
    """
    output_dim: int

    def encode(self, x: TensorLike) -> None:
        """Load input into the backend input registers/state."""

    def run(self, params: Mapping[str, Any] | None = None) -> Array:
        """Execute and return current state vector ``S_t`` with shape ``(output_dim,)``."""

    def project_future(self, s_t: TensorLike, branches: int = 2) -> Array:
        """Generate K candidate futures with shape ``(branches, output_dim)``."""

    def capabilities(self) -> Capabilities:
        """Report static backend capabilities."""


@runtime_checkable
class ProjectionPolicy(Protocol):
    """
    Map a set of candidate futures to a single representative.

    Returns
    -------
    tuple
        ``(Ŝ_{t+1}, diagnostics)`` where:
        - ``Ŝ_{t+1}`` has shape ``(D,)``
        - diagnostics is a mapping with auxiliary information
    """
    def select(self, futures: Array) -> tuple[Array, Mapping[str, Any]]:
        # Returns (Ŝ_{t+1}, diagnostics)
        ...


@runtime_checkable
class HypercausalNode(Protocol):
    """
    Single node performing triadic forward propagation.

    Returns
    -------
    tuple
        ``(S_t, Ŝ_{t+1}, info)`` where:
        - ``S_t`` has shape ``(D,)``
        - ``Ŝ_{t+1}`` has shape ``(D,)``
        - ``info`` may include
        ``{'s_tm1': Array, 'branches': Array(K, D), 'policy': str, ...}``
    """
    def forward(
        self,
        x_t: TensorLike,
        s_tm1: TensorLike | None,
        branches: int = 2,
    ) -> tuple[Array, Array, Mapping[str, Any]]:
        # Returns (S_t, Ŝ_{t+1}, info) where:
        # - S_t has shape (D,)
        # - Ŝ_{t+1} has shape (D,)
        # - info may include {'s_tm1': Array, 'branches': Array(K, D), 'policy': str, ...}
        ...


@runtime_checkable
class LossFn(Protocol):
    """
    Numerical objective with a float return.
    """
    def __call__(self, *args: Any, **kwargs: Any) -> Scalar:
        ...
