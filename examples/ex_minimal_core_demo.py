#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example 01 – Minimal Core Demo
------------------------------
Minimal flow of a hyper-causal model:

    x_t → Backend (encode/run) → HCNode (project + policy) → Ŝ_{t+1} → ConsistencyLoss

Modules exercised
-----------------
- qmlhc.core: contracts, BackendConfig, QuantumBackend (base), HCModel
- qmlhc.hc: HCNode, MeanPolicy
- qmlhc.loss: ConsistencyLoss
"""

from __future__ import annotations
import numpy as np

# Core and contracts
from qmlhc.core import BackendConfig, QuantumBackend as BaseBackend, HCModel
# Node + policy
from qmlhc.hc import HCNode, MeanPolicy
# Triadic consistency loss
from qmlhc.loss import ConsistencyLoss


# ---------------------------------------------------------------------------
# 1) Minimal concrete backend (inherits from base and implements run/project_future)
# ---------------------------------------------------------------------------

class ToyBackend(BaseBackend):
    """
    Demonstrative backend, deterministic and numerically stable.

    This backend defines the minimal deterministic behavior for demonstration
    purposes. It applies a smooth nonlinear transformation to the encoded
    input and generates future projections through uniform perturbations.

    Methods
    -------
    run(params=None)
        Applies a tanh-based transformation to the encoded input.
    project_future(s_t, branches=2)
        Generates K possible future states.
    """

    def run(self, params: dict | None = None) -> np.ndarray:
        """
        Apply a smooth nonlinear transformation to the last encoded input.

        Parameters
        ----------
        params : dict or None, optional
            Optional parameters (not used in this minimal demo).

        Returns
        -------
        np.ndarray
            The transformed state vector ``s_t``.
        """
        # Ensure that encode(x) was called before and fetch the last input.
        x = self._require_input()

        # Simple and stable transformation (tanh of a small affine bias).
        w = 0.95
        b = 0.05
        s_t = np.tanh(w * x + b)

        # Validate shape against output_dim (for real adapters that may change D at runtime).
        s_t = self._validate_state(s_t)
        return s_t

    def project_future(self, s_t: np.ndarray, branches: int = 2) -> np.ndarray:
        """
        Generate K future states around the current state.

        Parameters
        ----------
        s_t : np.ndarray
            Current state vector.
        branches : int, optional
            Number of future branches (K). Default is 2.

        Returns
        -------
        np.ndarray
            Future states matrix with shape ``(K, D)``.
        """
        # Validate current state and K.
        s = self._validate_state(s_t)
        k = max(2, int(branches))

        # Uniform deltas in [-span, +span]; added to s_t and passed through tanh.
        span = 0.25
        deltas = np.linspace(-span, span, k, dtype=float)

        futures = np.stack([np.tanh(s + d) for d in deltas], axis=0)  # (K, D)
        futures = self._validate_branches(futures)
        return futures


# ---------------------------------------------------------------------------
# 2) Demo: forward pass of a single node + consistency loss
# ---------------------------------------------------------------------------

def minimal_core_demo() -> None:
    """
    Run a minimal demonstration of the hyper-causal model.

    This function performs a full pass through a minimal hyper-causal pipeline:
    it initializes a backend, defines a node with mean policy, executes the
    forward step, and computes the triadic consistency loss. The results are
    printed to the console for inspection.

    Returns
    -------
    None
        Prints the results directly to the console.
    """
    # State dimension and configuration.
    D = 3
    K = 3
    cfg = BackendConfig(output_dim=D, seed=42)

    # Instantiate backend and node with mean policy.
    backend = ToyBackend(cfg)
    policy = MeanPolicy()
    node = HCNode(backend=backend, policy=policy)

    # Example data: input x_t and previous state s_{t-1}.
    x_t = np.array([0.20, -0.10, 0.40], dtype=float)   # (D,)
    s_tm1 = np.array([0.15, -0.05, 0.35], dtype=float) # (D,)

    # Execute hyper-causal step in the node.
    s_t, s_tp1_hat, info = node.forward(x_t, s_tm1=s_tm1, branches=K)

    # Triadic consistency loss.
    consistency = ConsistencyLoss(alpha=1.0, beta=1.0)
    loss_val = consistency(s_tm1, s_t, s_tp1_hat)

    # Display results.
    print("=== Minimal Core Demo ===")
    print(f"output_dim (D):     {D}")
    print(f"branches (K):       {K}\n")

    print("x_t:                ", x_t)
    print("S_{t-1} (s_tm1):    ", s_tm1)
    print("S_t (from run):     ", s_t)
    print("Ŝ_{t+1} (selected): ", s_tp1_hat)
    print("\nNode information (summary):")
    print("  policy:           ", info.get("policy"))
    branches_mat = info.get("branches")
    if isinstance(branches_mat, np.ndarray):
        print("  branches shape:    ", branches_mat.shape)
        print("  branches[0]:       ", branches_mat[0])
    print("\nConsistencyLoss:")
    print("  L = α||S_t - S_{t-1}||^2 + β||S_t - Ŝ_{t+1}||^2")
    print("  α = 1.0, β = 1.0")
    print("  loss =", loss_val)

    # (Optional) Validate the same flow through HCModel with a single node.
    model = HCModel([node])
    s_t_m, s_tp1_hat_m, info_m = model.forward(x_t, s_tm1=s_tm1, branches=K)
    assert np.allclose(s_t, s_t_m) and np.allclose(s_tp1_hat, s_tp1_hat_m)
    print("\nHCModel.forward() matches single-node result ✔")


# ---------------------------------------------------------------------------
# 3) Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    minimal_core_demo()
