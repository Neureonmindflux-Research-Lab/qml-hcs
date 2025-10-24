#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example 02 – Hypercausal Chain Demo (fixed & improved)
------------------------------------------------------
Chain of hyper-causal nodes with future projection and one optimization step.

Includes
--------
- Three chained HCNode objects (sequential model).
- Simple parametric backend (w, b) per node.
- Linear projector to generate K future branches (K×D).
- Losses: MSE (task), Consistency (triadic), Coherence (dispersion among branches).
- One optimization step using finite-difference gradients and
  ``qmlhc.optim.make_gradient_descent()`` for parameter updates.

Improvements over previous version
----------------------------------
- Prints parameters BEFORE and AFTER updates (correct snapshot).
- Safe float conversion using ``.item()`` (avoids NumPy DeprecationWarning).
- Prints gradient L2 norm for diagnostic.

Modules exercised
-----------------
- qmlhc.core: BackendConfig, HCModel
- qmlhc.hc: HCNode, MeanPolicy
- qmlhc.predictors: LinearProjector
- qmlhc.loss: MSELoss, ConsistencyLoss, CoherenceLoss
- qmlhc.optim: make_gradient_descent
"""

from __future__ import annotations
import numpy as np

# Core and contracts
from qmlhc.core import BackendConfig, QuantumBackend as BaseBackend, HCModel
# Nodes and policies
from qmlhc.hc import HCNode, MeanPolicy
# Projectors
from qmlhc.predictors import LinearProjector
# Losses
from qmlhc.loss import MSELoss, ConsistencyLoss, CoherenceLoss
# Optimizer
from qmlhc.optim import make_gradient_descent


# ============================================================================
# 1) Parametric backend with projection via LinearProjector
# ============================================================================
class ParametricBackend(BaseBackend):
    """
    Deterministic backend with per-node parameters (w, b).

    The backend applies a tanh transformation and generates future projections
    centered around the current state.

    Methods
    -------
    run(params=None)
        Computes ``S_t = tanh(w * x + b)``.
    project_future(S_t, K)
        Uses a LinearProjector centered on ``S_t`` to generate K future branches.
    """

    def __init__(
        self,
        config: BackendConfig,
        w: float = 0.9,
        b: float = 0.05,
        proj_span: float = 0.25,
    ):
        super().__init__(config)
        self.w = float(w)
        self.b = float(b)
        # Internal linear projector: uses S_t as projection base (not x)
        self._projector = LinearProjector(weight=1.0, bias=0.0, span=float(proj_span))

    def get_params(self) -> dict:
        """
        Return parameters as arrays for compatibility with the Optimizer API.

        Returns
        -------
        dict
            Dictionary with keys ``"w"`` and ``"b"`` as NumPy arrays.
        """
        return {
            "w": np.array([self.w], dtype=float),
            "b": np.array([self.b], dtype=float),
        }

    def set_params(self, params: dict) -> None:
        """
        Update backend parameters if provided.

        Parameters
        ----------
        params : dict
            Dictionary that may contain keys ``"w"`` and/or ``"b"``.
        """
        if "w" in params:
            self.w = float(np.asarray(params["w"]).reshape(()))
        if "b" in params:
            self.b = float(np.asarray(params["b"]).reshape(()))

    def run(self, params: dict | None = None) -> np.ndarray:
        """
        Apply the backend transformation ``S_t = tanh(w * x + b)``.

        Parameters
        ----------
        params : dict or None, optional
            Optional parameter override for this run.

        Returns
        -------
        np.ndarray
            Transformed state vector ``S_t``.
        """
        if params:
            self.set_params(params)
        x = self._require_input()
        s_t = np.tanh(self.w * x + self.b)
        s_t = self._validate_state(s_t)
        return s_t

    def project_future(self, s_t: np.ndarray, branches: int = 2) -> np.ndarray:
        """
        Generate future states around ``s_t`` using a linear projector.

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
        s = self._validate_state(s_t)
        k = max(2, int(branches))
        fut = self._projector.project(s, branches=k)
        fut = self._validate_branches(fut)
        return fut


# ============================================================================
# 2) Experiment utilities
# ============================================================================
def build_model_chain(D=3, K=5):
    """
    Build an HCModel composed of three nodes, each with its own parametric backend.

    Parameters
    ----------
    D : int, optional
        Dimensionality of the state space. Default is 3.
    K : int, optional
        Number of branches for projection. Default is 5.

    Returns
    -------
    tuple
        (model, nodes, backends)
    """
    cfg = BackendConfig(output_dim=D, seed=123)

    # Three backends with slight differences to observe training effect
    b0 = ParametricBackend(cfg, w=0.9, b=0.05, proj_span=0.20)
    b1 = ParametricBackend(cfg, w=0.95, b=0.02, proj_span=0.25)
    b2 = ParametricBackend(cfg, w=1.05, b=0.00, proj_span=0.30)

    policy = MeanPolicy()

    n0 = HCNode(backend=b0, policy=policy)
    n1 = HCNode(backend=b1, policy=policy)
    n2 = HCNode(backend=b2, policy=policy)

    model = HCModel([n0, n1, n2])
    nodes = [n0, n1, n2]
    backends = [b0, b1, b2]
    return model, nodes, backends


def params_pack(backends):
    """
    Flatten parameters of all backends into a single dictionary.

    Parameters
    ----------
    backends : list
        List of backend objects.

    Returns
    -------
    dict
        Flattened parameters for optimizer consumption.
    """
    packed = {}
    for idx, be in enumerate(backends):
        p = be.get_params()
        for k, v in p.items():
            packed[f"b{idx}_{k}"] = np.array(v, dtype=float)
    return packed


def params_unpack(backends, packed):
    """
    Distribute flat parameters back into each backend.

    Parameters
    ----------
    backends : list
        Backend instances to update.
    packed : dict
        Flattened parameter dictionary.
    """
    for idx, be in enumerate(backends):
        sub = {}
        for k in ("w", "b"):
            key = f"b{idx}_{k}"
            if key in packed:
                sub[k] = packed[key]
        be.set_params(sub)


def finite_diff_grads(loss_fn, params, apply_params_fn, eps=1e-3):
    """
    Compute finite-difference (forward diff) gradients for all parameters.

    Parameters
    ----------
    loss_fn : callable
        Function that recomputes the full loss reading current parameters.
    params : dict
        Parameter dictionary ``{str: np.ndarray}`` with scalar arrays.
    apply_params_fn : callable
        Function that applies a new parameter set to the model before calling ``loss_fn``.
    eps : float, optional
        Perturbation step for finite difference. Default is 1e-3.

    Returns
    -------
    dict
        Gradients for each parameter, same keys as ``params``.
    """
    grads = {}
    base_params = {k: v.copy() for k, v in params.items()}
    apply_params_fn(base_params)
    base_loss = loss_fn()

    for k, v in base_params.items():
        perturbed = {kk: vv.copy() for kk, vv in base_params.items()}
        perturbed[k] = v + eps
        apply_params_fn(perturbed)
        l_eps = loss_fn()
        grad = (l_eps - base_loss) / eps
        grads[k] = np.array([grad], dtype=float)

    apply_params_fn(base_params)
    return grads


def dict_to_scalars(d: dict) -> dict:
    """
    Convert scalar ndarray values to safe Python floats for printing.

    Parameters
    ----------
    d : dict
        Dictionary of parameter arrays.

    Returns
    -------
    dict
        Dictionary with all values converted to floats.
    """
    out = {}
    for k, v in d.items():
        arr = np.asarray(v)
        if arr.shape == () or arr.size == 1:
            out[k] = float(arr.reshape(()).item())
        else:
            out[k] = arr.tolist()
    return out


def grad_l2_norm(grads: dict) -> float:
    """
    Compute L2 norm of all scalar gradients.

    Parameters
    ----------
    grads : dict
        Dictionary with scalar gradient arrays.

    Returns
    -------
    float
        L2 norm of gradients.
    """
    sq_sum = 0.0
    for v in grads.values():
        g = float(np.asarray(v).reshape(()).item())
        sq_sum += g * g
    return float(np.sqrt(sq_sum))


# ============================================================================
# 3) Hyper-causal chain demo + optimization step
# ============================================================================
def chain_demo_step():
    """
    Run a hyper-causal chain demo with one optimization step.

    Builds a sequential model of three nodes, executes multiple time steps,
    computes task, consistency, and coherence losses, and applies a single
    gradient-descent update using finite-difference gradients.

    Returns
    -------
    tuple
        (losses_before, losses_after)
    """
    D = 3
    K = 5
    T = 6  # Temporal sequence length

    model, nodes, backends = build_model_chain(D=D, K=K)

    # Input data (simple oscillatory pattern) and task targets
    t = np.arange(T, dtype=float)
    x_seq = np.stack(
        [
            0.3 * np.sin(0.7 * t + 0.0),
            0.2 * np.sin(0.7 * t + 0.8),
            0.1 * np.cos(0.7 * t + 0.3),
        ],
        axis=1,
    )

    target_seq = np.zeros((T, D), dtype=float)

    mse = MSELoss()
    cons = ConsistencyLoss(alpha=0.8, beta=1.2)
    coh = CoherenceLoss(mode="variance")

    params = params_pack(backends)
    opt = make_gradient_descent(lr=5e-2)
    state = opt.initialize(params)

    # Helper functions for one full epoch
    def apply_params_fn(packed):
        params_unpack(backends, packed)

    def forward_and_losses():
        """Return (total_loss, details) evaluating full sequence with the chained model."""
        total_task = 0.0
        total_cons = 0.0
        total_coh = 0.0
        s_tm1 = None

        for ti in range(T):
            x_t = x_seq[ti]
            s_t, s_tp1_hat, infos = model.forward_chain(x_t, s_tm1=s_tm1, branches=K)

            total_task += mse(s_t, target_seq[ti])
            if s_tm1 is not None:
                total_cons += cons(s_tm1, s_t, s_tp1_hat)

            coh_vals = []
            for info in infos:
                branches = info.get("branches", None)
                if isinstance(branches, np.ndarray) and branches.ndim == 2:
                    coh_vals.append(coh(branches))
            if coh_vals:
                total_coh += float(np.mean(coh_vals))
            s_tm1 = s_t

        n_task = T
        n_cons = max(1, T - 1)
        n_coh = T
        task_loss = total_task / n_task
        cons_loss = total_cons / n_cons
        coh_loss = total_coh / n_coh
        total = task_loss + 0.5 * cons_loss + 0.3 * coh_loss

        details = {
            "task": float(task_loss),
            "cons": float(cons_loss),
            "coh": float(coh_loss),
            "total": float(total),
        }
        return total, details

    total0, det0 = forward_and_losses()
    params_before = dict_to_scalars(params_pack(backends))

    def loss_wrapper():
        l, _ = forward_and_losses()
        return l

    grads = finite_diff_grads(loss_wrapper, params, apply_params_fn, eps=1e-3)
    grad_norm = grad_l2_norm(grads)

    params, state = opt.step(params, grads, state)
    apply_params_fn(params)

    total1, det1 = forward_and_losses()
    params_after = dict_to_scalars(params_pack(backends))

    print("=== Hypercausal Chain Demo ===")
    print(f"D={D}, K={K}, T={T}")
    print("\nParameters (before):")
    print(params_before)
    print("\nLosses BEFORE update:")
    print(det0)
    print("\nUpdating parameters with GD (finite-diff grads)...")
    print(f"||grad||_2 ≈ {grad_norm:.6e}")
    print("\nParameters (after):")
    print(params_after)
    print("\nLosses AFTER update:")
    print(det1)

    return det0, det1


# ============================================================================
# 4) Entry point
# ============================================================================
if __name__ == "__main__":
    before, after = chain_demo_step()
    print("\nSummary:")
    print(f"total BEFORE:  {before['total']:.6f}")
    print(f"total AFTER :  {after['total']:.6f}")
