#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example 03b – Stable Training Demo
----------------------------------
Stable version that exercises the system with good practices:

- DepthScheduler limited (≤ 3) and invoked OUTSIDE the logger
- Adaptive LR: lr_eff = base_lr / depth^2
- CENTRAL finite-difference gradients (more stable)
- Gradient clipping (trust region)
- Early stopping (patience = 1)
- Safe metrics: SMAPE (target = 0), RMSE, Overshoot, Robustness
- JSON-safe telemetry (JSONL) + MemoryLogger
"""

from __future__ import annotations
import json
from pathlib import Path
import numpy as np

# Core and contracts
from qmlhc.core import BackendConfig, QuantumBackend as BaseBackend, HCModel
# Nodes and policies
from qmlhc.hc import HCNode, MeanPolicy
# Projectors
from qmlhc.predictors import LinearProjector
# Losses
from qmlhc.loss import MSELoss, ConsistencyLoss, CoherenceLoss
# Metrics
from qmlhc.metrics import overshoot, robustness
# Optimizer
from qmlhc.optim import make_gradient_descent
# Callbacks
from qmlhc.callbacks import CallbackList, TelemetryLogger, MemoryLogger, DepthScheduler


# ------------------------- Safe metrics -------------------------
def smape_safe(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    """
    Symmetric Mean Absolute Percentage Error (SMAPE) that is safe for zeros.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth values.
    y_pred : np.ndarray
        Predicted values.
    eps : float, optional
        Small constant to avoid division by zero, by default 1e-8.

    Returns
    -------
    float
        SMAPE in percent.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.maximum(np.abs(y_true) + np.abs(y_pred), eps)
    return float(np.mean(np.abs(y_pred - y_true) / denom) * 100.0)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Root Mean Squared Error (RMSE).

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth values.
    y_pred : np.ndarray
        Predicted values.

    Returns
    -------
    float
        RMSE value.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


# ------------------- Backend with adaptive depth -------------------
class DepthAwareBackend(BaseBackend):
    """
    Backend with depth-adaptive recursion and span.

    Parameters
    ----------
    config : BackendConfig
        Backend configuration (e.g., output_dim, seed).
    w : float, optional
        Weight coefficient for the tanh transformation, by default 0.9.
    b : float, optional
        Bias term for the tanh transformation, by default 0.05.
    proj_span : float, optional
        Base span for the linear projector, by default 0.25.

    Attributes
    ----------
    w : float
        Weight parameter.
    b : float
        Bias parameter.
    depth : int
        Current recursion depth (>= 1).
    _base_span : float
        Base span for projection.
    _span_floor : float
        Minimum span to preserve branch diversity.
    _projector : LinearProjector
        Internal linear projector centered on the current state.
    """

    def __init__(self, config: BackendConfig, w: float = 0.9, b: float = 0.05, proj_span: float = 0.25):
        super().__init__(config)
        self.w = float(w)
        self.b = float(b)
        self.depth = 1
        self._base_span = float(proj_span)
        self._span_floor = 0.10  # higher floor to maintain branch diversity
        self._projector = LinearProjector(weight=1.0, bias=0.0, span=self._base_span)

    def get_params(self):
        """
        Return parameters as arrays for optimizer compatibility.

        Returns
        -------
        dict
            Dictionary with keys ``"w"`` and ``"b"`` as 1-element float arrays.
        """
        return {"w": np.array([self.w], dtype=float),
                "b": np.array([self.b], dtype=float)}

    def set_params(self, params: dict):
        """
        Update parameters if present in the provided dictionary.

        Parameters
        ----------
        params : dict
            Parameter dictionary that may include keys ``"w"`` and/or ``"b"``.
        """
        if "w" in params:
            self.w = float(np.asarray(params["w"]).reshape(()))
        if "b" in params:
            self.b = float(np.asarray(params["b"]).reshape(()))

    def run(self, params: dict | None = None) -> np.ndarray:
        """
        Apply depth-recursive tanh transformation.

        Parameters
        ----------
        params : dict or None, optional
            Optional parameter override for this call.

        Returns
        -------
        np.ndarray
            Validated current state vector.
        """
        if params:
            self.set_params(params)
        x = self._require_input().astype(float)
        s = x
        for _ in range(max(1, int(self.depth))):
            s = np.tanh(self.w * s + self.b)
        return self._validate_state(s)

    def project_future(self, s_t: np.ndarray, branches: int = 2) -> np.ndarray:
        """
        Project future states around ``s_t`` using a depth-adjusted span.

        Parameters
        ----------
        s_t : np.ndarray
            Current state vector.
        branches : int, optional
            Number of future branches (K), by default 2.

        Returns
        -------
        np.ndarray
            Future branch matrix with shape ``(K, D)``.
        """
        s = self._validate_state(s_t)
        k = max(2, int(branches))
        # span reduced with depth, but with a high floor (0.10)
        span = max(self._span_floor, self._base_span / (1.0 + 0.3 * (self.depth - 1)))
        self._projector = LinearProjector(weight=1.0, bias=0.0, span=span)
        fut = self._projector.project(s, branches=k)
        return self._validate_branches(fut)


# ------------------------ Construction utils ------------------------
def build_model_chain(D=3):
    """
    Build a three-node HCModel with depth-aware backends.

    Parameters
    ----------
    D : int, optional
        State dimensionality, by default 3.

    Returns
    -------
    tuple
        (model, nodes, backends)
    """
    cfg = BackendConfig(output_dim=D, seed=11)
    b0 = DepthAwareBackend(cfg, w=0.90, b=0.03, proj_span=0.22)
    b1 = DepthAwareBackend(cfg, w=0.97, b=0.02, proj_span=0.25)
    b2 = DepthAwareBackend(cfg, w=1.05, b=0.00, proj_span=0.30)
    pol = MeanPolicy()
    n0, n1, n2 = HCNode(b0, pol), HCNode(b1, pol), HCNode(b2, pol)
    model = HCModel([n0, n1, n2])
    return model, [n0, n1, n2], [b0, b1, b2]


def params_pack(backends):
    """
    Flatten parameters from all backends into a single dictionary.

    Parameters
    ----------
    backends : list
        List of backend instances.

    Returns
    -------
    dict
        Flattened parameter dictionary suitable for the optimizer.
    """
    packed = {}
    for i, be in enumerate(backends):
        for k, v in be.get_params().items():
            packed[f"b{i}_{k}"] = np.array(v, dtype=float)
    return packed


def params_unpack(backends, packed):
    """
    Distribute flat parameters back to their corresponding backends.

    Parameters
    ----------
    backends : list
        Backend instances to update.
    packed : dict
        Flattened parameter dictionary.
    """
    for i, be in enumerate(backends):
        sub = {}
        for k in ("w", "b"):
            key = f"b{i}_{k}"
            if key in packed:
                sub[k] = packed[key]
        be.set_params(sub)


# ------------------- CENTRAL finite-difference grads -------------------
def central_diff_grads(loss_fn, params, apply_params_fn, eps: float):
    """
    Central finite-difference gradients (more stable than forward diff).

    Gradient ≈ (f(x + eps) - f(x - eps)) / (2 * eps)

    Parameters
    ----------
    loss_fn : callable
        Function that recomputes the full loss given current parameters.
    params : dict
        Parameter dictionary (values are scalar arrays).
    apply_params_fn : callable
        Function to apply a parameter dictionary to the model.
    eps : float
        Step size for the finite-difference estimation.

    Returns
    -------
    dict
        Dictionary of gradients with the same keys as ``params``.
    """
    grads = {}
    base = {k: v.copy() for k, v in params.items()}

    def setp(p):  # alias
        apply_params_fn(p)

    setp(base)
    _ = loss_fn()  # warm up cache if applicable

    for k, v in base.items():
        vp = {kk: vv.copy() for kk, vv in base.items()}
        vm = {kk: vv.copy() for kk, vv in base.items()}
        vp[k] = v + eps
        vm[k] = v - eps
        setp(vp)
        lp = loss_fn()
        setp(vm)
        lm = loss_fn()
        g = (lp - lm) / (2.0 * eps)
        grads[k] = np.array([g], dtype=float)

    setp(base)
    return grads


def clip_grads(grads: dict, max_norm: float) -> dict:
    """
    Global L2-norm gradient clipping.

    Parameters
    ----------
    grads : dict
        Gradients by parameter key.
    max_norm : float
        Maximum allowed L2 norm.

    Returns
    -------
    dict
        Possibly scaled gradients that respect the trust region.
    """
    sq = 0.0
    for g in grads.values():
        val = float(np.asarray(g).reshape(()))
        sq += val * val
    norm = float(np.sqrt(sq))
    if norm <= max_norm or norm == 0.0:
        return grads
    scale = max_norm / norm
    return {k: np.array([float(np.asarray(v).reshape(())) * scale], dtype=float) for k, v in grads.items()}


# ---------------------------- Training ----------------------------
def stable_training_demo():
    """
    Run a stable training loop with depth scheduling, clipping, and telemetry.

    Returns
    -------
    dict
        Dictionary with the best snapshot and final metrics.
    """
    # Config
    D, K, T = 3, 5, 48
    EPOCHS = 12
    BASE_LR = 5e-2
    BASE_EPS = 1e-3
    LOG_PATH = Path("runs/telemetry_stable.jsonl")

    # Model
    model, nodes, backends = build_model_chain(D=D)

    # Data
    t = np.arange(T, dtype=float)
    x_seq = np.stack([
        0.30 * np.sin(0.35 * t + 0.00),
        0.20 * np.sin(0.35 * t + 0.70),
        0.10 * np.cos(0.35 * t + 0.30),
    ], axis=1)
    target_seq = np.zeros((T, D), dtype=float)

    # Losses
    loss_task = MSELoss()
    loss_cons = ConsistencyLoss(alpha=0.8, beta=1.2)
    loss_coh = CoherenceLoss(mode="variance")

    # Optimizer + telemetry
    params = params_pack(backends)
    opt = make_gradient_descent(lr=BASE_LR)  # base LR; recalibrated by depth each epoch
    state = opt.initialize(params)

    callbacks = CallbackList([
        TelemetryLogger(path=LOG_PATH, flush_interval=8),
        MemoryLogger(),
    ])
    # Real DepthScheduler (1 → 3 across EPOCHS-1; clamped at 3)
    depth_cb = DepthScheduler(target_attr="depth", start=1, end=3, epochs=EPOCHS - 1)

    def apply_params_fn(packed):
        params_unpack(backends, packed)

    def forward_and_losses():
        """
        Compute forward pass over the sequence and all loss terms.

        Returns
        -------
        tuple
            (total_loss, details_dict, last_state_sequence)
        """
        total_task = total_cons = total_coh = 0.0
        s_tm1 = None
        y_last = []
        for step in range(T):
            callbacks.on_step_begin(step, {"step": int(step)})
            s_t, s_hat, infos = model.forward_chain(x_seq[step], s_tm1=s_tm1, branches=K)
            y_last.append(s_t)
            total_task += loss_task(s_t, target_seq[step])
            if s_tm1 is not None:
                total_cons += loss_cons(s_tm1, s_t, s_hat)
            coh_vals = []
            for info in infos:
                br = info.get("branches", None)
                if isinstance(br, np.ndarray) and br.ndim == 2:
                    coh_vals.append(loss_coh(br))
            if coh_vals:
                total_coh += float(np.mean(coh_vals))
            s_tm1 = s_t
            callbacks.on_step_end(step, {"step": int(step)})
        task = total_task / T
        cons = total_cons / max(1, T - 1)
        coh = total_coh / T
        total = task + 0.5 * cons + 0.3 * coh
        return total, {"task": float(task), "cons": float(cons), "coh": float(coh), "total": float(total)}, np.asarray(y_last)

    best = None
    patience = 1
    bad_epochs = 0

    for epoch in range(EPOCHS):
        # 1) Adjust depth (real)
        for be in backends:
            depth_cb.on_epoch_begin(epoch, {"backend": be})
        depth_mean = float(np.mean([be.depth for be in backends]))

        # 2) Depth-adaptive LR and EPS
        lr_eff = BASE_LR / (depth_mean ** 2)
        eps_eff = BASE_EPS / (1.0 + 0.5 * (depth_mean - 1.0))
        opt = make_gradient_descent(lr=lr_eff)  # recreate with effective LR

        # 3) JSON-safe telemetry
        callbacks.on_epoch_begin(epoch, {"epoch": int(epoch), "depth": float(depth_mean), "lr_eff": float(lr_eff), "eps_eff": float(eps_eff)})

        # 4) Forward before update
        total0, det0, _ = forward_and_losses()

        # 5) Central gradients + clipping + step
        def loss_wrapper():
            l, _, _ = forward_and_losses()
            return l

        grads = central_diff_grads(loss_wrapper, params, apply_params_fn, eps=eps_eff)
        grads = clip_grads(grads, max_norm=5e-2)
        params, state = opt.step(params, grads, state)
        apply_params_fn(params)

        # 6) Forward after update
        total1, det1, _ = forward_and_losses()

        # 7) Telemetry end
        callbacks.on_epoch_end(epoch, {
            "epoch": int(epoch),
            "loss_before": det0,
            "loss_after": det1,
        })

        # 8) Early stopping
        if best is None or det1["total"] < best["total"]:
            best = {"epoch": int(epoch), **det1}
            bad_epochs = 0
        else:
            bad_epochs += 1

        depths = [int(be.depth) for be in backends]
        print(f"[Epoch {epoch}] total_before={det0['total']:.6f} total_after={det1['total']:.6f} depth={depths} lr_eff={lr_eff:.3e} eps_eff={eps_eff:.3e}")

        if bad_epochs > patience:
            print(f"Early stopping activated at epoch {epoch}. Best total={best['total']:.6f} (epoch {best['epoch']}).")
            break

    # Final metrics using the last forward
    _, _, y_pred_seq = forward_and_losses()
    smape_val = smape_safe(target_seq[:, 0], y_pred_seq[:, 0])
    rmse_val = rmse(target_seq[:, 0], y_pred_seq[:, 0])
    over_val = overshoot(target_seq[:, 0], y_pred_seq[:, 0])
    rob_val = robustness(target_seq[:, 0], y_pred_seq[:, 0])

    print("\n=== Final metrics (channel 0) ===")
    print(f"SMAPE:      {smape_val:.6f} %")
    print(f"RMSE:       {rmse_val:.6f}")
    print(f"Overshoot:  {over_val:.6f}")
    print(f"Robustness: {rob_val:.6f}")
    print("\nBest epoch snapshot:", best)

    if LOG_PATH.exists():
        print(f"\nTelemetry JSONL → {LOG_PATH.resolve()}")

    return {"best": best, "metrics": {"smape": smape_val, "rmse": rmse_val, "overshoot": over_val, "robustness": rob_val}}


# ---------------------------- Entry point ----------------------------
if __name__ == "__main__":
    out = stable_training_demo()
    print("\nSummary:")
    print(json.dumps(out, indent=2))
