#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example 03 – Advanced Quantum-Hypercausal Training Demo (Depth REAL + Freeze + Relaxed Clipping)
------------------------------------------------------------------------------------------------
Advanced demonstration of the hyper-causal engine with:

- REAL DepthScheduler (invoked OUTSIDE the logger) up to depth ≤ 3
- TelemetryLogger + MemoryLogger (JSON-safe)
- Losses: MSE + Consistency + Coherence
- Optimization with CENTRAL finite-difference gradients
- LR/EPS with LINEAR scaling by depth + epoch decay
- RELAXED and adaptive gradient clipping by depth
- One “FREEZE” epoch after each depth jump (no update)
- Early stopping (patience = 1) + saving of best parameters (JSON)
- Metrics: SMAPE (zero-safe), RMSE, Overshoot, Robustness
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
# Base metrics
from qmlhc.metrics import overshoot, robustness
# Optimizer
from qmlhc.optim import make_gradient_descent
# Callbacks
from qmlhc.callbacks import CallbackList, TelemetryLogger, MemoryLogger, DepthScheduler


# ----------------------------------------------------------------------
# Safe metrics
# ----------------------------------------------------------------------
def smape_safe(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    """
    Symmetric Mean Absolute Percentage Error (SMAPE) that is safe for zero targets.

    Parameters
    ----------
    y_true, y_pred : np.ndarray
        True and predicted arrays.
    eps : float, optional
        Small constant to avoid division by zero.

    Returns
    -------
    float
        SMAPE value in percent.
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
    y_true, y_pred : np.ndarray
        True and predicted arrays.

    Returns
    -------
    float
        RMSE value.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


# ----------------------------------------------------------------------
# Backend with adaptive depth
# ----------------------------------------------------------------------
class DepthAwareBackend(BaseBackend):
    """
    Backend that adapts its depth during training.

    Attributes
    ----------
    w, b : float
        Backend parameters.
    depth : int
        Current recursion depth.
    _base_span : float
        Base span for the projector.
    _span_floor : float
        Lower bound for span to maintain branch diversity.
    """

    def __init__(self, config: BackendConfig, w: float = 0.9, b: float = 0.05, proj_span: float = 0.25):
        super().__init__(config)
        self.w = float(w)
        self.b = float(b)
        self.depth = 1
        self._base_span = float(proj_span)
        self._span_floor = 0.10
        self._projector = LinearProjector(weight=1.0, bias=0.0, span=self._base_span)

    def get_params(self) -> dict:
        """Return parameters as arrays."""
        return {"w": np.array([self.w], dtype=float),
                "b": np.array([self.b], dtype=float)}

    def set_params(self, params: dict) -> None:
        """Update parameters if present in dictionary."""
        if "w" in params:
            self.w = float(np.asarray(params["w"]).reshape(()))
        if "b" in params:
            self.b = float(np.asarray(params["b"]).reshape(()))

    def run(self, params: dict | None = None) -> np.ndarray:
        """
        Apply recursive tanh transformation according to depth.

        Returns
        -------
        np.ndarray
            Current state vector.
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
        Generate future states with depth-dependent span.

        Returns
        -------
        np.ndarray
            Projected future states of shape (K, D).
        """
        s = self._validate_state(s_t)
        k = max(2, int(branches))
        span = max(self._span_floor, self._base_span / (1.0 + 0.3 * (self.depth - 1)))
        self._projector = LinearProjector(weight=1.0, bias=0.0, span=span)
        fut = self._projector.project(s, branches=k)
        return self._validate_branches(fut)


# ----------------------------------------------------------------------
# Model-building utilities
# ----------------------------------------------------------------------
def build_model_chain(D=3):
    """Build a three-node model chain with depth-aware backends."""
    cfg = BackendConfig(output_dim=D, seed=11)
    b0 = DepthAwareBackend(cfg, w=0.90, b=0.03, proj_span=0.22)
    b1 = DepthAwareBackend(cfg, w=0.97, b=0.02, proj_span=0.25)
    b2 = DepthAwareBackend(cfg, w=1.05, b=0.00, proj_span=0.30)
    pol = MeanPolicy()
    n0, n1, n2 = HCNode(b0, pol), HCNode(b1, pol), HCNode(b2, pol)
    model = HCModel([n0, n1, n2])
    return model, [n0, n1, n2], [b0, b1, b2]


def params_pack(backends):
    """Flatten parameters of all backends into a single dictionary."""
    packed = {}
    for i, be in enumerate(backends):
        for k, v in be.get_params().items():
            packed[f"b{i}_{k}"] = np.array(v, dtype=float)
    return packed


def params_unpack(backends, packed):
    """Distribute flat parameters back to each backend."""
    for i, be in enumerate(backends):
        sub = {}
        for k in ("w", "b"):
            key = f"b{i}_{k}"
            if key in packed:
                sub[k] = packed[key]
        be.set_params(sub)


# ----------------------------------------------------------------------
# Finite-difference gradients (central)
# ----------------------------------------------------------------------
def central_diff_grads(loss_fn, params, apply_params_fn, eps: float):
    """
    Compute central finite-difference gradients for better stability.

    Gradient ≈ (f(x + ε) − f(x − ε)) / (2ε)
    """
    grads = {}
    base = {k: v.copy() for k, v in params.items()}

    def setp(p): apply_params_fn(p)
    setp(base)
    _ = loss_fn()

    for k, v in base.items():
        vp = {kk: vv.copy() for kk, vv in base.items()}
        vm = {kk: vv.copy() for kk, vv in base.items()}
        vp[k] = v + eps
        vm[k] = v - eps
        setp(vp); lp = loss_fn()
        setp(vm); lm = loss_fn()
        g = (lp - lm) / (2.0 * eps)
        grads[k] = np.array([g], dtype=float)

    setp(base)
    return grads


def grad_norm(grads: dict) -> float:
    """Compute L2 norm of gradients."""
    sq = 0.0
    for g in grads.values():
        val = float(np.asarray(g).reshape(()))
        sq += val * val
    return float(np.sqrt(sq))


def clip_grads_adaptive(grads: dict, depth_mean: float) -> tuple[dict, float, float]:
    """
    Adaptive gradient clipping based on mean depth.

    Returns
    -------
    tuple
        (clipped_gradients, norm_before, norm_after)
    """
    if depth_mean < 1.5:
        max_norm = 5e-2
    elif depth_mean < 2.5:
        max_norm = 7.5e-2
    else:
        max_norm = 1e-1

    n_before = grad_norm(grads)
    if n_before <= max_norm or n_before == 0.0:
        return grads, n_before, n_before

    scale = max_norm / n_before
    clipped = {k: np.array([float(np.asarray(v).reshape(())) * scale], dtype=float) for k, v in grads.items()}
    n_after = grad_norm(clipped)
    return clipped, n_before, n_after


# ----------------------------------------------------------------------
# Training procedure
# ----------------------------------------------------------------------
def advanced_training_with_freeze():
    """
    Perform advanced hyper-causal training with depth freeze and adaptive clipping.

    Returns
    -------
    dict
        Dictionary containing best loss snapshot and final metrics.
    """
    D, K, T = 3, 5, 48
    EPOCHS = 16
    BASE_LR = 5e-2
    BASE_EPS = 1e-3
    LOG_PATH = Path("runs/telemetry_stable.jsonl")
    SAVE_BEST = True
    BEST_PATH = Path("runs/best_params.json")

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

    # Optimizer and callbacks
    params = params_pack(backends)
    opt = make_gradient_descent(lr=BASE_LR)
    state = opt.initialize(params)

    callbacks = CallbackList([
        TelemetryLogger(path=LOG_PATH, flush_interval=8),
        MemoryLogger(),
    ])
    depth_cb = DepthScheduler(target_attr="depth", start=1, end=3, epochs=EPOCHS - 1)

    def apply_params_fn(packed): params_unpack(backends, packed)

    def forward_and_losses():
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
    best_params = None
    patience = 1
    bad_epochs = 0
    last_depth_signature = None

    for epoch in range(EPOCHS):
        for be in backends:
            depth_cb.on_epoch_begin(epoch, {"backend": be})
        depths = [int(be.depth) for be in backends]
        depth_mean = float(np.mean(depths))

        epoch_decay_lr = 1.0 / (1.0 + 0.5 * epoch)
        epoch_decay_eps = 1.0 / (1.0 + 0.3 * epoch)
        lr_eff = (BASE_LR / (1.0 + 0.5 * (depth_mean - 1.0))) * epoch_decay_lr
        eps_eff = (BASE_EPS / (1.0 + 0.3 * (depth_mean - 1.0))) * epoch_decay_eps
        opt = make_gradient_descent(lr=lr_eff)

        callbacks.on_epoch_begin(epoch, {
            "epoch": int(epoch),
            "depth": depths,
            "lr_eff": float(lr_eff),
            "eps_eff": float(eps_eff)
        })

        total0, det0, _ = forward_and_losses()

        depth_sig = tuple(depths)
        freeze_epoch = (last_depth_signature is not None) and (depth_sig != last_depth_signature)
        last_depth_signature = depth_sig

        if freeze_epoch:
            callbacks.on_epoch_end(epoch, {
                "epoch": int(epoch),
                "loss_before": det0,
                "loss_after": det0,
                "freeze": True,
            })
            if best is None or det0["total"] < best["total"]:
                best = {"epoch": int(epoch), **det0}
                best_params = {k: float(np.asarray(v).reshape(())) for k, v in params.items()}
                bad_epochs = 0
            else:
                bad_epochs += 1
            print(f"[Epoch {epoch}] FREEZE depth={depths} total={det0['total']:.6f} lr_eff={lr_eff:.3e} eps_eff={eps_eff:.3e}")
            if bad_epochs > patience:
                print(f"Early stopping activated at epoch {epoch} (freeze). Best total={best['total']:.6f} (epoch {best['epoch']}).")
                break
            continue

        def loss_wrapper():
            l, _, _ = forward_and_losses()
            return l

        grads = central_diff_grads(loss_wrapper, params, apply_params_fn, eps=eps_eff)
        grads, gnorm_before, gnorm_after = clip_grads_adaptive(grads, depth_mean)
        params, state = opt.step(params, grads, state)
        apply_params_fn(params)

        total1, det1, _ = forward_and_losses()

        callbacks.on_epoch_end(epoch, {
            "epoch": int(epoch),
            "loss_before": det0,
            "loss_after": det1,
            "grad_norm_before": float(gnorm_before),
            "grad_norm_after": float(gnorm_after),
            "freeze": False,
        })

        if best is None or det1["total"] < best["total"]:
            best = {"epoch": int(epoch), **det1}
            best_params = {k: float(np.asarray(v).reshape(())) for k, v in params.items()}
            bad_epochs = 0
        else:
            bad_epochs += 1

        print(
            f"[Epoch {epoch}] total_before={det0['total']:.6f} total_after={det1['total']:.6f} "
            f"depth={depths} lr_eff={lr_eff:.3e} eps_eff={eps_eff:.3e} "
            f"||g||_before={gnorm_before:.3e} ||g||_after={gnorm_after:.3e}"
        )

        if bad_epochs > patience:
            print(f"Early stopping activated at epoch {epoch}. Best total={best['total']:.6f} (epoch {best['epoch']}).")
            break

    if SAVE_BEST and best_params is not None:
        BEST_PATH.parent.mkdir(parents=True, exist_ok=True)
        with BEST_PATH.open("w") as f:
            json.dump(best_params, f, indent=2)
        print(f"\nBest parameters saved at: {BEST_PATH.resolve()}")

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
    print("\nBest epoch snapshot:", {
        "epoch": int(best["epoch"]),
        "task": float(best["task"]),
        "cons": float(best["cons"]),
        "coh":  float(best["coh"]),
        "total": float(best["total"]),
    })

    if LOG_PATH.exists():
        print(f"\nTelemetry JSONL → {LOG_PATH.resolve()}")

    return {
        "best": {
            "epoch": int(best["epoch"]),
            "task": float(best["task"]),
            "cons": float(best["cons"]),
            "coh":  float(best["coh"]),
            "total": float(best["total"]),
        },
        "metrics": {
            "smape": smape_val,
            "rmse": rmse_val,
            "overshoot": over_val,
            "robustness": rob_val
        }
    }


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------
if __name__ == "__main__":
    out = advanced_training_with_freeze()
    print("\nSummary:")
    print(json.dumps(out, indent=2))
