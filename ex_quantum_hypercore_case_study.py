#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example 04 – Quantum Hypercore Case Study
-----------------------------------------
Realistic scenario: prediction/stabilization of a dynamic signal with
counterfactual “what-if” projections and integrated evaluation.

Key updates
------------
- HCGraph now receives nodes as a dictionary {"A": HCNode, "B": ..., "C": ...}
  and edges as Edge("A", "B"), etc.
- TelemetryLogger contexts are JSON-safe (no direct objects or ndarrays passed).
- DepthScheduler is kept separate (not inside CallbackList) to allow passing
  backend objects without breaking JSON serialization.
"""

from __future__ import annotations
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Callable, Sequence
import numpy as np

# Core (contracts, types)
from qmlhc.core import BackendConfig, QuantumBackend as BaseBackend, Array
# Nodes / Graphs / Policies
from qmlhc.hc import HCNode
from qmlhc.hc.graph import HCGraph, Edge
from qmlhc.hc.policy import MinRiskPolicy
# Predictors / Anticipators
from qmlhc.predictors import LinearProjector, ContrafactualAnticipator
# Losses
from qmlhc.loss import MSELoss, ConsistencyLoss, CoherenceLoss
# Metrics
from qmlhc.metrics import (
    mape, overshoot, settling_time, robustness,
    early_roc_auc, recall_at_lag
)
# Optimizer
from qmlhc.optim import make_gradient_descent
# Callbacks
from qmlhc.callbacks import CallbackList, TelemetryLogger, MemoryLogger, DepthScheduler


# ============================================================================
# 1) Synthetic system with anomalies (for downstream evaluation)
# ============================================================================
@dataclass
class SyntheticScenario:
    """Synthetic dynamic environment with anomalies for testing."""
    D: int = 3
    T: int = 60
    seed: int = 123
    anomaly_times: tuple = (30, 45)
    anomaly_mag: float = 0.5

    def build(self):
        """
        Build a synthetic temporal signal with controlled anomalies.

        Returns
        -------
        tuple
            (x_seq, y_target, y_anom)
        """
        rng = np.random.default_rng(self.seed)
        t = np.arange(self.T, dtype=float)

        # Base smooth signals (3 channels)
        base = np.stack([
            0.35 * np.sin(0.25 * t + 0.10),
            0.20 * np.sin(0.25 * t + 0.75),
            0.15 * np.cos(0.25 * t + 0.40),
        ], axis=1)

        # Small Gaussian noise
        noise = 0.02 * rng.standard_normal(size=base.shape)

        x_seq = base + noise
        y_target = np.zeros_like(x_seq)

        # Inject anomalies into channel 0
        y_anom = np.zeros(self.T, dtype=float)
        for at in self.anomaly_times:
            if 0 <= at < self.T:
                y_anom[at] = 1.0
                x_seq[at:, 0] += self.anomaly_mag

        return x_seq, y_target, y_anom


# ============================================================================
# 2) Parametric backend with counterfactual anticipator
# ============================================================================
def shock_up(center: Array) -> Array:
    """What-if scenario: channel 0 increases slightly."""
    v = center.copy()
    v[0] += 0.10
    return v


def shock_down(center: Array) -> Array:
    """What-if scenario: channel 0 decreases slightly."""
    v = center.copy()
    v[0] -= 0.10
    return v


def lateral_shift(center: Array) -> Array:
    """What-if scenario: small lateral shift in channel 1."""
    v = center.copy()
    v[1] += 0.06
    return v


class CFBackend(BaseBackend):
    """
    Parametric backend with counterfactual anticipation.

    - ``run()``: applies a tanh transformation depth times.
    - ``project_future()``: linear projection around s_t with counterfactual perturbations.
    """

    def __init__(
        self, config: BackendConfig, w: float = 0.95, b: float = 0.02,
        proj_span: float = 0.25,
        perts: Sequence[Callable[[Array], Array]] | None = None,
        symmetric: bool = True,
    ):
        super().__init__(config)
        self.w = float(w)
        self.b = float(b)
        self.depth = 1
        self._base_span = float(proj_span)
        self._projector = LinearProjector(weight=1.0, bias=0.0, span=self._base_span)
        self._perts = list(perts or [])
        self._anticipator = ContrafactualAnticipator(
            projector=self._projector,
            perturbations=self._perts,
            config=None,
        )
        self._symmetric = bool(symmetric)

    def get_params(self):
        """Return current parameters as arrays."""
        return {"w": np.array([self.w], dtype=float),
                "b": np.array([self.b], dtype=float)}

    def set_params(self, params: dict):
        """Set backend parameters."""
        if "w" in params:
            self.w = float(np.asarray(params["w"]).reshape(()))
        if "b" in params:
            self.b = float(np.asarray(params["b"]).reshape(()))

    def run(self, params: dict | None = None) -> Array:
        """Apply recursive tanh transformation."""
        if params:
            self.set_params(params)
        x = self._require_input()
        s = x.astype(float)
        for _ in range(max(1, int(self.depth))):
            s = np.tanh(self.w * s + self.b)
        return self._validate_state(s)

    def project_future(self, s_t: Array, branches: int = 2) -> Array:
        """Generate future states using updated projector span and perturbations."""
        s = self._validate_state(s_t)
        k = max(2, int(branches))
        span = max(0.05, self._base_span / (1.0 + 0.25 * (self.depth - 1)))
        self._projector = LinearProjector(weight=1.0, bias=0.0, span=span)
        self._anticipator = ContrafactualAnticipator(
            projector=self._projector,
            perturbations=self._perts,
            config=None,
        )
        fut = self._anticipator.generate(s)
        if fut.shape[0] >= k:
            fut = fut[:k]
        else:
            rep = np.repeat(fut[-1][None, :], k - fut.shape[0], axis=0)
            fut = np.concatenate([fut, rep], axis=0)
        return self._validate_branches(fut)


# ============================================================================
# 3) Building the hyper-causal DAG and training utilities
# ============================================================================
def build_hc_dag(D=3, seed=7):
    """Construct a minimal 3-node hyper-causal DAG."""
    cfg = BackendConfig(output_dim=D, seed=seed)
    bA = CFBackend(cfg, w=0.95, b=0.03, proj_span=0.22, perts=[shock_up, lateral_shift])
    bB = CFBackend(cfg, w=1.00, b=0.01, proj_span=0.25, perts=[shock_down])
    bC = CFBackend(cfg, w=1.05, b=0.00, proj_span=0.28, perts=[lateral_shift])

    def l2_risk(branch: Array) -> float:
        return float(np.sqrt(np.sum(branch ** 2)))

    policy = MinRiskPolicy(risk=l2_risk)
    nA, nB, nC = HCNode(bA, policy), HCNode(bB, policy), HCNode(bC, policy)
    node_map = {"A": nA, "B": nB, "C": nC}
    edges = [Edge("A", "B"), Edge("A", "C"), Edge("B", "C")]
    dag = HCGraph(nodes=node_map, edges=edges)
    return dag, node_map, [bA, bB, bC]


def params_pack(backends):
    """Flatten parameters of all backends into a single dict."""
    packed = {}
    for i, be in enumerate(backends):
        for k, v in be.get_params().items():
            packed[f"b{i}_{k}"] = np.array(v, dtype=float)
    return packed


def params_unpack(backends, packed):
    """Restore parameters from flat dictionary."""
    for i, be in enumerate(backends):
        sub = {}
        for k in ("w", "b"):
            key = f"b{i}_{k}"
            if key in packed:
                sub[k] = packed[key]
        be.set_params(sub)


def finite_diff_grads(loss_fn, params, apply_params_fn, eps=1e-3):
    """Compute forward finite-difference gradients for all parameters."""
    grads = {}
    base = {k: v.copy() for k, v in params.items()}
    apply_params_fn(base)
    base_loss = loss_fn()
    for k, v in base.items():
        perturbed = {kk: vv.copy() for kk, vv in base.items()}
        perturbed[k] = v + eps
        apply_params_fn(perturbed)
        l_eps = loss_fn()
        grads[k] = np.array([(l_eps - base_loss) / eps], dtype=float)
    apply_params_fn(base)
    return grads


# ============================================================================
# 4) Experiment loop (training + evaluation)
# ============================================================================
def quantum_hypercore_case_study():
    """
    Run full hyper-causal DAG training and evaluation with counterfactual logic.

    Returns
    -------
    dict
        Dictionary with best epoch snapshot and evaluation metrics.
    """
    D, T, K = 3, 60, 6
    EPOCHS = 6
    LR = 4e-2
    LOG_PATH = Path("runs/quantum_hypercore_case_telemetry.jsonl")

    scenario = SyntheticScenario(D=D, T=T, anomaly_times=(30, 45), anomaly_mag=0.5)
    x_seq, y_target, y_anom = scenario.build()

    dag, node_map, backends = build_hc_dag(D=D, seed=11)
    loss_task, loss_cons, loss_coh = MSELoss(), ConsistencyLoss(0.7, 1.3), CoherenceLoss(mode="variance")

    params = params_pack(backends)
    opt = make_gradient_descent(lr=LR)
    opt_state = opt.initialize(params)

    callbacks = CallbackList([
        TelemetryLogger(path=LOG_PATH, flush_interval=4),
        MemoryLogger(),
    ])
    depth_cb = DepthScheduler(target_attr="depth", start=1, end=4, epochs=EPOCHS - 1)

    def apply_params_fn(packed):
        params_unpack(backends, packed)

    def step_dag(x_t, s_prev):
        """Execute one DAG step given input and previous states."""
        x_map = {"A": x_t}
        s_prev_map = {} if s_prev is None else s_prev
        return dag.step(x_map=x_map, s_tm1_map=s_prev_map, branches=K)

    def forward_sequence():
        """Iterate over the full sequence and accumulate losses."""
        total_task = total_cons = total_coh = 0.0
        s_prev, y_pred_seq, scores = None, [], []
        for t_idx in range(T):
            callbacks.on_step_begin(t_idx, {"step": int(t_idx)})
            s_map, s_hat_map, info_map = step_dag(x_seq[t_idx], s_prev)
            sA, sB, sC = s_map["A"], s_map["B"], s_map["C"]
            y_pred_seq.append(sC)
            scores.append(float(np.linalg.norm(sC)))
            total_task += loss_task(sC, y_target[t_idx])
            if s_prev is not None:
                cons_vals = [loss_cons(s_prev[k], s_map[k], s_hat_map[k]) for k in ("A", "B", "C")]
                total_cons += float(np.mean(cons_vals))
            coh_vals = []
            for key in ("A", "B", "C"):
                branches = info_map[key].get("branches", None)
                if isinstance(branches, np.ndarray) and branches.ndim == 2:
                    coh_vals.append(loss_coh(branches))
            if coh_vals:
                total_coh += float(np.mean(coh_vals))
            s_prev = s_map
            callbacks.on_step_end(t_idx, {
                "step": int(t_idx),
                "norm_sC": float(np.linalg.norm(sC)),
                "task_sC": float(loss_task(sC, y_target[t_idx])),
            })
        task, cons, coh = total_task / T, total_cons / max(1, T - 1), total_coh / T
        total = task + 0.5 * cons + 0.3 * coh
        return total, {"task": task, "cons": cons, "coh": coh, "total": total}, np.asarray(y_pred_seq), np.asarray(scores)

    best = None
    for epoch in range(EPOCHS):
        for be in backends:
            depth_cb.on_epoch_begin(epoch, {"backend": be})
        callbacks.on_epoch_begin(epoch, {"epoch": int(epoch)})
        total0, det0, _, _ = forward_sequence()

        def loss_wrapper():
            l, _, _, _ = forward_sequence()
            return l

        grads = finite_diff_grads(loss_wrapper, params, apply_params_fn)
        params, opt_state = opt.step(params, grads, opt_state)
        apply_params_fn(params)
        total1, det1, _, _ = forward_sequence()
        callbacks.on_epoch_end(epoch, {"epoch": int(epoch), "loss_before": det0, "loss_after": det1})

        if best is None or det1["total"] < best["total"]:
            best = {"epoch": int(epoch), **{k: float(v) for k, v in det1.items()}}

        print(f"[Epoch {epoch}] total_before={det0['total']:.6f} total_after={det1['total']:.6f} depth={[int(be.depth) for be in backends]}")

    _, _, y_pred, scores = forward_sequence()

    # Evaluation
    mape_val = mape(y_target[:, 0], y_pred[:, 0])
    over_val = overshoot(y_target[:, 0], y_pred[:, 0])
    setl_val = settling_time(y_target[:, 0], y_pred[:, 0], tol=0.05)
    rob_val = robustness(y_target[:, 0], y_pred[:, 0])
    auc_early = early_roc_auc(y_true=y_anom, scores=scores, horizon=3)
    thr = np.percentile(scores, 80)
    rec_lag = recall_at_lag(y_true=y_anom, y_pred=(scores > thr).astype(float), lag=3)

    print("\n=== Final Metrics ===")
    print(f"MAPE (%):          {mape_val:.6f}")
    print(f"Overshoot:         {over_val:.6f}")
    print(f"Settling Time:     {int(setl_val)} samples")
    print(f"Robustness:        {rob_val:.6f}")
    print(f"Early ROC-AUC@H=3: {auc_early:.6f}")
    print(f"Recall@Lag=3:      {rec_lag:.6f}")
    print("\nBest epoch snapshot:", best)

    if LOG_PATH.exists():
        print("\nTelemetry JSONL →", LOG_PATH.resolve())

    return {
        "best": best,
        "metrics": {
            "mape": float(mape_val),
            "overshoot": float(over_val),
            "settling_time": int(setl_val),
            "robustness": float(rob_val),
            "early_auc_h3": float(auc_early),
            "recall_lag3": float(rec_lag),
        },
    }


# ============================================================================
# 5) Entry point
# ============================================================================
if __name__ == "__main__":
    summary = quantum_hypercore_case_study()
    print("\nSummary:")
    print(json.dumps(summary, indent=2))
