#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hypercausal Core Demo — physical drift (phase + detuning + readout bias)
"""

import os
import sys
import re
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from typing import List
from pandas.plotting import parallel_coordinates


# ======================================================================
# Ensure src/ path
# ======================================================================
def ensure_src_path():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(script_dir, ".."))
    src_path = os.path.join(repo_root, "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    return repo_root, src_path

# ======================================================================
# Import  modules
# ======================================================================
repo_root, src_path = ensure_src_path()

from qmlhc.backends.pennylane_backend import PennyLaneBackend
from qmlhc.hc.node import HCNode
from qmlhc.hc.policy import MeanPolicy
from qmlhc.loss.task import MSELoss
from qmlhc.loss.consistency import ConsistencyLoss
from qmlhc.loss.coherence import CoherenceLoss
from qmlhc.callbacks.telemetry import MemoryLogger
from qmlhc.callbacks.depth_control import DepthScheduler
from qmlhc.callbacks.base import CallbackList
from qmlhc.core.model import HCModel
from qmlhc.core.backend import BackendConfig
from qmlhc.optim.registry_numpy import create_optimizer_numpy


# =====================================================================
# Utility
# =====================================================================
def ensure_dirs(*dirs):
    for d in dirs:
        os.makedirs(d, exist_ok=True)

def _next_numbered_path(output_dir: str, base: str) -> str:
    """
    Return a path like <output_dir>/<base>_NNN.png using the next available index.
    """
    os.makedirs(output_dir, exist_ok=True)
    pat = re.compile(rf"^{re.escape(base)}_(\d+)\.png$")
    max_n = 0
    for name in os.listdir(output_dir):
        m = pat.match(name)
        if m:
            max_n = max(max_n, int(m.group(1)))
    return os.path.join(output_dir, f"{base}_{max_n+1:03d}.png")

def _savefig_numbered(output_dir: str, base: str, fig=None, dpi: int = 160):
    """
    Save the current (or provided) figure as <base>_NNN.png with a tight bounding box.
    """
    path = _next_numbered_path(output_dir, base)
    (fig or plt.gcf()).savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"[FIG] Saved: {path}")

# --- drift (hardware-style) helpers --------------------------------
def hardware_drift_emulation (epoch, total_epochs, qubits,
                   freq_ppm=12e-6,      # “slow” drift ~ tens of ppm
                   phase_max=0.03,      # maximum accumulated phase in radians (1 sinusoidal cycle over the full run)
                   readout_bias_max=0.12):  # readout bias (e.g., up to ~12%)
    """
    Emulate hardware-level drift commonly observed in QPUs:
      - Accumulated phase (due to frequency drift) -> additive offset in parameters
      - Detuning/frequency (ppm) -> slow multiplicative scaling
      - Readout bias -> bias applied post-measurement
    """
    # 1) Slow per-qubit phase (simulates frequency drift -> phase)
    phase = 2.0 * np.pi * (epoch / max(1, total_epochs - 1))
    phase_drift = phase_max * np.sin(phase) * np.ones(qubits)

    # 2) Micro detuning/frequency interpreted as a mild amplitude scale (accumulated in ppm)
    detuning_scale = 1.0 + freq_ppm * epoch
    amp_drift_scale = np.full(qubits, detuning_scale, dtype=float)

    # 3) Small oscillatory readout bias
    readout_bias = readout_bias_max * (0.5 + 0.5 * np.sin(phase + np.pi/3.0))

    return phase_drift, amp_drift_scale, float(readout_bias)

# =====================================================================
# Build system 
# =====================================================================
def build_system(qubits: int = 7, shots: int = 1024, branches: int = 20,
                 sched_epochs: int = 350):  # <-- spread depth schedule across full training
    cfg = BackendConfig(output_dim=qubits, shots=shots)
    backend = PennyLaneBackend(cfg, num_qubits=qubits, shots=shots)

    policy = MeanPolicy()
    node = HCNode(backend=backend, policy=policy)
    model = HCModel(nodes=[node])

    task_loss = MSELoss()
    cons_loss = ConsistencyLoss(alpha=1.0, beta=0.8)
    coh_loss = CoherenceLoss(mode="variance")

    telemetry = MemoryLogger()
    depth_sched = DepthScheduler(target_attr="depth", start=1, end=5, epochs=sched_epochs)
    callbacks = CallbackList([telemetry, depth_sched])
    return model, task_loss, cons_loss, coh_loss, callbacks

# === Optimizer wiring helpers (SPSA / Trust-KL / MPC ready) ===
def evaluate_stats(model, params, ctx):
    """Compute task/consistency/coherence/total losses and return model info (branches)."""
    x0 = np.asarray(ctx["x0"], dtype=float)
    drift = np.asarray(ctx["drift"], dtype=float)                # phase (additive)
    target = np.asarray(ctx["target"], dtype=float)
    branches = int(ctx["branches"])
    task_loss, cons_loss, coh_loss = ctx["losses"]

    # Additional signals
    amp_scale = np.asarray(ctx.get("amp_scale", 1.0), dtype=float)  # detuning (multiplicative)
    readout_bias = float(ctx.get("readout_bias", 0.0))              # readout bias

    alpha = float(params["alpha"])
    x = alpha * x0
    s_tm1 = np.zeros_like(x)

    # Forward pass with hardware-style physics: detuning + phase
    x_in = amp_scale * (x + drift)
    s_t, s_hat, info = model.forward(x_in, s_tm1, branches)

    # Readout bias (post-measurement)
    s_t  = (1.0 - readout_bias) * s_t  + readout_bias * np.sign(s_t)
    s_hat = (1.0 - readout_bias) * s_hat + readout_bias * np.sign(s_hat)

    lt = float(task_loss(s_t, target))
    lc = float(cons_loss(s_tm1, s_t, s_hat))
    lq = float(coh_loss(info.get("branches", np.vstack([s_t, s_hat]))))
    total = lt + 0.5 * (lc + lq)
    return {"task": lt, "cons": lc, "coh": lq, "total": total, "info": info}

def refresh_info(model, params, ctx):
    """Recompute info(dict) used by the trust-region KL guard."""
    return evaluate_stats(model, params, ctx)["info"]

def kl_fn(old_info, new_info):
    """Symmetric KL-like proxy based on branch statistics."""
    from qmlhc.optim.numpy_optim.utils import kl_proxy
    return float(kl_proxy(old_info, new_info))

# =====================================================================
# Run experiment
# =====================================================================
def run_experiment(epochs: int = 350, qubits: int = 7, branches: int = 20, drift_amp: float = 0.30, shots: int = 1024):
    # Note: drift_amp remains as a “legacy” parameter (unused in hardware-style mode) to preserve the call signature.
    model, task_loss, cons_loss, coh_loss, callbacks = build_system(
        qubits=qubits,
        shots=shots,
        branches=branches,
        sched_epochs=epochs
    )

    x0 = np.linspace(0.1, 0.3, qubits)
    alpha = 1.0

    # === Initialize SPSA + Trust-KL Optimizer ===
    base_opt = create_optimizer_numpy("spsa", lr0=0.05, eps0=0.10, antithetic=True, clip=4.0)
    opt = create_optimizer_numpy("trust-kl", base_opt=base_opt, delta_kl=0.02, backtrack=0.7, max_backtracks=8)
    params = {"alpha": 1.0}
    opt.initialize(params)

    rows = []
    header = [
        "epoch", "alpha",
        "task_loss", "cons_loss", "coh_loss", "total_loss",
        "mean_s", "mean_mu", "drift_real"
    ]

    for epoch in range(epochs):
        ctx0 = {"epoch": epoch, "model": model}
        callbacks.on_epoch_begin(epoch, ctx0)

        # --- drift (hardware-style) ---
        phase_drift, amp_scale, readout_bias = hardware_drift_emulation (epoch, epochs, qubits)

        # Nominal target
        target = np.linspace(0.4, 0.9, qubits)

        # Physical signals for this epoch
        drift = phase_drift  # “drift_real” to log and pass to the optimizer

        # --- Build optimizer context for this epoch (with physics) ---
        context = {
            "epoch": epoch,
            "epochs": epochs,
            "model": model,
            "x0": x0,
            "drift": drift,                 # phase (additive)
            "amp_scale": amp_scale,         # detuning (multiplicative)
            "readout_bias": readout_bias,   # readout bias
            "target": target,
            "losses": (task_loss, cons_loss, coh_loss),
            "branches": branches,
            "info": {},
            "kl_fn": kl_fn,
            "refresh_info": refresh_info,
        }
        context["info"] = refresh_info(model, {"alpha": float(params["alpha"])}, context)

        # === Optimizer-driven update of alpha ===
        params, opt_state = opt.step_params(model, params, context)
        alpha = float(params["alpha"])

        # === Official forward pass with the same physics ===
        x = alpha * x0
        s_tm1 = np.zeros_like(x)
        x_in = amp_scale * (x + drift)
        s_t, s_hat, info = model.forward(x_in, s_tm1, branches)

        # Readout bias (post-measurement)
        s_t  = (1.0 - readout_bias) * s_t  + readout_bias * np.sign(s_t)
        s_hat = (1.0 - readout_bias) * s_hat + readout_bias * np.sign(s_hat)

        # Losses and metrics
        l_task = task_loss(s_t, target)
        l_cons = cons_loss(s_tm1, s_t, s_hat)
        l_coh = coh_loss(info["branches"])
        total_loss = l_task + 0.5 * (l_cons + l_coh)

        mean_s = float(np.mean(s_t))
        mean_mu = float(np.mean(s_hat))

        step_context = {"epoch": epoch, "loss": total_loss, "alpha": alpha}
        callbacks.on_step_end(epoch, step_context)

        rows.append([
            epoch, alpha,
            l_task, l_cons, l_coh, total_loss,
            mean_s, mean_mu,
            float(drift[0])  # “drift_real” used in plots
        ])

        print(f"Epoch {epoch:04d} | α={alpha:.4f} | "
              f"Task={l_task:.5f} Cons={l_cons:.5f} Coh={l_coh:.5f} Tot={total_loss:.5f}")

        callbacks.on_epoch_end(epoch, {"epoch": epoch, "loss": total_loss})

    return rows, header

# ======================================================================
# Save CSV & Plots
# ======================================================================
def save_and_plot_quantum_style(rows, header, output_dir="runs/hc_full_demo"):
    ensure_dirs(output_dir)
    df = pd.DataFrame(rows, columns=header)
    csv_path = os.path.join(output_dir, "results.csv")
    df.to_csv(csv_path, index=False)
    print(f"[OK] Saved CSV: {csv_path}")

    # 1) Loss & Coherence (dual y-axes)
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax2 = ax1.twinx()
    ax1.plot(df["epoch"], df["total_loss"], color="tab:red", linewidth=2)
    ax2.plot(df["epoch"], df["coh_loss"],   color="tab:blue", linewidth=2)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss", color="tab:red")
    ax2.set_ylabel("Coherence", color="tab:blue")
    fig.tight_layout()
    _savefig_numbered(output_dir, "loss_coherence", fig)

    # 2) Consistency vs Coherence (scatter), viridis + colorbar
    plt.figure(figsize=(6, 6))
    sc = plt.scatter(df["coh_loss"], df["cons_loss"], c=df["epoch"], cmap="viridis", s=40)
    plt.xlabel("Coherence")
    plt.ylabel("Consistency")
    plt.title("Consistency vs Coherence")
    cbar = plt.colorbar(sc)
    cbar.set_label("Epoch")
    plt.grid(True)
    _savefig_numbered(output_dir, "consistency_vs_coherence")

    # 3) Alpha (Feedback) Over Epochs — green
    plt.figure(figsize=(10, 4))
    plt.plot(df["epoch"], df["alpha"], color="tab:green", linewidth=2)
    plt.title("Alpha (Feedback) Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Alpha")
    plt.grid(True)
    _savefig_numbered(output_dir, "alpha_over_epochs")

    # 4) State Alignment — purple vs orange
    plt.figure(figsize=(12, 5))
    plt.plot(df["epoch"], df["mean_s"],  color="tab:purple", linewidth=2, label="mean(S_t)")
    plt.plot(df["epoch"], df["mean_mu"], color="tab:orange", linewidth=2, label="mean(mu_fut)")
    plt.title("State Alignment")
    plt.xlabel("Epoch")
    plt.ylabel("Mean Values")
    plt.legend()
    plt.grid(True)
    _savefig_numbered(output_dir, "state_alignment")

    # 5) 3D Causal Space — plasma, colorbar 'Epoch'
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    sc3 = ax.scatter(df["coh_loss"], df["cons_loss"], df["alpha"],
                     c=df["epoch"], cmap="plasma", s=60, alpha=0.9)
    ax.set_xlabel("Coherence")
    ax.set_ylabel("Consistency")
    ax.set_zlabel("Alpha")
    ax.set_title("3D Causal Space")
    cb3 = fig.colorbar(sc3, ax=ax, shrink=0.65)
    cb3.set_label("Epoch")
    plt.tight_layout()
    _savefig_numbered(output_dir, "causal_space_3d")

    # 6) Drift Signals Only (Real vs Proxy)
    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax2 = ax1.twinx()

    drift_proxy = np.abs(df["alpha"].diff().fillna(0.0))
    drift_proxy_s = drift_proxy.rolling(5, min_periods=1).mean()

    ax1.plot(df["epoch"], df["drift_real"], color="tab:purple", linewidth=2, label="Drift Real (phase-like)")
    ax2.plot(df["epoch"], drift_proxy_s, color="tab:red", linestyle="--", linewidth=1.8, label="|ΔAlpha| Proxy (roll=5)")

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Drift Real")
    ax2.set_ylabel("|ΔAlpha| Proxy")
    ax1.set_title("Drift Signals Only")

    l1, lab1 = ax1.get_legend_handles_labels()
    l2, lab2 = ax2.get_legend_handles_labels()
    ax1.legend(l1 + l2, lab1 + lab2, loc="upper right", frameon=False)

    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    _savefig_numbered(output_dir, "drift_signals_only")

    # 7) Alpha Sensitivity (Δα per Epoch)
    plt.figure(figsize=(10, 5))
    d_alpha = df["alpha"].diff().fillna(0.0)
    plt.plot(df["epoch"], d_alpha, color="tab:gray", linewidth=2)
    plt.title("Alpha Sensitivity (Δα per Epoch)")
    plt.xlabel("Epoch")
    plt.ylabel("ΔAlpha")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    _savefig_numbered(output_dir, "alpha_sensitivity")

    # 8) Causal Phase Portrait (Temporal Path)
    plt.figure(figsize=(7, 6))
    plt.plot(df["coh_loss"], df["cons_loss"], color="tab:purple", linewidth=1.5, alpha=0.8)
    sc_path = plt.scatter(df["coh_loss"], df["cons_loss"],
                          c=df["epoch"], cmap="plasma", s=28, alpha=0.9)
    plt.xlabel("Coherence")
    plt.ylabel("Consistency")
    plt.title("Causal Phase Portrait (Temporal Path)")
    cbp = plt.colorbar(sc_path)
    cbp.set_label("Epoch")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    _savefig_numbered(output_dir, "causal_phase_portrait")

    # 9) Drift vs Coherence Dynamics
    plt.figure(figsize=(9, 5))
    drift_proxy = np.abs(df["alpha"].diff().fillna(0.0))
    plt.plot(df["epoch"], df["coh_loss"],   color="tab:blue",  linewidth=2, label="Coherence")
    plt.plot(df["epoch"], drift_proxy,      color="tab:red",   linewidth=2, label="|ΔAlpha| (Drift Proxy)")
    plt.title("Drift vs Coherence Dynamics")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.legend(frameon=False)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    _savefig_numbered(output_dir, "drift_vs_coherence")

    # 10) Parallel Causal Dimensions (epoch grouped)
    plt.figure(figsize=(8,6))
    subset = df[["coh_loss", "cons_loss", "alpha", "epoch"]].copy()
    subset["epoch_group"] = pd.cut(subset["epoch"], bins=5, labels=False)
    parallel_coordinates(subset, "epoch_group", color=plt.cm.plasma(np.linspace(0,1,5)), alpha=0.6)
    plt.title("Parallel Causal Dimensions")
    plt.xlabel("Metrics")
    plt.ylabel("Value")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    _savefig_numbered(output_dir, "causal_parallel")


    return df

# ======================================================================
# Main
# ======================================================================
if __name__ == "__main__":
    # You may adjust epochs/qubits/branches according to your test profile
    rows, header = run_experiment(epochs=300, qubits=7, branches=20, drift_amp=0.30, shots=1024)
    save_and_plot_quantum_style(rows, header, output_dir="runs/hypercausal_physical_drift_resilience")
