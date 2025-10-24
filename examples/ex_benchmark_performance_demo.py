#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example 05 – Hypercausal Benchmark Performance
----------------------------------------------
Measures performance and stability of the hypercausal engine (qmlhc) under
different model and sequence sizes. Records:

- Time per forward and per epoch
- Peak memory (via tracemalloc)
- Quality metrics (RMSE, Overshoot, Robustness)
- Stability (std of total loss across runs)

Outputs
-------
- JSONL: .benchmarks/qmlhc_benchmarks.jsonl
- CSV  : .benchmarks/qmlhc_benchmarks.csv
- PNGs : .benchmarks/bench_times.png, .benchmarks/bench_scaling.png (if matplotlib is available)

Run:
    python src/examples/05_hypercausal_benchmark_performance.py
"""

from __future__ import annotations
from pathlib import Path
import json
import csv
import time
import tracemalloc
import itertools
import numpy as np

# Core components and contracts
from qmlhc.core import BackendConfig, QuantumBackend as BaseBackend, HCModel
# Nodes and policies
from qmlhc.hc import HCNode, MeanPolicy
# Projectors
from qmlhc.predictors import LinearProjector
# Losses
from qmlhc.loss import MSELoss, ConsistencyLoss, CoherenceLoss
# Optimizer
from qmlhc.optim import make_gradient_descent
# Metrics
from qmlhc.metrics import overshoot, robustness

# ------------------------------------------------------------
# Optional: plotting utilities (if matplotlib available)
# ------------------------------------------------------------
_HAS_MPL = False
try:
    import matplotlib.pyplot as plt  # type: ignore
    _HAS_MPL = True
except Exception:
    _HAS_MPL = False


# ------------------------------------------------------------
# Depth-aware lightweight backend
# ------------------------------------------------------------
class DepthAwareBackend(BaseBackend):
    """Backend with tunable recursive depth and projection span."""

    def __init__(self, config: BackendConfig, w: float = 0.95, b: float = 0.02, proj_span: float = 0.25):
        super().__init__(config)
        self.w = float(w)
        self.b = float(b)
        self.depth = 1
        self._base_span = float(proj_span)
        self._span_floor = 0.08
        self._projector = LinearProjector(weight=1.0, bias=0.0, span=self._base_span)

    def get_params(self):
        """Return parameters as 1-element arrays."""
        return {"w": np.array([self.w], dtype=float), "b": np.array([self.b], dtype=float)}

    def set_params(self, params: dict):
        """Set internal parameters."""
        if "w" in params:
            self.w = float(np.asarray(params["w"]).reshape(()))
        if "b" in params:
            self.b = float(np.asarray(params["b"]).reshape(()))

    def run(self, params: dict | None = None) -> np.ndarray:
        """Apply tanh recursion for the configured depth."""
        if params:
            self.set_params(params)
        s = self._require_input().astype(float)
        for _ in range(max(1, int(self.depth))):
            s = np.tanh(self.w * s + self.b)
        return self._validate_state(s)

    def project_future(self, s_t: np.ndarray, branches: int = 2) -> np.ndarray:
        """Generate future projections with adaptive span."""
        s = self._validate_state(s_t)
        k = max(2, int(branches))
        span = max(self._span_floor, self._base_span / (1.0 + 0.3 * (self.depth - 1)))
        self._projector = LinearProjector(weight=1.0, bias=0.0, span=span)
        fut = self._projector.project(s, branches=k)
        return self._validate_branches(fut)


# ------------------------------------------------------------
# Model builder (3-node chain)
# ------------------------------------------------------------
def build_model_chain(D: int, seed: int = 7):
    """Build a simple 3-node chain with depth-aware backends."""
    cfg = BackendConfig(output_dim=D, seed=seed)
    b0 = DepthAwareBackend(cfg, w=0.90, b=0.03, proj_span=0.22)
    b1 = DepthAwareBackend(cfg, w=0.97, b=0.02, proj_span=0.25)
    b2 = DepthAwareBackend(cfg, w=1.05, b=0.00, proj_span=0.30)
    for be in (b0, b1, b2):
        be.depth = 1
    pol = MeanPolicy()
    n0, n1, n2 = HCNode(b0, pol), HCNode(b1, pol), HCNode(b2, pol)
    model = HCModel([n0, n1, n2])
    backends = [b0, b1, b2]
    return model, backends


# ------------------------------------------------------------
# Forward + loss computation over a full sequence
# ------------------------------------------------------------
def forward_epoch(model: HCModel, backends, x_seq: np.ndarray, target_seq: np.ndarray, K: int):
    """
    Compute full-sequence forward pass and all loss components.

    Returns
    -------
    tuple
        (total_loss, task_loss, consistency_loss, coherence_loss, predictions)
    """
    T, D = x_seq.shape
    mse, cons, coh = MSELoss(), ConsistencyLoss(0.8, 1.2), CoherenceLoss(mode="variance")

    total_task = total_cons = total_coh = 0.0
    s_tm1 = None
    y_last = []
    for t in range(T):
        s_t, s_hat, infos = model.forward_chain(x_seq[t], s_tm1=s_tm1, branches=K)
        y_last.append(s_t)
        total_task += mse(s_t, target_seq[t])
        if s_tm1 is not None:
            total_cons += cons(s_tm1, s_t, s_hat)
        coh_vals = []
        for info in infos:
            br = info.get("branches", None)
            if isinstance(br, np.ndarray) and br.ndim == 2:
                coh_vals.append(coh(br))
        if coh_vals:
            total_coh += float(np.mean(coh_vals))
        s_tm1 = s_t

    task = total_task / T
    cns = total_cons / max(1, T - 1)
    ch = total_coh / T
    total = task + 0.5 * cns + 0.3 * ch
    return total, task, cns, ch, np.asarray(y_last)


# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------
def synthetic_sequence(T: int, D: int, seed: int = 11):
    """Generate a synthetic time series with low noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(T, dtype=float)
    x = np.stack([
        0.30 * np.sin(0.35 * t + 0.00),
        0.20 * np.sin(0.35 * t + 0.70),
        0.10 * np.cos(0.35 * t + 0.30),
    ], axis=1)
    if D > 3:
        reps = int(np.ceil(D / 3))
        x = np.tile(x, (1, reps))[:, :D]
    x += 0.01 * rng.standard_normal(size=x.shape)
    target = np.zeros((T, D), dtype=float)
    return x, target


def rmse_1d(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute RMSE for 1D arrays."""
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def benchmark_once(D: int, K: int, T: int, depth_schedule=(1, 2, 3), seed=123):
    """
    Execute a single benchmark run with light “training” (one update per depth).
    Measures time, memory, and stability metrics.

    Returns
    -------
    dict
        Benchmark summary with performance statistics.
    """
    x_seq, target_seq = synthetic_sequence(T, D, seed=seed)
    model, backends = build_model_chain(D, seed=seed)

    times_epoch, times_forward_avg, mem_peaks = [], [], []
    totals, rmses, overs, robs = [], [], [], []

    params = {}
    for i, be in enumerate(backends):
        p = be.get_params()
        for k, v in p.items():
            params[f"b{i}_{k}"] = v.astype(float)
    opt = make_gradient_descent(lr=2e-2)
    state = opt.initialize(params)

    def set_params(packed):
        for i, be in enumerate(backends):
            sub = {}
            for k in ("w", "b"):
                key = f"b{i}_{k}"
                if key in packed:
                    sub[k] = packed[key]
            be.set_params(sub)

    tracemalloc.start()

    for depth in depth_schedule:
        for be in backends:
            be.depth = int(depth)
        t0 = time.perf_counter()
        total, task, cns, ch, y_pred = forward_epoch(model, backends, x_seq, target_seq, K)
        t1 = time.perf_counter()
        epoch_time = t1 - t0
        times_epoch.append(epoch_time)
        times_forward_avg.append(epoch_time / T)

        rmse_val = rmse_1d(target_seq[:, 0], y_pred[:, 0])
        over_val = overshoot(target_seq[:, 0], y_pred[:, 0])
        rob_val = robustness(target_seq[:, 0], y_pred[:, 0])

        totals.append(float(total))
        rmses.append(rmse_val)
        overs.append(over_val)
        robs.append(rob_val)

        _, peak = tracemalloc.get_traced_memory()
        mem_peaks.append(int(peak))

        grads = {k: np.array([0.0], dtype=float) for k in params.keys()}
        params, state = opt.step(params, grads, state)
        set_params(params)

    tracemalloc.stop()

    return {
        "D": D,
        "K": K,
        "T": T,
        "depths": list(depth_schedule),
        "time_epoch_mean": float(np.mean(times_epoch)),
        "time_epoch_std": float(np.std(times_epoch)),
        "time_forward_mean": float(np.mean(times_forward_avg)),
        "time_forward_std": float(np.std(times_forward_avg)),
        "mem_peak_bytes_max": int(max(mem_peaks) if mem_peaks else 0),
        "total_mean": float(np.mean(totals)),
        "total_std": float(np.std(totals)),
        "rmse_mean": float(np.mean(rmses)),
        "rmse_std": float(np.std(rmses)),
        "overshoot_mean": float(np.mean(overs)),
        "robustness_mean": float(np.mean(robs)),
    }


def run_benchmarks():
    """
    Run benchmark sweeps over multiple (D, K, T) configurations.

    Returns
    -------
    tuple
        (results, output_directory)
    """
    bench_dir = Path(".benchmarks")
    bench_dir.mkdir(parents=True, exist_ok=True)

    Ds, Ks, Ts = [3, 6], [3, 5, 9], [48, 96]
    depth_schedule = (1, 2, 3)
    configs = list(itertools.product(Ds, Ks, Ts))
    results = []
    REPEATS = 3
    run_id = int(time.time())

    jsonl_path = bench_dir / "qmlhc_benchmarks.jsonl"
    csv_path = bench_dir / "qmlhc_benchmarks.csv"

    with jsonl_path.open("w") as jf, csv_path.open("w", newline="") as cf:
        fieldnames = [
            "run_id", "D", "K", "T", "depths",
            "time_epoch_mean", "time_epoch_std",
            "time_forward_mean", "time_forward_std",
            "mem_peak_bytes_max",
            "total_mean", "total_std",
            "rmse_mean", "rmse_std",
            "overshoot_mean", "robustness_mean",
        ]
        writer = csv.DictWriter(cf, fieldnames=fieldnames)
        writer.writeheader()

        for (D, K, T) in configs:
            reps = [benchmark_once(D, K, T, depth_schedule=depth_schedule, seed=123 + rep) for rep in range(REPEATS)]
            agg = {}
            for k in reps[0].keys():
                vals = [r[k] for r in reps]
                if isinstance(vals[0], (int, float, np.floating)):
                    agg[k] = float(np.mean(vals))
                else:
                    agg[k] = vals[0]
            row = {"run_id": run_id, **agg}
            jf.write(json.dumps(row) + "\n")
            writer.writerow(row)
            results.append(row)

    print(f"\nBenchmark complete. Results saved to:\n- {jsonl_path.resolve()}\n- {csv_path.resolve()}")
    return results, bench_dir


def make_plots(results, bench_dir: Path):
    """Generate benchmark performance plots if matplotlib is installed."""
    if not _HAS_MPL:
        print("matplotlib not available: skipping plots.")
        return

    labels = [f"D{r['D']}-K{r['K']}-T{r['T']}" for r in results]
    times = [r["time_epoch_mean"] for r in results]

    plt.figure(figsize=(10, 4))
    plt.plot(range(len(times)), times, marker="o")
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
    plt.ylabel("Time per epoch (s)")
    plt.title("QMLHC Benchmark – Average Epoch Time")
    plt.tight_layout()
    p1 = bench_dir / "bench_times.png"
    plt.savefig(p1, dpi=150)
    plt.close()
    print(f"Plot saved: {p1.resolve()}")

    Ds_sorted = sorted(set(r["D"] for r in results))
    Ks_sorted = sorted(set(r["K"] for r in results))
    Z = np.zeros((len(Ds_sorted), len(Ks_sorted)), dtype=float)
    for i, D in enumerate(Ds_sorted):
        for j, K in enumerate(Ks_sorted):
            vals = [r["time_epoch_mean"] for r in results if r["D"] == D and r["K"] == K]
            Z[i, j] = float(np.mean(vals)) if vals else np.nan

    plt.figure(figsize=(6, 4))
    im = plt.imshow(Z, aspect="auto")
    plt.colorbar(im, label="Time per epoch (s)")
    plt.xticks(range(len(Ks_sorted)), [f"K={k}" for k in Ks_sorted])
    plt.yticks(range(len(Ds_sorted)), [f"D={d}" for d in Ds_sorted])
    plt.title("QMLHC Benchmark – Scaling Map (Avg over T)")
    plt.tight_layout()
    p2 = bench_dir / "bench_scaling.png"
    plt.savefig(p2, dpi=150)
    plt.close()
    print(f"Plot saved: {p2.resolve()}")


def main():
    """Run benchmarks and display a quick summary."""
    results, bench_dir = run_benchmarks()
    make_plots(results, bench_dir)

    best = min(results, key=lambda r: r["time_epoch_mean"])
    worst = max(results, key=lambda r: r["time_epoch_mean"])
    print("\nQuick Summary:")
    print(f"- Fastest config : D={best['D']} K={best['K']} T={best['T']}  "
          f"time/epoch={best['time_epoch_mean']:.4f}s  RMSE={best['rmse_mean']:.4f}  Robustness={best['robustness_mean']:.3f}")
    print(f"- Slowest config : D={worst['D']} K={worst['K']} T={worst['T']} "
          f"time/epoch={worst['time_epoch_mean']:.4f}s  RMSE={worst['rmse_mean']:.4f}  Robustness={worst['robustness_mean']:.3f}")


if __name__ == "__main__":
    main()
