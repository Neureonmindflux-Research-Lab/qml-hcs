====================================================
Full Hypercausal System Demo (PennyLane + SPSA + KL)
====================================================
.. |Δα| unicode:: U+0394 U+03B1
   :trim:
   
Overview
========
This example demonstrates a **full hypercausal system** on a PennyLane backend,
combining causal feedback with **SPSA derivative-free optimization** and a
**KL-constrained trust-region guard**.
Dynamic depth scheduling and multi-metric telemetry track how the internal
feedback parameter :math:`\alpha` co-evolves with coherence and consistency
under non-stationary drift.

The model evaluates the stability of quantum-inspired feedback systems under
drift, using adaptive control of internal causal parameters to maintain
temporal coherence and counterfactual consistency.

Core Objectives
===============
- **Hypercausal Feedback Modeling:** multi-directional causal propagation.
- **Quantum-Inspired Efficiency:** simulated superposition for reduced cost.
- **Deterministic–Stochastic Integration:** deterministic and probabilistic backends.
- **Causal Stability Measurement:** tracking coherence, consistency, and feedback drift.
- **Scientific Transparency:** reproducible, open experimentation.

Experimental Setup
==================
We simulate a minimal quantum-like node structure using PennyLane circuits.
The internal state :math:`S_t` evolves via recursive updates modulated by
a causal feedback coefficient :math:`\alpha`, representing the degree of
self-consistent causal correction.

The environment introduces a small non-stationary drift (sinusoidal) to
evaluate robustness of the feedback mechanism.

System Architecture & Modules
=============================
This demo wires the high-level components of **QML-HCS** as follows:

- **Backend:** ``PennyLaneBackend`` with ``BackendConfig(output_dim=qubits, shots=shots)``.
- **Policy:** ``MeanPolicy`` (branch aggregation by mean).
- **Node / Model:** a single ``HCNode`` wrapped by ``HCModel``.
- **Losses:** ``MSELoss`` (task), ``ConsistencyLoss`` (temporal), ``CoherenceLoss`` (branch coherence).
- **Callbacks:** ``MemoryLogger`` (telemetry) and ``DepthScheduler`` (progressive circuit depth).
- **Optimizers:** base **SPSA** (derivative-free) wrapped by a **KL-bounded trust-region** controller.

Data & Non-Stationary Drift
===========================
We use a fixed seed input vector ``x0 ∈ ℝ^{qubits}`` and inject a **sinusoidal drift**
with amplitude ``drift_amp`` and phase coupled to the epoch:

.. math::

   \text{drift}_t \;=\; A \cdot \sin\!\Big(\frac{2\pi\,t}{T-1}\Big)\,\mathbf{1},

where :math:`A=\texttt{drift\_amp}` and :math:`T` is the number of epochs.
The **effective input** is the feedback-scaled signal

.. math::

   x_t \;=\; \alpha_t \, x_0 \;+\; \text{drift}_t.

A simple linear target ramp :math:`y_t` is used to evaluate task loss, while
consistency/coherence are computed from the model's present state :math:`S_t`
and one-step future projection :math:`\mu_{\mathrm{fut}}`.

Backend, Node & Depth Scheduling
================================
- ``PennyLaneBackend`` instantiates the quantum-inspired circuit with the given
  number of qubits and shots.
- ``HCNode`` + ``MeanPolicy`` produce a vector state and per-branch statistics.
- ``DepthScheduler(start=1, end=5, epochs=E)`` gradually increases circuit depth
  **across the full training** (``E = epochs``), enabling a mild curriculum:
  shallow circuits during early exploration and deeper circuits once feedback stabilizes.

Mathematical Formulation
========================
Let :math:`x_t` be the input at epoch :math:`t`. The internal state and one–step
future projection are

.. math::

   S_t \;=\; f_\theta(x_t, S_{t-1}, \alpha_t),
   \qquad
   \mu_{\mathrm{fut}} \;=\; g_\theta(x_t, S_t, \alpha_t),

where :math:`\theta` denotes circuit parameters and :math:`\alpha_t` is the
feedback coefficient.

**Task loss (MSE).**

.. math::

   \mathcal{L}_{\mathrm{task}}
   \;=\;
   \tfrac{1}{N}\sum_{t=1}^{N}\,\|\,h_\theta(S_t) - y_t\,\|_2^2.

**Temporal consistency loss.**

.. math::

   \mathcal{L}_{\mathrm{consistency}}
   \;=\;
   \tfrac{1}{N}\sum_{t=1}^{N}\,\|\,S_t - \mu_{\mathrm{fut}}\,\|_2^2.

**Coherence loss (branch coherence).**
Let :math:`p_t(b)` be the normalized branch activation:

.. math::

   \mathcal{L}_{\mathrm{coherence}}
   \;=\;
   \tfrac{1}{N}\sum_{t=1}^{N}
   \Big[ - \sum_{b} p_t(b)\,\log p_t(b) \Big].

**Total objective (minimized).**

.. math::

   \mathcal{L}_{\mathrm{total}}
   \;=\;
   \mathcal{L}_{\mathrm{task}}
   \;+\;
   \tfrac12\!\left(
     \mathcal{L}_{\mathrm{consistency}}
     +
     \mathcal{L}_{\mathrm{coherence}}
   \right).

Optimizers & Trust-Region (SPSA + KL)
=====================================
SPSA updates the feedback parameter :math:`\alpha` using two antithetic cost
evaluations, while a **KL-bounded trust-region** ensures distributional
stability of the state branches between epochs.

- **SPSA (antithetic) step:**

  .. math::

      \hat{g} = \frac{\mathcal{L}(\alpha+\varepsilon \Delta)
      - \mathcal{L}(\alpha-\varepsilon \Delta)}{2\varepsilon}\,\Delta,\quad
      \alpha_{\text{new}} \leftarrow \alpha - \eta\,\hat{g}

  with :math:`\Delta \in \{-1,+1\}^{d}`. This derivative-free estimate is
  robust to measurement noise and stochastic loss surfaces.

- **KL trust-region guard:** accept the proposal only if the symmetric KL proxy
  between **old** and **new** branch statistics satisfies

  .. math::

      D_{\mathrm{KL}}^{\mathrm{sym}}(p_{\text{old}}\parallel p_{\text{new}})
      \le \delta_{\mathrm{KL}},

  otherwise perform backtracking line-search (factor 0.7, up to 8 attempts).

This separation lets SPSA **explore**, while the trust-region **preserves
temporal coherence**, which is key to hypercausal stability.

Trust-Region Stability Metric (KL)
==================================
To prevent disruptive jumps between epochs, we bound the **symmetric KL** on
branch statistics:

.. math::

   D_{\mathrm{KL}}^{\mathrm{sym}}
   \big(p_{\text{old}} \parallel p_{\text{new}}\big)
   \;=\;
   D_{\mathrm{KL}}(p_{\text{old}} \parallel p_{\text{new}})
   +
   D_{\mathrm{KL}}(p_{\text{new}} \parallel p_{\text{old}}).

A step is **accepted** only if  
:math:`D_{\mathrm{KL}}^{\mathrm{sym}} \le \delta_{\mathrm{KL}}`.
Otherwise a **backtracking line-search** (factor 0.7, up to 8 trials) scales
the proposal until the constraint is met.


Choosing and Swapping Optimizers
================================
This framework allows you to easily switch between multiple optimizers implemented
in ``qmlhc.optim.numpy_optim`` using ``create_optimizer_numpy(name, **kwargs)``.

Each optimizer explores a different adaptation strategy for the feedback coefficient
:math:`\alpha_t`, offering varied robustness and convergence properties depending
on the noise level, drift intensity, or causal stability required.

Below is a concise summary of each available optimizer:

- **SPSA (Simultaneous Perturbation Stochastic Approximation)** - Derivative-free
  stochastic optimization with antithetic perturbations; highly robust to shot
  noise and measurement errors, ideal for quantum-like experiments.
- **Finite Differences (FD)** - Deterministic gradient approximation via finite
  perturbations; more precise in low-noise regimes but computationally heavier.
- **Adam (ADAM)** - Adaptive-momentum update rule that accelerates convergence
  on smooth loss surfaces when analytical or approximated gradients are available.
- **Natural Gradient (NGD)** - Rescales parameter updates according to the local
  information geometry (Fisher metric), improving step stability.
- **K-FAC (Kronecker-Factored Approximation of Curvature)** - Efficient block-wise
  approximation of the natural gradient; suited for structured or layered models.
- **Dual Ascent (DA)** - Alternating primal–dual updates following a Lagrangian
  scheme, enforcing constraints on feedback or coherence terms.
- **Model Predictive Control (MPC)** - Forward-planning optimizer that predicts
  a short-horizon trajectory for :math:`\alpha_t` to maintain causal stability
  under strong non-stationary drift.
- **Trust-Region (KL)** - A stability wrapper that bounds the inter-epoch
  distributional shift via a symmetric KL-divergence criterion; it can wrap any
  base optimizer to enforce temporal coherence.

Example: selecting an optimizer
-------------------------------
The demo uses ``SPSA`` wrapped by a ``Trust-Region (KL)`` controller.
To experiment with others, replace the optimizer initialization in the script:

.. code-block:: python

   # === Choose your optimizer ===
   base_opt = create_optimizer_numpy("spsa", lr0=0.05, eps0=0.10, antithetic=True, clip=4.0)
   # base_opt = create_optimizer_numpy("finite-diff", lr=0.02, eps=1e-2, clip=4.0)
   # base_opt = create_optimizer_numpy("adam", lr=0.02)
   # base_opt = create_optimizer_numpy("natural-grad", lr=0.05)
   # base_opt = create_optimizer_numpy("kfac", lr=0.05)
   # base_opt = create_optimizer_numpy("dual-ascent", lr=0.05)
   # base_opt = create_optimizer_numpy("mpc", horizon=5, lr=0.03)

   # Optional: wrap with Trust-Region (KL) for stability
   opt = create_optimizer_numpy("trust-kl",
                                base_opt=base_opt,
                                delta_kl=0.02, backtrack=0.7, max_backtracks=8)

Each optimizer modifies how :math:`\alpha_t` adapts to environmental drift and
causal feedback, providing a platform to study robustness, adaptation speed,
and stability across hypercausal configurations.

Summary and Further Exploration
===============================
The optimization architecture of QML-HCS is modular and extensible.  
Beyond the examples above, the following components form the complete **Optimization Suite**,  
allowing researchers to customize, register, and explore adaptive strategies in detail:

- :ref:`optim_api` - Defines the abstract interface and parameter flow used by all optimizers.
- :ref:`optim_registry` - Factory/registry (`create_optimizer_numpy`) for dynamic construction from string identifiers.
- :ref:`optim_numpy_index` - Concrete implementations (SPSA, Finite-Diff, Adam, Natural-Grad, K-FAC, Dual-Ascent, MPC, Trust-Region).

Together, these modules enable a flexible experimental workflow and form the backbone  
of adaptive dynamics within the **QML-HCS Optimization Suite**.

Telemetry & Training Loop
=========================
- **MemoryLogger** stores per-epoch scalars (losses, α, means), enabling
  downstream plots and CSV export.
- **CallbackList** orchestrates hooks:
  ``on_epoch_begin`` → optimizer step → loss eval → ``on_step_end`` → ``on_epoch_end``.

Pseudo-flow per epoch:

.. code-block:: text

   for t in range(E):
       callbacks.on_epoch_begin(t)
       build context (x0, drift_t, target_t, losses, branches, info)
       info_old = refresh_info(model, α_t)
       α_{t+1} = optimizer.step(model, α_t | info_old, KL_guard)
       forward pass → S_t, μ_fut, branches
       compute losses → task, cons, coh, total
       log metrics & means; callbacks.on_step_end(t)
       callbacks.on_epoch_end(t)

Saving & Numbered Figures
=========================
All results are saved under ``runs/hc_full_demo/``:

- **CSV:** ``results.csv`` with columns  
  ``[epoch, alpha, task_loss, cons_loss, coh_loss, total_loss, mean_s, mean_mu]``.
- **Numbered plots:** helper ``_savefig_numbered(dir, base)`` produces files
  like ``<base>_001.png``, ``<base>_002.png``, preserving previous runs.

Hyperparameters
===============
====================  ===========================
Parameter             Value / Description
--------------------  ---------------------------
qubits                4  (demo)
branches              6
epochs                101
drift_amp             0.05  (sinusoidal)
DepthScheduler        start=1, end=5, epochs=E
SPSA                  lr0=0.05, eps0=0.10, antithetic=True, clip=4.0
Trust-KL              δ_KL=0.02, backtrack=0.7, max_backtracks=8
shots                 500
====================  ===========================

How to Run
==========
.. code-block:: bash

   python ex_full_hypercausal_Pennylane_demo.py

Typical Console Output (Summarized)
===================================
Below is a condensed example of the console output produced by a 101-epoch run.
User/host paths are omitted for portability; figures and CSV are saved to
``runs/hc_full_demo/``.

.. code-block:: text

   /.../pennylane/devices/device_api.py:193: PennyLaneDeprecationWarning: Setting shots on device is deprecated. Please use the `set_shots` transform on the respective QNode instead.
     warnings.warn(
   Epoch 00 | α=1.0000 | Task=0.14699 Cons=0.97379 Coh=0.00133 Tot=0.63454
   Epoch 01 | α=1.0000 | Task=0.14109 Cons=0.95652 Coh=0.00136 Tot=0.62003
   ...
   Epoch 99 | α=1.0503 | Task=0.14837 Cons=0.98211 Coh=0.00131 Tot=0.64008
   Epoch 100 | α=1.0503 | Task=0.14188 Cons=0.94497 Coh=0.00139 Tot=0.61506
   [OK] Saved CSV: runs/hc_full_demo/results.csv
   [FIG] Saved: runs/hc_full_demo/loss_coherence_001.png
   [FIG] Saved: runs/hc_full_demo/consistency_vs_coherence_001.png
   [FIG] Saved: runs/hc_full_demo/alpha_over_epochs_001.png
   [FIG] Saved: runs/hc_full_demo/state_alignment_001.png
   [FIG] Saved: runs/hc_full_demo/causal_space_3d_001.png
   [FIG] Saved: runs/hc_full_demo/causal_parallel_001.png
   [FIG] Saved: runs/hc_full_demo/alpha_sensitivity_001.png
   [FIG] Saved: runs/hc_full_demo/causal_phase_portrait_001.png
   [FIG] Saved: runs/hc_full_demo/drift_vs_coherence_001.png

Figures and Results
===================

.. image:: ../../runs/hc_full_demo/loss_coherence_001.png
   :alt: Loss vs Coherence Dynamics
   :align: center
   :width: 90%

**Figure 1 – Loss vs Coherence Dynamics.**  
Coherence remains bounded while loss oscillates mildly under drifted conditions,
revealing the damping effect of KL-bounded control.

.. image:: ../../runs/hc_full_demo/consistency_vs_coherence_001.png
   :alt: Consistency vs Coherence
   :align: center
   :width: 90%

**Figure 2 – Consistency vs Coherence.**  
A near-linear inverse relationship indicates consistent causal compensation
between coherence and consistency metrics.

.. image:: ../../runs/hc_full_demo/alpha_over_epochs_001.png
   :alt: Alpha (Feedback) Over Epochs
   :align: center
   :width: 90%

**Figure 3 – Alpha (Feedback) Over Epochs.**  
The :math:`\alpha` trajectory reflects **SPSA-driven exploration**
modulated by the **KL trust-region**. After the initial exploration phase,
:math:`\alpha` typically **stabilizes** as the KL guard prevents disruptive
jumps, indicating convergence to a **hypercausal resonance**.

.. image:: ../../runs/hc_full_demo/state_alignment_001.png
   :alt: State Alignment
   :align: center
   :width: 90%

**Figure 4 – State Alignment.**  
Mean internal state :math:`S_t` (purple) and future projection
:math:`\mu_\mathrm{fut}` (orange). Under KL-guarded SPSA updates,
both remain **phase-aligned** even with sinusoidal drift,
evidencing **temporal-causal coherence** rather than mere stationarity.

.. image:: ../../runs/hc_full_demo/alpha_sensitivity_001.png
   :alt: Alpha Sensitivity (Δα per Epoch)
   :align: center
   :width: 90%

**Figure 5 – Alpha Sensitivity (Δα per Epoch).**  
Derivative of :math:`\alpha` per step; spikes mark the trust-region’s active
regulation boundaries where KL exceeded δ_KL and backtracking engaged.

.. image:: ../../runs/hc_full_demo/drift_vs_coherence_001.png
   :alt: Drift vs Coherence
   :align: center
   :width: 90%

**Figure 6 – Drift vs Coherence Dynamics.**  
The blue curve (coherence) stays flat while the red line (|Δα| drift proxy)
shows constrained bursts, confirming that the feedback channel absorbs drift
without coherence collapse.

.. image:: ../../runs/hc_full_demo/causal_phase_portrait_001.png
   :alt: Causal Phase Portrait
   :align: center
   :width: 90%

**Figure 7 – Causal Phase Portrait (Temporal Path).**  
A nearly one-dimensional manifold in (coherence, consistency) space indicates
that causal self-correction preserves phase alignment over epochs.

.. image:: ../../runs/hc_full_demo/causal_parallel_001.png
   :alt: Parallel Causal Dimensions
   :align: center
   :width: 90%

**Figure 8 – Parallel Causal Dimensions.**  
Parallel coordinate plot visualizing joint evolution of coherence,
consistency, feedback (α), and epoch index.

.. image:: ../../runs/hc_full_demo/causal_space_3d_001.png
   :alt: 3D Causal Space
   :align: center
   :width: 90%

**Figure 9 – 3D Causal Space.**  
Temporal trajectory of (coherence, consistency, α); the smooth spiral descent
shows coherent convergence within the hypercausal basin.

Discussion
==========
The simulation confirms that **SPSA + KL-bounded trust-region** achieves
stabilized adaptation under non-stationary drift.
The KL constraint enables learning progress while constraining
inter-epoch distributional shifts of branch statistics, avoiding decoherence
spikes and ensuring **smooth convergence** in non-stationary settings.

This demonstrates how hypercausal feedback architectures can self-correct
through dynamic, information-theoretic constraints rather than fixed regularization.

Reproducibility Notes
=====================
- Results vary slightly with randomness (device shots, SPSA noise).
- Fixing a global seed and locking ``shots`` improves repeatability, while the
  KL trust-region stabilizes inter-epoch distributional shifts.
- The numbered saving scheme avoids overwriting figures across runs.

Source Code Reference
=====================
.. literalinclude:: ../../examples/ex_full_hypercausal_Pennylane_demo.py
   :language: python
   :linenos:
   :caption: Full demo script with SPSA + Trust-KL optimization

