Optimization Module
===================

The **qmlhc.optim** package provides both a general **Optimizer API**
(:mod:`qmlhc.optim.api`) and a NumPy-based optimizer registry
(:mod:`qmlhc.optim.registry_numpy`) that wire multiple optimization
strategies to a unified project interface.

It enables interchangeable use of **gradient-free**, **gradient-based**, and
**trust-region** algorithms for training or adaptive control within the
hypercausal learning flow.

Architecture Overview
---------------------

::

   +--------------------------+
   |     qmlhc.optim          |
   +-------------+------------+
                 |
        +--------v--------+           +------------------------------+
        |      api        |  --->     |     registry_numpy           |
        | (Optimizer API) |           | (factory for NumPy backends) |
        +-----------------+           +------------------------------+

Unified Optimizer Interface
---------------------------

All optimizers returned by the NumPy registry expose a unified interface:

- ``initialize(params) -> state``
- ``step_params(model, params, context) -> (new_params, state)``

Where:

- ``params`` - possibly nested dictionary of numeric parameters.
- ``model`` - callable that accepts ``params`` and returns a scalar loss.
- ``context`` - optional dictionary with auxiliary signals (gradients, KL, rollouts, etc.).

Registry Keys (NumPy)
---------------------

Use ``create_optimizer_numpy(name, **kwargs)`` with:

- ``"finite-diff"`` - finite-difference gradient estimation  
- ``"spsa"`` - simultaneous perturbation stochastic approximation (SPSA)
- ``"adam"`` - Adam optimizer (using provided or estimated gradients)
- ``"natural-grad"`` - natural gradient optimizer with statistical preconditioning
- ``"trust-kl"`` - trust-region optimizer with KL constraint (wrapper)
- ``"dual-ascent"`` - dual-ascent optimizer with Lagrange multiplier (wrapper)
- ``"mpc"`` - short-horizon model predictive control optimizer (MPC)
- ``"kfac"`` - Kronecker-factored approximation (blockwise preconditioner)

Core Contracts
--------------

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Optimizer
     - Summary
   * - ``finite-diff``
     - Estimates gradients via **finite differences** on ``params`` (configurable step) and performs a standard descent update.
   * - ``spsa``
     - **SPSA** with simultaneous perturbations (optional antithetic mode).  
       **Constant evaluation cost (~2 function calls)** regardless of parameter dimensionality.
   * - ``adam``
     - **Adam** optimizer applied to either **estimated gradients** (e.g., FD/SPSA) or **provided ones** through ``context['grads']``. Maintains moment estimates and bias correction.
   * - ``natural-grad``
     - **Natural Gradient** optimizer using statistical preconditioning (e.g., covariance of branch states).  
       Consumes ``context['grads']`` or a gradient estimator; can leverage ``info['branches']`` for covariance statistics if available.
   * - ``trust-kl`` (wrapper)
     - **Trust-region** wrapper enforcing a **KL divergence constraint** on state transitions.  
       Wraps a base optimizer and applies **backtracking line-search** until ``ΔKL ≤ δ``.
   * - ``dual-ascent`` (wrapper)
     - **Dual-ascent** wrapper that updates primal parameters using a base optimizer and a **Lagrange multiplier** for constraint enforcement (e.g., KL).  
       Requires a constraint function in the execution context.
   * - ``mpc``
     - **Short-horizon MPC** optimizer performing multi-step rollouts using ``context["rollout_fn"]`` and computing gradient-free policy updates.  
       Supports projection or clipping functions for constrained domains.
   * - ``kfac``
     - **K-FAC preconditioner** performing Kronecker-factored blockwise updates to accelerate second-order optimization, using statistics from structured parameter sets when available.

Required Context (by Optimizer)
-------------------------------

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Optimizer
     - Required ``context`` keys
   * - ``finite-diff``
     - (Optional) ``'eps'`` for finite-difference step size.
   * - ``spsa``
     - (Optional) ``'antithetic'`` in constructor; no per-step context required.
   * - ``adam``
     - Requires either ``'grads'`` (gradient dict) or a ``grad_estimator`` defined at initialization.
   * - ``natural-grad``
     - Requires ``'grads'`` or ``grad_estimator``; optionally uses ``'info'`` with branch statistics for preconditioning.
   * - ``trust-kl`` (wrapper)
     - Requires ``'kl_fn'`` (callable returning ΔKL) and optionally ``'delta_kl'`` (threshold).  
       May use ``'refresh_info'`` for state recomputation. Needs ``'base_opt'`` in constructor.
   * - ``dual-ascent`` (wrapper)
     - Requires ``'constraint_fn'`` (constraint value) and optionally initial ``'lambda'`` and update rule.  
       Needs ``'base_opt'`` in constructor.
   * - ``mpc``
     - Requires ``'rollout_fn'`` (cumulative cost evaluator) and optionally ``'project_fn'`` (projection function), ``'horizon'``, or random seed.
   * - ``kfac``
     - Optionally uses blockwise statistics in ``'info'`` to construct the preconditioner; falls back to simple descent if missing.

Shared Utilities
----------------

- Parameter **flattening/unflattening** helpers and **covariance/CG solvers** for preconditioning and K-FAC.  
- **KL-divergence proxies** and numerical tools for step validation, clipping, and trust-region stability control.

Implementation Notes
--------------------
- Wrapper optimizers (``trust-kl``, ``dual-ascent``) **do not perform updates directly**; they **delegate** the actual parameter step to their **``base_opt``** and apply additional logic (KL bound, constraint enforcement, etc.) around it.
- If neither gradients nor a gradient estimator are provided to ``adam`` or ``natural-grad``, the optimizer **cannot proceed**.  
  Provide either a ``grad_estimator`` in the constructor or ``context['grads']`` at runtime.


Minimal Implementation Example
------------------------------

The following snippet illustrates how to integrate an optimizer within
the QML-HCS pipeline using the unified NumPy registry.  
It shows the initialization and parameter-update logic conceptually,
without executing a full training loop.

.. code-block:: python

   from qmlhc.optim.registry_numpy import create_optimizer_numpy

   # Define a derivative-free SPSA optimizer
   opt = create_optimizer_numpy("spsa", lr0=0.05, eps0=0.10, antithetic=True)

   # Initialize optimizer state
   params = {"alpha": 1.0}
   opt_state = opt.initialize(params)

   # Example single update (model function provided externally)
   new_params, opt_state = opt.step_params(
       model=lambda p: (p["alpha"] - 1.0) ** 2,   # illustrative cost
       params=params,
       context={"step": 0}
   )

.. note::

   This example illustrates only **how optimizers are conceptually integrated**
   into the hypercausal learning flow.  
   For a **fully runnable integration** combining SPSA, KL trust-region control,
   and adaptive drift resilience under PennyLane backends, see
   :ref:`Full Hypercausal System Demo <ex-full-hypercausal-pennylane-demo>`.  
   The physical variant extending drift into hardware-level perturbations is
   documented under :ref:`Alternative Drift Mode (Hardware-Style) <physical_drift_mode>`.

Invariants and Notes
--------------------

- All optimizers expose the same interface (`initialize`, `step_params`).
- Registry keys are case-insensitive (e.g., “SPSA” == “spsa”).
- Wrappers like ``trust-kl`` or ``dual-ascent`` accept a ``base_opt`` argument.
- NumPy serves as the canonical backend for deterministic optimization.
- Improper names trigger ``KeyError`` listing all available optimizers.

Module References
-----------------

.. toctree::
   :maxdepth: 1
   :titlesonly:

   Optimizer API <api>
   NumPy Registry <registry_numpy>
   NumPy Optimizers <numpy_optim/index>
