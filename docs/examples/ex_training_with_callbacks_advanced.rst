Advanced Training with Callbacks
================================

Introduction
------------

This example presents an advanced configuration of the hyper-causal training framework.  
It extends the previous demo by incorporating **external depth scheduling**, **freeze epochs**,  
and **adaptive gradient clipping** controlled by recursion depth.  
The design emphasizes deterministic, depth-aware optimization behavior with stable convergence properties.

Key components include:

1. **External DepthScheduler** – adjusts recursion depth independently of the logger.  
2. **Freeze epochs** – disable updates after depth transitions to stabilize new configurations.  
3. **Adaptive gradient clipping** – scales gradient bounds dynamically with mean recursion depth.  
4. **Epoch-dependent learning decay** – combines depth and time scaling for step-size control.  
5. **Parameter checkpointing** – stores best-performing parameters in JSON format.  
6. **Robust metric evaluation** – computes SMAPE, RMSE, Overshoot, and Robustness.

---

General Flow Structure
----------------------

The training loop performs controlled optimization with three main backends,  
each using recursive transformations and linear projections.  
Depth increases are scheduled externally, triggering *freeze epochs* where parameters remain static,  
allowing gradient statistics to stabilize before continuing updates.

- **DepthAwareBackend**: recursive causal unit using :math:`S_t^{(d)} = \tanh(W^{(d)}S_{t-1} + b^{(d)})`.  
- **Projection step**: generates K possible futures with span scaled by depth.  
- **Central finite-difference gradients**: provide stable numerical estimation.  
- **Adaptive learning control**: combines recursion-based scaling with epoch decay.  
- **Clipping**: applied adaptively based on the mean depth to limit the L2 norm of the gradient.  

---

How to Run
----------

.. code-block:: console

   # From the project root
   python -m examples.ex_training_with_callbacks_advanced

   # Or directly
   python examples/ex_training_with_callbacks_advanced.py

---

Relevant Code Snippets
----------------------

.. literalinclude:: ../../examples/ex_training_with_callbacks_advanced.py
   :language: python
   :linenos:
   :lines: 130-310
   :caption: Definition of the DepthAwareBackend class and gradient control utilities.

.. literalinclude:: ../../examples/ex_training_with_callbacks_advanced.py
   :language: python
   :linenos:
   :lines: 370-471
   :caption: Main training procedure advanced_training_with_freeze() implementing freeze logic and adaptive clipping.

---

Functional Explanation
----------------------

This training routine enhances the baseline model with **depth adaptation and parameter freezing**,  
resulting in a more controlled optimization process that preserves gradient stability.  
All components operate deterministically, and the system can reproduce results across runs.

1. **Recursive Backend Dynamics**

   Each backend updates its state using depth-controlled recursion:

   .. math::

      S_t^{(d)} = \tanh(W^{(d)} S_{t-1} + b^{(d)})

   Depth :math:`d` determines the number of recursive evaluations per step.
   Increasing depth raises representational capacity but requires finer gradient control.

2. **Future Projection and Span Scaling**

   Future states are generated through a linear projection mechanism with depth-dependent span:

   .. math::

      S_{t+1}^{(k)} = S_t + \Delta_d \cdot \mathcal{P}_k(S_t)

   where :math:`\Delta_d` decreases with increasing depth to maintain bounded perturbations.
   This ensures consistent diversity among causal branches without instability.

3. **Composite Loss Function**

   The loss combines predictive, consistency, and coherence terms:

   .. math::

      \mathcal{L}_{total} = \mathcal{L}_{task} + 0.5\,\mathcal{L}_{consistency} + 0.3\,\mathcal{L}_{coherence}

   Each term regulates a specific property:
   - **Task**: prediction accuracy.
   - **Consistency**: smooth temporal evolution.
   - **Coherence**: branch uniformity at projection level.

4. **Finite-Difference Gradient Estimation**

   Gradients are computed numerically:

   .. math::

      g_i = \frac{\mathcal{L}(\theta_i + \epsilon) - \mathcal{L}(\theta_i - \epsilon)}{2\epsilon}

   This avoids dependency on differentiable computation graphs and maintains stability under recursion.

5. **Adaptive Learning and Perturbation Scaling**

   Learning parameters decay with both depth and epoch index:

   .. math::

      \eta_{\text{eff}} = \frac{\eta_0}{(1 + 0.5(d - 1))(1 + 0.5e)}, \quad
      \epsilon_{\text{eff}} = \frac{\epsilon_0}{(1 + 0.3(d - 1))(1 + 0.3e)}

   where :math:`e` is the current epoch.
   This provides temporal damping, ensuring smaller updates as the system stabilizes.

6. **Freeze Epochs**

   After each depth increase, one epoch executes without parameter updates:

   .. math::

      \theta_{t+1} = \theta_t \quad \text{if depth\_changed=True}

   This step prevents transient gradient noise from destabilizing new recursion levels.

7. **Adaptive Gradient Clipping**

   The clipping threshold increases with mean depth :math:`\bar{d}`:

   .. math::

      \tau =
      \begin{cases}
        5\times10^{-2}, & \bar{d} < 1.5 \\
        7.5\times10^{-2}, & 1.5 \le \bar{d} < 2.5 \\
        1\times10^{-1}, & \bar{d} \ge 2.5
      \end{cases}

   The gradients are then rescaled:

   .. math::

      g_i' = g_i \cdot \min\left(1, \frac{\tau}{\|g\|_2}\right)

   providing consistent control across all recursion depths.

8. **Metric Evaluation**

   After training, four metrics summarize performance:

   - **SMAPE**: symmetric mean absolute percentage error.  
   - **RMSE**: root mean square error.  
   - **Overshoot**: excess deviation in prediction amplitude.  
   - **Robustness**: correlation-based stability ratio.

   All metrics are computed on the first output channel for reproducibility.

---

Exact Output
------------

.. code-block:: console

   [Epoch 0] total_before=0.031141 total_after=0.030663 depth=[1, 1, 1] lr_eff=5.000e-02 eps_eff=1.000e-03 ||g||_before=1.971e-01 ||g||_after=5.000e-02
   [Epoch 1] total_before=0.030663 total_after=0.030362 depth=[1, 1, 1] lr_eff=3.333e-02 eps_eff=7.692e-04 ||g||_before=1.848e-01 ||g||_after=5.000e-02
   [Epoch 2] total_before=0.030362 total_after=0.030145 depth=[1, 1, 1] lr_eff=2.500e-02 eps_eff=6.250e-04 ||g||_before=1.767e-01 ||g||_after=5.000e-02
   [Epoch 3] total_before=0.030145 total_after=0.029977 depth=[1, 1, 1] lr_eff=2.000e-02 eps_eff=5.263e-04 ||g||_before=1.707e-01 ||g||_after=5.000e-02
   [Epoch 4] FREEZE depth=[2, 2, 2] total=0.026492 lr_eff=1.111e-02 eps_eff=3.497e-04
   [Epoch 5] total_before=0.026492 total_after=0.026122 depth=[2, 2, 2] lr_eff=9.524e-03 eps_eff=3.077e-04 ||g||_before=5.258e-01 ||g||_after=7.500e-02
   [Epoch 6] total_before=0.026122 total_after=0.025806 depth=[2, 2, 2] lr_eff=8.333e-03 eps_eff=2.747e-04 ||g||_before=5.114e-01 ||g||_after=7.500e-02
   [Epoch 7] total_before=0.025806 total_after=0.025532 depth=[2, 2, 2] lr_eff=7.407e-03 eps_eff=2.481e-04 ||g||_before=4.988e-01 ||g||_after=7.500e-02
   [Epoch 8] total_before=0.025532 total_after=0.025291 depth=[2, 2, 2] lr_eff=6.667e-03 eps_eff=2.262e-04 ||g||_before=4.876e-01 ||g||_after=7.500e-02
   [Epoch 9] total_before=0.025291 total_after=0.025076 depth=[2, 2, 2] lr_eff=6.061e-03 eps_eff=2.079e-04 ||g||_before=4.775e-01 ||g||_after=7.500e-02
   [Epoch 10] total_before=0.025076 total_after=0.024883 depth=[2, 2, 2] lr_eff=5.556e-03 eps_eff=1.923e-04 ||g||_before=4.683e-01 ||g||_after=7.500e-02
   [Epoch 11] total_before=0.024883 total_after=0.024707 depth=[2, 2, 2] lr_eff=5.128e-03 eps_eff=1.789e-04 ||g||_before=4.599e-01 ||g||_after=7.500e-02
   [Epoch 12] FREEZE depth=[3, 3, 3] total=0.023982 lr_eff=3.571e-03 eps_eff=1.359e-04
   [Epoch 13] total_before=0.023982 total_after=0.023682 depth=[3, 3, 3] lr_eff=3.333e-03 eps_eff=1.276e-04 ||g||_before=9.096e-01 ||g||_after=1.000e-01
   [Epoch 14] total_before=0.023682 total_after=0.023404 depth=[3, 3, 3] lr_eff=3.125e-03 eps_eff=1.202e-04 ||g||_before=8.949e-01 ||g||_after=1.000e-01
   [Epoch 15] total_before=0.023404 total_after=0.023147 depth=[3, 3, 3] lr_eff=2.941e-03 eps_eff=1.136e-04 ||g||_before=8.810e-01 ||g||_after=1.000e-01

   Best parameters saved at: runs/best_params.json

   === Final metrics (channel 0) ===
   SMAPE:      100.000000 %
   RMSE:       0.167734
   Overshoot:  0.000000
   Robustness: 0.972635

   Best epoch snapshot: {'epoch': 15, 'task': 0.01889342623335241, 'cons': 0.0010430885080397983, 'coh': 0.01243963263570178, 'total': 0.023146860278082843}

   Telemetry JSONL → runs/telemetry_stable.jsonl

   Summary:
   {
     "best": {
       "epoch": 15,
       "task": 0.01889342623335241,
       "cons": 0.0010430885080397983,
       "coh": 0.01243963263570178,
       "total": 0.023146860278082843
     },
     "metrics": {
       "smape": 100.0,
       "rmse": 0.16773380360977846,
       "overshoot": 0.0,
       "robustness": 0.9726352677136917
     }
   }
