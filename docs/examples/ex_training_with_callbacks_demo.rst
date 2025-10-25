Training with Callbacks Demo
============================

Introduction
------------

This example presents a stable training framework for a three-node hyper-causal model.  
It demonstrates a complete adaptive optimization cycle using depth-aware recursion,
finite-difference gradients, and callback-driven monitoring.  
The goal is to illustrate how a causal system can be trained under dynamic conditions
without relying on classical backpropagation.

Key mechanisms include:

1. **DepthScheduler** – controls recursion depth dynamically per epoch.  
2. **Adaptive learning parameters** – rescale learning rate and perturbation with recursion depth.  
3. **Finite-difference gradient estimation** – numerical gradient replacement for stability.  
4. **Gradient clipping** – limits the trust region to prevent divergence.  
5. **Callback telemetry** – monitors all metrics and parameters across epochs.  
6. **Early stopping** – detects convergence based on stability in total loss.

---

General Flow Structure
----------------------

The training loop coordinates recursive backends connected through causal dependencies.
Each backend adjusts its transformation depth, projects future states, and contributes
to a composite loss.  
The callback system regulates both the internal optimization and the external logging.

- **DepthAwareBackend**: applies recursive transformations :math:`S_t^{(d)} = \tanh(W^{(d)}S_{t-1} + b^{(d)})`.  
- **Projection**: expands each state into :math:`K` future branches via a linear projector.  
- **Loss aggregation**: combines task, consistency, and coherence components.  
- **Optimizer**: updates parameters using finite-difference gradients and gradient clipping.  
- **Scheduler**: increases recursion depth gradually to control system complexity.

---

How to Run
----------

.. code-block:: console

   # From the project root
   python -m examples.ex_training_with_callbacks_demo

   # Or directly
   python examples/ex_training_with_callbacks_demo.py

---

Relevant Code Snippets
----------------------

.. literalinclude:: ../../examples/ex_training_with_callbacks_demo.py
   :language: python
   :linenos:
   :lines: 147-279
   :caption: Definition of the DepthAwareBackend class (recursive tanh transformation and projection)

.. literalinclude:: ../../examples/ex_training_with_callbacks_demo.py
   :language: python
   :linenos:
   :lines: 353-573
   :caption: Main function stable_training_demo() (training loop with callbacks and adaptive learning)

---

Functional Explanation
----------------------

The model trains through controlled recursion and adaptive numerical optimization.  
Each component has a defined mathematical role in stabilizing and guiding the learning process.

1. **Recursive Depth Evolution**

   Each backend performs a recursive update of the internal state:

   .. math::

      S_t^{(d)} = \tanh(W^{(d)} S_{t-1} + b^{(d)})

   The recursion depth :math:`d` determines the number of internal evaluations per epoch.
   Increasing :math:`d` allows the model to capture higher-order temporal dependencies.

2. **Future Projection**

   Each current state generates :math:`K` predicted future states:

   .. math::

      S_{t+1}^{(k)} = S_t + \Delta_d \cdot \mathcal{P}_k(S_t)

   where :math:`\mathcal{P}_k` is a projection operator and :math:`\Delta_d` scales with depth.
   This projection step introduces local temporal uncertainty and allows causal branching.

3. **Loss Structure**

   The total loss integrates three objectives:

   .. math::

      \mathcal{L}_{total} = \mathcal{L}_{task} + 0.5\,\mathcal{L}_{consistency} + 0.3\,\mathcal{L}_{coherence}

   - **Task loss** :math:`\mathcal{L}_{task} = \frac{1}{T}\sum_t \|S_t - Y_t\|^2` minimizes prediction error.  
   - **Consistency loss** maintains temporal smoothness:  
     :math:`\mathcal{L}_{consistency} = \alpha\|S_t - S_{t-1}\|^2 + \beta\|S_t - \hat{S}_{t+1}\|^2`.  
   - **Coherence loss** enforces similarity among projected branches:  
     :math:`\mathcal{L}_{coherence} = \mathrm{Var}(S_{t+1}^{(k)})`.

4. **Gradient Estimation**

   Finite differences are used to estimate local gradients:

   .. math::

      g_i = \frac{\mathcal{L}(\theta_i + \epsilon) - \mathcal{L}(\theta_i - \epsilon)}{2\epsilon}

   This method avoids symbolic differentiation and remains stable under non-smooth operations.

5. **Adaptive Learning Parameters**

   The effective parameters adjust with recursion depth:

   .. math::

      \eta_{\text{eff}} = \frac{\eta_0}{d^2}, \qquad
      \epsilon_{\text{eff}} = \frac{\epsilon_0}{1 + 0.5(d - 1)}

   These relations reduce step size and perturbation magnitude as depth increases,
   improving convergence stability for deeper causal recursions.

6. **Gradient Clipping**

   All gradient vectors are constrained within a trust region:

   .. math::

      g_i' = g_i \cdot \min\left(1, \frac{\tau}{\|g\|_2}\right)

   where :math:`\tau` is the clipping threshold. This ensures controlled parameter updates.

7. **Callback Coordination**

   - `DepthScheduler`: adjusts recursion depth at specific epochs.  
   - `TelemetryLogger`: records per-epoch statistics to JSONL.  
   - `MemoryLogger`: stores metrics in memory for later visualization.

   These components synchronize the optimization and provide complete training traceability.

---

Exact Output
------------

.. code-block:: console

   [Epoch 0] total_before=0.031141 total_after=0.030663 depth=[1, 1, 1] lr_eff=5.000e-02 eps_eff=1.000e-03
   [Epoch 1] total_before=0.030663 total_after=0.030217 depth=[1, 1, 1] lr_eff=5.000e-02 eps_eff=1.000e-03
   [Epoch 2] total_before=0.030217 total_after=0.029800 depth=[1, 1, 1] lr_eff=5.000e-02 eps_eff=1.000e-03
   [Epoch 3] total_before=0.025941 total_after=0.025629 depth=[2, 2, 2] lr_eff=1.250e-02 eps_eff=6.667e-04
   [Epoch 4] total_before=0.025629 total_after=0.025326 depth=[2, 2, 2] lr_eff=1.250e-02 eps_eff=6.667e-04
   [Epoch 5] total_before=0.025326 total_after=0.025030 depth=[2, 2, 2] lr_eff=1.250e-02 eps_eff=6.667e-04
   [Epoch 6] total_before=0.025030 total_after=0.024742 depth=[2, 2, 2] lr_eff=1.250e-02 eps_eff=6.667e-04
   [Epoch 7] total_before=0.024742 total_after=0.024462 depth=[2, 2, 2] lr_eff=1.250e-02 eps_eff=6.667e-04
   [Epoch 8] total_before=0.024462 total_after=0.024190 depth=[2, 2, 2] lr_eff=1.250e-02 eps_eff=6.667e-04
   [Epoch 9] total_before=0.022960 total_after=0.022723 depth=[3, 3, 3] lr_eff=5.556e-03 eps_eff=5.000e-04
   [Epoch 10] total_before=0.022723 total_after=0.022489 depth=[3, 3, 3] lr_eff=5.556e-03 eps_eff=5.000e-04
   [Epoch 11] total_before=0.022489 total_after=0.022259 depth=[3, 3, 3] lr_eff=5.556e-03 eps_eff=5.000e-04

   === Final metrics (channel 0) ===
   SMAPE:      100.000000 %
   RMSE:       0.165016
   Overshoot:  0.000000
   Robustness: 0.973491

   Best epoch snapshot: {'epoch': 11, 'task': 0.018001068416014097, 'cons': 0.0010439829584105837, 'coh': 0.012454292234950454, 'total': 0.022259347565704524}

   Telemetry JSONL → runs/telemetry_stable.jsonl

   Summary:
   {
     "best": {
       "epoch": 11,
       "task": 0.018001068416014097,
       "cons": 0.0010439829584105837,
       "coh": 0.012454292234950454,
       "total": 0.022259347565704524
     },
     "metrics": {
       "smape": 100.0,
       "rmse": 0.1650163483217719,
       "overshoot": 0.0,
       "robustness": 0.9734914432630335
     }
   }
