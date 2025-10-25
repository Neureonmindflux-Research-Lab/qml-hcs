Quantum Hypercore Case Study
============================

Introduction
------------

This case study demonstrates a complete **hyper-causal DAG (Directed Acyclic Graph)** experiment that integrates
counterfactual projection, finite-difference training, and post-training evaluation metrics.
It simulates a dynamic system with injected anomalies to assess the **stability**, **consistency**, and
**predictive robustness** of a multi-node hyper-causal architecture.

Main objectives:

1. Build a synthetic dynamic environment with controlled anomalies.  
2. Train a three-node hyper-causal DAG using counterfactual backends.  
3. Evaluate robustness through quantitative metrics such as MAPE, Settling Time, and ROC-AUC.

---

General Flow Structure
----------------------

The experiment combines deterministic simulation with causal inference and finite-difference optimization.  
Each backend models a dynamic node connected through directed edges forming a DAG, where causal
dependencies propagate forward in time.

- **SyntheticScenario**: generates a temporal dataset with sinusoidal base signals and anomaly injections.  
- **CFBackend**: a parametric backend with recursive :math:`\tanh` transformation and counterfactual anticipation.  
- **HCGraph**: connects nodes via directed edges to form a fully causal structure.  
- **Loss functions**: combine MSE, consistency, and coherence components.  
- **Finite-difference optimization**: updates parameters numerically per epoch.  
- **DepthScheduler**: progressively increases recursion depth through epochs.

---

How to Run
----------

.. code-block:: console

   # From the project root
   python -m examples.ex_quantum_hypercore_case_study

   # Or directly
   python examples/ex_quantum_hypercore_case_study.py

---

Relevant Code Snippets
----------------------

.. literalinclude:: ../../examples/ex_quantum_hypercore_case_study.py
   :language: python
   :linenos:
   :lines: 70-198
   :caption: Definition of CFBackend (parametric backend with counterfactual anticipation)

.. literalinclude:: ../../examples/ex_quantum_hypercore_case_study.py
   :language: python
   :linenos:
   :lines: 300-381
   :caption: Main experiment function quantum_hypercore_case_study()

---

Functional Explanation
----------------------

This experiment evaluates a hyper-causal DAG under perturbative dynamics and counterfactual projections.
It integrates predictive, causal, and diagnostic components in a single computational framework.

1. **Synthetic Dynamic Environment**

   The `SyntheticScenario` generates temporal signals with injected anomalies:

   .. math::

      x_t = \sin(\omega t + \phi) + \epsilon_t, \qquad
      x_{t,\text{anom}} = x_t + \delta \mathbf{1}_{t \in \mathcal{A}}

   where :math:`\epsilon_t` represents Gaussian noise and :math:`\mathcal{A}` marks anomaly indices.
   This allows controlled evaluation of the model’s ability to adapt to irregular perturbations.

2. **Counterfactual Backend**

   Each backend implements recursive dynamics:

   .. math::

      S_t = \tanh(w S_{t-1} + b)

   and generates counterfactual future branches:

   .. math::

      S_{t+1}^{(k)} = S_t + \mathcal{P}_k(S_t)

   where :math:`\mathcal{P}_k` are perturbation operators encoding “what-if” transformations.
   This enables simultaneous evaluation of nominal and hypothetical state trajectories.

3. **Hyper-Causal Graph Structure**

   The DAG is defined as:

   .. math::

      \mathcal{G} = (\mathcal{V}, \mathcal{E}), \quad
      \mathcal{V} = \{A, B, C\}, \quad
      \mathcal{E} = \{A \to B, A \to C, B \to C\}

   Each node receives upstream states, applies its transformation, and emits a projected state to downstream nodes.
   This simulates hierarchical causal propagation within a directed structure.

4. **Loss Composition**

   The total loss function integrates predictive, temporal, and coherence constraints:

   .. math::

      \mathcal{L}_{total} = \mathcal{L}_{task} + 0.5\,\mathcal{L}_{consistency} + 0.3\,\mathcal{L}_{coherence}

   - **Task loss**: prediction accuracy.  
   - **Consistency loss**: stability of consecutive states.  
   - **Coherence loss**: alignment across projected branches.

   Each loss term contributes to maintaining causal smoothness and coherent evolution.

5. **Training via Finite-Difference Gradients**

   Instead of backpropagation, gradients are computed numerically:

   .. math::

      g_i = \frac{\mathcal{L}(\theta_i + \epsilon) - \mathcal{L}(\theta_i - \epsilon)}{2\epsilon}

   Parameters are updated using simple gradient descent with adaptive learning rate control.

6. **Evaluation Metrics**

   After training, several metrics are computed to quantify system stability and sensitivity:

   - **MAPE** – Mean Absolute Percentage Error.  
   - **Overshoot** – amplitude deviation beyond target.  
   - **Settling Time** – number of samples to reach stability within tolerance.  
   - **Robustness** – correlation between predicted and target signals.  
   - **Early ROC-AUC@H=3** – anomaly discrimination within a 3-step horizon.  
   - **Recall@Lag=3** – sensitivity to delayed anomaly detection.

   These indicators evaluate both predictive accuracy and control stability under dynamic perturbations.

---

Exact Output
------------

.. code-block:: console

   [Epoch 0] total_before=0.098290 total_after=0.092576 depth=[1, 1, 1]
   [Epoch 1] total_before=0.069731 total_after=0.062847 depth=[2, 2, 2]
   [Epoch 2] total_before=0.062847 total_after=0.056539 depth=[2, 2, 2]
   [Epoch 3] total_before=0.042616 total_after=0.041622 depth=[3, 3, 3]
   [Epoch 4] total_before=0.041622 total_after=0.039550 depth=[3, 3, 3]
   [Epoch 5] total_before=0.030317 total_after=0.029850 depth=[4, 4, 4]

   === Final Metrics ===
   MAPE (%):          19094053004148.468750
   Overshoot:         0.000000
   Settling Time:     60 samples
   Robustness:        0.957343
   Early ROC-AUC@H=3: 0.224138
   Recall@Lag=3:      0.000000

   Best epoch snapshot: {'epoch': 5, 'task': 0.019989936715902698, 'cons': 0.015169491076939604, 'coh': 0.007584946879521136, 'total': 0.02985016631822884}

   Telemetry JSONL → runs/quantum_hypercore_case_telemetry.jsonl

   Summary:
   {
     "best": {
       "epoch": 5,
       "task": 0.019989936715902698,
       "cons": 0.015169491076939604,
       "coh": 0.007584946879521136,
       "total": 0.02985016631822884
     },
     "metrics": {
       "mape": 19094053004148.47,
       "overshoot": 0.0,
       "settling_time": 60,
       "robustness": 0.9573428590759466,
       "early_auc_h3": 0.22413793103448276,
       "recall_lag3": 0.0
     }
   }
