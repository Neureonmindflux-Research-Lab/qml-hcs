Minimal Core Demo
=================

Introduction
------------

This example demonstrates the minimal flow of a hyper-causal model using a deterministic
demonstration backend (**ToyBackend**). It focuses on three core parts:

1. Deterministic backend (`ToyBackend`).
2. Forward step and loss computation (`minimal_core_demo`).
3. Model consistency verification (`HCModel`).

---

General Flow Structure
----------------------

The example implements a minimal architecture:

- **Backend (`ToyBackend`)**: processes the input through a stable and validated :math:`\tanh(w \cdot x + b)` transformation.  
- **Node (`HCNode`)**: combines the backend with a decision policy (`MeanPolicy`) to generate future branches.  
- **Loss (`ConsistencyLoss`)**: compares temporal coherence between past, present, and predicted future states.

How to Run
----------

.. code-block:: console

   # From the project root
   python -m examples.ex_minimal_core_demo

   # Or directly
   python examples/ex_minimal_core_demo.py

Relevant Code Snippets
----------------------

.. literalinclude:: ../../examples/ex_minimal_core_demo.py
   :language: python
   :linenos:
   :lines: 25-96
   :caption: Definition of the ToyBackend class

.. literalinclude:: ../../examples/ex_minimal_core_demo.py
   :language: python
   :linenos:
   :lines: 99-177
   :caption: Function minimal_core_demo (main flow and loss computation)

Functional Explanation
----------------------

The minimal hyper-causal flow describes how an input state evolves over time in a simulated
quantum-like system. Each component plays a precise mathematical role in maintaining **temporal coherence** and **causal consistency**.

1. **State Update — Backend Transformation**

   The backend computes the internal system state as a nonlinear and bounded transformation:

   .. math::

      s_t = \tanh(w \cdot x_t + b)

   Here, :math:`x_t` is the input at time :math:`t`, while :math:`w` and :math:`b` act as fixed
   transformation parameters.  
   The hyperbolic tangent ensures the state remains numerically stable within the range
   :math:`(-1, 1)`, simulating a controlled energy state.

2. **Future Projection — Hypercausal Branches**

   The model predicts multiple potential future states (“branches”) by perturbing the current state
   :math:`s_t` with small offsets :math:`\delta_k`:

   .. math::

      S_{t+1}^{(k)} = \tanh(s_t + \delta_k), \quad \delta_k \in [-0.25, 0.25]

   Each :math:`S_{t+1}^{(k)}` represents a possible trajectory the system could take.
   This branching behavior emulates **quantum superposition**, where multiple possible futures exist
   simultaneously before being collapsed by a selection policy.

3. **Policy Selection — Mean Collapse**

   The `MeanPolicy` acts as the “collapse” function, averaging across branches to select the most
   coherent and representative state :math:`\hat{S}_{t+1}`.

4. **Consistency Loss — Temporal Coherence Evaluation**

   To ensure the system evolves smoothly, the **Consistency Loss** penalizes large deviations between
   consecutive temporal states:

   .. math::

      \mathcal{L} = \alpha \| S_t - S_{t-1} \|^2 + \beta \| S_t - \hat{S}_{t+1} \|^2

   - The first term (:math:`\alpha`) ensures internal continuity between the current and previous states.  
   - The second term (:math:`\beta`) enforces predictive alignment between the present and the expected next state.  

   The lower the loss, the more **temporally consistent** the hyper-causal chain.

5. **Global Validation — HCModel Check**

   Finally, the `HCModel` wrapper reproduces the single-node computation to confirm equivalence between
   the modular (node-level) and aggregated (model-level) behaviors, guaranteeing deterministic correctness.

Exact Output
------------

.. code-block:: console

   === Minimal Core Demo ===
   output_dim (D):     3
   branches (K):       3

   x_t:                 [ 0.2 -0.1  0.4]
   S_{t-1} (s_tm1):     [ 0.15 -0.05  0.35]
   S_t (from run):      [ 0.23549575 -0.04496965  0.40532131]
   Ŝ_{t+1} (selected):  [ 0.22245591 -0.04314564  0.3712728 ]

   Node information (summary):
     policy:            MeanPolicy
     branches shape:     (3, 3)
     branches[0]:        [-0.01450323 -0.28670244  0.15408422]

   ConsistencyLoss:
     L = α||S_t - S_{t-1}||^2 + β||S_t - Ŝ_{t+1}||^2
     α = 1.0, β = 1.0
     loss = 0.003909313321780935

   HCModel.forward() matches single-node result ✔
