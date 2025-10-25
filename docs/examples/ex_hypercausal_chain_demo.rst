Hypercausal Chain Demo
======================

Introduction
------------

This example demonstrates a multi-node hyper-causal chain simulation composed of three
connected nodes (**HCNode**) sharing a sequential temporal flow. Each node uses a
parametric backend (**ParametricBackend**) and cooperates through causal propagation
and gradient-based optimization.

---

General Flow Structure
----------------------

The model represents a **temporal hyper-causal system** where each node contributes to a
sequential information chain:

- **Parametric Backend**: transforms inputs via :math:`S_t = \tanh(w \cdot x_t + b)` and projects multiple possible futures.  
- **Linear Projector**: expands :math:`S_t` into :math:`K` candidate branches :math:`S_{t+1}^{(k)}`.  
- **Loss functions**: combine *task accuracy*, *temporal consistency*, and *branch coherence* for optimization.  
- **Gradient Descent**: updates all backends using finite-difference approximations.

How to Run
----------

.. code-block:: console

   # From the project root
   python -m examples.ex_hypercausal_chain_demo

   # Or directly
   python examples/ex_hypercausal_chain_demo.py

Relevant Code Snippets
----------------------

.. literalinclude:: ../../examples/ex_hypercausal_chain_demo.py
   :language: python
   :linenos:
   :lines: 40-137
   :caption: Definition of the ParametricBackend class (tanh transformation and projection)

.. literalinclude:: ../../examples/ex_hypercausal_chain_demo.py
   :language: python
   :linenos:
   :lines: 250-350
   :caption: Main function chain_demo_step() (chain flow, loss computation, and optimization)

Functional Explanation
----------------------

The hypercausal chain operates as a **multi-node causal model**, where each node processes,
projects, and corrects its state based on local losses and temporal dependencies.

1. **Parametric Transformation**

   Each node computes its local state:

   .. math::

      S_t = \tanh(w \cdot x_t + b)

   Here, :math:`w` and :math:`b` are node-specific parameters learned through gradient descent.
   The nonlinear :math:`\tanh` activation ensures numerical stability, bounding all internal states within :math:`(-1, 1)`.

2. **Future Projection (Linear Projector)**

   Each state generates :math:`K` possible futures using a linear projector centered at the current
   state:

   .. math::

      S_{t+1}^{(k)} = \text{LinearProjector}(S_t), \quad k \in \{1, \dots, K\}

   This projection expands the local state into a hypercausal "fan" of possibilities,
   representing multiple potential outcomes for the next time step.

3. **Loss Composition**

   The total loss combines three complementary objectives:

   .. math::

      \mathcal{L}_{total} = \mathcal{L}_{task} + 0.5 \, \mathcal{L}_{consistency} + 0.3 \, \mathcal{L}_{coherence}

   - **Task Loss (MSE):**

     .. math::

        \mathcal{L}_{task} = \frac{1}{T} \sum_{t=1}^{T} \| S_t - Y_t \|^2

     Measures how close the node’s output is to the desired target trajectory.

   - **Consistency Loss (Triadic):**

     .. math::

        \mathcal{L}_{consistency} = \alpha \| S_t - S_{t-1} \|^2 + \beta \| S_t - \hat{S}_{t+1} \|^2

     Ensures smooth temporal evolution between past, present, and predicted future states.

   - **Coherence Loss:**

     .. math::

        \mathcal{L}_{coherence} = \text{Var}(S_{t+1}^{(k)})

     Penalizes excessive divergence among projected branches, maintaining causal stability.

4. **Gradient Estimation and Parameter Update**

   Instead of backpropagation, the example uses a **finite-difference gradient estimator**:

   .. math::

      g_i = \frac{\mathcal{L}(\theta_i + \epsilon) - \mathcal{L}(\theta_i)}{\epsilon}

   Each parameter update follows a simple gradient-descent rule:

   .. math::

      \theta_i \leftarrow \theta_i - \eta \, g_i

   where :math:`\eta` is the learning rate.

5. **Optimization Loop**

   - The model runs for multiple time steps (:math:`T = 6`), accumulating losses.  
   - The optimizer (`make_gradient_descent`) applies one parameter update across all nodes.  
   - Losses and parameters before/after the update are displayed for interpretability.

Exact Output
------------

.. code-block:: console

   === Hypercausal Chain Demo ===
   D=3, K=5, T=6

   Parameters (before):
   {'b0_w': 0.9, 'b0_b': 0.05, 'b1_w': 0.95, 'b1_b': 0.02, 'b2_w': 1.05, 'b2_b': 0.0}

   Losses BEFORE update:
   {'task': 0.02533261887044361, 'cons': 0.006637497192550465, 'coh': 0.029526745779767466, 'total': 0.037509391200649084}

   Updating parameters with GD (finite-diff grads)...
   ||grad||_2 ≈ 3.556270e-01

   Parameters (after):
   {'b0_w': 0.8979223132592518, 'b0_b': 0.040790416564867205, 'b1_w': 0.9474490593467247, 'b1_b': 0.009730820141449475, 'b2_w': 1.0473899389089636, 'b2_b': -0.010405162840116874}

   Losses AFTER update:
   {'task': 0.019773491535848887, 'cons': 0.006664418126248786, 'coh': 0.02973869208514816, 'total': 0.03202730822451773}

   Summary:
   total BEFORE:  0.037509
   total AFTER :  0.032027
