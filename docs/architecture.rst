Architecture
============

This section presents the internal design of QML-HCS (Quantum Machine-Learning 
Hypercausal System), organized from the **lowest foundational layers** to the 
**highest orchestration layers**, and visualized through layered diagrams.

QML-HCS is fully modular. No single execution flow is mandatory. Instead, the 
system exposes interoperable components that can be assembled into custom 
pipelines depending on the experiment or backend.

Main Architecture Diagram
-------------------------

The following diagram shows the entire architecture from bottom (foundations) 
to top (orchestration):

.. graphviz::
   :align: center

    digraph Architecture {
        rankdir=TB;
        node [shape=box, style="filled,rounded", color="#1E1E1E", fillcolor="#3A3A3A", fontcolor="white"];

        Core       [label="Level 1 - Core Module\n(Contracts, Types, Registry, HCModel)"];
        Backends   [label="Level 2 - Backends\n(PennyLane, Qiskit, C++, pybind11)"];
        HC         [label="Level 3 - Hypercausal Layer\n(HCNode, HCGraph, Policies)"];
        Predictors [label="Level 4 - Predictors\n(Deterministic & Counterfactual)"];
        Loss       [label="Level 5 - Loss Module\n(Task • Coherence • Consistency)"];
        Metrics    [label="Level 6 - Metrics Module\n(Anomaly • Control • Forecasting)"];
        Optim      [label="Level 7 - Optimizers\n(SPSA, Adam, Natural-Grad, Trust-KL, KFAC)"];
        Callbacks  [label="Level 8 - Callbacks\n(Telemetry, DepthScheduler)"];

        Core -> Backends -> HC -> Predictors -> Loss -> Metrics -> Optim -> Callbacks;
    }

This is the **canonical structural view**, not a required flow of execution.

.. note::

   The diagram presents the canonical structural organization of QML-HCS. It is 
   intended as a conceptual layering model rather than an operational pipeline. 
   Component order in this diagram does *not* imply a mandatory execution sequence; 
   QML-HCS supports modular use and flexible composition.

Level 1 - Core Module (Structural Foundation)
---------------------------------------------
**See also the** :doc:`full Core documentation <modules/qmlhc/core/index>`

The core defines:

- Typed interfaces for backends, nodes, projection policies
- Registry system
- Formal validation
- HCModel: optional high-level orchestrator

.. graphviz::
  :align: center

    digraph Core {
        rankdir=LR;
        node [shape=box, style="rounded,filled", fillcolor="#2E2E2E", fontcolor="white"];

        Types      [label="Types & Protocols"];
        BackendIfc [label="QuantumBackend Interface"];
        Registry   [label="Backend Registry"];
        Model      [label="HCModel"];

        Types -> BackendIfc -> Registry -> Model;
    }

Level 2 - Backend Layer (Execution Engines)
-------------------------------------------
**See also the** :doc:`full Backends documentation <modules/qmlhc/backends/index>`

Backends provide actual numerical or quantum-based execution.

Supported adapters:

- PennyLane (analytic/differentiable)
- Qiskit (shot-based, stochastic)
- C++ pybind11 backend (deterministic, HPC speed)

.. graphviz::
   :align: center

    digraph Backends {
        rankdir=LR;
        node [shape=box, style="rounded,filled", fillcolor="#334455", fontcolor="white"];

        PL   [label="PennyLane Backend"];
        QK   [label="Qiskit Backend"];
        CPP  [label="C++ Backend"];

        PL -> QK -> CPP;
    }


Level 3 - Hypercausal Layer (Logical Evolution Engine)
------------------------------------------------------
**See also the** :doc:`full Hypercausal Layer documentation <modules/qmlhc/hc/index>`

Defines how states evolve over time and across branches.

Components:

- HCNode (input → state → futures → representative)
- HCGraph (causal connections)
- Projection policies

.. graphviz::
   :align: center

    digraph HC {
        rankdir=LR;
        node [shape=box, style="rounded,filled", fillcolor="#556677", fontcolor="white"];

        HCNode   [label="HCNode"];
        HCGraph  [label="HCGraph"];
        Policy   [label="Projection Policies"];

        HCNode  -> HCGraph;
        HCGraph -> Policy;
    }


Level 4 - Predictors (Deterministic & Counterfactual)
-----------------------------------------------------
**See also the** :doc:`full Predictors documentation <modules/qmlhc/predictors/index>`

Predictors extend computation with:

- LinearProjector
- Counterfactual anticipators
- Mirrored perturbations

.. graphviz::
   :align: center

    digraph Predictors {
        rankdir=LR;
        node [shape=box, style="rounded,filled", fillcolor="#774444", fontcolor="white"];

        Proj  [label="LinearProjector"];
        CF    [label="Counterfactual Anticipator"];
        Pert  [label="Perturbations"];

        Proj -> CF -> Pert;
    }


Level 5 - Loss Evaluation Layer
-------------------------------
**See also the** :doc:`full Loss Module documentation <modules/qmlhc/loss/index>`

Measures:

- **Task accuracy**
- **Inter-branch coherence**
- **Triadic temporal consistency**

.. graphviz::
   :align: center

    digraph Loss {
        rankdir=LR;
        node [shape=box, style="rounded,filled", fillcolor="#664466", fontcolor="white"];

        Task [label="Task Losses"];
        Coh  [label="Coherence"];
        Cons [label="Consistency"];

        Task -> Coh -> Cons;
    }


Level 6 - Metrics (Advanced Temporal & Statistical Analysis)
------------------------------------------------------------
**See also the** :doc:`full Metrics Module documentation <modules/qmlhc/metrics/index>`

Provides higher-order diagnostics:

- anomaly detection
- control stability
- forecasting alignment

.. graphviz::
   :align: center

    digraph Metrics {
        rankdir=LR;
        node [shape=box, style="rounded,filled", fillcolor="#334433", fontcolor="white"];

        Anom [label="Anomaly Metrics"];
        Ctrl [label="Control Metrics"];
        Fcast[label="Forecast Metrics"];

        Anom -> Ctrl -> Fcast;
    }


Level 7 - Optimization Module
-----------------------------
**See also the** :doc:`full Optimization documentation <modules/qmlhc/optim/index>`

Algorithms:

- finite-diff
- SPSA
- Adam
- Natural-Grad
- Trust-KL
- Dual-Ascent
- MPC
- KFAC

.. graphviz::
   :align: center

    digraph Optim {
        rankdir=LR;
        node [shape=box, style="rounded,filled", fillcolor="#444444", fontcolor="white"];

        GD   [label="Finite-Diff"];
        SPSA [label="SPSA"];
        Adam [label="Adam"];
        NG   [label="Natural Grad"];
        KL   [label="Trust-KL"];
        KFAC [label="KFAC"];

        GD -> SPSA -> Adam -> NG -> KL -> KFAC;
    }


Level 8 - Callbacks Layer
-------------------------
**See also the** :doc:`full Callbacks documentation <modules/qmlhc/callbacks/index>`

Provides live monitoring and state adaptation.

- TelemetryLogger  
- MemoryLogger  
- DepthScheduler  

.. graphviz::
   :align: center
   
    digraph Callbacks {
        rankdir=LR;
        node [shape=box, style="rounded,filled", fillcolor="#555555", fontcolor="white"];

        TL [label="TelemetryLogger"];
        ML [label="MemoryLogger"];
        DS [label="DepthScheduler"];

        TL -> ML -> DS;
    }


Optional Execution Flows (Not Mandatory)
----------------------------------------

QML-HCS allows multiple flows; these are optional examples.

**Hypercausal Flow Example**

.. graphviz::
   :align: center

    digraph FlowHC {
        rankdir=LR;
        node [shape=oval, style="filled", fillcolor="#335577", fontcolor="white"];

        X  [label="Input"];
        S  [label="State"];
        K  [label="Future Branches"];
        R  [label="Representative"];

        X -> S -> K -> R;
    }


**Predictor/Counterfactual Flow Example**

.. graphviz::
   :align: center

    digraph FlowCF {
        rankdir=LR;
        node [shape=oval, style="filled", fillcolor="#773333", fontcolor="white"];

        St [label="State"];
        LP [label="LinearProjector"];
        CF [label="Counterfactuals"];
        Ev [label="Evaluation"];

        St -> LP -> CF -> Ev;
    }


**Optimization Loop Example**

.. graphviz::
   :align: center
   
    digraph FlowOptim {
        rankdir=LR;
        node [shape=oval, style="filled", fillcolor="#555555", fontcolor="white"];

        P  [label="Params"];
        M  [label="Model"];
        L  [label="Loss/Metrics"];
        O  [label="Optimizer Step"];

        P -> M -> L -> O -> P;
    }


Summary
-------

QML-HCS is structured according to increasing abstraction:

1. Core Module - structural base  
2. Backends - execution engines  
3. Hypercausal Layer - logical evolution framework  
4. Predictors - deterministic & counterfactual expansion  
5. Loss Layer - coherence and task evaluation  
6. Metrics - statistical and temporal interpretation  
7. Optimization - adaptation and learning  
8. Callbacks - runtime monitoring and control  

The diagrams above reflect modular organization, not a required pipeline.  
Users may assemble custom computation flows depending on their needs.
