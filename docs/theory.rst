.. _theory:

Theory & Concepts
=================

This section explains the theoretical foundations of QML-HCS, covering
quantum principles, causal inference, and the mathematical structure behind
hypercausal hybrid learning systems.

Associated Paper
----------------

.. raw:: html

   <div style="
     border: 2px solid #991b1b;
     background: #ffffffff;
     padding: 18px 20px;
     border-radius: 10px;
     box-shadow: 0 8px 20px rgba(0,0,0,0.35);
     margin-top: 1rem;
     margin-bottom: 1.5rem;
   ">

     <h3 style="margin-top:0; color:#f87171;">
       QML-HCS: A Hypercausal Quantum Machine Learning Framework for
       Non-Stationary Environments
     </h3>

     <p style="margin: 0.4rem 0 0.8rem 0; color: #757575ff;">
       <strong>Hector E. Mozo</strong>
     </p>

     <p style="margin-bottom: 1rem; color: #686868ff; line-height:1.6;">
      This paper introduces the theoretical foundations of QML-HCS by formalizing
      a hypercausal learning model for non-stationary environments. It defines
      the core execution semantics, including multi-branch future generation,
      projection policies, and continuous causal feedback mechanisms, and
      presents the mathematical structure used to maintain coherence and
      stability under distributional drift. The paper focuses on establishing
      the architectural and conceptual framework that informs the design and
      behavior of the QML-HCS software.
     </p>

     <div style="display:flex; gap:12px; flex-wrap:wrap;">

       <a href="https://arxiv.org/abs/2511.17624"
          target="_blank"
          style="
            padding: 10px 14px;
            background: #991b1b;
            color: #ffffff;
            border-radius: 6px;
            text-decoration: none;
            font-weight: 600;
          ">
          arXiv:2511.17624
       </a>

       <div style="
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 8px 12px;
        background: #0f0f0f;
        border: 1.5px solid #991b1b;
        border-radius: 6px;
        font-family: monospace;
        font-size: 0.95em;
        color: #ffffff;
      ">
        <span>DOI: 10.48550/arXiv.2511.17624</span>

        <button onclick="
          navigator.clipboard.writeText('10.48550/arXiv.2511.17624');
          this.innerText='✓';
          setTimeout(() => this.innerText='📋', 1200);
        "
        style="
          background: transparent;
          border: none;
          color: #f87171;
          cursor: pointer;
          font-size: 1.1em;
        "
        title="Copy DOI">
          ⧉
        </button>
      </div>

     </div>
   </div>



.. raw:: html

   <div style="
     border: 2px solid #1c1758ff;
     background: #ffffffff;
     padding: 18px 20px;
     border-radius: 10px;
     box-shadow: 0 8px 20px rgba(0,0,0,0.35);
     margin-top: 1rem;
     margin-bottom: 1.5rem;
   ">

     <h3 style="margin-top:0; color:#4f46e5;">
       Pre-Temporal Model of Quantum Causal Order
     </h3>

     <p style="margin: 0.4rem 0 0.8rem 0; color: #757575ff;">
       <strong>Hector E. Mozo</strong>
     </p>

     <p style="margin-bottom: 1rem; color: #686868ff; line-height:1.6;">
      This paper establishes the theoretical foundation for treating causal order
      as a continuous, quantifiable resource within computational and learning
      frameworks. It introduces the causal-indefiniteness measure <em>λ(W)</em>,
      defined via the trace distance between a quantum process and the convex set
      of causally separable processes, providing a principled scalar that
      interpolates between indefinite and definite causal structure.
      <br><br>
      Within the context of QML-HCS, this pre-temporal formulation supplies a
      rigorous conceptual layer for modeling systems whose causal structure
      evolves over time. The measure <em>λ(W)</em> functions as an operational
      signal that can be tracked, optimized, or regularized within hypercausal
      learning loops, enabling QML-HCS to reason about causal consolidation,
      stability, and regime transitions in non-stationary environments.
      <br><br>
      In this way, the framework leverages pre-temporal causal dynamics not as an
      abstract phenomenon, but as a computable control variable that informs
      prediction, adaptation, and system-level coherence.
     </p>

     <div style="display:flex; gap:12px; flex-wrap:wrap;">

       <a href="https://ssrn.com/abstract=5993818"
          target="_blank"
          style="
            padding: 10px 14px;
            background: #1b176eff;
            color: #ffffff;
            border-radius: 6px;
            text-decoration: none;
            font-weight: 600;
          ">
          SSRN: 5993818
       </a>

       <div style="
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 8px 12px;
        background: #0f172a;
        border: 1.5px solid #201d64ff;
        border-radius: 6px;
        font-family: monospace;
        font-size: 0.95em;
        color: #e5e7eb;
      ">
        <span>DOI: 10.2139/ssrn.5993818</span>

        <button onclick="
          navigator.clipboard.writeText('10.2139/ssrn.5993818');
          this.innerText='✓';
          setTimeout(() => this.innerText='⧉', 1200);
        "
        style="
          background: transparent;
          border: none;
          color: #a5b4fc;
          cursor: pointer;
          font-size: 1.1em;
        "
        title="Copy DOI">
          ⧉
        </button>
      </div>

     </div>
   </div>
