.. _contributing:

Contributing
============

Contributions are welcome!  
You can help by improving documentation, adding new examples,
enhancing the quantum–hypercausal core, or fixing bugs.

**How to get started**

1. **Fork the repository** on GitHub and create a dedicated branch for your contribution.
2. Follow the established **PEP 8**, **type hinting**, and **linting** standards.
3. Add or update **unit tests** to maintain reliability and coverage above **75 %**.
4. Write clear, concise **commit messages** and include references to related issues or papers when applicable.
5. **Open a pull request (PR)** describing your update and its motivation.

---

Extended Guidelines
-------------------

Before contributing, ensure your environment is properly configured in **editable mode** as shown in the  
:ref:`Getting Started <getting_started>` section. This setup allows your modifications to be reflected immediately  
without reinstallation, facilitating rapid iteration during development.

If your contribution involves significant updates to the **QML-HCS core modules** (such as `HCModel`, `QuantumBackend`,  
`ConsistencyLoss`, or causal projection mechanisms), please include a brief theoretical rationale or pseudocode  
to ensure that your addition aligns with the framework’s scientific and causal structure.

Each pull request should be **self-contained and reproducible**.  
For experimental extensions or research components, it is encouraged to include a minimal runnable example under  
``src/qmlhc/examples/`` that demonstrates the intended use or theoretical impact of the modification.

---

Testing and Verification
------------------------

After implementing your changes, run the project’s automated test suite to confirm that the system remains stable  
and compliant. Testing instructions are detailed in the :ref:`Getting Started <getting_started>` guide.

Consistent testing not only ensures functional reliability but also supports reproducibility in collaborative research,  
a key principle in scientific software design.

---

Documentation and Scientific Transparency
-----------------------------------------

Contributors are strongly encouraged to update or extend the documentation when adding new features or theoretical components.  
Follow the same structure and tone used throughout this manual to maintain clarity and accessibility.

To preview your documentation changes locally:

.. code-block:: bash

   sphinx-build -E -a -b html docs/ docs/_build/html

Open the generated page in your browser:

.. code-block:: text

   docs/_build/html/index.html

Ensure that your examples and math expressions render correctly, and that new content follows the same  
academic rigor and stylistic consistency demonstrated in other sections.

---

Ethical and Research Standards
------------------------------

QML-HCS aims to serve as a **scientific framework** for advancing *quantum-inspired machine learning*  
and **hypercausal inference**. To preserve integrity and reproducibility:

- Provide references for any external methods, datasets, or algorithms used.
- Document all experimental parameters clearly.
- Avoid introducing proprietary or closed-source dependencies.
- Ensure that benchmark or performance claims are supported by verifiable data.

---

Reporting Issues and Collaboration
----------------------------------

If you encounter a bug, have a feature proposal, or wish to discuss integration ideas, please open an issue on GitHub:

`QML-HCS – GitHub Issues <https://github.com/Neureonmindflux-Research-Lab/qml-hcs/issues>`_

When reporting, include:
- A brief description of the issue or enhancement.
- The environment and Python version used.
- Any relevant logs or error messages.

Meaningful feedback helps maintain the long-term robustness of the system.

---

Thank you for helping advance **QML-HCS**.  
Your collaboration contributes to a transparent and forward-looking ecosystem for  
**quantum-inspired machine learning and hypercausal modeling**.
Together, we can push the boundaries of research and innovation!