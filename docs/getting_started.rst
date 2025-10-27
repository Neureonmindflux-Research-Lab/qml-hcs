.. _getting_started:

Installation
============

This section guides you through installing, configuring, and verifying **QML-HCS** —  
the *Quantum Machine Learning Hypercausal System*.  

It covers installation from source and via PyPI (package mode), environment setup,  
and first-run validation for both users and contributors.

---

Overview
--------

**QML-HCS** provides a research-grade environment for building *quantum-inspired causal architectures*  
and exploring **hypercausal inference**, **quantum learning**, and **non-stationary systems**.

You can install QML-HCS in two ways:

1. **As a local editable project** (recommended for development and experiments)
2. **As a standard installed package** (for users or deployment environments)

The project follows a modern Python packaging layout (PEP 621 + `src/` structure)  
and includes optional extras for development, documentation, and testing.

---

Requirements
------------

- **Python 3.9+** (tested on 3.9–3.12)
- **pip >= 23.0**
- **Git** for cloning the repository
- Optional: **virtual environment manager** (venv, Poetry, or Conda)

It is highly recommended to use a **virtual environment** to isolate dependencies
and avoid system-wide conflicts.

---

Installation (Editable Source Mode)
-----------------------------------

1. **Clone the repository**

   .. code-block:: bash

      git clone https://github.com/Neureonmindflux-Research-Lab/qml-hcs.git
      cd qml-hcs

2. **Create and activate a virtual environment**

   .. code-block:: bash

      python3 -m venv .venv
      source .venv/bin/activate

   *(Windows PowerShell)*

   .. code-block:: powershell

      python -m venv .venv
      .\.venv\Scripts\Activate.ps1

3. **Upgrade pip and install the package in editable mode**

   .. code-block:: bash

      pip install -U pip
      pip install -e .

This editable mode links your local `src/qmlhc` folder directly into the environment,
so code changes are immediately reflected without reinstalling.

---

Installation (Package Mode)
---------------------------

The package can be installed directly from **PyPI** using the standard Python installation command:

.. code-block:: bash

   pip install qml-hcs


or specify a particular version:

.. code-block:: bash

   pip install qml-hcs==0.1.0

You can verify installation by running:

.. code-block:: bash

   python -c "import qmlhcs; print(qmlhcs.__version__)"

This method is ideal for end-users or production deployments where the source code is not being modified, but the full library functionality and API features remain available for direct use.

---

Optional Installation Profiles
------------------------------

QML-HCS supports modular *extras* to tailor the installation
to your workflow or research needs.

- **Development tools** (linting, typing, packaging):

  .. code-block:: bash

     pip install -e .[dev]

- **Documentation stack** (Sphinx + themes + MyST + rendering tools):

  .. code-block:: bash

     pip install -e .[docs]

- **Testing suite** (pytest + coverage):

  .. code-block:: bash

     pip install -e .[test]

- **Visualization utilities** (matplotlib):

  .. code-block:: bash

     pip install -e .[viz]

- **All-in-one environment (everything)**:

  .. code-block:: bash

     pip install -e .[all]

> Each of these groups is defined under `[project.optional-dependencies]`
> in `pyproject.toml`, allowing you to install only what you need.

---

Verifying Installation
----------------------

To confirm that QML-HCS is correctly recognized by Python, run:

.. code-block:: bash

   python -c "import qmlhc, sys; print('QML-HCS imported successfully on Python', sys.version)"

You should see an output similar to:

.. code-block:: text

   QML-HCS imported successfully on Python 3.11.8

If you see ``ModuleNotFoundError: No module named 'qmlhc'``,
double-check that your virtual environment is active and the installation succeeded.

---

Running the Minimal Demo
------------------------

Once installed, try the minimal example to confirm correct functionality:

.. code-block:: bash

   qmlhc-demo

or run it directly as a Python module:

.. code-block:: bash

   python -m qmlhc.examples.ex_minimal_core_demo

This script demonstrates the basic architecture and telemetry output
for the quantum hypercausal backend.

Example output

.. code-block:: text
   :caption: Minimal Core Demo run

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
     α = 1.0, β = 1.0
     loss = 0.003909313321780935

   HCModel.forward() matches single-node result ✔

**Consistency loss definition**

.. math::

   L \;=\; \alpha \,\lVert S_t - S_{t-1} \rVert^2
   \;+\;
   \beta \,\lVert S_t - \hat{S}_{t+1} \rVert^2

**Explanation**

This minimal example demonstrates the core execution of the *Quantum Machine Learning Hypercausal Core* (QMLHC).  
It shows how the system evolves between two consecutive states, evaluates internal consistency, and confirms that the forward operation matches the theoretical expectation.  

You can continue exploring this and other runnable examples in the :ref:`Examples <examples>` section for more advanced demonstrations of QMLHC models and configurations.


---

Creating Your Own Pipelines
===========================

This section explains how to start building your own code using QMLHC’s core modules.
It follows the same structure as the official examples under ``src/qmlhc/examples/``.

**Quick steps**

1. **Define a backend**: subclass ``QuantumBackend`` and implement the logic of ``run`` and ``project_future``.
2. **Wrap the backend** in an ``HCNode`` and attach a policy such as ``MeanPolicy``.
3. **Combine nodes** into an ``HCModel`` or ``HCGraph`` depending on the desired topology.
4. **Add a loss** such as ``ConsistencyLoss`` or ``TriadicLoss``.
5. **Optionally train** using a lightweight optimizer and callbacks (see training examples).

**Minimal skeleton**

.. code-block:: python

   from qmlhc.core import BackendConfig, QuantumBackend
   from qmlhc.hc import HCNode, MeanPolicy
   from qmlhc.loss import ConsistencyLoss
   import numpy as np

   class MyBackend(QuantumBackend):
       def run(self, params=None):
           x = self._require_input().astype(float)
           s = np.tanh(0.9 * x + 0.1)
           return self._validate_state(s)

       def project_future(self, s_t, branches=3):
           s = self._validate_state(s_t)
           deltas = np.linspace(-0.25, 0.25, branches)
           fut = np.stack([np.tanh(s + d) for d in deltas], axis=0)
           return self._validate_branches(fut)

   cfg = BackendConfig(output_dim=3)
   node = HCNode(backend=MyBackend(cfg), policy=MeanPolicy())
   model = HCModel([node])

   x_t, s_tm1 = np.array([0.2, -0.1, 0.4]), np.array([0.15, -0.05, 0.35])
   s_t, s_hat, info = node.forward(x_t, s_tm1=s_tm1, branches=3)

   loss_fn = ConsistencyLoss(alpha=1.0, beta=1.0)
   loss = loss_fn(s_tm1, s_t, s_hat)

You can explore more runnable implementations in the :ref:`Examples <examples>` section,
which includes advanced training, multi-node graphs, and benchmarking demos.


Building the Documentation
--------------------------

If you installed the `docs` extras, you can generate the local HTML documentation:

.. code-block:: bash

   sphinx-build -E -a -b html docs/ docs/_build/html

Then open:

.. code-block:: text

   docs/_build/html/index.html

to view the generated site in your browser.





---

Repository Structure
--------------------

The QML-HCS repository follows a clean modular layout:

.. code-block:: text

      qml-hcs/
      ├── src/
      │   └── qmlhc/                     # Core Python package
      │       ├── __init__.py
      │       ├── core/                  # Quantum hypercausal core modules
      │       ├── hc/                    # Hypercausal policies and dynamics
      │       ├── predictors/            # Predictors, operators, and projection layers
      │       ├── loss/                  # Loss and metric definitions
      │       ├── metrics/               # Evaluation metrics and consistency checks
      │       ├── optim/                 # Optimizers and training utilities
      │       ├── callbacks/             # Telemetry and training callbacks
      │       ├── backends/              # Interfaces to quantum / hybrid backends
      │       └── examples/              # Runnable minimal and advanced examples
      │
      ├── tests/                         # Unit and integration tests
      │   ├── test_core.py
      │   ├── test_loss.py
      │   └── ...
      │
      ├── docs/                          # Sphinx documentation
      │   ├── conf.py
      │   ├── index.rst
      │   ├── getting_started.rst
      │   └── examples.rst
      │
      ├── pyproject.toml                 # Build metadata (PEP 621)
      ├── Makefile                       # Build, test, and docs automation
      └── README.md                      # Project overview and installation guide


This layout ensures a clear separation between the library code (`src/qmlhc`),
tests, documentation, and build configuration.

---

Troubleshooting
===============

1. **“ModuleNotFoundError: No module named ‘qmlhcs’”**  
   This error may occur if the installation did not complete successfully or if the virtual environment is not active.  
   Activate the environment and reinstall using:

   .. code-block:: bash

      pip install qml-hcs

2. **“Command ‘qmlhc-demo’ not found”**  
   This issue typically occurs when the package’s console scripts are not available in the system path.  
   Reinstall the package or ensure that the Python environment’s ``bin/`` directory is included in the ``PATH`` variable.

3. **“ImportError: cannot import name ...”**  
   This problem may arise when using an outdated version of the library.  
   Upgrade to the latest release with:

   .. code-block:: bash

      pip install -U qml-hcs

4. **“Documentation build fails”**  
   If Sphinx fails to build the documentation due to missing extensions or duplicated ``.. toctree::`` directives, verify that all dependencies are installed and that only one ``:caption:`` directive appears per section.

5. **“Version mismatch or API not found”**  
   Cached or partially installed versions can cause inconsistencies.  
   Clear the environment and reinstall cleanly using:

   .. code-block:: bash

      pip uninstall qml-hcs -y && pip install qml-hcs

.. note::

   If problems persist, report installation or runtime issues through the official issue tracker:  
   `GitHub Issues – QML-HCS <https://github.com/Neureonmindflux-Research-Lab/qml-hcs/issues>`_


---

Next Steps
-----------

Once QML-HCS is installed and verified, explore:

- The **Examples** section for runnable demos
- The **Core API Reference** for detailed class and function documentation
- The **Benchmark Studies** for experimental comparisons

These resources will help you understand how QML-HCS integrates *quantum-inspired logic*
into adaptive, hypercausal neural architectures.

---

By following this setup guide, your environment will be ready for  
**quantum-hypercausal research and development** using QML-HCS.

Collaborating and Contributing
==============================

If you’d like to contribute new features, documentation, or examples to QMLHC,
please review the :ref:`Contributing <contributing>` section for basic guidelines
on branching, style, and pull-request workflow.

Once your environment is ready, you can verify everything with the test suite below.

Running the Test Suite
======================

To verify that everything is working correctly, you can run the full test suite or a single test file.

**Run all tests**

.. code-block:: bash

   pytest -q --cov=qmlhc --cov-report=term-missing

**Run a single test file**

.. code-block:: bash

   pytest tests/test_core.py

**Example output**

.. code-block:: text

   .....................................................                                                                                               [100%]
   ===================================================================== tests coverage =====================================================================
   _____________________________________________ coverage: platform linux, python 3.13.5-final-0 _______________________________________________

   Name                                         Stmts   Miss Branch BrPart  Cover   Missing
   ----------------------------------------------------------------------------------------
   src/qmlhc/backends/cpp_backend.py               37      2      8      0    96%   158-159
   src/qmlhc/callbacks/base.py                     28      1     10      0    97%   118
   src/qmlhc/callbacks/depth_control.py            30      6      4      0    82%   93-96, 101, 105, 109, 113
   src/qmlhc/callbacks/telemetry.py                52      1      6      2    95%   70->exit, 82
   src/qmlhc/core/backend.py                       42      3     10      1    92%   120, 142, 234
   src/qmlhc/examples/ex_minimal_core_demo.py      56     56      4      0     0%   17-174
   src/qmlhc/hc/graph.py                           69      0     26      1    99%   221->219
   src/qmlhc/metrics/control.py                    26      1      6      1    94%   82
   ----------------------------------------------------------------------------------------
   TOTAL                                          847     70    144      5    92%

   25 files skipped due to complete coverage.
   Required test coverage of 75.0% reached. Total coverage: 92.03%
   53 passed in 0.32s
   (.venv) (base) mozoh@pop-os:~/Desktop/P1/qml-hcs$

A 100% test success and a coverage above 90% confirm that the QMLHC package is correctly installed and all major modules are functioning as expected.

Acknowledgments
===============

We appreciate your interest in **QML-HCS** and the time taken to review and explore the library.  
Your feedback and contributions are essential to help us continue refining and expanding the  
**Quantum Machine Learning Hypercausal System** as a reliable and innovative research framework.

For suggestions, feature requests, or academic collaborations, please open a discussion or  
report an issue on our official repository:

`QML-HCS – GitHub Repository <https://github.com/Neureonmindflux-Research-Lab/qml-hcs>`_

Thank you for supporting open research and advancing the development of **quantum-inspired machine learning**.

---