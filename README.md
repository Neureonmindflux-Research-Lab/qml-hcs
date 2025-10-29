[![PyPI version](https://badge.fury.io/py/qml-hcs.svg)](https://pypi.org/project/qml-hcs/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17445484.svg)](https://doi.org/10.5281/zenodo.17445484)
[![Documentation Status](https://readthedocs.org/projects/qml-hcs/badge/?version=latest)](https://qml-hcs.readthedocs.io/en/latest/?badge=latest)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/qml-hcs?style=flat-square)](https://pypi.org/project/qml-hcs/)
[![PyPI - Wheel](https://img.shields.io/pypi/wheel/qml-hcs?style=flat-square)](https://pypi.org/project/qml-hcs/)
[![License](https://img.shields.io/pypi/l/qml-hcs?style=flat-square)](https://github.com/Neureonmindflux-Research-Lab/qml-hcs/blob/main/LICENSE)
![Status](https://img.shields.io/badge/status-active-brightgreen?style=flat-square)
![Framework](https://img.shields.io/badge/framework-quantum--machine--learning-purple?style=flat-square)
![Backend](https://img.shields.io/badge/backend-python%20%7C%20C%2B%2B-lightgrey?style=flat-square)
[![Build](https://github.com/Neureonmindflux-Research-Lab/qml-hcs/actions/workflows/python-package.yml/badge.svg)](https://github.com/Neureonmindflux-Research-Lab/qml-hcs/actions)
![OS Support](https://img.shields.io/badge/OS-Linux%20%7C%20macOS%20%7C%20Windows-blue?style=flat-square)
[![CodeQL](https://github.com/Neureonmindflux-Research-Lab/qml-hcs/actions/workflows/codeql.yml/badge.svg)](https://github.com/Neureonmindflux-Research-Lab/qml-hcs/actions/workflows/codeql.yml)
[![codecov](https://codecov.io/gh/Neureonmindflux-Research-Lab/qml-hcs/graph/badge.svg?token=<TOKEN>)](https://codecov.io/gh/Neureonmindflux-Research-Lab/qml-hcs)
[![PyPI Downloads](https://static.pepy.tech/personalized-badge/qml-hcs?period=total&units=INTERNATIONAL_SYSTEM&left_color=LIGHTGREY&right_color=BLACK&left_text=downloads)](https://pepy.tech/projects/qml-hcs)






<p align="center">
  <img src="https://raw.githubusercontent.com/Neureonmindflux-Research-Lab/qml-hcs/main/qml-hcs-logo.svg"
       alt="QML-HCS Logo"
       width="1200"
       height="auto">
</p>

<p align="center">
  <b>Quantum Machine Learning Hypercausal System</b>  
  <br>
  <i>A research-grade library for quantum-inspired machine learning with hypercausal feedback.</i>
</p>

# Quantum Machine Learning Hypercausal System 

**QML-HCS** is a research-grade framework for constructing, simulating, and analyzing quantum-inspired machine learning architectures with hypercausal feedback mechanisms.  
It integrates deterministic computation with causal inference and quantum-like superposition principles to explore emerging paradigms in Quantum Machine Learning (QML) and Causal Systems Theory.

---

## Overview

QML-HCS provides a modular and extensible environment for the study of hypercausal quantum models—systems that unify classical causal inference with quantum-inspired dynamics such as superposition, reversible transformations, and probabilistic branching.  
It supports research into information propagation, causal stability, and consistency across interconnected quantum-like networks.

The framework is intended for scientific and engineering research in the following domains:

- **Quantum Machine Learning:** Development of quantum-inspired learning architectures.  
- **Causal Dynamics and Feedback Modeling:** Formalization of recursive multi-branch causal systems.  
- **Hybrid Quantum–Classical Computation:** Simulation of efficient causal propagation in hybrid systems.  
- **Counterfactual Simulation:** Modeling systems capable of evaluating alternative causal scenarios.  
- **Algorithmic Benchmarking:** Studying quantum-efficient learning and reasoning processes on classical hardware.

---

## Core Objectives

1. **Hypercausal Feedback Modeling:** Implement layered feedback systems capable of multi-directional causal propagation.  
2. **Quantum-Inspired Efficiency:** Apply principles of superposition and entanglement to reduce computational cost.  
3. **Deterministic–Stochastic Integration:** Provide configurable backends for deterministic, probabilistic, and mixed causal engines.  
4. **Scientific Transparency:** Ensure reproducibility and open experimentation through standardized interfaces.  
5. **Scalability and Extensibility:** Support modular expansion for backends, loss functions, and causal evaluators.

---

## Installation

Install the package directly from PyPI:

```bash
pip install qml-hcs
```

Install a specific version:

```bash
pip install qml-hcs==0.1.0
```

Verify installation:

```bash
python -c "import qmlhc; print(qmlhc.__version__)"
```

This mode is recommended for research and production environments where the source code remains static but full access to all APIs and modules is required.

---

## Getting Started

To verify installation, execute the minimal example:

```bash
qmlhc-demo
```

or run directly as a module:

```bash
python -m qmlhc.examples.ex_minimal_core_demo
```

Expected output (abridged):

```
=== Minimal Core Demo ===
output_dim (D):     3
branches (K):       3
...
HCModel.forward() matches single-node result ✔
```

Refer to the [Getting Started Guide](https://qml-hcs.readthedocs.io/en/latest/getting_started.html) for further instructions.

---

## Examples

The repository provides several scientifically oriented demonstrations:

- Minimal hypercausal core operation  
- Depth-dependent evaluation of feedback models  
- Quantum-inspired benchmarking and stability testing  
- Training with callback telemetry and adaptive losses  
- Coherence and consistency experiments under stochastic variation  

All examples are documented in the [Examples Section](https://qml-hcs.readthedocs.io/en/latest/examples.html).

---

## Intended Research Applications

QML-HCS serves as a research platform for the theoretical and experimental study of quantum-inspired machine learning.  
It facilitates investigations in:

- Quantum-efficient learning architectures  
- Simulation of adaptive feedback systems  
- Analysis of causal consistency and information stability  
- Hybrid quantum–classical training methodologies  
- Exploration of hypercausal structures for predictive and inferential modeling  

By providing a deterministic yet quantum-compatible environment, QML-HCS enables the testing of emerging theories in quantum-causal computation without the need for specialized quantum hardware.

---

## Contributing

Contributions are welcome.  
Researchers and developers can improve QML-HCS by adding new modules, extending the documentation, or enhancing the quantum-hypercausal backends.

### Contribution Guidelines

1. Fork the repository and create a feature branch.  
2. Follow PEP 8 conventions and maintain typing annotations.  
3. Ensure test coverage remains above 75%.  
4. Provide detailed documentation and minimal runnable examples.  
5. Submit a well-described pull request for review.

Further details are available in the [Contributing Section](https://qml-hcs.readthedocs.io/en/latest/contributing.html).

---

## Testing and Documentation

To execute the test suite:

```bash
pytest -v
```

To build documentation locally:

```bash
sphinx-build -E -a -b html docs/ docs/_build/html
```

View the generated site by opening:

```
docs/_build/html/index.html
```

---

## Issues and Feedback

For bug reports or feature suggestions, please use the official issue tracker:

[QML-HCS Issue Tracker](https://github.com/Neureonmindflux-Research-Lab/qml-hcs/issues)

When reporting, include:
- Operating system and Python version  
- Steps to reproduce  
- Logs or traceback if available  

---

## Research Vision

QML-HCS is part of the NeureonMindFlux Research Lab initiative to formalize quantum–causal computational frameworks.  
It seeks to unify Quantum Machine Learning, Causal Inference, and Deterministic Modeling into a single, reproducible platform for scientific investigation and applied experimentation.

---

## Acknowledgments

Developed under the NeureonMindFlux Research Initiative in quantum-inspired and hypercausal computation.  
The project benefits from ongoing collaboration and peer feedback within the open scientific community.

---
## 📚 Documentation

Full documentation is available here:

[**QML-HCS Official Documentation**](https://qml-hcs.readthedocs.io/en/latest/index.html)

---
## Contact

For inquiries, collaboration proposals, or research-related communication regarding QML-HCS, please use the following contact:

**Email:** contact@neureonmindfluxlab.org

---

**QML-HCS** — advancing research in Quantum Machine Learning with Hypercausal Feedback Systems.

---
