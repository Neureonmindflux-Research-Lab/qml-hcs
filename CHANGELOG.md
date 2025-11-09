# Changelog
All notable changes to this project will be documented in this file.  
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)  
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2025-11-09
### Added
- Complete NumPy-based optimizer suite under `qmlhc.optim.numpy_optim/`:
  `adam`, `spsa`, `finite-diff`, `natural-grad`, `dual-ascent`, `mpc`, `kfac`, and `trust-kl`.
- Unified optimizer factory and registry system `create_optimizer_numpy()` for flexible configuration.
- Added `run_batch()` method for consistent batched execution `(B, D)` across backends.
- Added a minimal optimizer usage example integrated with existing demo pipelines.
- Added backend examples and documentation for the `capabilities()` interface.

### Changed
- Unified backend adapter layer for `PennyLane`, `Qiskit`, and `Cpp` backends with standardized conventions for
  capability reporting, batch execution, and noise handling.
- Added `supports_noise` keyword parameter to the PennyLane backend for explicit compatibility and control.
- Refined internal logic of `pennylane_backend.qnode_shot` for compatibility with the latest PennyLane versions (no API break).

### Docs
- Added minimal optimizer documentation (table + quick example).
- Updated RST for `qmlhc.backends` and optimizer sections.
- Verified Sphinx build without critical warnings.

### Testing
- Expanded unit and integration tests for optimizers, registry, and backend adapters.
- Achieved >99% overall test coverage with `pytest --cov`.

### Removed
- Removed outdated figures and deprecated references (e.g., `qmlhc.optim.api.rst`).

### Migration Notes
- No breaking changes introduced; existing APIs like `run()` remain functional.
- New parameters and features are optional and backward compatible.

---

## [0.2.1] - (Planned)
### Docs
- Extended optimizer documentation including parameter definitions, formulas, and detailed use cases.
- Added extended examples and Jupyter notebooks.

---

[0.2.0]: https://github.com/HectorMozo3110/qml-hcs/compare/v0.1.0...v0.2.0
