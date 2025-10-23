
# Package entry-point: defines public metadata and stable namespace anchors only.

# Semantic version synchronized with pyproject.toml.
__version__: str = "0.1.0"

# Short package descriptor for tooling and logging.
__package_name__: str = "qmlhc"
__description__: str = "Quantum Machine Learning with hypercausal feedback for non-stationary environments."

# Public namespace policy:
# - No eager imports to avoid side-effects and heavy dependencies at import time.
# - Subpackages (core, hc, predictors, loss, optim, backends, callbacks, metrics)
#   are discovered via standard 'import qmlhc.<submodule>' usage.

__all__ = [
    "__version__",
    "__package_name__",
    "__description__",
]
