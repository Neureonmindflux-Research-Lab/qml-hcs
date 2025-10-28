
# Public re-exports for backend adapters.

__all__ = ["QiskitBackend", "PennyLaneBackend", "CppBackend"]

import importlib
import warnings

def __getattr__(name):
    if name == "PennyLaneBackend":
        mod = importlib.import_module(".pennylane_backend", __package__)
        return getattr(mod, name)
    elif name == "QiskitBackend":
        try:
            mod = importlib.import_module(".qiskit_backend", __package__)
            return getattr(mod, name)
        except Exception as e:
            warnings.warn(f"Qiskit backend not available: {e}")
            return None
    elif name == "CppBackend":
        mod = importlib.import_module(".cpp_backend", __package__)
        return getattr(mod, name)
    raise AttributeError(f"No backend named {name!r}")
