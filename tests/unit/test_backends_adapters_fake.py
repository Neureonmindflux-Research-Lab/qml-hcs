from __future__ import annotations

# --- Standard library imports grouped first
import importlib
import sys
import types

# --- Third-party imports
import numpy as np
import pytest

# --- Local imports
from qmlhc.core.backend import BackendConfig


# =============================================================================
# Fake SDKs install: minimal Qiskit & PennyLane stubs to make adapters import-safe
# =============================================================================

def _install_fake_qiskit() -> None:
    if "qiskit" in sys.modules:
        return

    qiskit = types.ModuleType("qiskit")
    primitives = types.ModuleType("qiskit.primitives")

    class QuantumCircuit:
        def __init__(self, n: int):
            self.n = n

        def ry(self, *_args, **_kwargs):
            return

        def barrier(self):
            return

    class _ResElem:
        def __init__(self, counts: dict[str, int]):
            self.data = types.SimpleNamespace(meas={"counts": counts})

    class _Res:
        def __init__(self, counts: dict[str, int]):
            self._counts = counts

        def result(self):
            return [_ResElem(self._counts)]

    class Sampler:
        def run(self, _qc, shots: int = 1024):
            return _Res({"000": shots // 2, "111": shots // 2})

    qiskit.QuantumCircuit = QuantumCircuit
    primitives.Sampler = Sampler

    sys.modules["qiskit"] = qiskit
    sys.modules["qiskit.primitives"] = primitives


def _install_fake_pennylane() -> None:
    if "pennylane" in sys.modules:
        return

    qml = types.ModuleType("pennylane")
    qml.__version__ = "0.0.0-fake"

    def device(name: str, wires: int, shots: int | None = None):
        return types.SimpleNamespace(name=name, wires=wires, shots=shots)

    def qnode(dev):
        def deco(fn):
            def wrapped(x):
                _ = fn(x)
                return tuple(0.0 for _ in range(dev.wires))
            return wrapped
        return deco

    def RY(_theta, wires=None):
        return None

    def CNOT(wires=None):
        return None

    class PauliZ:
        def __init__(self, _wire):
            pass

    def expval(_op):
        return 0.0

    qml.device = device
    qml.qnode = qnode
    qml.RY = RY
    qml.CNOT = CNOT
    qml.PauliZ = PauliZ
    qml.expval = expval

    sys.modules["pennylane"] = qml


# =============================================================================
# Autouse fixture: ensure fake SDKs exist before importing any backends
# =============================================================================

@pytest.fixture(autouse=True, scope="module")
def _autouse_install_fakes():
    _install_fake_qiskit()
    _install_fake_pennylane()
    yield


# =============================================================================
# Public re-exports: verify backends namespace surface
# =============================================================================

def test_backends_public_reexports():
    import qmlhc.backends as B

    assert hasattr(B, "QiskitBackend")
    assert hasattr(B, "PennyLaneBackend")
    assert hasattr(B, "CppBackend")


# =============================================================================
# Qiskit backend: behavior with custom StatevectorSampler and validations
# =============================================================================

def test_qiskit_backend_fake_runs_and_projects():
    prim = types.ModuleType("qiskit.primitives")

    class StatevectorSampler:
        def __init__(self):
            self.mode = 0

        def run(self, _qc, shots: int = 100):
            if self.mode == 0:
                self.mode = 1
                return types.SimpleNamespace(
                    result=lambda: types.SimpleNamespace(
                        quasi_dists=[{"000": 0.5, "111": 0.5}]
                    )
                )
            elif self.mode == 1:
                self.mode = 2
                return types.SimpleNamespace(
                    result=lambda: types.SimpleNamespace(
                        data=[
                            types.SimpleNamespace(
                                meas={"counts": {"000": shots // 2, "111": shots // 2}}
                            )
                        ]
                    )
                )
            else:
                return types.SimpleNamespace(result=lambda: types.SimpleNamespace())

    sys.modules["qiskit.primitives"] = prim
    prim.StatevectorSampler = StatevectorSampler

    qiskit_mod = types.ModuleType("qiskit")

    class QuantumCircuit:
        def __init__(self, n: int):
            self.n = n

        def ry(self, *_a, **_k):
            pass

        def barrier(self):
            pass

        def measure_all(self):
            pass

    qiskit_mod.QuantumCircuit = QuantumCircuit
    sys.modules["qiskit"] = qiskit_mod

    import qmlhc.backends.qiskit_backend as qb
    importlib.reload(qb)

    be = qb.QiskitBackend(BackendConfig(output_dim=3, shots=100), num_qubits=3)
    be._sampler = StatevectorSampler()  # type: ignore[attr-defined]

    x = np.array([0.2, -0.1, 0.3], dtype=float)
    be.encode(x)

    _ = be.run()
    s = be.run()
    _ = be.run()

    fut = be.project_future(s, branches=4)
    caps = be.capabilities()

    assert s.shape == (3,)
    assert np.all(np.isfinite(s))
    assert fut.shape == (4, 3)
    assert caps["backend_name"] == "QiskitSampler"
    assert caps["supports_shots"] is True


def test_qiskit_backend_output_dim_validation():
    from qmlhc.backends.qiskit_backend import QiskitBackend

    with pytest.raises(ValueError):
        _ = QiskitBackend(BackendConfig(output_dim=2), num_qubits=3)


# =============================================================================
# PennyLane backend: basic runs, projections, and validations
# =============================================================================

def test_pennylane_backend_fake_runs_and_projects():
    from qmlhc.backends.pennylane_backend import PennyLaneBackend

    be = PennyLaneBackend(
        BackendConfig(output_dim=2, shots=50),
        num_qubits=2,
        device_name="default.qubit",
    )
    x = np.array([0.1, 0.3], dtype=float)
    be.encode(x)

    s = be.run()
    fut = be.project_future(s, branches=5)
    fut2 = be.project_future(s, branches=0)
    caps = be.capabilities()

    assert s.shape == (2,)
    assert fut.shape == (5, 2)
    assert fut2.shape == (2, 2)
    assert "PennyLaneDevice" in caps["backend_name"]
    assert isinstance(caps["backend_version"], str)
    assert caps["supports_batch"] is True


def test_pennylane_backend_output_dim_validation():
    from qmlhc.backends.pennylane_backend import PennyLaneBackend

    with pytest.raises(ValueError):
        _ = PennyLaneBackend(BackendConfig(output_dim=1), num_qubits=2)


# =============================================================================
# C++ bridge backend: fake bridge for shape checks and error paths
# =============================================================================

class _FakeCppBridge:
    def __init__(self, dim: int = 2):
        self._dim = int(dim)
        self._last: np.ndarray | None = None

    def encode(self, arr: np.ndarray) -> None:
        self._last = np.asarray(arr, dtype=float).reshape(-1)

    def run(self, params=None) -> np.ndarray:
        if self._last is None:
            return np.zeros(self._dim, dtype=float)
        return np.tanh(self._last + 0.01)

    def project_future(self, s: np.ndarray, k: int) -> np.ndarray:
        k = int(max(2, k))
        noise = np.linspace(-0.05, 0.05, k)
        return np.stack([np.tanh(s + d) for d in noise], axis=0)

    def capabilities(self) -> dict:
        return {
            "backend_name": "FakeCpp",
            "backend_version": "1.0",
            "output_dim": self._dim,
            "max_qubits": 0,
            "supports_shots": False,
            "supports_noise": False,
            "supports_batch": True,
            "gradient": "none",
        }


def test_cpp_backend_happy_path_and_shapes():
    from qmlhc.backends.cpp_backend import CppBackend

    bridge = _FakeCppBridge(dim=2)
    be = CppBackend(BackendConfig(output_dim=2), bridge_module=bridge)

    with pytest.raises(ValueError):
        be.encode([0.1])

    be.encode([0.1, -0.2])
    s = be.run()
    assert s.shape == (2,)

    fut = be.project_future(s, branches=3)
    assert fut.shape == (3, 2)

    caps = be.capabilities()
    assert caps["backend_name"] == "FakeCpp"
    assert caps["supports_batch"] is True
    assert caps["gradient"].value in ("none", "parameter-shift", "finite-diff", "adjoint")

def test_cpp_backend_output_dim_mismatch_raises():
    from qmlhc.backends.cpp_backend import CppBackend

    bad_bridge = _FakeCppBridge(dim=3)
    with pytest.raises(ValueError):
        _ = CppBackend(BackendConfig(output_dim=2), bridge_module=bad_bridge)


def test_cpp_backend_missing_method_raises_attributeerror():
    from qmlhc.backends.cpp_backend import CppBackend

    class BrokenBridge:
        def encode(self, x):  # noqa: ANN001
            pass

        def run(self, p=None):  # noqa: ANN001
            return np.zeros(1)

        def capabilities(self):
            return {"output_dim": 1}

    with pytest.raises(AttributeError):
        _ = CppBackend(BackendConfig(output_dim=1), bridge_module=BrokenBridge())


# =============================================================================
# PennyLane backend: batch execution shapes and capability flags
# =============================================================================

def test_pennylane_backend_run_batch_shapes():
    from qmlhc.backends.pennylane_backend import PennyLaneBackend

    be = PennyLaneBackend(BackendConfig(output_dim=3, shots=None), num_qubits=3)
    X = np.stack(
        [np.zeros(3), np.ones(3) * 0.1, np.arange(3) * 0.01],
        axis=0,
    )
    out = be.run_batch(X)

    assert out.shape == (3, 3)
    assert np.all(np.isfinite(out))

    with pytest.raises(ValueError):
        be.run_batch(np.zeros(3))

    with pytest.raises(ValueError):
        be.run_batch(np.zeros((2, 2)))


def test_pennylane_backend_capabilities_flags_shots():
    from qmlhc.backends.pennylane_backend import PennyLaneBackend

    be = PennyLaneBackend(BackendConfig(output_dim=2, shots=None), num_qubits=2)
    caps = be.capabilities()
    assert caps["supports_shots"] is True
    assert caps["using_shots"] is False

    be2 = PennyLaneBackend(BackendConfig(output_dim=2, shots=100), num_qubits=2)
    caps2 = be2.capabilities()
    assert caps2["supports_shots"] is True
    assert caps2["using_shots"] is True


def test_pennylane_backend_capabilities_noise_override():
    from qmlhc.backends.pennylane_backend import PennyLaneBackend

    be = PennyLaneBackend(BackendConfig(output_dim=2), num_qubits=2, supports_noise=True)
    caps = be.capabilities()
    assert caps["supports_noise"] is True

    be2 = PennyLaneBackend(BackendConfig(output_dim=2), num_qubits=2, device_name="default.mixed")
    caps2 = be2.capabilities()
    assert caps2["supports_noise"] is True

# =============================================================================
# Backends __getattr__ coverage: warn branch (Qiskit), Cpp path, unknown raise
# =============================================================================

def test_backends_dunder_getattr_qiskit_missing_warns(monkeypatch):
    """Trigger the except branch: Qiskit import fails -> warns and returns None."""
    import qmlhc.backends as B

    real_import = B.importlib.import_module
    def fake_import(name, package=None):
        if name == ".qiskit_backend" and package == B.__package__:
            raise RuntimeError("boom")
        return real_import(name, package)

    monkeypatch.setattr(B.importlib, "import_module", fake_import)
    with pytest.warns(UserWarning):
        val = getattr(B, "QiskitBackend")
    assert val is None

def test_backends_dunder_getattr_cpp_backend_loaded():
    """Exercise elif name == 'CppBackend' branch (partial line coverage)."""
    import qmlhc.backends as B
    Cpp = getattr(B, "CppBackend")
    assert Cpp is not None and Cpp.__name__ == "CppBackend"

def test_backends_dunder_getattr_unknown_raises():
    """Exercise final raise AttributeError branch for unknown backend names."""
    import qmlhc.backends as B
    with pytest.raises(AttributeError):
        getattr(B, "NopeBackend")

# =============================================================================
# CppBackend Exception Path Coverage: capabilities() runtime failure tolerance
# =============================================================================

def test_cpp_backend_capabilities_handles_failure_min():
    from qmlhc.backends.cpp_backend import CppBackend
    from qmlhc.core.backend import BackendConfig

    class Bridge(_FakeCppBridge):
        def __init__(self, dim=2):
            super().__init__(dim); self.ok = True
        def capabilities(self):
            if self.ok: self.ok = False; return {"output_dim": self._dim}
            raise RuntimeError("boom")

    be = CppBackend(BackendConfig(output_dim=2), bridge_module=Bridge())
    caps = be.capabilities()  
    assert "backend_name" in caps

# =============================================================================
# QiskitBackend Branch Coverage: fallback when meas has no "counts"
# =============================================================================
def test_qiskit_backend_fallback_when_meas_has_no_counts():
    import sys, types, importlib, numpy as np
    sys.modules.pop("qiskit", None); sys.modules.pop("qiskit.primitives", None)

    class _QC:
        def __init__(self, n): self.n = n
        def ry(self, *_a, **_k): pass
        def barrier(self): pass
        def measure_all(self): pass

    class _Sampler:
        def run(self, *_a, **_k):
            res = types.SimpleNamespace(data=[types.SimpleNamespace(meas={"foo": 1})])
            return types.SimpleNamespace(result=lambda: res)

    qiskit = types.ModuleType("qiskit"); prim = types.ModuleType("qiskit.primitives")
    qiskit.QuantumCircuit = _QC; prim.Sampler = _Sampler
    sys.modules["qiskit"] = qiskit; sys.modules["qiskit.primitives"] = prim

    import qmlhc.backends.qiskit_backend as qb
    importlib.reload(qb)
    from qmlhc.core.backend import BackendConfig
    be = qb.QiskitBackend(BackendConfig(output_dim=3, shots=64), num_qubits=3)
    be.encode(np.zeros(3))
    out = be.run()
    assert out.shape == (3,)