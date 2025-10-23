
# Exercises QiskitBackend, PennyLaneBackend (with faked SDKs via sys.modules),


from __future__ import annotations

import sys
import types
import numpy as np
import pytest

from qmlhc.core.backend import BackendConfig


# ---------- Install fake Qiskit & PennyLane BEFORE importing adapters ----------

def _install_fake_qiskit():
    if "qiskit" in sys.modules:
        return
    qiskit = types.ModuleType("qiskit")
    primitives = types.ModuleType("qiskit.primitives")

    class QuantumCircuit:
        def __init__(self, n):
            self.n = n
        def ry(self, *_args, **_kwargs):
            return
        def barrier(self):
            return

    class _ResElem:
        def __init__(self, counts):
            # Mimic result[0].data.meas.get("counts", {})
            self.data = types.SimpleNamespace(meas={"counts": counts})

    class _Res:
        def __init__(self, counts):
            self._counts = counts
        def result(self):
            # QiskitBackend expects a list-like
            return [_ResElem(self._counts)]

    class Sampler:
        def run(self, _qc, shots=1024):
            # Deterministic counts for 3 qubits; only "000" and "111"
            return _Res({"000": shots // 2, "111": shots // 2})

    qiskit.QuantumCircuit = QuantumCircuit
    primitives.Sampler = Sampler

    sys.modules["qiskit"] = qiskit
    sys.modules["qiskit.primitives"] = primitives


def _install_fake_pennylane():
    if "pennylane" in sys.modules:
        return
    qml = types.ModuleType("pennylane")
    qml.__version__ = "0.0.0-fake"

    def device(name, wires, shots=None):
        return types.SimpleNamespace(name=name, wires=wires, shots=shots)

    def qnode(dev):
        def deco(fn):
            def wrapped(x):
                # Execute to touch RY/CNOT/expval; return zeros (len = wires)
                _ = fn(x)
                return tuple(0.0 for _ in range(dev.wires))
            return wrapped
        return deco

    # Gate + measurement stubs
    def RY(_theta, wires=None): return None
    def CNOT(wires=None): return None
    class PauliZ:
        def __init__(self, _wire): pass
    def expval(_op): return 0.0

    qml.device = device
    qml.qnode = qnode
    qml.RY = RY
    qml.CNOT = CNOT
    qml.PauliZ = PauliZ
    qml.expval = expval

    sys.modules["pennylane"] = qml


@pytest.fixture(autouse=True, scope="module")
def _autouse_install_fakes():
    # Ensure both fakes exist so importing qmlhc.backends.* never crashes.
    _install_fake_qiskit()
    _install_fake_pennylane()
    yield


# ---------- Tests for public re-exports ---------------------------------------

def test_backends_public_reexports():
    import qmlhc.backends as B
    assert hasattr(B, "QiskitBackend")
    assert hasattr(B, "PennyLaneBackend")
    assert hasattr(B, "CppBackend")


# ---------- QiskitBackend (fake SDK) ------------------------------------------

def test_qiskit_backend_fake_runs_and_projects():
    from qmlhc.backends.qiskit_backend import QiskitBackend

    be = QiskitBackend(BackendConfig(output_dim=3, shots=100), num_qubits=3)
    x = np.array([0.2, -0.1, 0.3])
    be.encode(x)
    s = be.run()
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


# ---------- PennyLaneBackend (fake SDK) ---------------------------------------

def test_pennylane_backend_fake_runs_and_projects():
    from qmlhc.backends.pennylane_backend import PennyLaneBackend

    be = PennyLaneBackend(BackendConfig(output_dim=2, shots=50), num_qubits=2, device_name="default.qubit")
    x = np.array([0.1, 0.3])
    be.encode(x)
    s = be.run()
    fut = be.project_future(s, branches=5)
    caps = be.capabilities()

    assert s.shape == (2,)
    assert fut.shape == (5, 2)
    assert "PennyLaneDevice" in caps["backend_name"]
    assert isinstance(caps["backend_version"], str)
    assert caps["supports_batch"] is True


def test_pennylane_backend_output_dim_validation():
    from qmlhc.backends.pennylane_backend import PennyLaneBackend
    with pytest.raises(ValueError):
        _ = PennyLaneBackend(BackendConfig(output_dim=1), num_qubits=2)


# ---------- CppBackend (fake bridge) ------------------------------------------

class _FakeCppBridge:
    def __init__(self, dim=2):
        self._dim = int(dim)
        self._last = None
    def encode(self, arr: np.ndarray) -> None:
        self._last = np.asarray(arr, dtype=float).reshape(-1)
    def run(self, params=None) -> np.ndarray:
        # Return tanh of last input + small bias
        if self._last is None:
            # Mimic extension behavior; QuantumBackend._require_input will guard anyway
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

    # encode size must match
    with pytest.raises(ValueError):
        be.encode([0.1])  # wrong size

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
    # Bridge declares output_dim=3 but config asks for 2 -> should raise on init
    bad_bridge = _FakeCppBridge(dim=3)
    with pytest.raises(ValueError):
        _ = CppBackend(BackendConfig(output_dim=2), bridge_module=bad_bridge)


def test_cpp_backend_missing_method_raises_attributeerror():
    from qmlhc.backends.cpp_backend import CppBackend

    class BrokenBridge:
        # Missing 'project_future' on purpose
        def encode(self, x): pass
        def run(self, p=None): return np.zeros(1)
        def capabilities(self): return {"output_dim": 1}

    with pytest.raises(AttributeError):
        _ = CppBackend(BackendConfig(output_dim=1), bridge_module=BrokenBridge())
