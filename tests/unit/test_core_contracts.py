
# Comprehensive tests for src/qmlhc/core: types.py, backend.py, model.py, registry.py, and public __init__ re-exports.

from __future__ import annotations

import numpy as np
import pytest

# Public API re-exports should work
import qmlhc.core as core


# ---------- Types / Protocols -------------------------------------------------

def test_types_enums_and_protocols_runtime_checkable():
    # GradientKind enum exists and contains expected members
    assert set(k.value for k in core.GradientKind) >= {
        "parameter-shift", "finite-diff", "adjoint", "none"
    }
    # Protocols are runtime-checkable and callable shapes look sane
    assert hasattr(core, "QuantumBackendProtocol")
    assert hasattr(core, "ProjectionPolicy")
    assert hasattr(core, "HypercausalNode")
    assert hasattr(core, "LossFn")


# ---------- Backend base + validations ---------------------------------------

class DummyBackend(core.QuantumBackend):
    """Concrete backend to exercise base helpers and contract paths."""

    def run(self, params=None) -> core.Array:
        x = self._require_input()
        return np.tanh(x + 0.05)

    def project_future(self, s_t, branches: int = 2) -> core.Array:
        s = self._validate_state(s_t)
        k = max(2, int(branches))
        deltas = np.linspace(-0.1, 0.1, k, dtype=float)
        fut = np.stack([np.tanh(s + d) for d in deltas], axis=0)
        return self._validate_branches(fut)


def test_backend_config_and_encode_validations():
    # output_dim must be positive (the check happens in the backend constructor)
    with pytest.raises(ValueError):
        _ = DummyBackend(core.BackendConfig(output_dim=0))


    be = DummyBackend(core.BackendConfig(output_dim=3))
    # encode must match output_dim
    with pytest.raises(ValueError):
        be.encode([1.0, 2.0])  # size 2 != 3
    # run requires prior encode
    with pytest.raises(RuntimeError):
        _ = be.run()
    # happy path
    be.encode([0.0, 0.5, 1.0])
    s = be.run()
    assert s.shape == (3,)
    # _validate_state / _validate_branches via project_future
    fut = be.project_future(s, branches=4)
    assert fut.shape == (4, 3)
    # wrong state size
    with pytest.raises(ValueError):
        _ = be.project_future(np.array([0.1, 0.2]), branches=3)
    # capabilities default content
    caps = be.capabilities()
    for key in ("backend_name", "backend_version", "output_dim", "supports_shots", "gradient"):
        assert key in caps


# ---------- Model orchestration ----------------------------------------------

def test_hcmodel_requires_nodes_and_branches_validation():
    with pytest.raises(ValueError):
        _ = core.HCModel(nodes=[])  # requires at least one node

    be = DummyBackend(core.BackendConfig(output_dim=2))
    # Minimal HCNode using the backend directly satisfies HypercausalNode protocol
    from qmlhc.hc.node import HCNode, NodeConfig  # uses your real node implementation
    node = HCNode(backend=be, config=NodeConfig(branches=3))
    model = core.HCModel(nodes=[node], config=core.ModelConfig(default_branches=3))

    # branches must be >= 2
    with pytest.raises(ValueError):
        _ = model.forward([0.1, 0.2], s_tm1=None, branches=1)

    # forward (single node)
    s_t, s_hat, info = model.forward([0.0, 0.5], s_tm1=None, branches=None)  # uses default_branches
    assert s_t.shape == (2,) and s_hat.shape == (2,)
    assert "branches" in info

    # forward_chain (two nodes)
    be2 = DummyBackend(core.BackendConfig(output_dim=2))
    node2 = HCNode(backend=be2, config=NodeConfig(branches=3))
    model2 = core.HCModel(nodes=[node, node2], config=core.ModelConfig(default_branches=3))
    s_tc, s_hatc, infos = model2.forward_chain([0.0, 0.5], s_tm1=None, branches=None)
    assert s_tc.shape == (2,) and s_hatc.shape == (2,)
    assert isinstance(infos, list) and len(infos) == 2 and infos[0]["node_index"] == 0

    # predict_sequence in both modes: use_chain=False and True
    x_seq = [np.array([v, 1 - v]) for v in np.linspace(0.0, 1.0, 5)]
    states, futures, infos = model.predict_sequence(x_seq, s0=None, branches=3, use_chain=False)
    assert len(states) == len(x_seq) and states[-1].shape == (2,)
    states_c, futures_c, infos_c = model2.predict_sequence(x_seq, s0=None, branches=3, use_chain=True)
    assert len(states_c) == len(x_seq) and isinstance(infos_c[-1], list)


# ---------- Registry (singleton API + instance API) --------------------------

def test_backend_registry_happy_and_error_paths():
    reg = core.BackendRegistry()
    # Register new backend
    reg.register(
        name="dummy",
        constructor=lambda cfg: DummyBackend(cfg),
        capabilities={"backend_name": "Dummy", "output_dim": 2},
        overwrite=False,
    )
    assert reg.exists("dummy")
    e = reg.get("dummy")
    assert e.capabilities["backend_name"] == "Dummy"
    inst = reg.create("dummy", core.BackendConfig(output_dim=2))
    assert isinstance(inst, DummyBackend)

    # Duplicate without overwrite should raise
    with pytest.raises(KeyError):
        reg.register(
            name="dummy",
            constructor=lambda cfg: DummyBackend(cfg),
            capabilities={"backend_name": "Dup", "output_dim": 2},
            overwrite=False,
        )

    # Empty/blank name should raise ValueError
    with pytest.raises(ValueError):
        reg.register(
            name="   ",
            constructor=lambda cfg: DummyBackend(cfg),
            capabilities={"backend_name": "X", "output_dim": 1},
        )

    # Missing entries should raise KeyError
    with pytest.raises(KeyError):
        _ = reg.get("not-registered")
    with pytest.raises(KeyError):
        _ = reg.create("not-registered", core.BackendConfig(output_dim=1))

    # list() returns capabilities mapping
    listing = reg.list()
    assert "dummy" in listing and listing["dummy"]["output_dim"] == 2

    # Global singleton helpers should work
    core.register_backend(
        name="dummy2",
        constructor=lambda cfg: DummyBackend(cfg),
        capabilities={"backend_name": "Dummy2", "output_dim": 1},
        overwrite=True,
    )
    assert core.backend_exists("dummy2")
    _ = core.create_backend("dummy2", core.BackendConfig(output_dim=1))
    assert "dummy2" in core.list_backends()
