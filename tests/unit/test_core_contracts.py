from __future__ import annotations

import numpy as np
import pytest
import qmlhc.core as core


# =============================================================================
# Types / Protocols
# =============================================================================
def test_types_enums_and_protocols_runtime_checkable():
    assert set(k.value for k in core.GradientKind) >= {
        "parameter-shift", "finite-diff", "adjoint", "none"
    }
    assert hasattr(core, "QuantumBackendProtocol")
    assert hasattr(core, "ProjectionPolicy")
    assert hasattr(core, "HypercausalNode")
    assert hasattr(core, "LossFn")


# =============================================================================
# Backend base + validations
# =============================================================================
class DummyBackend(core.QuantumBackend):
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
    with pytest.raises(ValueError):
        _ = DummyBackend(core.BackendConfig(output_dim=0))

    be = DummyBackend(core.BackendConfig(output_dim=3))
    with pytest.raises(ValueError):
        be.encode([1.0, 2.0])
    with pytest.raises(RuntimeError):
        _ = be.run()
    be.encode([0.0, 0.5, 1.0])
    s = be.run()
    assert s.shape == (3,)
    fut = be.project_future(s, branches=4)
    assert fut.shape == (4, 3)
    with pytest.raises(ValueError):
        _ = be.project_future(np.array([0.1, 0.2]), branches=3)
    caps = be.capabilities()
    for key in ("backend_name", "backend_version", "output_dim", "supports_shots", "gradient"):
        assert key in caps


# =============================================================================
# Model orchestration
# =============================================================================
def test_hcmodel_requires_nodes_and_branches_validation():
    with pytest.raises(ValueError):
        _ = core.HCModel(nodes=[])

    be = DummyBackend(core.BackendConfig(output_dim=2))
    from qmlhc.hc.node import HCNode, NodeConfig
    node = HCNode(backend=be, config=NodeConfig(branches=3))
    model = core.HCModel(nodes=[node], config=core.ModelConfig(default_branches=3))

    with pytest.raises(ValueError):
        _ = model.forward([0.1, 0.2], s_tm1=None, branches=1)

    s_t, s_hat, info = model.forward([0.0, 0.5], s_tm1=None, branches=None)
    assert s_t.shape == (2,) and s_hat.shape == (2,)
    assert "branches" in info

    be2 = DummyBackend(core.BackendConfig(output_dim=2))
    node2 = HCNode(backend=be2, config=NodeConfig(branches=3))
    model2 = core.HCModel(nodes=[node, node2], config=core.ModelConfig(default_branches=3))
    s_tc, s_hatc, infos = model2.forward_chain([0.0, 0.5], s_tm1=None, branches=None)
    assert s_tc.shape == (2,) and s_hatc.shape == (2,)
    assert isinstance(infos, list) and len(infos) == 2 and infos[0]["node_index"] == 0

    x_seq = [np.array([v, 1 - v]) for v in np.linspace(0.0, 1.0, 5)]
    states, futures, infos = model.predict_sequence(x_seq, s0=None, branches=3, use_chain=False)
    assert len(states) == len(x_seq) and states[-1].shape == (2,)
    states_c, futures_c, infos_c = model2.predict_sequence(x_seq, s0=None, branches=3, use_chain=True)
    assert len(states_c) == len(x_seq) and isinstance(infos_c[-1], list)


# =============================================================================
# Registry (singleton API + instance API)
# =============================================================================
def test_backend_registry_happy_and_error_paths():
    reg = core.BackendRegistry()
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

    with pytest.raises(KeyError):
        reg.register(
            name="dummy",
            constructor=lambda cfg: DummyBackend(cfg),
            capabilities={"backend_name": "Dup", "output_dim": 2},
            overwrite=False,
        )

    with pytest.raises(ValueError):
        reg.register(
            name="   ",
            constructor=lambda cfg: DummyBackend(cfg),
            capabilities={"backend_name": "X", "output_dim": 1},
        )

    with pytest.raises(KeyError):
        _ = reg.get("not-registered")
    with pytest.raises(KeyError):
        _ = reg.create("not-registered", core.BackendConfig(output_dim=1))

    listing = reg.list()
    assert "dummy" in listing and listing["dummy"]["output_dim"] == 2

    core.register_backend(
        name="dummy2",
        constructor=lambda cfg: DummyBackend(cfg),
        capabilities={"backend_name": "Dummy2", "output_dim": 1},
        overwrite=True,
    )
    assert core.backend_exists("dummy2")
    _ = core.create_backend("dummy2", core.BackendConfig(output_dim=1))
    assert "dummy2" in core.list_backends()

def test_backend_base_methods_raise_notimplemented():
    class Bare(core.QuantumBackend):
        pass

    be = Bare(core.BackendConfig(output_dim=1))
    with pytest.raises(NotImplementedError):
        _ = be.run()
    with pytest.raises(NotImplementedError):
        _ = be.project_future(np.array([0.0]), branches=2)

def test_validate_branches_raises_on_not_2d():
    be = DummyBackend(core.BackendConfig(output_dim=2))
    arr_1d = np.array([0.1, 0.2])
    with pytest.raises(ValueError):
        be._validate_branches(arr_1d)

def test_validate_branches_raises_on_wrong_second_dim():
    be = DummyBackend(core.BackendConfig(output_dim=2))
    arr_bad_dim = np.ones((3, 3))
    with pytest.raises(ValueError):
        be._validate_branches(arr_bad_dim)
