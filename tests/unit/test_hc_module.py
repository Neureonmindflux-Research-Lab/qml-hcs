
# Comprehensive tests for src/qmlhc/hc: __init__.py, policy.py, node.py, graph.py


from __future__ import annotations

import numpy as np
import pytest

import qmlhc.hc as hc  # public re-exports


# ---------- Public API re-exports --------------------------------------------

def test_hc_public_reexports_exist():
    assert hasattr(hc, "HCNode")
    assert hasattr(hc, "NodeConfig")
    assert hasattr(hc, "HCGraph")
    assert hasattr(hc, "Edge")
    assert hasattr(hc, "MeanPolicy")
    assert hasattr(hc, "MedianPolicy")
    assert hasattr(hc, "MinRiskPolicy")


# ---------- Policies ----------------------------------------------------------

def test_policies_mean_median_shapes_and_values():
    fut = np.array([[0.0, 2.0], [2.0, 4.0], [4.0, 6.0]], dtype=float)
    rep_mean, d_mean = hc.MeanPolicy().select(fut)
    rep_median, d_median = hc.MedianPolicy().select(fut)
    assert np.allclose(rep_mean, np.array([2.0, 4.0]))
    assert np.allclose(rep_median, np.array([2.0, 4.0]))
    assert d_mean["branches"] == 3 and d_median["branches"] == 3


def test_policy_minrisk_selects_expected_branch():
    fut = np.array([[1.0, 5.0], [1.0, 0.1], [1.0, 3.0]], dtype=float)
    def risk(v):  # prefer the smallest 2nd coordinate
        return float(v[1])
    rep, diag = hc.MinRiskPolicy(risk).select(fut)
    assert np.allclose(rep, fut[1])
    assert diag["chosen_index"] == 1 and diag["branches"] == 3


def test_policies_raise_on_invalid_futures_rank():
    with pytest.raises(ValueError):
        _ = hc.MeanPolicy().select(np.array([1.0, 2.0], dtype=float))
    with pytest.raises(ValueError):
        _ = hc.MedianPolicy().select(np.array([1.0, 2.0], dtype=float))
    with pytest.raises(ValueError):
        _ = hc.MinRiskPolicy(lambda v: 0.0).select(np.array([1.0, 2.0], dtype=float))


# ---------- Dummy backend for node/graph tests --------------------------------

from qmlhc.core.backend import BackendConfig, QuantumBackend
from qmlhc.core.types import Array


class DummyBackend(QuantumBackend):
    """Deterministic backend to exercise HCNode and HCGraph."""
    def run(self, params=None) -> Array:
        x = self._require_input()
        return np.tanh(x + 0.05)
    def project_future(self, s_t, branches: int = 2) -> Array:
        s = self._validate_state(s_t)
        k = max(2, int(branches))
        deltas = np.linspace(-0.1, 0.1, k, dtype=float)
        fut = np.stack([np.tanh(s + d) for d in deltas], axis=0)
        return self._validate_branches(fut)


# ---------- Node --------------------------------------------------------------

def test_hcnode_forward_default_mean_policy_and_cfg_branches():
    be = DummyBackend(BackendConfig(output_dim=3))
    node = hc.HCNode(backend=be, policy=None, config=hc.NodeConfig(branches=5))

    x_t = np.array([0.0, 0.5, 1.0], dtype=float)
    s_t, s_hat, info = node.forward(x_t, s_tm1=None, branches=None)  # uses cfg.branches>=2
    assert s_t.shape == (3,) and s_hat.shape == (3,)
    assert "branches" in info and info["branches"].shape == (5, 3)
    assert info["policy"] == "Mean"  # default path


def test_hcnode_with_explicit_policy_used_over_default():
    be = DummyBackend(BackendConfig(output_dim=2))
    policy = hc.MedianPolicy()
    node = hc.HCNode(backend=be, policy=policy, config=hc.NodeConfig(branches=3))

    x_t = np.array([0.2, -0.1], dtype=float)
    s_t, s_hat, info = node.forward(x_t, s_tm1=np.array([0.0, 0.0]), branches=4)
    assert s_t.shape == (2,) and s_hat.shape == (2,)
    assert info["branches"].shape == (4, 2)
    assert info["policy"] == "MedianPolicy"
    assert "diagnostics" in info and info["diagnostics"]["branches"] == 4


# ---------- Graph (DAG) ------------------------------------------------------

def test_hcgraph_chain_and_step_with_parent_aggregation():
    be1 = DummyBackend(BackendConfig(output_dim=2))
    be2 = DummyBackend(BackendConfig(output_dim=2))
    n1 = hc.HCNode(backend=be1, config=hc.NodeConfig(branches=3))
    n2 = hc.HCNode(backend=be2, config=hc.NodeConfig(branches=3))
    g = hc.HCGraph.chain(names=["a", "b"], nodes=[n1, n2])

    # Provide x only for root 'a'; child 'b' should receive mean of parents' S_t (here, parent is only 'a').
    x_map = {"a": np.array([0.1, 0.2], dtype=float)}
    s_map, s_hat_map, info_map = g.step(x_map=x_map, s_tm1_map=None, branches=3)

    assert s_map["a"].shape == (2,) and s_map["b"].shape == (2,)
    assert s_hat_map["a"].shape == (2,) and s_hat_map["b"].shape == (2,)
    assert "branches" in info_map["a"] and "branches" in info_map["b"]


def test_hcgraph_constructor_and_topology_errors():
    be = DummyBackend(BackendConfig(output_dim=1))
    n = hc.HCNode(backend=be)

    # Graph requires at least one node
    with pytest.raises(ValueError):
        _ = hc.HCGraph(nodes={}, edges=[])

    # chain length mismatch
    with pytest.raises(ValueError):
        _ = hc.HCGraph.chain(names=["a", "b"], nodes=[n])

    # Unknown node in edge
    with pytest.raises(KeyError):
        _ = hc.HCGraph(nodes={"a": n}, edges=[hc.Edge("a", "b")])

    # Self-loop not allowed
    with pytest.raises(ValueError):
        _ = hc.HCGraph(nodes={"a": n}, edges=[hc.Edge("a", "a")])

    # Cycle detection (a->b, b->a)
    n2 = hc.HCNode(backend=be)
    with pytest.raises(ValueError):
        _ = hc.HCGraph(nodes={"a": n, "b": n2}, edges=[hc.Edge("a", "b"), hc.Edge("b", "a")])


def test_hcgraph_missing_root_input_raises():
    be = DummyBackend(BackendConfig(output_dim=1))
    n = hc.HCNode(backend=be)
    g = hc.HCGraph(nodes={"a": n}, edges=[])
    # No x for root 'a' -> KeyError
    with pytest.raises(KeyError):
        _ = g.step(x_map={}, s_tm1_map=None, branches=2)
