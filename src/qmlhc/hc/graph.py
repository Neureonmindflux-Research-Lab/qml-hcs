#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hypercausal Graphs
------------------
Directed acyclic graph (DAG) abstraction for deterministic evaluation of
hypercausal nodes.

This module defines:
- ``Edge``: a directed connection between nodes.
- ``HCGraph``: a lightweight executor for hypercausal DAGs and chains.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Tuple

import numpy as np

from ..core.types import Array, HypercausalNode, TensorLike


@dataclass(frozen=True)
class Edge:
    """
    Directed edge connecting two nodes.

    Parameters
    ----------
    src : str
        Source node name.
    dst : str
        Destination node name.
    """
    src: str
    dst: str


class HCGraph:
    """
    Directed acyclic graph (DAG) of hypercausal nodes with deterministic
    topological evaluation.

    Nodes are executed respecting causal order. Missing explicit inputs
    are automatically derived from the mean of parent states.

    Examples
    --------
    >>> dag = HCGraph.chain(["A", "B", "C"], [nodeA, nodeB, nodeC])
    >>> s_map, s_hat_map, info_map = dag.step({"A": x_t})
    """

    def __init__(self, nodes: Mapping[str, HypercausalNode], edges: Iterable[Edge]) -> None:
        """
        Initialize a hypercausal DAG.

        Parameters
        ----------
        nodes : Mapping[str, HypercausalNode]
            Dictionary mapping node names to ``HypercausalNode`` instances.
        edges : Iterable[Edge]
            Directed edges connecting nodes.

        Raises
        ------
        ValueError
            If no nodes are provided or if a cycle/self-loop is detected.
        KeyError
            If an edge references an unknown node.
        """
        if not nodes:
            raise ValueError("Graph requires at least one node.")
        self._nodes: Dict[str, HypercausalNode] = dict(nodes)
        self._edges: List[Edge] = list(edges)
        self._check_edges()
        self._order: List[str] = self._topological_order()

    # ----------------------------------------------------------------------
    # Step execution
    # ----------------------------------------------------------------------
    def step(
        self,
        x_map: Mapping[str, TensorLike],
        s_tm1_map: Mapping[str, TensorLike] | None = None,
        branches: int = 2,
    ) -> Tuple[Dict[str, Array], Dict[str, Array], Dict[str, Mapping[str, Any]]]:
        """
        Process one time step across the DAG.

        Parameters
        ----------
        x_map : Mapping[str, TensorLike]
            Per-node current inputs.
        s_tm1_map : Mapping[str, TensorLike] or None, optional
            Optional mapping of previous states per node, by default ``None``.
        branches : int, optional
            Number of candidate future branches (K), by default ``2``.

        Returns
        -------
        tuple
            ``(s_map, s_hat_map, info_map)`` where:
            - ``s_map`` : dict[str, Array]
            Per-node current state ``S_t``.
            - ``s_hat_map`` : dict[str, Array]
            Per-node projected future state ``Ŝ_{t+1}``.
            - ``info_map`` : dict[str, Mapping[str, Any]]
            Auxiliary per-node diagnostics.

        Raises
        ------
        KeyError
            If a root node has no explicit input and no parent.
        """
        s_map: Dict[str, Array] = {}
        s_hat_map: Dict[str, Array] = {}
        info_map: Dict[str, Mapping[str, Any]] = {}

        # Execute nodes respecting topological order
        for name in self._order:
            node = self._nodes[name]

            # Input resolution: prefer explicit input, otherwise average of parents
            if name in x_map:
                x_t = np.asarray(x_map[name], dtype=float).reshape(-1)
            else:
                parents = self._parents(name)
                if not parents:
                    raise KeyError(f"Missing input for root node '{name}'.")
                parent_states = [s_map[p] for p in parents]
                x_t = np.mean(parent_states, axis=0)

            s_prev = None if not s_tm1_map or name not in s_tm1_map else np.asarray(
                s_tm1_map[name], dtype=float
            ).reshape(-1)

            s_t, s_tp1_hat, info = node.forward(x_t, s_tm1=s_prev, branches=branches)
            s_map[name] = s_t
            s_hat_map[name] = s_tp1_hat
            info_map[name] = info

        return s_map, s_hat_map, info_map

    # ----------------------------------------------------------------------
    # Chain constructor
    # ----------------------------------------------------------------------
    @classmethod
    def chain(cls, names: List[str], nodes: List[HypercausalNode]) -> HCGraph:
        """
        Build a linear chain of nodes ``n0 -> n1 -> ... -> n_{L-1}``.

        Parameters
        ----------
        names : list[str]
            Node names (must be unique).
        nodes : list[HypercausalNode]
            Node instances corresponding to ``names``.

        Returns
        -------
        HCGraph
            Constructed linear chain graph.

        Raises
        ------
        ValueError
            If ``names`` and ``nodes`` have different lengths.
        """
        if len(names) != len(nodes):
            raise ValueError("Names and nodes must have the same length.")
        node_map = {n: nd for n, nd in zip(names, nodes)}
        edges = [Edge(src=names[i], dst=names[i + 1]) for i in range(len(names) - 1)]
        return cls(node_map, edges)

    # ----------------------------------------------------------------------
    # Topology utilities
    # ----------------------------------------------------------------------
    def _parents(self, node: str) -> List[str]:
        """Return the list of parent node names for a given node."""
        return [e.src for e in self._edges if e.dst == node]

    def _children(self, node: str) -> List[str]:
        """Return the list of child node names for a given node."""
        return [e.dst for e in self._edges if e.src == node]

    def _check_edges(self) -> None:
        """Validate all edges and ensure there are no self-loops or unknown references."""
        for e in self._edges:
            if e.src not in self._nodes or e.dst not in self._nodes:
                raise KeyError(f"Edge '{e.src}->{e.dst}' references an unknown node.")
            if e.src == e.dst:
                raise ValueError("Self-loops are not allowed in DAGs.")

    def _topological_order(self) -> List[str]:
        """
        Compute topological order using Kahn’s algorithm.

        Returns
        -------
        list[str]
            List of node names in execution order.

        Raises
        ------
        ValueError
            If the graph contains a cycle.
        """
        indeg: Dict[str, int] = {n: 0 for n in self._nodes}
        for e in self._edges:
            indeg[e.dst] += 1

        queue: List[str] = [n for n, d in indeg.items() if d == 0]
        order: List[str] = []

        while queue:
            u = queue.pop(0)
            order.append(u)
            for v in self._children(u):
                indeg[v] -= 1
                if indeg[v] == 0:
                    queue.append(v)

        if len(order) != len(self._nodes):
            raise ValueError("Graph contains a cycle.")
        return order
