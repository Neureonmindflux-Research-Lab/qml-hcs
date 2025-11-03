#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hypercausal Model Composition
-----------------------------
High-level orchestration of hypercausal nodes in single-node or chained setups.

This module defines the ``HCModel`` class, which manages the execution of one
or multiple hypercausal nodes across steps or temporal sequences. Each node
follows the hypercausal contract: given an input ``x_t`` and optional previous
state ``s_{t-1}``, it returns the current state ``s_t``, a projected future
state ``ŝ_{t+1}``, and auxiliary information.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Mapping, Sequence, Tuple
import numpy as np

from .types import Array, HypercausalNode, TensorLike


@dataclass(frozen=True)
class ModelConfig:
    """
    Static configuration defining model execution semantics.

    Parameters
    ----------
    default_branches : int, optional
        Number of candidate future branches (K) used by default
        when no explicit value is provided, by default 2.
    """

    default_branches: int = 2


class HCModel:
    """
    Composes one or more hypercausal nodes into an executable model.

    Provides both single-step and multi-step execution methods that support:
    - Sequential chaining of multiple nodes (``forward_chain``)
    - Temporal sequence processing (``predict_sequence``)
    """

    def __init__(self, nodes: Sequence[HypercausalNode], config: ModelConfig | None = None):
        """
        Initialize an HCModel with a sequence of hypercausal nodes.

        Parameters
        ----------
        nodes : Sequence[HypercausalNode]
            Ordered list of hypercausal nodes to be executed.
        config : ModelConfig or None, optional
            Model configuration. If ``None``, uses default settings.

        Raises
        ------
        ValueError
            If no nodes are provided.
        """
        if not nodes:
            raise ValueError("HCModel requires at least one node.")
        self._nodes: List[HypercausalNode] = list(nodes)
        self._cfg = config or ModelConfig()

    # ------------------------------------------------------------------
    # Single-step API
    # ------------------------------------------------------------------
    def forward(
        self,
        x_t: TensorLike,
        s_tm1: TensorLike | None = None,
        branches: int | None = None,
    ) -> Tuple[Array, Array, Mapping[str, Any]]:
        """
        Execute only the first node.

        Parameters
        ----------
        x_t : TensorLike
            Current input vector at time t.
        s_tm1 : TensorLike or None, optional
            Previous state (t−1), by default ``None``.
        branches : int or None, optional
            Number of future branches (K). Uses ``default_branches`` if ``None``.

        Returns
        -------
        tuple
            (``s_t``, ``ŝ_{t+1}``, ``info``)
            ``s_t`` : Array
            Current state.
            ``ŝ_{t+1}`` : Array
            Projected next-state prediction.
            ``info`` : dict
            Additional node diagnostics.
        """
        k = self._resolve_branches(branches)
        s_t, s_tp1_hat, info = self._nodes[0].forward(x_t, s_tm1=s_tm1, branches=k)
        return s_t, s_tp1_hat, info

    # ------------------------------------------------------------------
    # Multi-node chain API
    # ------------------------------------------------------------------
    def forward_chain(
        self,
        x_t: TensorLike,
        s_tm1: TensorLike | None = None,
        branches: int | None = None,
    ) -> Tuple[Array, Array, List[Mapping[str, Any]]]:
        """
        Execute all nodes sequentially in a forward chain.

        Each node receives the output ``s_t`` of the previous node as its
        next input ``x_t``. The previous state reference (``s_tm1``) is
        passed to the first node only; subsequent nodes use the previous
        node's state for consistency.

        Parameters
        ----------
        x_t : TensorLike
            Current input vector at time t.
        s_tm1 : TensorLike or None, optional
            Previous state (t-1), by default ``None``.
        branches : int or None, optional
            Number of future branches (K). Uses ``default_branches`` if ``None``.

        Returns
        -------
        tuple
            (``s_t``, ``ŝ_{t+1}``, ``infos``)
            ``s_t`` : Array
            Final state after the last node.
            ``ŝ_{t+1}`` : Array
            Projected next-state prediction from the last node.
            ``infos`` : list[dict]
            Per-node diagnostic information.
        """
        k = self._resolve_branches(branches)
        infos: List[Mapping[str, Any]] = []

        current_x = np.asarray(x_t, dtype=float).reshape(-1)
        prev_state = None if s_tm1 is None else np.asarray(s_tm1, dtype=float).reshape(-1)

        for idx, node in enumerate(self._nodes):
            s_t, s_tp1_hat, info = node.forward(current_x, s_tm1=prev_state, branches=k)
            infos.append({"node_index": idx, **info})
            current_x = s_t  # feed to next node
            prev_state = s_t

        return current_x, s_tp1_hat, infos

    # ------------------------------------------------------------------
    # Sequence-level API
    # ------------------------------------------------------------------
    def predict_sequence(
        self,
        x_seq: Sequence[TensorLike],
        s0: TensorLike | None = None,
        branches: int | None = None,
        use_chain: bool = False,
    ) -> Tuple[List[Array], List[Array], List[Any]]:
        """
        Process an entire temporal sequence of inputs.

        If ``use_chain`` is False, applies only the first node across all steps.
        If True, applies the full multi-node chain at each time step.

        Parameters
        ----------
        x_seq : Sequence[TensorLike]
            Input sequence (T × D).
        s0 : TensorLike or None, optional
            Initial state, by default ``None``.
        branches : int or None, optional
            Number of future branches (K). Uses ``default_branches`` if ``None``.
        use_chain : bool, optional
            Whether to execute all nodes sequentially per time step, by default ``False``.

        Returns
        -------
        tuple
            (``states``, ``futures``, ``infos``)
            ``states`` : list[Array]
            Sequence of current states.
            ``futures`` : list[Array]
            Sequence of projected next states.
            ``infos`` : list[Any]
            Per-step diagnostic information.
        """
        k = self._resolve_branches(branches)
        states: List[Array] = []
        futures: List[Array] = []
        infos: List[Any] = []

        s_tm1 = None if s0 is None else np.asarray(s0, dtype=float).reshape(-1)

        for x_t in x_seq:
            if use_chain:
                s_t, s_tp1_hat, info = self.forward_chain(x_t, s_tm1=s_tm1, branches=k)
            else:
                s_t, s_tp1_hat, info = self.forward(x_t, s_tm1=s_tm1, branches=k)

            states.append(s_t)
            futures.append(s_tp1_hat)
            infos.append(info)
            s_tm1 = s_t

        return states, futures, infos

    # ------------------------------------------------------------------
    # Internal utilities
    # ------------------------------------------------------------------
    def _resolve_branches(self, branches: int | None) -> int:
        """
        Determine the number of branches (K) for projection.

        Parameters
        ----------
        branches : int or None
            Optional override value.
            
        Returns
        -------
        int
            Final branch count (≥ 2).

        Raises
        ------
        ValueError
            If branches < 2.
        """
        k = self._cfg.default_branches if branches is None else int(branches)
        if k < 2:
            raise ValueError("branches must be >= 2")
        return k
