# -*- coding: utf-8 -*-
"""
Trust-Region (KL over States)
-----------------------------
Wrapper that enforces a trust-region constraint measured as a (symmetric) KL
proxy over state branches. If the KL bound is exceeded, performs backtracking
line-search along the update direction until satisfied.

Interface:
    - initialize(params) -> state
    - step_params(model, params, context) -> (new_params, state)

Requires:
    - base optimizer exposing step_params(...)
    - context["kl_fn"](old_info, new_info) -> float (KL or proxy)
    - context["info"]: current state's info dict (with 'branches' if possible)
    - context["refresh_info"](model, params, context) -> info (callable)
"""

from __future__ import annotations
from typing import Any, Callable, Dict, Mapping, Tuple
import numpy as np


class HCTrustRegion:
    """Trust-region wrapper with KL constraint over state-space."""

    def __init__(
        self,
        base_opt: Any,
        delta_kl: float = 2e-2,
        backtrack: float = 0.7,
        max_backtracks: int = 8,
    ):
        self.base = base_opt
        self.delta_kl = float(delta_kl)
        self.bt = float(backtrack)
        self.bt_max = int(max_backtracks)
        self._state: Dict[str, Any] = {}

    def initialize(self, params: Mapping[str, Any]) -> Dict[str, Any]:
        if hasattr(self.base, "initialize"):
            base_state = self.base.initialize(params)
        else:
            base_state = {}
        self._state = {"base_state": base_state, "steps": 0}
        return dict(self._state)

    def step_params(
        self, model: Any, params: Mapping[str, Any], context: Mapping[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        old_info = context.get("info", {})
        kl_fn: Callable[[Mapping[str, Any], Mapping[str, Any]], float] = context["kl_fn"]
        refresh_info: Callable[[Any, Mapping[str, Any], Mapping[str, Any]], Mapping[str, Any]] = context["refresh_info"]

        # propose update
        proposal, base_state = self.base.step_params(model, params, context)

        # line search with KL guard
        alpha = 1.0
        new_params = proposal
        for _ in range(self.bt_max + 1):
            new_info = refresh_info(model, new_params, context)
            kl_val = float(kl_fn(old_info, new_info))
            if kl_val <= self.delta_kl:
                self._state = {"base_state": base_state, "steps": self._state.get("steps", 0) + 1, "kl": kl_val, "alpha_bt": alpha}
                return new_params, dict(self._state)
            # backtrack
            alpha *= self.bt
            blended = {}
            for k in params.keys():
                blended[k] = (1 - alpha) * np.asarray(params[k], dtype=float) + alpha * np.asarray(proposal[k], dtype=float)
            new_params = blended

        # if failed to satisfy KL, return original params (conservative)
        self._state = {"base_state": base_state, "steps": self._state.get("steps", 0) + 1, "kl": float("inf"), "alpha_bt": 0.0}
        return dict(params), dict(self._state)
