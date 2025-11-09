# -*- coding: utf-8 -*-
"""
Dual-Ascent Lagrangian Wrapper
------------------------------
Treats Consistency and Coherence as inequality constraints and performs
dual ascent on their Lagrange multipliers. Useful when you want guarantees
on constraint violations instead of fixed weights.

Primal update comes from a base optimizer (e.g., FD/SPSA/Adam).

Interface:
    - initialize(params) -> state
    - step_params(model, params, context) -> (new_params, state)

Context:
    - context["evaluate"](model, params, context) -> dict with keys:
        {"task": float, "cons": float, "coh": float, "total": float, "info": dict}
    - bounds: pass at construction time (cons_bound, coh_bound)
"""

from __future__ import annotations
from typing import Any, Dict, Mapping, Tuple
import numpy as np


class HCDualAscent:
    """Dual-ascent wrapper for (cons, coh) inequality constraints."""

    def __init__(
        self,
        base_opt: Any,
        dual_lr: float = 1e-2,
        cons_bound: float = 0.5,
        coh_bound: float = 1e-2,
        clip_lambda: float | None = 10.0,
    ):
        self.base = base_opt
        self.dual_lr = float(dual_lr)
        self.cons_bound = float(cons_bound)
        self.coh_bound = float(coh_bound)
        self.clip_lambda = clip_lambda
        self.lmb_cons: float = 0.0
        self.lmb_coh: float = 0.0
        self._state: Dict[str, Any] = {}

    def initialize(self, params: Mapping[str, Any]) -> Dict[str, Any]:
        self.lmb_cons = 0.0
        self.lmb_coh = 0.0
        self._state = {"steps": 0, "lambda_cons": 0.0, "lambda_coh": 0.0}
        if hasattr(self.base, "initialize"):
            self.base.initialize(params)
        return dict(self._state)

    def step_params(
        self, model: Any, params: Mapping[str, Any], context: Mapping[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        # Evaluate current state
        evaluate = context["evaluate"]
        stats = evaluate(model, params, context)
        cons_violation = float(stats["cons"] - self.cons_bound)
        coh_violation = float(stats["coh"] - self.coh_bound)

        # Update dual variables (projected gradient ascent on multipliers)
        self.lmb_cons = max(0.0, self.lmb_cons + self.dual_lr * cons_violation)
        self.lmb_coh = max(0.0, self.lmb_coh + self.dual_lr * coh_violation)
        if self.clip_lambda is not None:
            self.lmb_cons = float(np.clip(self.lmb_cons, 0.0, self.clip_lambda))
            self.lmb_coh = float(np.clip(self.lmb_coh, 0.0, self.clip_lambda))

        # Modify context to reflect new Lagrangian weights before primal step
        # Caller should make total loss incorporate: L = task + lmb_cons*cons + lmb_coh*coh
        context_mod = dict(context)
        context_mod["dual"] = {"lambda_cons": self.lmb_cons, "lambda_coh": self.lmb_coh}

        new_params, base_state = self.base.step_params(model, params, context_mod)
        self._state = {
            "steps": self._state.get("steps", 0) + 1,
            "lambda_cons": self.lmb_cons,
            "lambda_coh": self.lmb_coh,
            "cons_violation": cons_violation,
            "coh_violation": coh_violation,
        }
        return new_params, dict(self._state)
