# -*- coding: utf-8 -*-
"""
NumPy Optimizer Registry
------------------------
Factory for NumPy-based optimizers, wired to the project's Optimizer API.

Usage:
    from qmlhc.optim.registry_numpy import create_optimizer_numpy
    opt = create_optimizer_numpy("spsa", lr0=0.05, eps0=0.1)

Returned objects implement:
    - initialize(params) -> state
    - step_params(model, params, context) -> (new_params, state)
"""

from __future__ import annotations
from typing import Any, Callable, Dict
from .numpy_optim.finite_diff import HCFiniteDiffOptimizer
from .numpy_optim.spsa import HCSPSAOptimizer
from .numpy_optim.adam import HCAdam
from .numpy_optim.natural_grad import HCNaturalGrad
from .numpy_optim.trust_region import HCTrustRegion
from .numpy_optim.dual_ascent import HCDualAscent
from .numpy_optim.mpc import HCMPCShortHorizon
from .numpy_optim.kfac import HCKFACLike


_CREATORS: Dict[str, Callable[..., Any]] = {
    "finite-diff":  lambda **kw: HCFiniteDiffOptimizer(**kw),
    "spsa":         lambda **kw: HCSPSAOptimizer(**kw),
    "adam":         lambda **kw: HCAdam(**kw),
    "natural-grad": lambda **kw: HCNaturalGrad(**kw),
    "trust-kl":     lambda **kw: HCTrustRegion(**kw),
    "dual-ascent":  lambda **kw: HCDualAscent(**kw),
    "mpc":          lambda **kw: HCMPCShortHorizon(**kw),
    "kfac":         lambda **kw: HCKFACLike(**kw),
}


def create_optimizer_numpy(name: str, **kwargs) -> Any:
    """
    Create a NumPy-based optimizer by name.

    Parameters
    ----------
    name : str
        One of {"finite-diff","spsa","adam","natural-grad","trust-kl",
                "dual-ascent","mpc","kfac"}.
    kwargs : dict
        Optimizer hyperparameters. For wrappers ("trust-kl","dual-ascent"),
        pass 'base_opt' (the underlying optimizer instance).

    Returns
    -------
    object
        Optimizer instance exposing initialize(...) and step_params(...).
    """
    key = name.strip().lower()
    try:
        return _CREATORS[key](**kwargs)
    except KeyError as e:
        raise KeyError(f"Unknown optimizer '{name}'. Available: {list(_CREATORS)}") from e
