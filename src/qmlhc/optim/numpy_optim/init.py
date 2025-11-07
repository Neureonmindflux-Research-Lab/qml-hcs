# -*- coding: utf-8 -*-
"""
NumPy-based optimizers for Hypercausal learning.

This package contains pure-NumPy implementations of advanced optimizers.
All optimizers follow a common interface:

    - initialize(params) -> state (optional)
    - step_params(model, params, context) -> (new_params, state)

They are designed to be used via the registry factory:
    from qmlhc.optim.registry_numpy import create_optimizer_numpy
"""
