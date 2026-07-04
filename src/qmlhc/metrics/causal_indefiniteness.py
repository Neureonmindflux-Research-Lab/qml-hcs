# -*- coding: utf-8 -*-
r"""

This module implements a computational metric for quantifying *causal
indefiniteness* based on the trace distance to classically ordered
(causally separable) reference processes, as introduced in the paper
:ref:`Pre-Temporal Model of Quantum Causal Order <theory>`

The implementation is intentionally minimal and framework-agnostic.
All objects are represented as square complex matrices, allowing the
metric to be applied not only to process matrices in the strict
process-matrix formalism, but also to abstract operators or effective
representations used in numerical simulations, diagnostics, or learning
frameworks.

Conceptual definition
---------------------
Causal indefiniteness is quantified by measuring how far a given operator
``W`` lies from the convex set of processes that admit a definite causal
order. This distance defines a continuous scale:

- ``λ(W) = 0`` indicates a causally separable (classically ordered) process.
- ``λ(W) > 0`` indicates the presence of genuinely indefinite causal structure.
- Intermediate values correspond to partial or weakening causal indefiniteness.

The metric is designed to behave continuously and monotonically under
physical decoherence and coarse-graining operations.

Computable form implemented here
--------------------------------

Rather than minimizing over the full convex set of causally separable
processes, this module implements the *computable instantiation* used in
the paper for the bipartite, two-order scenario.

Two definite-order reference matrices are assumed:

- ``W_AB`` representing the order :math:`A \prec B`
- ``W_BA`` representing the order :math:`B \prec A`

From these, a one-parameter family of classically ordered reference
processes is constructed:

.. math::

   V(q) = \frac{1}{2} \left[ q \, W_{AB} + (1 - q) \, W_{BA} \right],
   \qquad q \in [0, 1]

The causal-indefiniteness measure is then computed as the minimum trace
distance between ``W`` and this reference family:

.. math::

   \lambda(W) = \min_{q} \, D\!\left(W, V(q)\right)

where the trace distance is defined as

.. math::

   D(A, B) = \frac{1}{2} \left\| A - B \right\|_{1}

This formulation captures the transition between indefinite and definite
causal structure in a numerically stable and interpretable way.

Numerical and structural assumptions
------------------------------------
- All input matrices are assumed to be square and of equal shape.
- Inputs are expected to be Hermitian or approximately Hermitian.
  Optional symmetrization is provided to suppress numerical artifacts.
- No specific tensor-product structure is assumed. Any required embedding
  (e.g., control systems or ancillary spaces) must be handled by the caller.
- The minimization over ``q`` is performed using a deterministic grid search,
  favoring robustness and reproducibility over asymptotic optimality.

Intended use
------------
This metric is designed as a *computational primitive* rather than a full
causal-separability solver. Typical use cases include:

- Monitoring the decay of causal indefiniteness under noise or decoherence
- Characterizing intermediate (pre-temporal) causal regimes
- Providing a scalar diagnostic for simulations, optimization loops,
  or hybrid quantum-classical workflows
- Serving as a plug-in metric within larger numerical or learning frameworks

Public interface
----------------
- ``trace_distance``: Trace distance between two (approximately) Hermitian matrices
- ``reference_process``: Construction of the convex reference process :math:`V(q)`
- ``lambda_w_trace``: Computation of the causal-indefiniteness measure :math:`\lambda(W)`
"""



from __future__ import annotations

from typing import Tuple

import numpy as np

from ..core.types import Array


def _validate_square_same_shape(*mats: Array) -> Tuple[int, int]:
    """Validate that all matrices are square and share the same shape."""
    if not mats:
        raise ValueError("At least one matrix is required.")

    shape = mats[0].shape
    if len(shape) != 2 or shape[0] != shape[1]:
        raise ValueError(f"Expected a square matrix, got shape={shape}.")
    for M in mats[1:]:
        if M.shape != shape:
            raise ValueError(f"All matrices must share the same shape. Got {shape} vs {M.shape}.")
    return int(shape[0]), int(shape[1])


def trace_norm_hermitian(X: Array, *, symmetrize: bool = True) -> float:
    r"""
    Compute the trace norm :math:`\|X\|_{1}` for an (approximately) Hermitian matrix.

    For Hermitian :math:`X`, the trace norm equals the sum of the absolute values
    of its eigenvalues.

    Parameters
    ----------
    X:
        Square matrix.
    symmetrize:
        If ``True``, replace :math:`X` with
        :math:`(X + X^{\dagger}) / 2` before eigendecomposition to remove
        tiny numerical non-Hermiticity.

    Returns
    -------
    float
        The trace norm :math:`\|X\|_{1}`.
    """
    _validate_square_same_shape(X)
    H = X
    if symmetrize:
        H = 0.5 * (H + H.conj().T)
    eigs = np.linalg.eigvalsh(H)
    return float(np.sum(np.abs(eigs)))


def trace_distance(A: Array, B: Array, *, symmetrize: bool = True) -> float:
    """
    Trace distance D(A,B) = 1/2 ||A - B||_1.

    This function is intended for Hermitian (or nearly Hermitian) inputs,
    which is the standard case for process/Choi operators.

    Parameters
    ----------
    A, B:
        Square matrices with the same shape.
    symmetrize:
        If True, symmetrize (A-B) prior to computing eigenvalues.

    Returns
    -------
    float
        Trace distance D(A,B).
    """
    _validate_square_same_shape(A, B)
    return 0.5 * trace_norm_hermitian(A - B, symmetrize=symmetrize)


def reference_process(
    q: float,
    W_AB: Array,
    W_BA: Array,
    *,
    half_factor: bool = True,
) -> Array:
    """
    Build the paper's convex reference process V(q).

    V(q) = 1/2 [ q * W_AB + (1-q) * W_BA ].

    Parameters
    ----------
    q:
        Mixing parameter in [0,1].
    W_AB:
        Definite-order reference branch for A≺B, embedded in the same space as W.
    W_BA:
        Definite-order reference branch for B≺A, embedded in the same space as W.
    half_factor:
        Whether to include the explicit 1/2 factor used in the manuscript.

    Returns
    -------
    np.ndarray
        Reference matrix V(q).
    """
    if not (0.0 <= float(q) <= 1.0):
        raise ValueError(f"q must be in [0,1], got q={q}.")
    _validate_square_same_shape(W_AB, W_BA)
    V = float(q) * W_AB + (1.0 - float(q)) * W_BA
    if half_factor:
        V = 0.5 * V
    return V


def lambda_w_trace(
    W: Array,
    W_AB: Array,
    W_BA: Array,
    *,
    q_grid: int = 81,
    symmetrize: bool = True,
    half_factor: bool = True,
) -> float:
    r"""
    Compute the causal-indefiniteness measure :math:`\lambda(W)` via trace distance.

    This matches the manuscript's computable instantiation:

    - :math:`\lambda(W) = \min_{q \in [0,1]} D\!\left(W, V(q)\right)`.
    - :math:`V(q) = \frac{1}{2}\left[q\,W_{AB} + (1-q)\,W_{BA}\right]` (when ``half_factor=True``).
    - :math:`D(A,B) = \frac{1}{2}\left\|A - B\right\|_{1}`.

    The minimization is performed by a deterministic grid search over
    :math:`q \in [0, 1]`. For most practical uses (monitoring, diagnostics,
    regularization), this provides robust and deterministic behavior.

    Parameters
    ----------
    W:
        Target process/operator matrix.
    W_AB:
        Definite-order reference branch (:math:`A \prec B`), embedded in ``W``'s matrix space.
    W_BA:
        Definite-order reference branch (:math:`B \prec A`), embedded in ``W``'s matrix space.
    q_grid:
        Number of grid points for :math:`q \in [0,1]`. Typical resolutions are in the 50–80 range;
        the default ``81`` provides a convenient inclusive grid.
    symmetrize:
        If ``True``, symmetrize :math:`(W - V(q))` before eigendecomposition.
    half_factor:
        If ``True``, use the manuscript convention
        :math:`V(q) = \frac{1}{2}\left[q\,W_{AB} + (1-q)\,W_{BA}\right]`.

    Returns
    -------
    float
        The minimized trace distance :math:`\lambda(W)`.
    """

    _validate_square_same_shape(W, W_AB, W_BA)
    if q_grid < 2:
        raise ValueError(f"q_grid must be >= 2, got q_grid={q_grid}.")

    qs = np.linspace(0.0, 1.0, int(q_grid))
    best = float("inf")

    for q in qs:
        Vq = reference_process(q, W_AB, W_BA, half_factor=half_factor)
        d = trace_distance(W, Vq, symmetrize=symmetrize)
        if d < best:
            best = d

    return float(best)
