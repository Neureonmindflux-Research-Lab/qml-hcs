#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Contrafactual Anticipators
--------------------------
Wrappers that synthesize structured counterfactual futures on top of a base
projector and optional perturbations. 
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from ..core.types import Array, TensorLike
from .projector import Projector

# Maps a (D,) vector to a (D,) vector
Perturb = Callable[[Array], Array]


@dataclass(frozen=True)
class AnticipatorConfig:
    """
    Static configuration controlling counterfactual generation semantics.

    Parameters
    ----------
    branches : int, optional
        Number of base branches K produced by the projector, by default ``3``.
        Must satisfy ``K >= 2`` in downstream usage.
    symmetric : bool, optional
        When ``True``, for every perturbation ``v`` around the center, also add
        its mirrored counterpart ``(2 * center - v)``. Default is ``True``.
    """
    branches: int = 3
    symmetric: bool = True


class ContrafactualAnticipator:
    """
    Generate structured counterfactual futures on top of a base projector.

    Given a current state ``s_t``, first obtains a base set of futures from
    ``Projector.project``. Then, for each user-provided perturbation, it adds a
    single variant (and optionally its symmetric mirror) around the base center.

    Notes
    -----
    The final future set is the concatenation of:
    - the projector's base set ``(K, D)``,
    - one row per perturbation,
    - (optionally) one mirrored row per perturbation.
    """

    def __init__(
        self,
        projector: Projector,
        perturbations: Sequence[Perturb] | None = None,
        config: AnticipatorConfig | None = None,
    ) -> None:
        self._proj = projector
        self._perts = list(perturbations or [])
        self._cfg = config or AnticipatorConfig()

    def generate(self, s_t: TensorLike) -> Array:
        """
        Produce a combined future set ``(K', D)`` from base projection and variants.

        Steps
        -----
        1. Call the base projector to obtain ``base_set`` with shape ``(K, D)``.
        2. If perturbations are provided, compute the center as ``mean(base_set, axis=0)``.
        3. For each perturbation ``p``, append ``p(center)`` as a new row.
        4. If ``symmetric`` is enabled, also append the mirrored row
           ``2 * center - p(center)``.

        Parameters
        ----------
        s_t : TensorLike
            Current state vector ``(D,)``.

        Returns
        -------
        Array
            Concatenated futures matrix with shape ``(K', D)``, where
            ``K' >= K`` depends on the number of perturbations and symmetry.
        """
        base_set = self._proj.project(s_t, branches=self._cfg.branches)
        variants = [base_set]

        if self._perts:
            center = base_set.mean(axis=0)
            for p in self._perts:
                v = p(center)
                variants.append(np.expand_dims(v, axis=0))
                if self._cfg.symmetric:
                    variants.append(np.expand_dims(2 * center - v, axis=0))

        fut = np.concatenate(variants, axis=0)
        return fut
