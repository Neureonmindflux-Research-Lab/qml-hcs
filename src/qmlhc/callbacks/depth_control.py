#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Depth Scheduling Callback
-------------------------
Adaptive control of circuit or model depth across epochs or iterations.

This callback linearly interpolates an integer-valued depth attribute from a
starting value to an ending value over a given number of epochs. It updates
either a backend or a model object, depending on which one is provided in the
callback context.
"""

from __future__ import annotations

from typing import Any, Mapping

from .base import Callback


class DepthScheduler(Callback):
    """
    Gradually adjust a depth-like attribute during training.

    The scheduler linearly interpolates the target attribute (default: ``"depth"``)
    from ``start`` to ``end`` over ``epochs`` steps. At the beginning of each
    epoch, it writes the new integer depth into the object passed via the
    callback context (``context["model"]`` or ``context["backend"]``).

    Parameters
    ----------
    target_attr : str, optional
        Name of the attribute to update, by default ``"depth"``.
    start : int, optional
        Starting depth value, by default ``1``.
    end : int, optional
        Final depth value reached at or before the last epoch, by default ``6``.
    epochs : int, optional
        Number of epochs over which to interpolate (minimum 1), by default ``10``.
    """

    def __init__(
        self,
        target_attr: str = "depth",
        start: int = 1,
        end: int = 6,
        epochs: int = 10,
    ) -> None:
        self._attr = target_attr
        self._start = int(start)
        self._end = int(end)
        self._epochs = max(1, int(epochs))

    def _interpolate(self, epoch: int) -> int:
        """
        Compute the current depth via linear interpolation.

        Parameters
        ----------
        epoch : int
            Current epoch index (0-based).

        Returns
        -------
        int
            Interpolated integer depth, clamped to the range ``[start, end]``.
        """
        ratio = min(1.0, max(0.0, epoch / float(self._epochs)))
        return int(round(self._start + (self._end - self._start) * ratio))

    def on_epoch_begin(self, epoch: int, context: Mapping[str, Any]) -> None:
        """
        Update the target object's depth attribute at the start of each epoch.

        The method looks for a ``"model"`` or ``"backend"`` entry in ``context``.
        If present and the object exposes the target attribute, it is updated
        with the interpolated integer value.

        Parameters
        ----------
        epoch : int
            Current epoch index (0-based).
        context : Mapping[str, Any]
            Callback context, expected to include ``"model"`` or ``"backend"``.
        """
        obj = context.get("model") or context.get("backend")
        if obj is None:
            return
        new_depth = self._interpolate(epoch)
        if hasattr(obj, self._attr):
            try:
                setattr(obj, self._attr, new_depth)
            except Exception:
                # Silently ignore attribute assignment issues to avoid
                # breaking the training loop.
                pass

    # The remaining callbacks are no-ops by default.
    def on_step_begin(self, step: int, context: Mapping[str, Any]) -> None:
        """No-op: provided for interface completeness."""
        pass

    def on_step_end(self, step: int, context: Mapping[str, Any]) -> None:
        """No-op: provided for interface completeness."""
        pass

    def on_epoch_end(self, epoch: int, context: Mapping[str, Any]) -> None:
        """No-op: provided for interface completeness."""
        pass

    def on_error(self, error: Exception, context: Mapping[str, Any]) -> None:
        """No-op: provided for interface completeness."""
        pass
