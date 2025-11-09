#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Callback Base Interface
-----------------------
Defines the minimal callback protocol used for training, evaluation, and
telemetry integration. Provides a unified interface for hook registration
and event dispatch during iterative processes (e.g., epochs, steps).

Includes:

- ``Callback``: protocol defining the event hook signatures.
- ``CallbackList``: container that manages multiple callbacks and dispatches
  events sequentially.
"""

from __future__ import annotations

from typing import Any, Mapping, Protocol


class Callback(Protocol):
    """
    Minimal interface for runtime hooks used during training or evaluation.

    A callback provides optional implementations for any of the event methods
    below. They can be combined to track progress, modify state, or log
    telemetry data across steps or epochs.
    """

    def on_step_begin(self, step: int, context: Mapping[str, Any]) -> None:
        """
        Called before each optimization or update step.

        Parameters
        ----------
        step : int
            Index of the current step (0-based).
        context : Mapping[str, Any]
            Mutable context dictionary shared among callbacks and the trainer.    
        """

    def on_step_end(self, step: int, context: Mapping[str, Any]) -> None:
        """
        Called after each optimization or update step.

        Parameters
        ----------
        step : int
            Index of the current step (0-based).
        context : Mapping[str, Any]
            Mutable context dictionary shared among callbacks and the trainer.
        """

    def on_epoch_begin(self, epoch: int, context: Mapping[str, Any]) -> None:
        """
        Called before each epoch begins.

        Parameters
        ----------
        epoch : int
            Current epoch number (0-based).
        context : Mapping[str, Any]
            Mutable context dictionary shared among callbacks and the trainer.
        """

    def on_epoch_end(self, epoch: int, context: Mapping[str, Any]) -> None:
        """
        Called after each epoch ends.

        Parameters
        ----------
        epoch : int
            Current epoch number (0-based).
        context : Mapping[str, Any]
            Mutable context dictionary shared among callbacks and the trainer.
        """

    def on_error(self, error: Exception, context: Mapping[str, Any]) -> None:
        """
        Called when an exception occurs within the training or evaluation loop.

        Parameters
        ----------
        error : Exception
            The raised exception object.
        context : Mapping[str, Any]
            Context snapshot at the time of the error.  
        """


class CallbackList:
    """
    Manages multiple callbacks and dispatches their events sequentially.

    Parameters
    ----------
    callbacks : list[Callback] or None, optional
        Initial list of callbacks. Defaults to an empty list.

    Notes
    -----
    - Callbacks are executed in the order they were added.
    - Each event propagates to all registered callbacks. 
    """

    def __init__(self, callbacks: list[Callback] | None = None) -> None:
        self._callbacks = list(callbacks or [])

    def append(self, cb: Callback) -> None:
        """
        Add a new callback to the list.

        Parameters
        ----------
        cb : Callback
            Instance implementing the ``Callback`` protocol.    
        """
        self._callbacks.append(cb)

    def on_step_begin(self, step: int, context: Mapping[str, Any]) -> None:
        """Trigger ``on_step_begin`` for all registered callbacks."""
        for cb in self._callbacks:
            cb.on_step_begin(step, context)

    def on_step_end(self, step: int, context: Mapping[str, Any]) -> None:
        """Trigger ``on_step_end`` for all registered callbacks."""
        for cb in self._callbacks:
            cb.on_step_end(step, context)

    def on_epoch_begin(self, epoch: int, context: Mapping[str, Any]) -> None:
        """Trigger ``on_epoch_begin`` for all registered callbacks."""
        for cb in self._callbacks:
            cb.on_epoch_begin(epoch, context)

    def on_epoch_end(self, epoch: int, context: Mapping[str, Any]) -> None:
        """Trigger ``on_epoch_end`` for all registered callbacks."""
        for cb in self._callbacks:
            cb.on_epoch_end(epoch, context)

    def on_error(self, error: Exception, context: Mapping[str, Any]) -> None:
        """Trigger ``on_error`` for all registered callbacks."""
        for cb in self._callbacks:
            cb.on_error(error, context)
