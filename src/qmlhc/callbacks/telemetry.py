#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Telemetry callback utilities
----------------------------
Structured logging of metrics, losses, and state during training/evaluation.

This module provides two callbacks:

- ``TelemetryLogger``: appends JSON records to a JSONL file with periodic
  flush, suitable for long runs.
- ``MemoryLogger``: collects records in memory for debugging or tests.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Mapping, Optional  # Optional kept for external type usage

from .base import Callback


class TelemetryLogger(Callback):
    """
    JSONL-based telemetry logger.

    Records tagged events (e.g., step/epoch begin/end, errors) into an
    internal buffer and flushes to a JSONL file either when the buffer
    reaches a given length or when a time threshold elapses.

    Parameters
    ----------
    path : str or Path, optional
        Output path for the JSONL file. Defaults to ``"telemetry.jsonl"``.
    flush_interval : int, optional
        Flush after this many buffered entries (minimum = 1). Defaults to ``1``.

    Notes
    -----
    Each JSON line includes:
        - ``ts`` (float): UNIX timestamp (seconds).
        - ``tag`` (str): event tag (``"step_begin"``, ``"epoch_end"``, etc.).
        - Any extra payload fields provided by the caller.
    """

    def __init__(self, path: str | Path = "telemetry.jsonl", flush_interval: int = 1):
        self._path = Path(path)
        self._flush_every = max(1, int(flush_interval))
        self._buffer: list[dict[str, Any]] = []
        self._last_flush = time.time()

    # ----------------------------- internals -----------------------------
    def _record(self, tag: str, payload: Mapping[str, Any]) -> None:
        """
        Append a tagged entry to the buffer and flush if needed.

        Parameters
        ----------
        tag : str
            Event tag.
        payload : Mapping[str, Any]
            Additional fields to be merged into the record.
        """
        entry = {"ts": time.time(), "tag": tag, **payload}
        self._buffer.append(entry)

        now = time.time()
        if len(self._buffer) >= self._flush_every or (now - self._last_flush) > 5.0:
            self._flush()
            self._last_flush = now

    def _flush(self) -> None:
        """
        Flush the buffered entries to the JSONL file (append mode).

        Creates parent directories if they do not exist. No-op if the buffer
        is empty.
        """
        if not self._buffer:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8") as f:
            for entry in self._buffer:
                f.write(json.dumps(entry) + "\n")
        self._buffer.clear()

    # -------------------------- callback hooks ---------------------------
    def on_step_begin(self, step: int, context: Mapping[str, Any]) -> None:
        """Record the beginning of a step."""
        self._record("step_begin", {"step": int(step)})

    def on_step_end(self, step: int, context: Mapping[str, Any]) -> None:
        """
        Record the end of a step.

        Notes
        -----
        ``context`` is converted to a plain ``dict`` to ensure JSON safety.
        """
        self._record("step_end", {"step": int(step), "context": dict(context)})

    def on_epoch_begin(self, epoch: int, context: Mapping[str, Any]) -> None:
        """Record the beginning of an epoch."""
        self._record("epoch_begin", {"epoch": int(epoch)})

    def on_epoch_end(self, epoch: int, context: Mapping[str, Any]) -> None:
        """
        Record the end of an epoch.

        Notes
        -----
        ``context`` is converted to a plain ``dict`` to ensure JSON safety.
        """
        self._record("epoch_end", {"epoch": int(epoch), "context": dict(context)})

    def on_error(self, error: Exception, context: Mapping[str, Any]) -> None:
        """
        Record an error event.

        Parameters
        ----------
        error : Exception
            The encountered exception.
        context : Mapping[str, Any]
            Additional context at the time of the error.
        """
        self._record(
            "error",
            {"type": type(error).__name__, "message": str(error)},
        )


class MemoryLogger(Callback):
    """
    In-memory telemetry collector.

    Useful for unit tests or quick debugging sessions where file I/O is
    undesirable. Each call appends a small dictionary to ``records``.
    """

    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []

    # -------------------------- helper internals -------------------------
    def _log(self, tag: str, payload: Mapping[str, Any]) -> None:
        """
        Append a tagged entry to the in-memory ``records`` list.

        Parameters
        ----------
        tag : str
            Event tag (e.g., ``"step_begin"``).
        payload : Mapping[str, Any]
            Additional fields to store with the record.
        """
        self.records.append({"tag": tag, **payload})

    # -------------------------- callback hooks ---------------------------
    def on_step_begin(self, step: int, context: Mapping[str, Any]) -> None:
        """Record the beginning of a step."""
        self._log("step_begin", {"step": int(step)})

    def on_step_end(self, step: int, context: Mapping[str, Any]) -> None:
        """
        Record the end of a step.

        Notes
        -----
        ``context`` is converted to a plain ``dict`` to avoid surprises
        with non-serializable payloads during debugging.
        """
        self._log("step_end", {"step": int(step), "context": dict(context)})

    def on_epoch_begin(self, epoch: int, context: Mapping[str, Any]) -> None:
        """Record the beginning of an epoch."""
        self._log("epoch_begin", {"epoch": int(epoch)})

    def on_epoch_end(self, epoch: int, context: Mapping[str, Any]) -> None:
        """
        Record the end of an epoch.

        Notes
        -----
        ``context`` is converted to a plain ``dict`` to keep a JSON-safe record.
        """
        self._log("epoch_end", {"epoch": int(epoch), "context": dict(context)})

    def on_error(self, error: Exception, context: Mapping[str, Any]) -> None:
        """
        Record an error event.

        Parameters
        ----------
        error : Exception
            The encountered exception.
        context : Mapping[str, Any]
            Additional context at the time of the error.
        """
        self._log("error", {"type": type(error).__name__, "message": str(error)})
