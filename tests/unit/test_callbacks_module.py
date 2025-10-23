
# Comprehensive tests for src/qmlhc/callbacks: __init__.py, base.py, telemetry.py, depth_control.py

from __future__ import annotations

from typing import Any, Mapping
import json
from pathlib import Path

import pytest

import qmlhc.callbacks as C


# ---------- Public API re-exports --------------------------------------------

def test_callbacks_public_reexports_exist():
    expected = {
        "Callback", "CallbackList",
        "TelemetryLogger", "MemoryLogger",
        "DepthScheduler",
    }
    for name in expected:
        assert hasattr(C, name), f"{name} missing from qmlhc.callbacks"


# ---------- CallbackList dispatch --------------------------------------------

class _DummyCB(C.Callback):
    def __init__(self):
        self.events: list[str] = []

    def on_step_begin(self, step: int, context: Mapping[str, Any]) -> None:
        self.events.append(f"sb:{step}")

    def on_step_end(self, step: int, context: Mapping[str, Any]) -> None:
        self.events.append(f"se:{step}")

    def on_epoch_begin(self, epoch: int, context: Mapping[str, Any]) -> None:
        self.events.append(f"eb:{epoch}")

    def on_epoch_end(self, epoch: int, context: Mapping[str, Any]) -> None:
        self.events.append(f"ee:{epoch}")

    def on_error(self, error: Exception, context: Mapping[str, Any]) -> None:
        self.events.append(f"er:{type(error).__name__}")


def test_callbacklist_dispatches_all_hooks():
    cb = _DummyCB()
    cl = C.CallbackList([cb])
    cl.on_step_begin(1, {})
    cl.on_step_end(1, {})
    cl.on_epoch_begin(0, {})
    cl.on_epoch_end(0, {})
    cl.on_error(RuntimeError("x"), {})
    assert cb.events == ["sb:1", "se:1", "eb:0", "ee:0", "er:RuntimeError"]


# ---------- DepthScheduler behavior ------------------------------------------

def test_depthscheduler_noop_and_with_attr():
    sch = C.DepthScheduler(target_attr="depth", start=1, end=4, epochs=3)

    class NoTarget:  # lacks 'depth'
        pass

    class WithDepth:
        def __init__(self):
            self.depth = 1

    # No model/backend in context -> no-op
    sch.on_epoch_begin(0, {})
    # Backend present but missing attribute -> no-op (no exception)
    sch.on_epoch_begin(0, {"backend": NoTarget()})

    # Model present with attribute -> depth should change with epoch
    obj = WithDepth()
    sch.on_epoch_begin(0, {"model": obj})  # epoch 0 -> start (~1)
    d0 = obj.depth
    sch.on_epoch_begin(2, {"model": obj})  # epoch near end -> closer to 'end'
    d2 = obj.depth
    sch.on_epoch_begin(10, {"model": obj})  # beyond epochs -> end (clamped)
    dF = obj.depth

    assert d0 >= 1 and d0 <= 2        # rounded start
    assert d2 >= 2 and d2 <= 4        # mid interpolation
    assert dF == 4                    # clamped to end


# ---------- TelemetryLogger and MemoryLogger ---------------------------------

def test_memorylogger_records_all_tags():
    mem = C.MemoryLogger()
    mem.on_step_begin(1, {})
    mem.on_step_end(1, {"loss": 0.1})
    mem.on_epoch_begin(0, {})
    mem.on_epoch_end(0, {"metric": 0.9})
    mem.on_error(RuntimeError("boom"), {})
    tags = [r["tag"] for r in mem.records]
    assert tags == ["step_begin", "step_end", "epoch_begin", "epoch_end", "error"]


def test_telemetrylogger_writes_and_flushes(tmp_path: Path):
    path = tmp_path / "run" / "telemetry.jsonl"
    tel = C.TelemetryLogger(path=path, flush_interval=1)  # flush every record

    tel.on_step_begin(1, {})
    tel.on_step_end(1, {"loss": 0.123})
    tel.on_epoch_begin(0, {})
    tel.on_epoch_end(0, {"acc": 0.99})
    tel.on_error(ValueError("bad"), {})

    # File should exist and contain 5 JSON lines with expected tags
    assert path.exists()
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 5
    parsed = [json.loads(l) for l in lines]
    tags = [p["tag"] for p in parsed]
    assert tags == ["step_begin", "step_end", "epoch_begin", "epoch_end", "error"]
    # Ensure extra payloads are present in some entries
    assert "context" in parsed[1] and "epoch" in parsed[2]





