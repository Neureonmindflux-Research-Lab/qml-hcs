from __future__ import annotations
from typing import Any, Mapping
import json
from pathlib import Path
import pytest
import qmlhc.callbacks as C


# =============================================================================
# Public API re-exports
# =============================================================================
def test_callbacks_public_reexports_exist():
    expected = {
        "Callback", "CallbackList",
        "TelemetryLogger", "MemoryLogger",
        "DepthScheduler",
    }
    for name in expected:
        assert hasattr(C, name), f"{name} missing from qmlhc.callbacks"


# =============================================================================
# CallbackList dispatch behavior
# =============================================================================
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


# =============================================================================
# DepthScheduler interpolation and clamping
# =============================================================================
def test_depthscheduler_noop_and_with_attr():
    sch = C.DepthScheduler(target_attr="depth", start=1, end=4, epochs=3)

    class NoTarget:
        pass

    class WithDepth:
        def __init__(self):
            self.depth = 1

    sch.on_epoch_begin(0, {})  # no-op (no model/backend)
    sch.on_epoch_begin(0, {"backend": NoTarget()})  # no-op (missing attr)

    obj = WithDepth()
    sch.on_epoch_begin(0, {"model": obj})
    d0 = obj.depth
    sch.on_epoch_begin(2, {"model": obj})
    d2 = obj.depth
    sch.on_epoch_begin(10, {"model": obj})
    dF = obj.depth

    assert d0 >= 1 and d0 <= 2
    assert d2 >= 2 and d2 <= 4
    assert dF == 4


# =============================================================================
# MemoryLogger: in-memory logging
# =============================================================================
def test_memorylogger_records_all_tags():
    mem = C.MemoryLogger()
    mem.on_step_begin(1, {})
    mem.on_step_end(1, {"loss": 0.1})
    mem.on_epoch_begin(0, {})
    mem.on_epoch_end(0, {"metric": 0.9})
    mem.on_error(RuntimeError("boom"), {})
    tags = [r["tag"] for r in mem.records]
    assert tags == ["step_begin", "step_end", "epoch_begin", "epoch_end", "error"]


# =============================================================================
# TelemetryLogger: file writing and flush behavior
# =============================================================================
def test_telemetrylogger_writes_and_flushes(tmp_path: Path):
    path = tmp_path / "run" / "telemetry.jsonl"
    tel = C.TelemetryLogger(path=path, flush_interval=1)

    tel.on_step_begin(1, {})
    tel.on_step_end(1, {"loss": 0.123})
    tel.on_epoch_begin(0, {})
    tel.on_epoch_end(0, {"acc": 0.99})
    tel.on_error(ValueError("bad"), {})

    assert path.exists()
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 5
    parsed = [json.loads(l) for l in lines]
    tags = [p["tag"] for p in parsed]
    assert tags == ["step_begin", "step_end", "epoch_begin", "epoch_end", "error"]
    assert "context" in parsed[1] and "epoch" in parsed[2]


def test_depthscheduler_and_basecallback_all_noops_and_exceptions():
    import qmlhc.callbacks as C
    sch = C.DepthScheduler(target_attr="depth", start=1, end=2, epochs=2)

    class BadAssign:
        def __init__(self):
            self._depth = 0
        @property
        def depth(self):
            return self._depth
        @depth.setter
        def depth(self, _):
            raise RuntimeError("cannot set")

    obj = BadAssign()
    sch.on_epoch_begin(0, {"model": obj})
    sch.on_step_begin(1, {})
    sch.on_step_end(1, {})
    sch.on_epoch_end(0, {})
    sch.on_error(RuntimeError("x"), {})

    class BaseImpl(C.Callback):
        pass

    base = BaseImpl()
    base.on_step_begin(1, {})
    base.on_step_end(1, {})
    base.on_epoch_begin(0, {})
    base.on_epoch_end(0, {})
    base.on_error(RuntimeError("x"), {})

def test_telemetrylogger_flush_no_buffer(tmp_path):
    import qmlhc.callbacks as C
    path = tmp_path / "run" / "t.jsonl"
    tel = C.TelemetryLogger(path=path, flush_interval=999)
    tel._flush()
    assert not path.exists()

def test_callbacklist_append_adds_and_dispatches():
    import qmlhc.callbacks as C
    class CB(C.Callback):
        def __init__(self): self.events=[]
        def on_step_begin(self, step, context): self.events.append(step)
    cb = CB()
    cl = C.CallbackList([])
    cl.append(cb)
    cl.on_step_begin(7, {})
    assert cb.events == [7]

