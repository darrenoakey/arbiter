"""GPU Memory Manager for Arbiter.

Tracks model load states, manages VRAM budget, handles LRU eviction,
and supports pipeline overlap (loading one model while running another).
"""
from __future__ import annotations

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .adapters.base import ModelAdapter
    from .log import EventLogger

logger = logging.getLogger(__name__)


class ModelState(str, Enum):
    UNLOADED = "unloaded"
    LOADING = "loading"
    LOADED = "loaded"
    EVICTING = "evicting"
    ERROR = "error"


@dataclass
class ModelSlot:
    model_id: str
    adapter: ModelAdapter
    memory_gb: float
    keep_alive_s: float
    state: ModelState = ModelState.UNLOADED
    active_count: int = 0
    last_active: float = 0.0
    load_event: asyncio.Event = field(default_factory=asyncio.Event)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class EvictionImpossible(Exception):
    """Raised when we can't free enough VRAM even after evicting all idle models."""
    pass


class MemoryManager:
    """Manages GPU VRAM budget and model lifecycle.

    Usage:
        mm = MemoryManager(budget_gb=100, executor=pool, event_logger=logger)
        mm.register("flux", adapter, memory_gb=12, keep_alive_s=300)

        await mm.ensure_loaded("flux")  # loads if needed, increments active_count
        try:
            # ... run inference ...
        finally:
            mm.release("flux")  # decrements active_count, starts keep-alive timer
    """

    def __init__(
        self,
        budget_gb: float,
        executor: ThreadPoolExecutor,
        event_logger: Optional[EventLogger] = None,
    ):
        self._budget_gb = budget_gb
        self._used_gb = 0.0
        self._executor = executor
        self._event_logger = event_logger
        self._slots: dict[str, ModelSlot] = {}
        self._global_lock = asyncio.Lock()
        self._keepalive_task: Optional[asyncio.Task] = None

    @property
    def budget_gb(self) -> float:
        return self._budget_gb

    @property
    def used_gb(self) -> float:
        return self._used_gb

    @property
    def free_gb(self) -> float:
        return self._budget_gb - self._used_gb

    def register(
        self,
        model_id: str,
        adapter: ModelAdapter,
        memory_gb: float,
        keep_alive_s: float = 300,
    ):
        """Register a model slot. Must be called before ensure_loaded."""
        self._slots[model_id] = ModelSlot(
            model_id=model_id,
            adapter=adapter,
            memory_gb=memory_gb,
            keep_alive_s=keep_alive_s,
        )

    def get_slot(self, model_id: str) -> Optional[ModelSlot]:
        return self._slots.get(model_id)

    def is_loaded(self, model_id: str) -> bool:
        slot = self._slots.get(model_id)
        return slot is not None and slot.state in (ModelState.LOADED,)

    def get_all_slots(self) -> dict[str, ModelSlot]:
        return dict(self._slots)

    async def ensure_loaded(self, model_id: str) -> None:
        """Ensure model is loaded and increment active_count.

        If already loaded: immediate.
        If loading: wait for load to complete.
        If unloaded: check budget, evict if needed, load in thread pool.

        Raises EvictionImpossible if can't free enough VRAM.
        """
        slot = self._slots.get(model_id)
        if slot is None:
            raise KeyError(f"Model not registered: {model_id}")

        async with slot._lock:
            if slot.state == ModelState.LOADED:
                slot.active_count += 1
                return

            if slot.state == ModelState.LOADING:
                # Release per-slot lock, wait for load, then re-acquire
                pass  # handled below

            if slot.state == ModelState.ERROR:
                # Reset and try again
                slot.state = ModelState.UNLOADED

        # If loading, wait for it
        if slot.state == ModelState.LOADING:
            await slot.load_event.wait()
            async with slot._lock:
                if slot.state == ModelState.LOADED:
                    slot.active_count += 1
                    return
                elif slot.state == ModelState.ERROR:
                    raise RuntimeError(f"Model {model_id} failed to load")

        # Need to load — acquire global lock for budget arithmetic
        async with self._global_lock:
            # Double-check after acquiring global lock
            if slot.state == ModelState.LOADED:
                async with slot._lock:
                    slot.active_count += 1
                return

            needed = slot.memory_gb
            if self._used_gb + needed > self._budget_gb:
                await self._evict_for(needed)

            # Reserve the memory and start loading
            slot.load_event = asyncio.Event()  # reset event
            slot.state = ModelState.LOADING
            self._used_gb += needed

        self._log("model.load_start", model_id=model_id, memory_gb=slot.memory_gb)

        # Load in thread pool (blocking I/O)
        loop = asyncio.get_event_loop()
        try:
            await loop.run_in_executor(self._executor, slot.adapter.load, "cuda")
        except Exception as e:
            async with self._global_lock:
                self._used_gb -= slot.memory_gb
            async with slot._lock:
                slot.state = ModelState.ERROR
                slot.load_event.set()
            self._log("model.load_error", model_id=model_id, error=str(e))
            raise

        async with slot._lock:
            slot.state = ModelState.LOADED
            slot.active_count += 1
            slot.load_event.set()

        self._log("model.load_done", model_id=model_id, memory_gb=slot.memory_gb)

    def release(self, model_id: str):
        """Decrement active_count and record last_active time. Thread-safe (sync)."""
        slot = self._slots.get(model_id)
        if slot is None:
            return
        slot.active_count = max(0, slot.active_count - 1)
        if slot.active_count == 0:
            slot.last_active = time.monotonic()

    async def _evict_for(self, needed_gb: float):
        """Evict idle models (LRU) to free at least needed_gb.

        Must be called while holding _global_lock.
        Raises EvictionImpossible if can't free enough.
        """
        deficit = (self._used_gb + needed_gb) - self._budget_gb
        if deficit <= 0:
            return

        # Collect evictable models: loaded, no active inferences
        evictable = [
            s for s in self._slots.values()
            if s.state == ModelState.LOADED and s.active_count == 0
        ]
        # Sort by last_active ascending (oldest idle first = LRU)
        evictable.sort(key=lambda s: s.last_active)

        freed = 0.0
        to_evict = []
        for slot in evictable:
            to_evict.append(slot)
            freed += slot.memory_gb
            if freed >= deficit:
                break

        if freed < deficit:
            raise EvictionImpossible(
                f"Need {deficit:.1f}GB but can only free {freed:.1f}GB "
                f"({len(evictable)} idle models)"
            )

        # Evict selected models
        loop = asyncio.get_event_loop()
        for slot in to_evict:
            slot.state = ModelState.EVICTING
            self._log("model.evict_start", model_id=slot.model_id, memory_gb=slot.memory_gb)
            try:
                await loop.run_in_executor(self._executor, slot.adapter.unload)
            except Exception as e:
                logger.error("Failed to unload %s: %s", slot.model_id, e)
            self._used_gb -= slot.memory_gb
            slot.state = ModelState.UNLOADED
            slot.active_count = 0
            self._log("model.evict_done", model_id=slot.model_id)

    async def run_keepalive_loop(self, interval: float = 10.0):
        """Background task: evict models idle past their keep_alive_s."""
        while True:
            await asyncio.sleep(interval)
            now = time.monotonic()
            to_evict = []
            for slot in self._slots.values():
                if (
                    slot.state == ModelState.LOADED
                    and slot.active_count == 0
                    and slot.last_active > 0
                    and (now - slot.last_active) > slot.keep_alive_s
                ):
                    to_evict.append(slot)

            if to_evict:
                async with self._global_lock:
                    loop = asyncio.get_event_loop()
                    for slot in to_evict:
                        # Re-check under lock
                        if slot.state != ModelState.LOADED or slot.active_count > 0:
                            continue
                        slot.state = ModelState.EVICTING
                        self._log(
                            "model.evict_start",
                            model_id=slot.model_id,
                            reason="keepalive_expired",
                            memory_gb=slot.memory_gb,
                        )
                        try:
                            await loop.run_in_executor(self._executor, slot.adapter.unload)
                        except Exception as e:
                            logger.error("Keepalive unload failed for %s: %s", slot.model_id, e)
                        self._used_gb -= slot.memory_gb
                        slot.state = ModelState.UNLOADED
                        self._log("model.evict_done", model_id=slot.model_id)

    def start_keepalive(self):
        """Start the keepalive background task."""
        self._keepalive_task = asyncio.create_task(self.run_keepalive_loop())

    def stop_keepalive(self):
        """Stop the keepalive background task."""
        if self._keepalive_task:
            self._keepalive_task.cancel()
            self._keepalive_task = None

    def snapshot(self) -> dict:
        """Return current state for logging/API."""
        models = []
        for slot in self._slots.values():
            idle_s = None
            if slot.state == ModelState.LOADED and slot.active_count == 0 and slot.last_active > 0:
                idle_s = round(time.monotonic() - slot.last_active, 1)
            models.append({
                "id": slot.model_id,
                "state": slot.state.value if slot.active_count == 0 else "active",
                "memory_gb": slot.memory_gb,
                "active_jobs": slot.active_count,
                "idle_seconds": idle_s,
            })
        return {
            "vram_budget_gb": self._budget_gb,
            "vram_used_gb": round(self._used_gb, 2),
            "models": models,
        }

    def _log(self, event: str, **kwargs):
        if self._event_logger:
            self._event_logger.log(event, **kwargs)
