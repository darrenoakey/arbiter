"""GPU Memory Manager for Arbiter.

Tracks model load states, manages VRAM budget, handles LRU eviction,
and supports pipeline overlap (loading one model while running another).

Supports multi-instance models: a single model_id (e.g. "moondream") can
have N independent adapter instances, each with its own VRAM slot.  Instances
are loaded on-demand — a new instance is only loaded when all existing loaded
instances are busy.
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
    instance_id: str
    adapter: ModelAdapter
    memory_gb: float
    keep_alive_s: float
    max_concurrent: int = 1
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

        instance_id = mm.pick_instance("flux", max_concurrent=1)
        await mm.ensure_loaded(instance_id)
        try:
            # ... run inference using mm.get_slot(instance_id) ...
        finally:
            mm.release(instance_id)
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
        self._slots: dict[str, ModelSlot] = {}  # keyed by instance_id
        self._model_instances: dict[str, list[str]] = {}  # model_id -> [instance_ids]
        self._global_lock = asyncio.Lock()
        self._keepalive_task: Optional[asyncio.Task] = None
        self._condemned: set[str] = set()  # instance_ids pending removal after current jobs finish

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
        max_concurrent: int = 1,
        instance_id: Optional[str] = None,
    ):
        """Register a model slot. Must be called before ensure_loaded.

        For multi-instance models, call once per instance with a unique
        instance_id (e.g. "moondream#0", "moondream#1").  For single-instance
        models, instance_id defaults to model_id.
        """
        iid = instance_id or model_id
        self._slots[iid] = ModelSlot(
            model_id=model_id,
            instance_id=iid,
            adapter=adapter,
            memory_gb=memory_gb,
            keep_alive_s=keep_alive_s,
            max_concurrent=max_concurrent,
        )
        if model_id not in self._model_instances:
            self._model_instances[model_id] = []
        if iid not in self._model_instances[model_id]:
            self._model_instances[model_id].append(iid)

    def get_slot(self, instance_id: str) -> Optional[ModelSlot]:
        return self._slots.get(instance_id)

    def get_model_instances(self, model_id: str) -> list[str]:
        """Return all instance_ids for a model."""
        return list(self._model_instances.get(model_id, []))

    def is_loaded(self, model_id: str) -> bool:
        """True if ANY instance of this model is loaded."""
        for iid in self._model_instances.get(model_id, []):
            slot = self._slots.get(iid)
            if slot is not None and slot.state == ModelState.LOADED:
                return True
        # Fallback: direct slot lookup (backward compat for tests)
        slot = self._slots.get(model_id)
        return slot is not None and slot.state == ModelState.LOADED

    def pick_instance(self, model_id: str, max_concurrent: int = 1) -> Optional[str]:
        """Pick the best instance for a new job.

        Preference order:
        1. Loaded instance with spare capacity (least busy first)
        2. Loading instance (job will wait for it via load_event)
        3. Unloaded instance (cold start — only when all others are busy)
        4. None if no instances registered

        This ordering ensures we don't start loading N instances simultaneously.
        A new instance only starts loading when existing loaded+loading instances
        are all at capacity.
        """
        instances = self._model_instances.get(model_id, [])
        if not instances:
            # Fallback: if model_id is itself a slot key (single-instance)
            if model_id in self._slots:
                return model_id
            return None

        # Categorize instances
        loaded = []
        loading = []
        unloaded = []
        for iid in instances:
            slot = self._slots[iid]
            if slot.state == ModelState.LOADED:
                if slot.active_count < max_concurrent:
                    loaded.append((iid, slot))
            elif slot.state == ModelState.LOADING:
                loading.append(iid)
            elif slot.state == ModelState.UNLOADED:
                unloaded.append(iid)

        # 1. Prefer loaded instances with capacity
        if loaded:
            loaded.sort(key=lambda x: x[1].active_count)
            return loaded[0][0]

        # 2. If an instance is already loading, wait for it rather than
        #    starting another cold load
        if loading:
            return loading[0]

        # 3. No loaded or loading instances available — cold start one
        if unloaded:
            return unloaded[0]

        # All errored — return first to retry
        return instances[0]

    def total_capacity(self, model_id: str, max_concurrent: int = 1) -> int:
        """Total job capacity across all instances of a model."""
        instances = self._model_instances.get(model_id, [])
        return len(instances) * max_concurrent

    def get_all_slots(self) -> dict[str, ModelSlot]:
        return dict(self._slots)

    async def ensure_loaded(self, instance_id: str) -> None:
        """Ensure instance is loaded and increment active_count.

        If already loaded: immediate.
        If loading: wait for load to complete.
        If unloaded: check budget, evict if needed, load in thread pool.

        Raises EvictionImpossible if can't free enough VRAM.
        """
        slot = self._slots.get(instance_id)
        if slot is None:
            raise KeyError(f"Model not registered: {instance_id}")

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
                    raise RuntimeError(f"Model {instance_id} failed to load")

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

        self._log(
            "model.load_start",
            model_id=slot.model_id,
            instance_id=instance_id,
            memory_gb=slot.memory_gb,
        )

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
            self._log(
                "model.load_error",
                model_id=slot.model_id,
                instance_id=instance_id,
                error=str(e),
            )
            raise

        async with slot._lock:
            slot.state = ModelState.LOADED
            slot.active_count += 1
            slot.load_event.set()

        self._log(
            "model.load_done",
            model_id=slot.model_id,
            instance_id=instance_id,
            memory_gb=slot.memory_gb,
        )

    def release(self, instance_id: str):
        """Decrement active_count and record last_active time. Thread-safe (sync).

        If the instance is condemned (marked for removal during scale-down) and
        now idle, schedules async eviction and slot removal.
        """
        slot = self._slots.get(instance_id)
        if slot is None:
            return
        slot.active_count = max(0, slot.active_count - 1)
        if slot.active_count == 0:
            slot.last_active = time.monotonic()
            if instance_id in self._condemned:
                self._condemned.discard(instance_id)
                try:
                    loop = asyncio.get_running_loop()
                    loop.create_task(self._evict_and_remove(instance_id))
                except RuntimeError:
                    pass  # No event loop — keepalive will catch it

    async def _evict_and_remove(self, instance_id: str):
        """Evict a condemned instance and remove its slot entirely."""
        async with self._global_lock:
            slot = self._slots.get(instance_id)
            if slot is None:
                return
            if slot.active_count > 0:
                # Got busy again somehow — re-condemn
                self._condemned.add(instance_id)
                return
            if slot.state == ModelState.LOADED:
                slot.state = ModelState.EVICTING
                self._log(
                    "model.evict_start",
                    model_id=slot.model_id,
                    instance_id=instance_id,
                    reason="condemned",
                    memory_gb=slot.memory_gb,
                )
                loop = asyncio.get_event_loop()
                try:
                    await loop.run_in_executor(self._executor, slot.adapter.unload)
                except Exception as e:
                    logger.error("Failed to unload condemned %s: %s", instance_id, e)
                self._used_gb -= slot.memory_gb
                self._log(
                    "model.evict_done",
                    model_id=slot.model_id,
                    instance_id=instance_id,
                )
            elif slot.state in (ModelState.LOADING, ModelState.EVICTING):
                # Still loading or already evicting — re-condemn and let it finish
                self._condemned.add(instance_id)
                return
            # Remove slot entirely
            model_id = slot.model_id
            del self._slots[instance_id]
            # Also clean from _model_instances if somehow still there
            if model_id in self._model_instances and instance_id in self._model_instances[model_id]:
                self._model_instances[model_id].remove(instance_id)
            self._log(
                "instance.removed",
                model_id=model_id,
                instance_id=instance_id,
            )

    async def scale_model(
        self,
        model_id: str,
        new_count: int,
        adapter_cls,
        memory_gb: float,
        keep_alive_s: float,
        max_concurrent: int,
    ) -> dict:
        """Scale the number of instances for a model.

        Scaling up: creates new unloaded instances (loaded on demand by scheduler).
        Scaling down: evicts idle excess immediately, condemns active excess
        (they auto-evict when their current jobs finish).

        Returns dict with results: added, removed, condemned.
        """
        current_ids = list(self._model_instances.get(model_id, []))
        current_count = len(current_ids)

        result = {"added": 0, "removed": 0, "condemned": 0}

        if new_count == current_count:
            return result

        if new_count > current_count:
            # Scale up: add new instances
            next_idx = self._next_instance_index(model_id)
            for i in range(new_count - current_count):
                idx = next_idx + i
                instance_id = f"{model_id}#{idx}"
                adapter = adapter_cls()
                self.register(
                    model_id=model_id,
                    adapter=adapter,
                    memory_gb=memory_gb,
                    keep_alive_s=keep_alive_s,
                    max_concurrent=max_concurrent,
                    instance_id=instance_id,
                )
                result["added"] += 1
                self._log(
                    "instance.added",
                    model_id=model_id,
                    instance_id=instance_id,
                )
            return result

        # Scale down: remove from the end of the list
        to_remove = current_ids[new_count:]

        async with self._global_lock:
            for iid in to_remove:
                slot = self._slots.get(iid)
                if slot is None:
                    continue

                # Remove from active instance list (prevents new job dispatch)
                if model_id in self._model_instances and iid in self._model_instances[model_id]:
                    self._model_instances[model_id].remove(iid)

                if slot.state == ModelState.UNLOADED:
                    # Not loaded — just remove the slot
                    del self._slots[iid]
                    result["removed"] += 1
                    self._log("instance.removed", model_id=model_id, instance_id=iid)
                elif slot.state == ModelState.LOADED and slot.active_count == 0:
                    # Loaded but idle — evict and remove immediately
                    slot.state = ModelState.EVICTING
                    self._log(
                        "model.evict_start",
                        model_id=model_id,
                        instance_id=iid,
                        reason="scale_down",
                        memory_gb=slot.memory_gb,
                    )
                    loop = asyncio.get_event_loop()
                    try:
                        await loop.run_in_executor(self._executor, slot.adapter.unload)
                    except Exception as e:
                        logger.error("Failed to unload %s during scale-down: %s", iid, e)
                    self._used_gb -= slot.memory_gb
                    del self._slots[iid]
                    result["removed"] += 1
                    self._log("instance.removed", model_id=model_id, instance_id=iid)
                elif slot.state == ModelState.ERROR:
                    # Errored — just remove
                    del self._slots[iid]
                    result["removed"] += 1
                    self._log("instance.removed", model_id=model_id, instance_id=iid)
                else:
                    # Active (running jobs) or loading — condemn
                    # Will be auto-evicted and removed when current jobs finish
                    self._condemned.add(iid)
                    result["condemned"] += 1
                    self._log(
                        "instance.condemned",
                        model_id=model_id,
                        instance_id=iid,
                        active_count=slot.active_count,
                        state=slot.state.value,
                    )

        return result

    def _next_instance_index(self, model_id: str) -> int:
        """Find the next available instance index for a model.

        Checks active instances, condemned slots still in _slots, and all
        known instance_ids to avoid collisions.
        """
        max_idx = -1
        # Check all known instance IDs (active list + any slots still alive)
        all_ids: set[str] = set(self._model_instances.get(model_id, []))
        for iid, slot in self._slots.items():
            if slot.model_id == model_id:
                all_ids.add(iid)

        for iid in all_ids:
            if "#" in iid:
                try:
                    idx = int(iid.rsplit("#", 1)[1])
                    max_idx = max(max_idx, idx)
                except ValueError:
                    pass
            else:
                # Bare model_id counts as occupying index 0
                max_idx = max(max_idx, 0)
        return max_idx + 1

    async def _evict_for(self, needed_gb: float):
        """Evict idle models (LRU) to free at least needed_gb.

        Must be called while holding _global_lock.
        Raises EvictionImpossible if can't free enough.
        """
        deficit = (self._used_gb + needed_gb) - self._budget_gb
        if deficit <= 0:
            return

        # Collect evictable slots: loaded, no active inferences
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

        # Evict selected slots
        loop = asyncio.get_event_loop()
        for slot in to_evict:
            slot.state = ModelState.EVICTING
            self._log(
                "model.evict_start",
                model_id=slot.model_id,
                instance_id=slot.instance_id,
                memory_gb=slot.memory_gb,
            )
            try:
                await loop.run_in_executor(self._executor, slot.adapter.unload)
            except Exception as e:
                logger.error("Failed to unload %s: %s", slot.instance_id, e)
            self._used_gb -= slot.memory_gb
            slot.state = ModelState.UNLOADED
            slot.active_count = 0
            self._log(
                "model.evict_done",
                model_id=slot.model_id,
                instance_id=slot.instance_id,
            )

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
                            instance_id=slot.instance_id,
                            reason="keepalive_expired",
                            memory_gb=slot.memory_gb,
                        )
                        try:
                            await loop.run_in_executor(self._executor, slot.adapter.unload)
                        except Exception as e:
                            logger.error("Keepalive unload failed for %s: %s", slot.instance_id, e)
                        self._used_gb -= slot.memory_gb
                        slot.state = ModelState.UNLOADED
                        self._log(
                            "model.evict_done",
                            model_id=slot.model_id,
                            instance_id=slot.instance_id,
                        )

    def start_keepalive(self):
        """Start the keepalive background task."""
        self._keepalive_task = asyncio.create_task(self.run_keepalive_loop())

    def stop_keepalive(self):
        """Stop the keepalive background task."""
        if self._keepalive_task:
            self._keepalive_task.cancel()
            self._keepalive_task = None

    def snapshot(self) -> dict:
        """Return current state for logging/API.

        Multi-instance models are grouped: each model appears once with
        aggregated stats and a per-instance breakdown.
        """
        # Group slots by model_id
        model_groups: dict[str, list[ModelSlot]] = {}
        for slot in self._slots.values():
            model_groups.setdefault(slot.model_id, []).append(slot)

        models = []
        for model_id, slots in model_groups.items():
            total_active = sum(s.active_count for s in slots)
            loaded_count = sum(1 for s in slots if s.state == ModelState.LOADED)
            total_memory = sum(s.memory_gb for s in slots if s.state != ModelState.UNLOADED)

            # Idle time: min idle across loaded idle instances
            idle_s = None
            for s in slots:
                if s.state == ModelState.LOADED and s.active_count == 0 and s.last_active > 0:
                    s_idle = round(time.monotonic() - s.last_active, 1)
                    idle_s = s_idle if idle_s is None else min(idle_s, s_idle)

            # Count active (non-condemned) instances
            active_instance_ids = set(self._model_instances.get(model_id, []))
            condemned_count = sum(1 for s in slots if s.instance_id in self._condemned)

            entry = {
                "id": model_id,
                "state": "active" if total_active > 0 else (
                    "loaded" if loaded_count > 0 else slots[0].state.value
                ),
                "memory_gb": round(total_memory, 2) if total_memory else slots[0].memory_gb,
                "active_jobs": total_active,
                "idle_seconds": idle_s,
            }

            # Add instance breakdown for multi-instance models or when condemned instances exist
            if len(slots) > 1 or condemned_count > 0:
                entry["instances"] = []
                for s in slots:
                    s_idle = None
                    if s.state == ModelState.LOADED and s.active_count == 0 and s.last_active > 0:
                        s_idle = round(time.monotonic() - s.last_active, 1)
                    inst_info = {
                        "instance_id": s.instance_id,
                        "state": s.state.value if s.active_count == 0 else "active",
                        "active_jobs": s.active_count,
                        "idle_seconds": s_idle,
                    }
                    if s.instance_id in self._condemned:
                        inst_info["condemned"] = True
                    entry["instances"].append(inst_info)
                entry["loaded_instances"] = loaded_count
                entry["total_instances"] = len(active_instance_ids)
                if condemned_count > 0:
                    entry["condemned_instances"] = condemned_count

            models.append(entry)

        # Read actual GPU memory from CUDA (not just bookkeeping)
        vram_actual_gb = self._used_gb  # fallback to bookkeeping
        try:
            import torch
            if torch.cuda.is_available():
                vram_actual_gb = torch.cuda.memory_allocated() / (1024**3)
        except Exception:
            pass

        return {
            "vram_budget_gb": self._budget_gb,
            "vram_used_gb": round(vram_actual_gb, 2),
            "vram_configured_gb": round(self._used_gb, 2),
            "models": models,
        }

    def _log(self, event: str, **kwargs):
        if self._event_logger:
            self._event_logger.log(event, **kwargs)
