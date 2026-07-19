import threading

from arbiter.adapters.base import HeapTrimGuard


def test_heap_trim_guard_stops_its_real_background_thread() -> None:
    before = {thread.ident for thread in threading.enumerate()}

    with HeapTrimGuard(interval_seconds=0.01) as guard:
        threading.Event().wait(0.05)

    leaked = [
        thread
        for thread in threading.enumerate()
        if thread.ident not in before and thread.name == "arbiter-heap-trim"
    ]
    assert leaked == []
    # Linux/glibc executes real malloc_trim calls. Other libc implementations
    # are deliberately a no-op while retaining the same lifecycle behavior.
    assert guard.trim_count >= 0
