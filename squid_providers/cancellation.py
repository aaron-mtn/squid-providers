"""Transport-owned cancellation watcher; Core owns the shared control signal."""

from __future__ import annotations

import threading
from collections.abc import Callable

from squid_core.providers.base import StreamCancelled


CANCELLATION_POLL_SECONDS = 0.02
CANCELLATION_MONITOR_JOIN_SECONDS = 0.2


class CancellationMonitor:
    """Close one active transport when a request's cancellation predicate fires.

    Provider I/O remains on the calling thread.  Only this small monitor runs in
    the background, which lets it close a socket/client while the caller is
    blocked waiting for response headers or the next body chunk.
    """

    def __init__(
        self,
        should_cancel: Callable[[], bool] | None,
        abort: Callable[[], object],
    ) -> None:
        self._should_cancel = should_cancel
        self._abort = abort
        self._stop = threading.Event()
        self._cancelled = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def cancelled(self) -> bool:
        return self._cancelled.is_set()

    def __enter__(self) -> "CancellationMonitor":
        if self._should_cancel is None:
            return self
        if self._should_cancel():
            self._cancelled.set()
            self._abort_safely()
            return self
        self._thread = threading.Thread(
            target=self._watch,
            name="squid-provider-cancellation",
            daemon=True,
        )
        self._thread.start()
        return self

    def __exit__(self, *_args) -> None:
        self._stop.set()
        if self._thread is not None:
            # The watcher uses Event.wait rather than sleep, so a normal stop
            # wakes it immediately. The bound protects the caller if a foreign
            # transport implements a pathological blocking close method.
            self._thread.join(timeout=CANCELLATION_MONITOR_JOIN_SECONDS)

    def raise_if_cancelled(self) -> None:
        if self.cancelled or (
            self._should_cancel is not None and self._should_cancel()
        ):
            raise StreamCancelled

    def _watch(self) -> None:
        while not self._stop.wait(CANCELLATION_POLL_SECONDS):
            try:
                cancelled = bool(self._should_cancel and self._should_cancel())
            except Exception:
                # Cancellation predicates are host control flow. A broken
                # predicate must not replace the active provider request.
                return
            if cancelled:
                self._cancelled.set()
                self._abort_safely()
                return

    def _abort_safely(self) -> None:
        try:
            self._abort()
        except Exception:
            # The blocked request will surface its own transport error. The
            # cancellation flag remains authoritative at the provider boundary.
            pass
