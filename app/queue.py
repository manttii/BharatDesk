"""Small queue interface with an in-process asyncio implementation (no Redis)."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Generic, Protocol, TypeVar

import structlog

log = structlog.get_logger()

T = TypeVar("T")
T_contra = TypeVar("T_contra", contravariant=True)


class Queue(Protocol[T_contra]):
    async def put(self, item: T_contra) -> None: ...


class InProcessQueue(Generic[T]):
    """One worker, so a chat's messages are handled in arrival order. A handler error is
    logged and never kills the worker."""

    def __init__(self, handler: Callable[[T], Awaitable[None]], maxsize: int = 1000) -> None:
        self._handler = handler
        self._q: asyncio.Queue[T] = asyncio.Queue(maxsize=maxsize)
        self._task: asyncio.Task[None] | None = None

    async def put(self, item: T) -> None:
        self._q.put_nowait(item)  # raises QueueFull -> caller returns 503, Telegram retries

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._run(), name="inprocess-queue")

    async def join(self) -> None:
        await self._q.join()

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
            self._task = None

    async def _run(self) -> None:
        while True:
            item = await self._q.get()
            try:
                await self._handler(item)
            except Exception:
                log.exception("queue_handler_failed")
            finally:
                self._q.task_done()
