"""Fan-out of pipeline packets to websocket clients.

The pipeline thread publishes; each client has a queue of size 1 where the
newest packet replaces an unread one — a slow client gets fewer frames but
never a growing delay.
"""

from __future__ import annotations

import asyncio
import threading


class Broadcaster:
    def __init__(self):
        self._loop: asyncio.AbstractEventLoop | None = None
        self._queues: set[asyncio.Queue] = set()
        self._lock = threading.Lock()

    def attach(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def register(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=1)
        with self._lock:
            self._queues.add(q)
        return q

    def unregister(self, q: asyncio.Queue) -> None:
        with self._lock:
            self._queues.discard(q)

    @property
    def client_count(self) -> int:
        with self._lock:
            return len(self._queues)

    def publish(self, data: bytes) -> None:
        """Called from the pipeline thread."""
        loop = self._loop
        if loop is None or loop.is_closed():
            return
        with self._lock:
            queues = list(self._queues)
        if queues:
            loop.call_soon_threadsafe(self._offer_all, queues, data)

    @staticmethod
    def _offer_all(queues: list[asyncio.Queue], data: bytes) -> None:
        for q in queues:
            if q.full():
                try:
                    q.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            try:
                q.put_nowait(data)
            except asyncio.QueueFull:
                pass


broadcaster = Broadcaster()
