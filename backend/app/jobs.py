"""In-process job runner with progress broadcast.

Stage A (generation) takes minutes and Stage B (assembly) takes seconds, but
both need to stream progress to the browser. A job is just a coroutine plus a
progress channel; the WebSocket layer subscribes to it.

Deliberately in-process and in-memory: this is a single-GPU desktop tool, not a
cluster. Swap for a real queue only if that changes.
"""

from __future__ import annotations

import asyncio
import logging
import traceback
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

log = logging.getLogger(__name__)


class JobState(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Job:
    id: str
    kind: str
    state: JobState = JobState.PENDING
    progress: float = 0.0
    message: str = ""
    result: Any = None
    error: str | None = None
    traceback: str | None = None
    _subscribers: list[asyncio.Queue] = field(default_factory=list, repr=False)
    _task: asyncio.Task | None = field(default=None, repr=False)

    def snapshot(self) -> dict:
        return {
            "id": self.id,
            "kind": self.kind,
            "state": self.state.value,
            "progress": round(self.progress, 4),
            "message": self.message,
            "result": self.result,
            "error": self.error,
            "traceback": self.traceback,
        }

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=64)
        self._subscribers.append(q)
        q.put_nowait(self.snapshot())
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        if q in self._subscribers:
            self._subscribers.remove(q)

    def publish(self) -> None:
        snap = self.snapshot()
        for q in list(self._subscribers):
            # Drop the oldest update rather than the newest, so a slow client
            # always converges on current state instead of lagging behind.
            # Backpressure, not error suppression: no failure is being hidden.
            while q.full():
                q.get_nowait()
            q.put_nowait(snap)


class JobManager:
    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        # One GPU: serialise generation so jobs can't fight over VRAM.
        self._gpu_lock = asyncio.Lock()

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def list(self) -> list[dict]:
        return [j.snapshot() for j in self._jobs.values()]

    def submit(self, kind: str,
               fn: Callable[[Job], Awaitable[Any]],
               exclusive: bool = False) -> Job:
        """Run ``fn`` as a job.

        Args:
            exclusive: take the GPU lock for the duration (Stage A work).
        """
        job = Job(id=uuid.uuid4().hex[:12], kind=kind)
        self._jobs[job.id] = job

        async def runner() -> None:
            try:
                if exclusive:
                    async with self._gpu_lock:
                        job.state = JobState.RUNNING
                        job.publish()
                        job.result = await fn(job)
                else:
                    job.state = JobState.RUNNING
                    job.publish()
                    job.result = await fn(job)
                job.state = JobState.DONE
                job.progress = 1.0
            except asyncio.CancelledError:
                job.state = JobState.CANCELLED
                job.publish()
                raise
            except Exception as exc:  # noqa: BLE001
                # Not a fallback: the job still fails, and the full traceback
                # goes to both the log and the client. Catching here only stops
                # the exception vanishing into a detached asyncio task.
                job.state = JobState.FAILED
                job.error = f"{type(exc).__name__}: {exc}"
                job.traceback = traceback.format_exc()
                log.error("job %s (%s) failed:\n%s", job.id, kind, job.traceback)
            finally:
                job.publish()

        job._task = asyncio.create_task(runner())
        return job

    def cancel(self, job_id: str) -> bool:
        job = self._jobs.get(job_id)
        if job and job._task and not job._task.done():
            job._task.cancel()
            return True
        return False


def progress_reporter(job: Job, loop: asyncio.AbstractEventLoop,
                      lo: float = 0.0, hi: float = 1.0):
    """Build a thread-safe progress callback for blocking render/generate code.

    The renderer runs in a worker thread; this marshals its callbacks back onto
    the event loop so subscribers get updates without touching asyncio from the
    wrong thread. ``lo``/``hi`` map a sub-task onto a slice of overall progress.
    """

    def report(frac: float, message: str = "") -> None:
        def apply() -> None:
            job.progress = lo + (hi - lo) * max(0.0, min(1.0, frac))
            if message:
                job.message = message
            job.publish()

        loop.call_soon_threadsafe(apply)

    return report


manager = JobManager()
