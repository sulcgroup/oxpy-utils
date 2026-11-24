"""
A multiprocessing queue whose put() sends synchronously, in the calling thread.

Why not multiprocessing.Queue: its put() only buffers the object; a background "feeder" thread sends it,
holding the queue's write lock, a semaphore shared by every process that writes to the queue. A feeder blocked
on that lock has released its process's GIL; when it gets the lock it must take the GIL back before it can send
and release the lock. If that process's main thread has meanwhile entered a long C call that holds the GIL, the
feeder holds the shared lock for that whole call and every other process's writes to the queue stall behind it.
oxpy's OxpyManager.run_complete() holds the GIL for a whole simulation, so in FFS (dozens of worker processes
logging and sending process-graph updates while running oxpy) logs and graph events stalled for hours, released
about one record per finished simulation (2026-11-24).

SyncQueue.put() takes the write lock, sends and releases it in the calling thread, which already holds its
process's GIL, so a writer never holds the lock while waiting for its own GIL. A send to a full pipe blocks until
the reader drains it: the queue must have a reader (QueueListener thread, graph updater thread) for as long as
writers use it.

Provides what logging.handlers.QueueHandler / QueueListener and the FFS graph updater use: put, put_nowait,
get(block, timeout) raising queue.Empty, empty, close.
"""
import multiprocessing
import queue
from multiprocessing.queues import SimpleQueue


class SyncQueue(SimpleQueue):
    def __init__(self, ctx=None):
        super().__init__(ctx=ctx if ctx is not None else multiprocessing.get_context())

    def put_nowait(self, obj):
        """same as put() (used by logging.handlers.QueueHandler); sends immediately"""
        self.put(obj)

    def get(self, block: bool = True, timeout: float = None):
        """
        Remove and return the next object. block=False or a timeout: raise queue.Empty if nothing arrives in
        time. Meant for a single reader per queue (as QueueListener and the FFS graph updater are): with several
        readers, one that saw data in poll() could block in the read if another took it first.
        """
        if not block:
            timeout = 0
        if timeout is not None and not self._poll(timeout):
            raise queue.Empty
        return super().get()
