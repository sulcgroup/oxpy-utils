"""
Tests for utils/sync_queue.py.

The scenario that motivated SyncQueue (FFS logging and process-graph updates stalling for hours, 2026-11-24): a
worker process puts to a shared multiprocessing queue and then holds its GIL in a long C call (oxpy running a
simulation). With multiprocessing.Queue, that worker's feeder thread can take the queue's cross-process write lock
and then wait for its own GIL until the call ends, blocking every other process's writes. Here the long call is a
sort of a large list of floats, which holds the GIL for its whole duration.
"""
import multiprocessing
import queue
import random
import time

import pytest

from oxpy_utils.utils.oxlog import OxLogHandler
from oxpy_utils.utils.sync_queue import SyncQueue

CTX = multiprocessing.get_context("fork")
HOG_ITEMS = 4_000_000  # one sort holds the GIL ~1 s on a typical machine; the hog does two


def test_put_get_and_empty():
    q = SyncQueue()
    assert q.empty()
    q.put(1)
    q.put_nowait({"a": 2})
    assert not q.empty()
    assert q.get() == 1
    assert q.get(timeout=1) == {"a": 2}
    with pytest.raises(queue.Empty):
        q.get(block=False)
    with pytest.raises(queue.Empty):
        q.get(timeout=0.05)


def put_from_child(q, n):
    for i in range(n):
        q.put(("child", i))


def test_forked_writers_arrive_in_order_per_writer():
    q = SyncQueue()
    procs = [CTX.Process(target=put_from_child, args=(q, 200)) for _ in range(4)]
    for p in procs:
        p.start()
    got = [q.get(timeout=30) for _ in range(800)]
    for p in procs:
        p.join(timeout=30)
        assert p.exitcode == 0
    assert sorted(i for _, i in got) == sorted(list(range(200)) * 4)


def gil_hog_writer(q, hogging):
    data = [random.random() for _ in range(HOG_ITEMS)]
    q.put(("A", time.time()))  # with multiprocessing.Queue: A's feeder thread now waits for the held write lock
    time.sleep(0.3)
    hogging.set()
    sorted(data)  # GIL held throughout, like OxpyManager.run_complete()
    sorted(data)


def quick_writer(q):
    q.put(("B", time.time()))


def delay_of_other_writer(q, wlock, sync: bool) -> float:
    """seconds between B's put and its arrival while A holds its GIL in a long C call (see module docstring)"""
    wlock.acquire()  # as if some other process were mid-send
    hogging = CTX.Event()
    a = CTX.Process(target=gil_hog_writer, args=(q, hogging))
    a.start()
    if sync:
        time.sleep(0.2)
        wlock.release()  # A's own put() was waiting for the lock: it sends before A starts its long call
        hogging.wait(timeout=30)
        time.sleep(0.3)
    else:
        hogging.wait(timeout=30)
        time.sleep(0.3)  # A is inside sorted(), holding its GIL
        wlock.release()  # A's feeder takes the lock, then needs A's GIL
    time.sleep(0.2)
    b = CTX.Process(target=quick_writer, args=(q,))
    b.start()
    t_put = time.time()
    arrived = {}
    while len(arrived) < 2:
        name, _ = q.get(timeout=60)
        arrived[name] = time.time()
    a.join(timeout=60)
    b.join(timeout=60)
    return arrived["B"] - t_put


def test_writer_holding_its_gil_does_not_stall_other_writers():
    q = SyncQueue()
    assert delay_of_other_writer(q, q._wlock, sync=True) < 0.5


@pytest.mark.xfail(reason="documents the multiprocessing.Queue behaviour SyncQueue avoids", strict=False)
def test_multiprocessing_queue_stalls_in_the_same_scenario():
    q = CTX.Queue()
    assert delay_of_other_writer(q, q._wlock, sync=False) < 0.5


def log_then_hog(handler, hogging):
    data = [random.random() for _ in range(HOG_ITEMS)]
    handler.spinoff("synctest_A").info("A before")
    hogging.set()
    sorted(data)
    sorted(data)


def log_quickly(handler):
    handler.spinoff("synctest_B").info("B quick")


def test_oxloghandler_records_arrive_while_another_worker_holds_its_gil(tmp_path):
    h = OxLogHandler("s", False, tmp_path)
    assert isinstance(h.log_queue, SyncQueue)
    hogging = CTX.Event()
    a = CTX.Process(target=log_then_hog, args=(h, hogging))
    a.start()
    hogging.wait(timeout=30)
    b = CTX.Process(target=log_quickly, args=(h,))
    b.start()
    b.join(timeout=30)
    deadline = time.time() + 1.0
    text = ""
    while time.time() < deadline and "B quick" not in text:
        time.sleep(0.05)
        text = (tmp_path / "s.log").read_text()
    a_still_hogging = a.is_alive()
    a.join(timeout=60)
    h.close()
    assert "B quick" in text and "A before" in text
    assert a_still_hogging, "the hog finished too early for this test to show anything"
