"""Tests for utils/oxlog.py: OxLogHandler shutdown (close, atexit, forked copies)."""
import logging
import multiprocessing
import subprocess
import sys
import textwrap
from pathlib import Path

from oxpy_utils.utils.oxlog import OxLogHandler, open_handlers

REPO = Path(__file__).resolve().parent.parent


def test_close_writes_queued_records_and_is_idempotent(tmp_path):
    h = OxLogHandler("t", False, tmp_path)
    logger = h.spinoff("oxlog_test_close")
    for i in range(50):
        logger.info(f"record {i}")
    h.close()
    h.close()
    lines = (tmp_path / "t.log").read_text().splitlines()
    assert len(lines) == 50 and lines[-1].endswith("record 49")
    assert h not in open_handlers


def close_copy(handler: OxLogHandler):
    handler.close()


def test_forked_copy_cannot_stop_parent_listener(tmp_path):
    h = OxLogHandler("f", False, tmp_path)
    p = multiprocessing.get_context("fork").Process(target=close_copy, args=(h,))
    p.start()
    p.join(timeout=30)
    assert p.exitcode == 0
    h.spinoff("oxlog_test_fork").info("after child close")
    h.close()
    assert "after child close" in (tmp_path / "f.log").read_text()


def test_handlers_alive_at_exit_shut_down_cleanly(tmp_path):
    # several handlers that have logged, still referenced when the interpreter exits: used to abort the
    # interpreter (listener threads erroring during finalization) and could drop queued records
    script = textwrap.dedent(f"""
        from pathlib import Path
        from oxpy_utils.utils.oxlog import OxLogHandler
        keep = [OxLogHandler(f"h{{i}}", False, Path({str(tmp_path)!r})) for i in range(8)]
        for i, h in enumerate(keep):
            h.spinoff(f"lg{{i}}").info("hello")
    """)
    for _ in range(3):
        r = subprocess.run([sys.executable, "-c", script], cwd=REPO, capture_output=True, text=True, timeout=120)
        assert r.returncode == 0, r.stderr[-2000:]
        assert "Exception in thread" not in r.stderr and "Fatal Python error" not in r.stderr, r.stderr[-2000:]
    for i in range(8):
        assert (tmp_path / f"h{i}.log").read_text().count("hello") == 3
