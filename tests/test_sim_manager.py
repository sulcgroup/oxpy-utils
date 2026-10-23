"""
Tests for SimulationManager's scheduling (process/GPU limits, failure handling) and its
CUDA MPS daemon lifecycle.

The scheduling tests use FakeSim stand-ins rather than real oxDNA runs, so they need neither
oxDNA nor a GPU: a fake simulation can sleep, report an error, or kill its own worker process,
and records what its worker saw. GPUs and nvidia-cuda-mps-control are faked the same way
(monkeypatched visible_gpus(), a stub nvidia-cuda-mps-control script on PATH); one test at the
end exercises the real daemon when this machine has one.
"""
import json
import os
import shutil
import signal
import stat
import subprocess as sp
import sys
import time
from pathlib import Path

import pytest

from oxpy_utils.oxdna_simulation import SimulationManager


class FakeOxpyRun:
    def __init__(self, sim: "FakeSim"):
        self.sim = sim
        self.error_message = None

    def run(self, subprocess=False, custom_observables=None):
        sim = self.sim
        start = time.time()
        (sim.sim_dir / "pid").write_text(str(os.getpid()))
        time.sleep(sim.duration)
        if sim.mode == "exit":
            os._exit(7)
        if sim.mode == "kill":
            os.kill(os.getpid(), signal.SIGKILL)
        if sim.mode == "error":
            self.error_message = f"fake oxDNA error in {sim.name}"
        else:
            (sim.sim_dir / "last_conf.dat").write_text("done")
        (sim.sim_dir / "worker.json").write_text(json.dumps({
            "start": start,
            "end": time.time(),
            "CUDA_device": sim.input.input_dict.get("CUDA_device"),
            "mps_pipe": os.environ.get("CUDA_MPS_PIPE_DIRECTORY"),
        }))


class FakeInput:
    def __init__(self):
        self.input_dict = {}


class FakeSim:
    """Just enough of Simulation for SimulationManager to schedule and run."""

    def __init__(self, sim_dir: Path, mode="ok", duration=0.0):
        self.sim_dir = sim_dir
        sim_dir.mkdir(parents=True, exist_ok=True)
        self.name = sim_dir.name
        self.mode = mode
        self.duration = duration
        self.input = FakeInput()
        self.oxpy_run = FakeOxpyRun(self)

    def input_file(self, updates: dict):
        self.input.input_dict.update(updates)

    def result(self) -> dict:
        return json.loads((self.sim_dir / "worker.json").read_text())

    def completed(self) -> bool:
        return (self.sim_dir / "last_conf.dat").exists()


@pytest.fixture
def make_manager():
    managers = []

    def make(**kwargs):
        kwargs.setdefault("sleep_time", 0.05)
        mgr = SimulationManager(**kwargs)
        managers.append(mgr)
        return mgr

    yield make
    for mgr in managers:
        try:
            mgr.shutdown()
        except Exception:
            pass
    # the manager keeps each MPS daemon's logs for debugging; their directory names start with
    # the creating process's pid, so these are exactly the ones this test run left behind
    for log_dir in Path("/tmp").glob(f"mps-log_{os.getpid()}_*"):
        shutil.rmtree(log_dir, ignore_errors=True)


@pytest.fixture(autouse=True)
def hang_guard():
    """A scheduling bug here shows up as the manager waiting forever: fail instead."""
    def on_alarm(signum, frame):
        raise TimeoutError("worker_manager() hung")

    previous = signal.signal(signal.SIGALRM, on_alarm)
    signal.alarm(60)
    yield
    signal.alarm(0)
    signal.signal(signal.SIGALRM, previous)


def overlap(results: list[dict]) -> int:
    """the largest number of the given worker runs that were in progress at the same moment"""
    events = sorted([(r["start"], 1) for r in results] + [(r["end"], -1) for r in results],
                    key=lambda e: (e[0], e[1]))
    peak = cur = 0
    for _, delta in events:
        cur += delta
        peak = max(peak, cur)
    return peak


class TestFailureHandling:
    def test_all_succeed(self, tmp_path, make_manager):
        mgr = make_manager(n_processes=2)
        sims = [FakeSim(tmp_path / f"s{i}") for i in range(4)]
        for sim in sims:
            mgr.queue_sim(sim)
        assert mgr.worker_manager(cpu_run=True) == []
        assert all(sim.completed() for sim in sims)

    @pytest.mark.parametrize("mode, expected", [("exit", "exit code 7"), ("kill", "SIGKILL")])
    def test_worker_dying_without_reporting_does_not_hang(self, tmp_path, make_manager, mode, expected):
        mgr = make_manager(n_processes=2)
        sims = [FakeSim(tmp_path / "ok0"), FakeSim(tmp_path / "bad", mode=mode),
                FakeSim(tmp_path / "ok1"), FakeSim(tmp_path / "ok2")]
        for sim in sims:
            mgr.queue_sim(sim)
        failures = mgr.worker_manager(cpu_run=True, run_when_failed=True)
        assert len(failures) == 1
        assert "bad" in failures[0] and expected in failures[0]
        assert [sim.completed() for sim in sims] == [True, False, True, True]

    def test_several_reported_failures_do_not_hang(self, tmp_path, make_manager):
        # each failing worker used to block forever putting its message on a 1-slot queue
        # nobody was reading, which also held its process slot
        mgr = make_manager(n_processes=2)
        sims = [FakeSim(tmp_path / f"s{i}", mode="error") for i in range(4)]
        for sim in sims:
            mgr.queue_sim(sim)
        failures = mgr.worker_manager(cpu_run=True, run_when_failed=True)
        assert len(failures) == 4
        assert all("fake oxDNA error" in f for f in failures)

    def test_failure_in_last_simulation_is_not_missed(self, tmp_path, make_manager):
        # once the queue is empty, the manager only waits for workers: it still has to notice
        mgr = make_manager(n_processes=4)
        sims = [FakeSim(tmp_path / "ok", duration=0.3), FakeSim(tmp_path / "bad", mode="error", duration=0.3)]
        for sim in sims:
            mgr.queue_sim(sim)
        with pytest.raises(SystemExit):
            mgr.worker_manager(cpu_run=True)

    def test_first_failure_stops_the_batch_by_default(self, tmp_path, make_manager):
        mgr = make_manager(n_processes=1)
        sims = [FakeSim(tmp_path / "bad", mode="error"), FakeSim(tmp_path / "never")]
        for sim in sims:
            mgr.queue_sim(sim)
        with pytest.raises(SystemExit):
            mgr.worker_manager(cpu_run=True)
        assert not sims[1].completed()

    def test_terminating_a_busy_worker_is_prompt(self, tmp_path, make_manager):
        # workers inherit worker_manager()'s SIGTERM handler, which can't run while a worker is
        # busy in C++ oxDNA code; they have to reset it to the default for terminate() to work
        mgr = make_manager(n_processes=2)
        sims = [FakeSim(tmp_path / "bad", mode="error"), FakeSim(tmp_path / "long", duration=30)]
        for sim in sims:
            mgr.queue_sim(sim)
        t = time.time()
        with pytest.raises(SystemExit):
            mgr.worker_manager(cpu_run=True)
        assert time.time() - t < 15

    def test_sigterm_handler_installed_only_during_batch(self, tmp_path, make_manager):
        before = signal.getsignal(signal.SIGTERM)
        mgr = make_manager(n_processes=1)
        assert signal.getsignal(signal.SIGTERM) is before  # not at construction
        mgr.queue_sim(FakeSim(tmp_path / "s"))
        mgr.worker_manager(cpu_run=True)
        assert signal.getsignal(signal.SIGTERM) is before


class TestConcurrencyLimits:
    def test_n_processes_limits_concurrency(self, tmp_path, make_manager):
        mgr = make_manager(n_processes=2)
        sims = [FakeSim(tmp_path / f"s{i}", duration=0.4) for i in range(6)]
        for sim in sims:
            mgr.queue_sim(sim)
        mgr.worker_manager(cpu_run=True)
        assert overlap([s.result() for s in sims]) == 2

    @pytest.fixture
    def two_gpus(self, monkeypatch):
        monkeypatch.setattr(SimulationManager, "visible_gpus", lambda self: [(0, 0), (1, 1)])
        monkeypatch.setattr(SimulationManager, "_free_memory_mb", lambda self, idx: 1000.0)

    def test_gpus_are_balanced_and_capped(self, tmp_path, make_manager, two_gpus):
        mgr = make_manager(n_processes=16, max_sims_per_gpu=2)
        sims = [FakeSim(tmp_path / f"s{i}", duration=0.6) for i in range(8)]
        for sim in sims:
            mgr.queue_sim(sim)
        mgr.worker_manager(use_mps=False)
        results = [s.result() for s in sims]
        by_gpu = {g: [r for r in results if r["CUDA_device"] == str(g)] for g in (0, 1)}
        assert [len(by_gpu[0]), len(by_gpu[1])] == [4, 4]
        assert overlap(by_gpu[0]) <= 2 and overlap(by_gpu[1]) <= 2
        # all four slots filled at once, i.e. the cap was reached rather than serialised
        assert overlap(results) == 4

    def test_cuda_device_reassigned_each_launch(self, tmp_path, make_manager, two_gpus):
        mgr = make_manager(n_processes=4)
        sim = FakeSim(tmp_path / "s")
        sim.input.input_dict["CUDA_device"] = "7"  # stale, e.g. from a previous run
        mgr.queue_sim(sim)
        mgr.worker_manager(use_mps=False)
        # `sim` here is our own copy (the manager's queue pickles); the worker wrote what it saw
        assert sim.result()["CUDA_device"] in ("0", "1")

    def test_pick_gpu(self, make_manager, monkeypatch):
        mgr = make_manager(n_processes=4)
        free = {0: 500.0, 1: 900.0, 2: 700.0}
        monkeypatch.setattr(mgr, "visible_gpus", lambda: [(0, 0), (1, 1), (2, 2)])
        monkeypatch.setattr(mgr, "_free_memory_mb", lambda idx: free[idx])
        # fewest running wins, regardless of free memory...
        assert mgr._pick_gpu({0: 1, 1: 2, 2: 1}, cap=None) == 2
        # ...with free memory only breaking ties
        assert mgr._pick_gpu({}, cap=None) == 1
        assert mgr._pick_gpu({1: 1}, cap=None) == 2
        # a full GPU is skipped, and None means wait when they're all full
        assert mgr._pick_gpu({0: 0, 1: 2, 2: 2}, cap=2) == 0
        assert mgr._pick_gpu({0: 2, 1: 2, 2: 2}, cap=2) is None

    def test_invalid_args(self):
        with pytest.raises(ValueError):
            SimulationManager(n_processes=2, max_sims_per_gpu=0)


class TestVisibleGpus:
    @pytest.fixture
    def mgr(self, make_manager, monkeypatch):
        mgr = make_manager(n_processes=1)
        monkeypatch.setattr(mgr, "get_num_gpus", lambda: 4)
        monkeypatch.setattr(mgr, "_gpu_uuid", lambda idx: f"GPU-{idx:08x}-aaaa")
        return mgr

    def test_all_gpus_by_default(self, mgr, monkeypatch):
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        assert mgr.visible_gpus() == [(0, 0), (1, 1), (2, 2), (3, 3)]

    def test_ordinal_is_position_in_cuda_visible_devices(self, mgr, monkeypatch):
        # slurm gave this job GPUs 2 and 3 of the node: CUDA_device must be 0 and 1, not 2 and 3
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3")
        assert mgr.visible_gpus() == [(0, 2), (1, 3)]

    def test_uuid_entries(self, mgr, monkeypatch):
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-00000003,GPU-00000001")
        assert mgr.visible_gpus() == [(0, 3), (1, 1)]

    def test_no_usable_gpus(self, mgr, monkeypatch):
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "9")
        with pytest.raises(RuntimeError):
            mgr.visible_gpus()


FAKE_MPS_CONTROL = """#!/bin/bash
# stand-in for nvidia-cuda-mps-control: a "daemon" is a file in the pipe directory
echo "$@ | pipe=$CUDA_MPS_PIPE_DIRECTORY" >> "$FAKE_MPS_LOG"
if [ "$1" = "-d" ]; then
    [ -n "$FAKE_MPS_START_FAILS" ] || touch "$CUDA_MPS_PIPE_DIRECTORY/alive"
    exit 0
fi
read -r cmd
echo "cmd: $cmd | pipe=$CUDA_MPS_PIPE_DIRECTORY" >> "$FAKE_MPS_LOG"
[ -e "$CUDA_MPS_PIPE_DIRECTORY/alive" ] || { echo "Cannot find MPS control daemon process"; exit 1; }
[ "$cmd" = "quit" ] && rm "$CUDA_MPS_PIPE_DIRECTORY/alive"
echo 100.0
"""


@pytest.fixture
def fake_mps(tmp_path, monkeypatch):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    script = bin_dir / "nvidia-cuda-mps-control"
    script.write_text(FAKE_MPS_CONTROL)
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    log = tmp_path / "mps_calls.log"
    log.touch()
    monkeypatch.setenv("PATH", f"{bin_dir}:{os.environ['PATH']}")
    monkeypatch.setenv("FAKE_MPS_LOG", str(log))
    monkeypatch.delenv("FAKE_MPS_START_FAILS", raising=False)
    monkeypatch.delenv("CUDA_MPS_PIPE_DIRECTORY", raising=False)
    monkeypatch.delenv("CUDA_MPS_LOG_DIRECTORY", raising=False)
    return log


class TestMpsLifecycle:
    def test_start_and_stop(self, make_manager, fake_mps):
        mgr = make_manager(n_processes=1)
        assert mgr.start_nvidia_cuda_mps_control() is True
        pipe_dir = mgr.mps_pipe_directory
        assert os.environ["CUDA_MPS_PIPE_DIRECTORY"] == pipe_dir
        assert Path(pipe_dir, "alive").exists()
        assert mgr.start_nvidia_cuda_mps_control() is True  # idempotent: no second daemon
        assert fake_mps.read_text().count("-d |") == 1

        mgr.stop_nvidia_cuda_mps_control()
        assert "cmd: quit" in fake_mps.read_text()
        assert not Path(pipe_dir).exists()  # sockets cleaned up, so /tmp doesn't fill with them
        assert "CUDA_MPS_PIPE_DIRECTORY" not in os.environ
        mgr.stop_nvidia_cuda_mps_control()  # safe to repeat

    def test_stop_restores_previous_environment(self, make_manager, fake_mps, monkeypatch, tmp_path):
        # a stale value that doesn't answer: we start our own daemon, then put theirs back
        monkeypatch.setenv("CUDA_MPS_PIPE_DIRECTORY", str(tmp_path / "stale"))
        monkeypatch.setenv("CUDA_MPS_LOG_DIRECTORY", str(tmp_path / "stale_log"))
        (tmp_path / "stale").mkdir()
        mgr = make_manager(n_processes=1)
        assert mgr.start_nvidia_cuda_mps_control() is True
        assert os.environ["CUDA_MPS_PIPE_DIRECTORY"] != str(tmp_path / "stale")
        mgr.stop_nvidia_cuda_mps_control()
        assert os.environ["CUDA_MPS_PIPE_DIRECTORY"] == str(tmp_path / "stale")
        assert os.environ["CUDA_MPS_LOG_DIRECTORY"] == str(tmp_path / "stale_log")

    def test_uses_but_does_not_stop_a_daemon_that_is_already_running(self, make_manager, fake_mps,
                                                                      monkeypatch, tmp_path):
        # e.g. started by the job script, as in the README
        pipe = tmp_path / "pipe"
        pipe.mkdir()
        (pipe / "alive").touch()
        monkeypatch.setenv("CUDA_MPS_PIPE_DIRECTORY", str(pipe))
        mgr = make_manager(n_processes=1)
        assert mgr.start_nvidia_cuda_mps_control() is True
        assert "-d |" not in fake_mps.read_text()
        mgr.stop_nvidia_cuda_mps_control()
        assert "cmd: quit" not in fake_mps.read_text()
        assert (pipe / "alive").exists()
        assert os.environ["CUDA_MPS_PIPE_DIRECTORY"] == str(pipe)

    def test_unresponsive_daemon_falls_back_without_mps(self, make_manager, fake_mps, monkeypatch):
        monkeypatch.setenv("FAKE_MPS_START_FAILS", "1")
        mgr = make_manager(n_processes=1)
        with pytest.warns(UserWarning, match="isn't responding"):
            assert mgr.start_nvidia_cuda_mps_control() is False
        assert mgr.mps_pipe_directory is None
        assert "CUDA_MPS_PIPE_DIRECTORY" not in os.environ

    def test_missing_binary_falls_back_without_mps(self, make_manager, monkeypatch, tmp_path):
        monkeypatch.setenv("PATH", str(tmp_path))
        mgr = make_manager(n_processes=1)
        with pytest.warns(UserWarning, match="not found on PATH"):
            assert mgr.start_nvidia_cuda_mps_control() is False

    def test_workers_see_the_daemon_and_batch_cleans_up(self, tmp_path, make_manager, fake_mps, monkeypatch):
        monkeypatch.setattr(SimulationManager, "visible_gpus", lambda self: [(0, 0)])
        monkeypatch.setattr(SimulationManager, "_free_memory_mb", lambda self, idx: 1000.0)
        mgr = make_manager(n_processes=2)
        sims = [FakeSim(tmp_path / f"s{i}") for i in range(3)]
        for sim in sims:
            mgr.queue_sim(sim)
        mgr.worker_manager()
        pipes = {sim.result()["mps_pipe"] for sim in sims}
        assert len(pipes) == 1 and None not in pipes
        assert not Path(pipes.pop()).exists()  # stopped and removed after the batch
        assert mgr.mps_pipe_directory is None

    def test_daemon_stopped_even_when_batch_fails(self, tmp_path, make_manager, fake_mps, monkeypatch):
        monkeypatch.setattr(SimulationManager, "visible_gpus", lambda self: [(0, 0)])
        monkeypatch.setattr(SimulationManager, "_free_memory_mb", lambda self, idx: 1000.0)
        mgr = make_manager(n_processes=2)
        mgr.queue_sim(FakeSim(tmp_path / "bad", mode="error"))
        with pytest.raises(SystemExit):
            mgr.worker_manager()
        assert "cmd: quit" in fake_mps.read_text()

    def test_no_daemon_for_cpu_runs(self, tmp_path, make_manager, fake_mps):
        mgr = make_manager(n_processes=1)
        mgr.queue_sim(FakeSim(tmp_path / "s"))
        mgr.worker_manager(cpu_run=True)
        assert fake_mps.read_text() == ""



CHILD_SCRIPT = """
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[2])
from test_sim_manager import FakeSim
from oxpy_utils.oxdna_simulation import SimulationManager
SimulationManager.visible_gpus = lambda self: [(0, 0)]
SimulationManager._free_memory_mb = lambda self, idx: 1000.0
mgr = SimulationManager(n_processes=2, sleep_time=0.05)
for i in range(2):
    mgr.queue_sim(FakeSim(Path(sys.argv[1]) / f"s{i}", duration=120))
mgr.worker_manager()
"""


def _process_alive(pid: int) -> bool:
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0] != "Z"
    except FileNotFoundError:
        return False


class TestSigterm:
    """what slurm's scancel / a time limit does to a running batch"""

    def test_sigterm_stops_workers_and_mps_daemon(self, tmp_path, fake_mps):
        child = sp.Popen([sys.executable, "-c", CHILD_SCRIPT, str(tmp_path / "sims"), str(Path(__file__).parent)])
        pid_files = [tmp_path / "sims" / f"s{i}" / "pid" for i in range(2)]
        try:
            deadline = time.time() + 30
            while not all(f.exists() for f in pid_files):
                assert time.time() < deadline, "workers never started"
                assert child.poll() is None, "batch ended early"
                time.sleep(0.1)
            worker_pids = [int(f.read_text()) for f in pid_files]
            assert all(_process_alive(pid) for pid in worker_pids)

            child.send_signal(signal.SIGTERM)
            assert child.wait(timeout=30) == 1
            deadline = time.time() + 10
            while any(_process_alive(pid) for pid in worker_pids):
                assert time.time() < deadline, "workers outlived the manager"
                time.sleep(0.1)
        finally:
            if child.poll() is None:
                child.kill()
        calls = fake_mps.read_text()
        assert "-d |" in calls and "cmd: quit" in calls  # daemon was started, and told to quit
        for log_dir in Path("/tmp").glob(f"mps-log_{child.pid}_*"):
            shutil.rmtree(log_dir, ignore_errors=True)
        assert not list(Path("/tmp").glob(f"mps-pipe_{child.pid}_*"))


@pytest.mark.skipif(shutil.which("nvidia-cuda-mps-control") is None or shutil.which("nvidia-smi") is None
                    or sp.run(["nvidia-smi", "-L"], capture_output=True).returncode != 0,
                    reason="needs a GPU and nvidia-cuda-mps-control")
def test_real_mps_daemon_starts_answers_and_stops(make_manager):
    mgr = make_manager(n_processes=1)
    assert mgr.start_nvidia_cuda_mps_control() is True
    pipe_dir, log_dir = mgr.mps_pipe_directory, mgr.mps_log_directory
    assert mgr._mps_daemon_responds(pipe_dir, log_dir)
    mgr.stop_nvidia_cuda_mps_control()
    assert not mgr._mps_daemon_responds(pipe_dir, log_dir)
    assert not Path(pipe_dir).exists()
