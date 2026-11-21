"""
Flux generation from a weighted pool of starting configurations.

FFSFluxGenerator equilibrates every worker from the single conf in its source directory. When the
basin A has slow internal degrees of freedom (e.g. a hairpin that rarely opens or closes on the
timescale of the flux run), one starting conf, or one per worker, under-samples A. This generator
keeps a pool of confs (which can be larger than the number of workers) and draws one, by weight,
every time a worker (re)starts from equilibration. With restart_after_success (the default here)
that is after every lambda_0 success, so the pool is sampled throughout the run.

Weights are typically chosen so that the drawn confs follow a known distribution of some slow
variable in A, e.g. a free-energy profile from umbrella sampling; see stratified_weights().

RestartOnFailFluxGenerator goes further: its pool confs are already past lambda_{-1}, and every
attempt (no equilibration) ends at lambda_0 or lambda_fail and restarts from a new draw. Use it when
the slow approach to lambda_{-1} is not of interest; its flux is biased accordingly.
"""
from __future__ import annotations

import csv
import logging
import random as rnd
import shutil
from collections import Counter
from pathlib import Path
from typing import Hashable, Mapping, Optional, Sequence, Union

from .base_flux_sampler import read_output, success_pattern
from .flux_generator import FFSFluxGenerator


def stratified_weights(labels: Sequence[Hashable],
                       label_probs: Mapping[Hashable, float]) -> list[float]:
    """
    Per-conf weights that make a weighted draw pick label L with probability label_probs[L]
    (renormalized over the labels that have confs), uniformly among the confs labelled L.

    weight_i = P(label_i) / count(label_i). Confs whose label is missing from label_probs get weight 0.
    Raises ValueError if no conf has a label with nonzero probability.

    Example: labels = hairpin bond count of each pool conf, label_probs = P(hairpin) from a
    free-energy profile, P(h) ∝ exp(-F(h)/kT).
    """
    counts = Counter(labels)
    weights = [float(label_probs.get(lab, 0.0)) / counts[lab] for lab in labels]
    if not any(w > 0 for w in weights):
        raise ValueError("no conf has a label with nonzero probability")
    return weights


class WeightedSourceFluxGenerator(FFSFluxGenerator):
    """
    FFSFluxGenerator that equilibrates each worker (re)start from a conf drawn by weight from a pool.

    Use set_source_pool() after source_directory is set (FFSProgram.use_source_pool does both).
    The chosen conf's name is written to source_conf.txt in each equilibration sim dir.
    """
    pool: list[str]  # conf file names inside self.source_directory
    pool_weights: list[float]

    def __init__(self, name: str, working_directory: Path, destination_dir: Path):
        super().__init__(name, working_directory, destination_dir)
        self.pool = []
        self.pool_weights = []
        self.restart_after_success = True

    def set_source_pool(self,
                        confs: Sequence[Union[str, Path]],
                        weights: Optional[Sequence[float]] = None):
        """
        confs: conf files. Files outside self.source_directory are copied into it (names must be
            unique); bare names are taken to be files already in it.
        weights: nonnegative, not necessarily normalized; None means uniform.
        """
        assert self.source_directory is not None, "set source_directory first"
        if weights is None:
            weights = [1.0] * len(confs)
        if len(weights) != len(confs):
            raise ValueError(f"{len(confs)} confs but {len(weights)} weights")
        if any(w < 0 for w in weights) or not any(w > 0 for w in weights):
            raise ValueError("weights must be nonnegative with at least one positive")
        self.source_directory.mkdir(parents=True, exist_ok=True)
        names = []
        for conf in confs:
            conf = Path(conf)
            if conf.parent == Path(".") or conf.parent.resolve() == self.source_directory.resolve():
                if not (self.source_directory / conf.name).is_file():
                    raise FileNotFoundError(f"{conf.name} not found in {self.source_directory}")
            else:
                shutil.copy(conf, self.source_directory / conf.name)
            names.append(conf.name)
        if len(set(names)) != len(names):
            raise ValueError("pool conf file names must be unique")
        self.pool = names
        self.pool_weights = [float(w) for w in weights]

    def choose_source_conf(self, rng: rnd.Random) -> Union[str, None]:
        if not self.pool:
            return None
        return rng.choices(self.pool, weights=self.pool_weights, k=1)[0]

    def write_pool(self, fname: str = "source_pool.csv"):
        """record the pool and normalized draw probabilities next to the flux output"""
        total = sum(self.pool_weights)
        with (self.working_directory / fname).open("w") as f:
            f.write("conf,weight,probability\n")
            for name, w in zip(self.pool, self.pool_weights):
                f.write(f"{name},{w},{w / total}\n")


ATTEMPT_LOG_COLUMNS = ["sim", "source_conf", "seed", "outcome", "steps", "success_index"]


def conf_time(conf: Path) -> int:
    """the step counter in the header of an oxDNA conf file"""
    with conf.open() as f:
        return int(float(f.readline().split("=")[1]))


class RestartOnFailFluxGenerator(WeightedSourceFluxGenerator):
    """
    Flux generator for when the approach to lambda_{-1} from deep in A is not of interest.

    Every attempt starts from a pool conf that is already past lambda_{-1} (e.g. confs saved where
    trajectories of an earlier flux run crossed lambda_{-1}) and runs until it crosses lambda_0
    (success) or lambda_fail (failure, by default back across lambda_{-1}). Either way the worker
    starts its next attempt from a freshly drawn pool conf, instead of equilibrating in A and waiting
    for the system to diffuse back across lambda_{-1}.

    This is a biased flux: time spent beyond lambda_fail is never simulated, so compute_flux() is not
    the flux out of A. It is the rate of reaching lambda_0 per step spent past lambda_{-1}, given the
    pool. success_probability() is the per-attempt probability of reaching lambda_0 before
    lambda_fail. Both are conditional on the pool's distribution of starting confs.

    Each worker appends one row per attempt to attempts_p<i>.csv in the working directory
    (ATTEMPT_LOG_COLUMNS), which compute_flux() and success_probability() read back, so the
    statistics survive restarts.
    """
    # draw fresh Maxwell-Boltzmann velocities at the start of each attempt, so repeated attempts from
    # the same pool conf diverge immediately
    refresh_velocities: bool = True

    def attempt_log(self, process_idx: int) -> Path:
        return self.working_directory / f"attempts_p{process_idx}.csv"

    def read_attempts(self) -> list[dict]:
        rows = []
        for log in sorted(self.working_directory.glob("attempts_p*.csv")):
            with log.open() as f:
                rows.extend(csv.DictReader(f))
        return rows

    def outcome_counts(self) -> Counter:
        return Counter(row["outcome"] for row in self.read_attempts())

    def success_probability(self) -> float:
        """successes / (successes + failures) over all logged attempts"""
        counts = self.outcome_counts()
        decided = counts["success"] + counts["failure"]
        return counts["success"] / decided if decided else float("nan")

    def compute_flux(self) -> float:
        """successes per step, counting the steps of every attempt (successful or not)"""
        rows = self.read_attempts()
        steps = sum(int(row["steps"]) for row in rows)
        successes = sum(row["outcome"] == "success" for row in rows)
        return successes / steps if steps else float("nan")

    def run(self):
        super().run()
        counts = self.outcome_counts()
        self.loghandler.spinoff("main").info(
            f"restart-on-fail attempts: {dict(counts)}; P(lambda_0 before fail) = {self.success_probability():.4g}")

    def ffs_process(self, process_idx: int, plogger: logging.Logger):
        if not self.pool:
            raise RuntimeError("RestartOnFailFluxGenerator needs a source pool; call set_source_pool() first")
        myrng = rnd.Random(self.initial_seed + process_idx)
        log_path = self.attempt_log(process_idx)
        is_new_log = not log_path.exists()
        with log_path.open("a", newline="") as log_file:
            log = csv.DictWriter(log_file, fieldnames=ATTEMPT_LOG_COLUMNS)
            if is_new_log:
                log.writeheader()
                sim_counter = 0
            else:
                # resuming: carry on numbering after the attempts already logged
                with log_path.open() as f:
                    sim_counter = sum(1 for _ in csv.DictReader(f))

            while self.success_count.value < self.desired_success_count:
                source_conf = self.choose_source_conf(myrng)
                seed = myrng.randint(1, int(5e6))
                sim_dir = self.tld() / f"p{process_idx}" / f"sim{sim_counter}"
                sim = self.make_ffs_simulation(self.source_directory, sim_dir, seed, {
                    "refresh_vel": int(self.refresh_velocities),
                    "restart_step_counter": 1,
                    "steps": 2e10,
                }, self.pass_or_fail, conf_file_name=source_conf)
                if self.update_queue is not None:
                    self.update_queue.put(("flux", process_idx, sim_counter, "flux_fwd", "origin"))
                sim.oxpy_run.run(subprocess=False)
                if sim.oxpy_run.error_message:
                    raise Exception(sim.oxpy_run.error_message)

                op_values = read_output(sim)
                success = self.lambda_n.test(op_values[self.lambda_n.op.name])
                failure = self.lambda_fail.test(op_values[self.lambda_fail.op.name])
                last_conf = sim.sim_dir / sim.input.input_dict["lastconf_file"]
                success_index = ""
                if success and not failure:
                    outcome = "success"
                    with self.success_lock:
                        self.success_count.value += 1
                        success_index = self.success_count.value
                        shutil.copy(last_conf,
                                    self.destination_directory / success_pattern.replace("*", str(success_index)))
                        if self.update_queue is not None:
                            self.update_queue.put(("CPY_CONF", self.name, process_idx, sim_counter,
                                                   success_index - 1))  # zero indexed
                    plogger.info(f"Worker {process_idx}: SUCCESS from {source_conf}")
                elif failure and not success:
                    outcome = "failure"
                else:
                    outcome = "undetermined"
                    plogger.warning(f"Worker {process_idx}: attempt from {source_conf} ended with "
                                    f"neither or both of success/failure: {op_values}")
                if self.update_queue is not None:
                    self.update_queue.put(("flux_report", process_idx, sim_counter, outcome == "success"))

                log.writerow({"sim": sim_dir.name, "source_conf": source_conf, "seed": seed, "outcome": outcome,
                              "steps": conf_time(last_conf), "success_index": success_index})
                log_file.flush()
                if not self.keep_sim_dirs:
                    shutil.rmtree(sim_dir)
                sim_counter += 1
        plogger.info(f"Achieved desired success count, process #{process_idx} terminating....")
