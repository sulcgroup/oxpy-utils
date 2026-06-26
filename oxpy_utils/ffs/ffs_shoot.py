import glob
import logging
import random
import shutil
from multiprocessing import Array, Process, Value, Lock
from pathlib import Path
import time
from typing import Union


from .base_flux_sampler import BaseFluxSampler, success_pattern, read_output, success_regex
from .ffs_interface import FFSInterface, write_order_params, Condition
from ..oxdna_simulation import Simulation, find_top_file
from ..utils.oxlog import OxLogHandler

import pandas as pd
import numpy as np

undetermined_pattern = './undefin_*.dat'

# for debugging
import threading


class FFSShooter(BaseFluxSampler):
    # whether to keep undetermined shoot results
    keep_undetermined: bool

    # list of starting conf files
    starting_confs: list[str]

    # success count is in superclass
    undetermined_count: Value
    undetermined_lock: Lock

    # we can't construct arrays until we have starting conf count
    # array of successes from each start conf
    success_from: Union[None, Array]
    # array of attempts from each start conf
    attempt_from: Union[None, Array]

    def __init__(self,
                 name: str,
                 working_directory: Path,
                 start_conf_dir: Path,
                 end_conf_dir: Path):
        super().__init__(name, working_directory, end_conf_dir)
        # assert start_conf_dir.is_dir(), f"No such directory `{str(start_conf_dir)}`!"
        self.keep_undetermined = False
        self.source_directory = start_conf_dir

        self.undetermined_count = Value("i", 0)
        self.undetermined_lock = Lock()

        self.attempt_lock = Lock()

        self.success_from = None
        self.attempt_from = None

        self.lambda_fail = None
        self.lambda_plus1 = None

    def init(self):
        super().init()

        self.starting_confs = sorted(
            glob.glob(str(self.source_directory / success_pattern)),
            key=lambda p: int(success_regex.search(p).group(1))
        )

        assert len(
            self.starting_confs) > 0, f"No files matching pattern {success_pattern} found in source directory {str(self.source_directory)}!"

        self.success_from = Array('i', len(self.starting_confs))  # zeroed by default
        self.attempt_from = Array('i', len(self.starting_confs))  # zeroed by default

        # load successes from csv
        if self.initial_success_count == 0:
            for i in range(len(self.starting_confs)):
                self.success_from[i] = 0
                self.attempt_from[i] = 0
        else:
            # load csv file with existing successes data
            successes_data: pd.DataFrame = pd.read_csv(self.destination_directory / f"{self.name}_success_log.csv")

            for i, row in successes_data.iterrows():
                file_name, n_attempts, n_successes = row.values[1:]
                # do we actually need this assert
                conf_file_path = f"{Path(self.starting_confs[i]).parent.name}/{Path(self.starting_confs[i]).name}"
                file_path = f"{Path(file_name).parent.name}/{Path(file_name).name}"
                assert conf_file_path == file_path, \
                    f"Mismatch between start conf name {self.starting_confs[i]} and logged file name {file_name}"
                self.success_from[i] = int(n_successes)
                self.attempt_from[i] = int(n_attempts)

    def set_interfaces(self,
                       lambda_f: FFSInterface,
                       lambda_m: FFSInterface
                       ):
        self.lambda_fail = lambda_f
        self.lambda_plus1 = lambda_m

        # only one condition, can init it at runtime

    def run(self):
        assert self.desired_success_count > 1

        assert len(self.starting_confs) > 0, "No starting confs found! you probably forgot to run init()"
        assert self.success_from is not None
        assert self.attempt_from is not None
        main_logger = self.loghandler.spinoff("main")
        main_logger.info(f"Starting {self.name} procedure")
        if self.desired_success_count <= self.success_count.value:
            main_logger.info("We already have enough successes! moving on")
            return
        # make directory for shoots
        # (self.tld() / self.shoot_name).mkdir()
        # find top file in tld
        top_file_name = find_top_file(self.source_directory).name
        # copy topology to source directory
        shutil.copy(self.source_directory / top_file_name,
                    self.destination_directory)

        processes = []
        main_logger.info(f"Found {len(self.starting_confs)} starting confs")
        tp = Process(target=self.timer)
        tp.start()

        # TODO: intermittantly write success count / attempts from each

        if self.ncpus > 1:
            for i in range(self.ncpus):
                p = Process(target=self.ffs_process, args=(i, self.loghandler.spinoff(f"Worker{i}")))
                processes.append(p)

            main_logger.info("starting processes...")
            for p in processes:
                p.start()

            main_logger.info("waiting for processes to finish")
            for p in processes:
                p.join()
        # for debugging purposes, allow single-thread executipn
        else:
            self.ffs_process(0, self.loghandler.spinoff("worker"))

        main_logger.info("Terminating timer")
        tp.terminate()  # terminate timer

        nsuccesses = self.success_count.value - self.initial_success_count
        assert nsuccesses == sum(self.success_from)
        main_logger.info("## log of successes probabilities from each starting conf")
        self.save_success_info()
        main_logger.info(f"# {self.name} SUMMARY")
        success_prob = nsuccesses / float(sum(self.attempt_from))
        main_logger.info(f"# nsuccesses: {nsuccesses} nattempts: {sum(self.attempt_from)} success_prob: {success_prob}"
                         f" undetermined: {self.undetermined_count.value}")

    def log_time(self, logger: logging.Logger, start_time: float):
        super().log_time(logger, start_time)
        # every 5 mins save success info
        # this method should only run every 10s so don't just use int(s)
        if int(time.time() - start_time) // 10 % 30 == 0:
            logger.info(f"Saving attempt & success info...")
            self.save_success_info()

    def save_success_info(self):
        with self.attempt_lock:
            successes_data = pd.DataFrame([{
                "File": self.starting_confs[i],
                "Attempts": self.attempt_from[i],
                "Successes": success_count
            } for i, success_count in enumerate(self.success_from)])
        successes_data.to_csv(self.destination_directory / f"{self.name}_success_log.csv")

    def load_success_info(self):
        if not (self.destination_directory / f"{self.name}_success_log.csv").exists():
            raise FileNotFoundError(f"No success log found at {str(self.destination_directory / f'{self.name}_success_log.csv')}")
        filepath = str(self.destination_directory / f"{self.name}_success_log.csv")
        successes_data = pd.read_csv(filepath, index_col=0)

        self.starting_confs = successes_data["File"].tolist()
        self.attempt_from = successes_data["Attempts"].tolist()
        self.success_from = successes_data["Successes"].tolist()
        self.success_count.value = sum(self.success_from) # todo: get rid of initial_success_count and just use this method to load from csv if it exists

    def ffs_process(self, process_idx: int, plogger: logging.Logger):
        plogger.info(f"Worker {process_idx} started")
        # sim counter is process-specific
        sim_counter = 0
        # while we haven't reached desired success count yet
        while self.success_count.value < self.desired_success_count:
            # choose a starting configuration index
            conf_index: int = random.choice(list(range(0, len(self.starting_confs))))
            conf_file = Path(self.starting_confs[conf_index]).name

            plogger.info(f"Chose starting configuration {conf_file}")

            # iter attempt count
            with self.attempt_lock:
                self.attempt_from[conf_index] += 1
                sim_dir = self.tld() / f"p{process_idx}" / f"sim{sim_counter}"
                seed = self.initial_seed + sum(self.attempt_from)

            sim = self.make_ffs_simulation(conf_file, sim_dir, seed)
            plogger.info("Executing shoot....")
            tstart = time.time()
            # tell process queue that shoot is starting
            # id for graph node
            if self.update_queue is not None:
                self.update_queue.put((
                    "shoot",
                    process_idx,
                    sim_counter,
                    self.name,
                    conf_index
                ))
            sim.oxpy_run.run(subprocess=False)
            plogger.info(f"Shoot complete. Execution time: {((time.time() - tstart) / 1000):.1f}ms")
            if sim.oxpy_run.error_message:
                raise Exception(sim.oxpy_run.error_message)

            op_values = read_output(sim)
            # success = is last conf past lambda+1
            success = self.lambda_plus1.test(op_values[self.lambda_plus1.op.name])
            # failure = is last conf behind lambda_fail
            failure = self.lambda_fail.test(op_values[self.lambda_fail.op.name])

            if success and not failure:
                with self.success_lock:
                    self.success_count.value += 1
                    self.success_from[conf_index] += 1
                    shutil.copy(
                        f"{sim.sim_dir}/{sim.input.input_dict['lastconf_file']}",
                        self.destination_directory / f"success_{str(self.success_count.value)}.dat"
                    )
                    plogger.info(f"SUCCESS: starting from conf_index {conf_index} and seed {seed}")
                    if self.update_queue is not None:
                        self.update_queue.put((
                            "CPY_CONF",
                            self.name,
                            process_idx,
                            sim_counter,
                            self.success_count.value - 1 # index from 0
                        ))
                        # command type, process idx, sim idx, success idx, success value
                        self.update_queue.put((
                            "shoot_report",
                            process_idx,
                            sim_counter,
                            self.name,
                            True
                        ))
            elif not success and failure:
                # do else
                plogger.info(f"FAILURE: starting from conf_index {process_idx} and seed {seed}")
                if self.update_queue is not None:
                    self.update_queue.put((
                        "shoot_report",
                        process_idx,
                        sim_counter,
                        self.name,
                        False
                    ))
            else:
                # do undetermined
                sim_log_file = sim.sim_dir / sim.input.input_dict["log_file"]
                with sim_log_file.open("r") as f:
                    txt = f.read()
                plogger.info(f"UNDETERMINED: starting from conf_index {conf_index} and seed {seed}"
                             f"\n{txt}")
                with self.undetermined_lock:
                    self.undetermined_count.value += 1
                    if self.keep_undetermined:
                        shutil.copy(
                            f"{sim.sim_dir}/{sim.input.input_dict['lastconf_file']}",
                            f"{undetermined_pattern + str(self.undetermined_count.value)}.dat"
                        )
                if self.update_queue is not None:
                    self.update_queue.put((
                        "shoot_report",
                        process_idx,
                        sim_counter,
                        self.name,
                        "undetermined"
                    ))
            if not self.keep_sim_dirs:
                shutil.rmtree(sim_dir)
                plogger.info(f"Deleted temporary simulation directory {str(sim_dir)}")
            sim_counter += 1

        plogger.info(f"Reached desired success count {self.desired_success_count}. Worker {process_idx} returning")
        threads = threading.enumerate()
        # if sending stuff to an update queue
        if self.update_queue is not None:
            try:
                # this prevents the process from trying to join the feeder thread on exit
                self.update_queue.cancel_join_thread()
                # close the writer-side in THIS process (safe; other processes can still use the queue)
                self.update_queue.close()
            except Exception as e:
                plogger.warning(f"Queue cleanup failed: {e}")

        plogger.info(f"[Worker {process_idx}] Alive threads at exit: {[t.name for t in threads]}")

    def make_ffs_simulation(self,
                            start_conf: str,
                            sim_dir: Path,
                            seed: int,
                            input_paras=dict) -> Simulation:
        assert self.lambda_fail is not None, "Fail interface has not been set!"
        assert self.lambda_plus1 is not None, "Lambda+1 interface has not been set!"
        sim = Simulation(self.source_directory, sim_dir)
        sim.build_sim.conf_file_name = start_conf
        sim.build(clean_build='force')

        sim.input.swap_default_input(self.default_input_name)
        sim.input["seed"] = seed
        sim.input["restart_step_counter"] = 0
        sim.input["steps"] = 2e10  # as good as forever
        assert (self.source_directory / start_conf).exists()

        # write order parameters file
        ffs_condition = Condition("shoot_condition", [self.lambda_fail, self.lambda_plus1])
        write_order_params(sim.sim_dir / "op.txt", *ffs_condition.get_order_params())
        # write ffs condition file
        ffs_condition.write(sim.sim_dir)
        sim.input["ffs_file"] = ffs_condition.file_name()
        sim.input["order_parameters_file"] = "op.txt"
        sim.input(**self.input_file_params)

        if self.seq_dependant:
            sim.make_sequence_dependant()

        return sim

    def is_complete(self) -> bool:
        return self.success_count >= self.desired_success_count
