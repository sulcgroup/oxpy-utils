import glob
import multiprocessing
import re
import time
from abc import ABC, abstractmethod
from multiprocessing import Lock, Value
from pathlib import Path
import logging
from typing import Union

from .ffs_interface import FFSInterface
from ..oxdna_simulation import Simulation, SimulationManager
from ..utils.oxlog import OxLogHandler

success_pattern = './success_*.dat'
success_regex = re.compile(r'success_(\d+)\.dat$')

class BaseFluxSampler(ABC):
    """
    abstract base class that functions as the base-class for both
    the initial-flux class and shoot class
    """
    T: float # todo ditch this for input_file_params
    name: str  # name of processor, used to label outputs & files

    desired_success_count: int
    initial_seed: int
    ncpus: int
    initial_success_count: int

    success_lock: Lock
    success_count: Value

    loghandler: OxLogHandler

    # the "working directory" is the top-level directory
    # for the location where simulations will actually be run
    working_directory: Path

    mgr: SimulationManager

    # parameters to add to input files
    # put temperature here, as well as others
    input_file_params: dict

    # directory where starting confs originate
    source_directory: Path
    # directory where completed confs will be deposited
    destination_directory: Path

    # fail interface = system moves across lambda_neg1 going backwards.
    # in the case of initial flux, this will be physically impossible in some systems
    # whether or not this interface includes the process start condition itself
    # depends on whether we are generating initial flux or shooting
    lambda_fail: FFSInterface

    # interface we are trying to reach within this step of the ffs
    lambda_plus1: FFSInterface

    # name of default input file (from ipy_oxDNA/defaults/input/) used for ffs process
    default_input_name: str

    update_queue: Union[multiprocessing.Queue, None]

    seq_dependant: bool

    # whether to preserve all simulation directories
    keep_sim_dirs: bool = False

    def __init__(self,
                 name: str,
                 working_directory: Path,
                 destination_dir: Path  # provide this here so we can count initial successes
                ):
        """
        constructor
        trying to minimize amount of stuff in the constructor
        """
        self.seq_dependant = True
        self.name = name
        self.working_directory = working_directory
        self.desired_success_count = -1 # obvious bullshit number
        self.initial_success_count = -1 # will be set in init

        self.initial_seed = int(time.time() + 0)
        # verbose = False
        self.ncpus = 1

        self.success_lock = Lock()
        self.success_count = Value('i')
        self.destination_directory = destination_dir
        # self.mgr = SimulationManager(self.ncpus) todo: use and reimplement

        # can set verbose later
        self.loghandler = OxLogHandler(self.name, False, working_directory)
        self.lambda_neg1 = None # will need to be set before shooting

        self.default_input_name = "ffs"

        # default value. we can't put this in input defaults b/c we need to call make_sequence_dependant
        self.seq_dependant = True

        # passing data back to main process is optional
        self.update_queue = None

        self.input_file_params = {}


    def init(self):
        """
        reads any existing data and assigns vars required for run
        """
        self.initial_success_count = len(glob.glob(str(self.destination_directory / success_pattern)))
        self.destination_directory.mkdir(exist_ok=True)
        self.success_count.value = self.initial_success_count

    def set_num_cpus(self, newVal: int):
        self.ncpus = newVal

    def set_initial_seed(self, newval: int):
        self.initial_seed = newval

    def set_desired_success_count(self, newVal: int):
        self.desired_success_count = newVal

    def alter_initial_seed(self, change: int):
        self.initial_seed += change

    def tld(self) -> Path:
        return self.working_directory

    def set_tld(self, new_path: Union[Path,str]):
        if isinstance(new_path, str):
            self.set_tld(Path(new_path))
        else:
            self.working_directory = new_path

    def set_working_directory(self, d: Path):
        self.working_directory = d

    @abstractmethod
    def set_interfaces(self, *args: tuple[Union[FFSInterface, None], ...]):
        raise NotImplementedError("you need to implement this")

    @abstractmethod
    def run(self):
        raise NotImplementedError("for the love of god override this abstract base-class method")

    @abstractmethod
    def ffs_process(self, idx: int, plogger: logging.Logger):
        raise NotImplementedError("you need to implement this")

    @abstractmethod
    def make_ffs_simulation(self,
                            origin,
                            sim_dir: Path,
                            seed: int,
                            input_params, *args, **kwargs) -> Simulation:
        """
        constructs a simulation as part of whatever it is we're doing here
        """
        raise NotImplementedError("you need to implement this method")


    # timer function: it spits out things
    def timer(self):
        """
        builds a timer process and logs time + status every 10 seconds
        todo: customized sleep period
        """
        logger = self.loghandler.spinoff("timer")
        logger.info(f"Timer started at {(time.asctime(time.localtime()))}")
        itime = time.time()
        while True:  # arbrgfgfgwse
            time.sleep(10)
            with self.success_lock:
                self.log_time(logger, itime)
                # if self.success_count.value() >= self.desired_success_count:
                #     break
        # logger.info("Timer Complete!")

    def log_time(self, logger: logging.Logger, start_time: float):
        """
        should only exec under self.success_lock
        """
        now = time.time()

        ns = self.success_count.value - self.initial_success_count
        logstr = f"runtime {(now - start_time):.2f}s:"
        if ns > 1:
            logstr += f" successes: {ns}, time per success: {((now - start_time) / float(ns)):.3f}s"
        else:
            logstr += f" no successes yet (at {self.success_count.value})"
        logger.info(logstr)

def read_output(init_sim: Simulation) -> dict[str, float]:
    """
    terrible code, but i'm making it its own terrible code method
    """
    data = False
    sim_log_file = init_sim.sim_dir / init_sim.input.input_dict["log_file"]
    if not sim_log_file.exists():
        raise Exception("No simulation run output!")
    with sim_log_file.open("r") as f:
        for line in f:
            words = line.split()
            if len(words) > 1:
                # jesus fucking christ
                if words[1] == 'FFS' and words[2] == 'final':
                    data = [w for w in words[4:]]
    if data is False:
        raise Exception("oxDNA output does not include requisite FFS information")
    op_names = data[::2]
    op_value = data[1::2]
    op_values = {}
    for ii, name in enumerate(op_names):
        op_values[name[:-1]] = float(op_value[ii][:-1])
    return op_values
