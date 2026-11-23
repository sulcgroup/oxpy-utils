import logging
import multiprocessing
import queue
import re
import threading
import time
from pathlib import Path
from time import sleep
from typing import Any, Union
import json

import hashlib
from collections import defaultdict
from matplotlib.patches import Patch

import networkx as nx
import numpy as np
import pandas as pd
from networkx.readwrite import json_graph
import matplotlib.pyplot as plt

from .base_flux_sampler import success_pattern
from .ffs_interface import FFSInterface
from .ffs_shoot import FFSShooter
from .flux_generator import FFSFluxGenerator
from .weighted_flux_generator import RestartOnFailFluxGenerator, WeightedSourceFluxGenerator
from ..defaults.defaults import default_input_exist
from ..defaults.defaults import get_default_input
from ..utils.oxlog import OxLogHandler
from ..utils.util import si_units, SI_TIME_UNITS

import hashlib
from collections import defaultdict
from matplotlib.patches import Patch

# process graph event log, in the program's root_dir (see FFSProgram.graph_events_path)
GRAPH_EVENTS_FILE = "process_graph_events.jsonl"


def as_node_id(x):
    """process graph node id from a message or the event log: JSON turns (step, process, sim) tuples into lists"""
    return tuple(x) if isinstance(x, list) else x


class FFSProgram:
    """
    bundles ffs flux and shoots
    """

    fluxer: FFSFluxGenerator
    shooters: list[FFSShooter]
    shooter_name_map: dict[str, FFSShooter]
    root_dir: Path
    n_cpus: int
    input_file_params: dict[str, Any]
    # other interaces will be defined in interfaces property
    lambda_neg1: FFSInterface
    # interfaces which partition the overall process
    interfaces: list[FFSInterface]
    desired_n_successes: int

    __ffs_default_input_name: str

    # use directed graph
    process_graph: nx.DiGraph
    # queue to send updates tp graph from processes to program
    graph_update_queue = multiprocessing.Queue()


    def __init__(self, name: str,
                 root_dir: Path,
                 n_cpus: int,
                 desired_n_successes: int,
                 source_directory: Union[Path, None]=None):
        self.root_dir = root_dir
        self.n_cpus = n_cpus
        self.desired_n_successes = desired_n_successes
        if source_directory is None:
            source_directory = root_dir
        self.input_file_params = dict()
        self.fluxer = FFSFluxGenerator("initial_flux",
                                       root_dir,
                                       root_dir/"initial_flux",
                                       )
        self.fluxer.set_num_cpus(n_cpus)
        self.fluxer.source_directory = source_directory
        self.shooters = []
        self.shooter_name_map = dict()

        self.loghandler = OxLogHandler(name, False, self.root_dir)
        self.graph_listener = None
        self.__ffs_default_input_name = "ffs"

        # use a graph to keep track of process (origin node = start conf)
        self.reset_graph()
        # next free sim index per worker process, shared by all steps; created by run()
        self.sim_counters = None
        # self.graph_listener = threading.Thread(target=self._graph_updater, daemon=True)
        self.keep_sim_dirs = True

        self.auto_save = False

    @property
    def ffs_default_input_name(self):
        return self.__ffs_default_input_name

    @ffs_default_input_name.setter
    def ffs_default_input_name(self, value: str):
        if not default_input_exist(value):
            # todo: custom exception
            raise Exception(f"ndefault input file named {value}")
        self.__ffs_default_input_name = value
        self.fluxer.default_input_name = value
        if len(self.shooters) > 0:
            for shooter in self.shooters:
                shooter.default_input_name = value

    def use_source_pool(self, confs, weights=None, restart_after_success: bool = True,
                        restart_on_fail: bool = False):
        """
        Replace the flux generator with a WeightedSourceFluxGenerator that equilibrates each worker
        (re)start from a conf drawn by weight from `confs` (see weighted_flux_generator.py and
        stratified_weights() for building weights from a distribution of some slow variable).
        The pool confs are copied into the source directory, which must also hold the topology.
        Call before set_interfaces().

        restart_on_fail: use a RestartOnFailFluxGenerator instead. The pool confs must already be
        past lambda_{-1}; every attempt runs from a drawn conf (no equilibration) to lambda_0 or
        lambda_fail, then restarts from a new draw. The resulting flux is biased (time beyond
        lambda_fail is never simulated); see RestartOnFailFluxGenerator.
        """
        assert len(self.shooters) == 0, "call use_source_pool() before set_interfaces()"
        old = self.fluxer
        cls = RestartOnFailFluxGenerator if restart_on_fail else WeightedSourceFluxGenerator
        fluxer = cls(old.name, old.working_directory, old.destination_directory)
        fluxer.set_num_cpus(old.ncpus)
        fluxer.source_directory = old.source_directory
        fluxer.default_input_name = old.default_input_name
        fluxer.restart_after_success = restart_after_success
        fluxer.set_source_pool(confs, weights)
        fluxer.write_pool()
        self.fluxer = fluxer

    def set_input_params(self, **kwargs):
        """
        sets key-value pairs that should be present in all simulation input files
        """
        self.input_file_params.update({key: str(kwargs[key]) for key in kwargs})
        self.fluxer.input_file_params = self.input_file_params
        for shooter in self.shooters:
            shooter.input_file_params = self.input_file_params

    def set_interfaces(self, *args: FFSInterface):
        """
        sets the interfaces that partition the process we are studying
        """
        self.interfaces = [*args]
        assert len(args) >= 3, "Need interfaces lambda_-1, lambda_0, lambda_s at minimum"
        if len(self.shooters) > 0:
            print("WARNING: shooters already set")

        self.fluxer.set_interfaces(
            lambda_neg1=args[0],
            lambda_0=args[1],
            lambda_s=args[-1],
        )
        self.fluxer.keep_sim_dirs = self.keep_sim_dirs
        self.fluxer.update_queue = self.graph_update_queue

        # iter interfaces
        for i, interface in enumerate(self.interfaces[2:]):
            shooter = FFSShooter(f"shoot{i+1}",
                                 self.root_dir,
                                 self.shooters[-1].destination_directory if len(self.shooters) > 0 else self.fluxer.destination_directory,
                                 self.root_dir / f"shoot{i+1}")
            shooter.keep_sim_dirs = self.keep_sim_dirs
            shooter.set_num_cpus(self.n_cpus)
            shooter.set_interfaces(~args[0], interface)
            shooter.input_file_params = {**self.input_file_params}
            shooter.set_desired_success_count(self.desired_n_successes)
            shooter.update_queue = self.graph_update_queue
            self.shooters.append(shooter)

    def is_ready(self) -> bool:
        """
        todo: implement checks
        """
        return True

    def run(self):
        """
        runs fluxer and shooters sequentially

        Simulation indices: all steps share one counter per worker process (sim_counters), so worker i's sims
        are numbered consecutively through the whole program, and each p<i>/sim<k> directory (in a step's working
        directory) and process graph node belongs to exactly one step.

        Resuming a run (same root_dir, e.g. after a crash or to add interfaces): the process graph is kept as an
        append-only event log (graph_events_path()). If one exists it is replayed first, so the graph keeps every
        earlier launch and shoot nodes can be linked to confs produced by earlier launches; the counters start
        past every sim index already used (next_sim_indices()).
        """
        self.fluxer.set_desired_success_count(self.desired_n_successes)
        self.fluxer.input_file_params = self.input_file_params
        assert self.is_ready()
        if self.graph_events_path().is_file():
            n_events = self.replay_graph_events()
            self.loghandler.spinoff("main").info(
                f"replayed {n_events} process graph events from {self.graph_events_path()}")
        start = self.next_sim_indices()
        # shared with the worker processes (inherited); slots for workers of earlier, larger launches are kept so
        # the indices stay readable, but only 0..n_cpus-1 are used
        self.sim_counters = multiprocessing.Array("i", max([self.n_cpus, *(p + 1 for p in start)]))
        for p, k in start.items():
            self.sim_counters[p] = k
        for step in [self.fluxer, *self.shooters]:
            step.sim_counters = self.sim_counters
        self.graph_update_queue.put(("LAUNCH", {"time": time.time(), "n_cpus": self.n_cpus,
                                                "desired_n_successes": self.desired_n_successes}))
        self.graph_listener = threading.Thread(target=self._graph_updater, daemon=True)
        self.graph_listener.start()

        if self.auto_save:
            self._update_daemon_thread = threading.Thread(target=self._update_daemon, daemon=True)
            self._update_daemon_thread.start()

        try:
            for step in [self.fluxer, *self.shooters]:
                if step.success_count.value == 0:
                    step.init()
                step.run()
        finally:
            # stop the listener even if a step fails, so everything already queued reaches the event log
            self.graph_update_queue.put(("TERMINATE",))
            self.graph_listener.join()

    def load(self):
        """
        loads fluxer and shooters from disk
        """
        if not self.is_ready():
            raise Exception("Program not ready to load - check configuration")
        if self.fluxer.destination_directory.is_dir():
            self.fluxer.init()
            for shooter in self.shooters:
                # if destination directory has been created
                if shooter.destination_directory.is_dir():
                    shooter.init()
                    try:
                        shooter.load_success_info()
                    except FileNotFoundError:
                        pass # skip

        # the event log is the complete record (every launch); process_graph.json is a snapshot
        if self.graph_events_path().is_file():
            self.replay_graph_events()
        elif (self.root_dir / "process_graph.json").is_file():
            self.load_graph()

    def _update_daemon(self):
        while True:
            self.save_graph()
            for shooter in self.shooters:
                if shooter.success_count > 0:
                    shooter.save_success_info()
            # only save data every 5m or so
            sleep(300)

    def graph_events_path(self) -> Path:
        """append-only log of every process graph update, across all launches in this root_dir"""
        return self.root_dir / GRAPH_EVENTS_FILE

    def reset_graph(self):
        """an empty process graph (just the origin node) and an empty conf -> producing node lookup"""
        self.process_graph = nx.DiGraph(launches=[])
        self.process_graph.add_node("origin", path=str(self.fluxer.source_directory))
        # (step name, 0-based success index) -> node that wrote that step's success_<index + 1>.dat, so that
        # shoot nodes can be connected to the node that produced their starting conf
        self.graph_source_map = {}

    def replay_graph_events(self) -> int:
        """
        Rebuild the process graph from the event log (graph_events_path()) written by earlier launches: one JSON
        line {"t": unix time, "cmd": [message]} per graph update, applied in order with apply_graph_event().
        Call set_interfaces() first (shoot events refer to shooters by name). A last line cut short by a crash
        is skipped. Returns the number of events replayed.
        """
        self.reset_graph()
        graph_logger = self.loghandler.spinoff("graph_updater")
        n_events = 0
        with self.graph_events_path().open() as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    graph_logger.warning(f"skipping unreadable process graph event line: {line[:200]!r}")
                    continue
                self.apply_graph_event(event["cmd"], graph_logger)
                n_events += 1
        return n_events

    def next_sim_indices(self) -> dict[int, int]:
        """
        Per worker process index, one past the largest simulation index already used by any step: in the process
        graph, or as a p<i>/sim<k> directory in any step's working directory (which also covers runs without an
        event log, and sims a crash cut short before they reached the graph).
        """
        start: dict[int, int] = {}

        def seen(p: int, k: int):
            start[p] = max(start.get(p, 0), k + 1)

        for node in self.process_graph.nodes:
            if isinstance(node, tuple):
                seen(node[1], node[2])
        for work_dir in {step.tld() for step in [self.fluxer, *self.shooters]}:
            for sim_dir in work_dir.glob("p*/sim*"):
                p, k = re.fullmatch(r"p(\d+)", sim_dir.parent.name), re.fullmatch(r"sim(\d+)", sim_dir.name)
                if p and k and sim_dir.is_dir():
                    seen(int(p.group(1)), int(k.group(1)))
        return start

    def step_by_name(self, name: str):
        for step in [self.fluxer, *self.shooters]:
            if step.name == name:
                return step
        raise KeyError(f"no flux generator or shooter named {name}")

    def apply_graph_event(self, cmd, graph_logger: logging.Logger):
        """
        Apply one graph update message (from the queue, or replayed from the event log, where tuples have become
        lists). The 1st element is the command type; the others depend on it:
        1. "LAUNCH" - info: dict (appended to process_graph.graph["launches"])
        2. "CPY_CONF" - step_name: str, process_idx: int, sim_idx: int, conf_count: int (0-based success index).
           The node gets success_index (conf_count + 1) and success_conf (the success_<n>.dat it wrote).
        3. "flux" - process_idx: int, sim_idx: int, fluxer_step: str, source node id, [attrs: dict]
           (the sim's "path"; "source_conf" for a sim started from a pool conf)
        4. "shoot" - process_idx: int, sim_idx: int, shoot_name: str, source_conf_idx: int (0-based index of the
           starting conf among the previous step's successes), [attrs: dict] ("path", "source_conf", "seed").
           The node gets source_step / source_success_index, and an edge from the node that wrote that conf.
        5. "shoot_report" - process_idx: int, sim_idx: int, shoot_name: str, status: bool or "undetermined"
        6. "flux_report" - process_idx: int, sim_idx: int, status: bool or "undetermined"
        7. "node_attrs" - process_idx: int, sim_idx: int, step_name: str, attrs: dict (added to an existing node,
           e.g. {"full_success_conf": path} from keep_conf())
        Best effort: a malformed/out-of-order message (e.g. one dropped by a race between a worker process exiting
        and its queue feeder thread flushing) is logged and skipped, so it can't stop the rest of the graph.
        """
        cmd_type, *args = cmd
        try:
            if cmd_type == "LAUNCH":
                info, = args
                self.process_graph.graph.setdefault("launches", []).append(info)
            elif cmd_type == "CPY_CONF":
                step_name, process_idx, sim_idx, conf_count = args
                node_id = (step_name, process_idx, sim_idx)
                self.graph_source_map[step_name, conf_count] = node_id
                if node_id not in self.process_graph:
                    raise KeyError(f"no node {node_id} for its success conf")
                success_index = conf_count + 1
                self.process_graph.nodes[node_id]["success_index"] = success_index
                self.process_graph.nodes[node_id]["success_conf"] = str(
                    self.step_by_name(step_name).destination_directory
                    / success_pattern.replace("*", str(success_index)))
            else:
                # process and sim idxs are universal. process_idx is not checked against n_cpus: replayed events
                # can come from launches that used more worker processes
                process_idx, sim_idx, *args = args
                assert process_idx >= 0, f"Invalid process idx {process_idx} in graph update"
                assert sim_idx >= 0, f"Invalid sim idx {sim_idx} in graph update"
                if cmd_type == "flux":
                    fluxer_step, source_sim, *extra = args
                    source_sim = as_node_id(source_sim)
                    assert fluxer_step in ["equilibrate", "reset", "to_l-1_fwd", "flux_fwd", "flux_back"]
                    # identify nodes with 3-length tuple of program step, process idx, sim idx
                    node_id: tuple[str, int, int] = (self.fluxer.name, process_idx, sim_idx)
                    # workers report the sim's path; older messages did not, and the flux sims were in root_dir
                    attrs = {"fluxer_step": fluxer_step,
                             "path": str(self.root_dir / f"p{process_idx}" / f"sim{sim_idx}"),
                             **(extra[0] if extra else {})}
                    self.process_graph.add_node(node_id, **attrs)
                    assert source_sim != node_id, "Cannot have self-loop in process graph - source sim cannot be the same node as the new flux node"
                    assert source_sim == "origin" or self.process_graph.nodes[node_id]["fluxer_step"] != self.process_graph.nodes[source_sim]["fluxer_step"], f"Cannot have two nodes with the same fluxer_step in parent-child relationship - check graph structure for errors. Node: {node_id}, source node: {source_sim}, fluxer_step of new node: {self.process_graph.nodes[node_id]['fluxer_step']}, fluxer_step of source node: {self.process_graph.nodes[source_sim]['fluxer_step']}"
                    self.process_graph.add_edge(source_sim, node_id)
                elif cmd_type == "shoot":
                    shoot_name, source_conf_idx, *extra = args
                    shoot_idx = next(i for i, s in enumerate(self.shooters) if s.name == shoot_name)
                    shooter_obj = self.shooters[shoot_idx]
                    # the first shooter starts from flux successes, the others from the previous shooter's
                    source_step = self.fluxer.name if shoot_idx == 0 else self.shooters[shoot_idx - 1].name
                    attrs = dict(extra[0]) if extra else {}
                    if "source_conf" not in attrs:
                        attrs["source_conf"] = str(shooter_obj.starting_confs[source_conf_idx])
                    # identify nodes with 3-length tuple of program step, process idx, sim idx
                    node_id: tuple[str, int, int] = (shoot_name, process_idx, sim_idx)
                    self.process_graph.add_node(
                        node_id,
                        shooter=shooter_obj,
                        source_step=source_step,
                        source_success_index=source_conf_idx + 1,
                        **attrs
                    )
                    source_sim_node = self.graph_source_map.get((source_step, source_conf_idx))
                    if source_sim_node is None:
                        raise KeyError(f"no node is known to have written {source_step} success "
                                       f"{source_conf_idx + 1}; {node_id} keeps source_step/source_success_index "
                                       f"but gets no parent edge")
                    self.process_graph.add_edge(source_sim_node, node_id)
                elif cmd_type == "shoot_report":
                    shoot_name, status = args
                    node_id = (shoot_name, process_idx, sim_idx)
                    # queue is FIFO, node should already be in graph
                    self.process_graph.nodes[node_id]["success"] = status  # can be true, false, or "undetermined"
                elif cmd_type == "flux_report":
                    status, = args
                    node_id = (self.fluxer.name, process_idx, sim_idx)
                    # queue is FIFO, node should already be in graph
                    self.process_graph.nodes[node_id]["success"] = status  # can be true, false, or "undetermined"
                elif cmd_type == "node_attrs":
                    step_name, attrs = args
                    node_id = (step_name, process_idx, sim_idx)
                    if node_id not in self.process_graph:
                        raise KeyError(f"no node {node_id} to add {attrs} to")
                    self.process_graph.nodes[node_id].update(attrs)
                else:
                    raise Exception(f"Unknown command type {cmd_type}")
        except Exception as e:
            graph_logger.warning(f"Dropping malformed/unresolvable graph update {tuple(cmd)!r}: {e}")

    def _graph_updater(self):
        """
        Listener thread: takes graph update messages (see apply_graph_event) off graph_update_queue until
        "TERMINATE", appends each to the event log (graph_events_path(), line-buffered, so it survives a crash)
        and applies it to the in-memory process graph.
        """
        graph_logger = self.loghandler.spinoff("graph_updater")
        with self.graph_events_path().open("a", buffering=1) as events:
            while True:
                try:
                    cmd = self.graph_update_queue.get(timeout=5)  # avoid indefinite hang
                except queue.Empty:
                    continue  # Timeout — continue listening
                if cmd[0] == "TERMINATE":
                    break  # end of process
                try:
                    events.write(json.dumps({"t": time.time(), "cmd": list(cmd)}) + "\n")
                except (TypeError, ValueError) as e:
                    graph_logger.warning(f"graph update {cmd!r} is not JSON-serialisable and was not logged: {e}")
                self.apply_graph_event(cmd, graph_logger)


    def save_graph(self):
        G = self.process_graph

        # build a json-safe copy
        data = {
            "graph": dict(G.graph),
            "nodes": [],
            "edges": [],
        }

        for node, attrs in G.nodes(data=True):
            # serialize node id: tuples become lists, "origin" stays as-is
            node_serial = list(node) if isinstance(node, tuple) else node

            # serialize attrs: replace shooter objects with shooter name
            safe_attrs = {}
            for k, v in attrs.items():
                if k == "shooter":
                    safe_attrs["shooter"] = v.name
                else:
                    safe_attrs[k] = v

            data["nodes"].append({"id": node_serial, "attrs": safe_attrs})

        for src, dst in G.edges():
            src_serial = list(src) if isinstance(src, tuple) else src
            dst_serial = list(dst) if isinstance(dst, tuple) else dst
            data["edges"].append({"src": src_serial, "dst": dst_serial})

        # no indentation: graphs of long runs have millions of nodes
        with open(self.root_dir / "process_graph.json", "w") as f:
            json.dump(data, f)

    def load_graph(self):
        """
        Load the process_graph.json snapshot written by save_graph(). It only holds what was in memory when it
        was saved; the event log (replay_graph_events(), used by load() and run()) holds every launch.
        """
        with open(self.root_dir / "process_graph.json", "r") as f:
            data = json.load(f)

        # build shooter name -> object lookup
        shooter_map = {s.name: s for s in self.shooters}

        self.reset_graph()
        self.process_graph.graph.update(data.get("graph", {}))

        for entry in data["nodes"]:
            raw_id = entry["id"]
            # restore tuples: lists of [str, int, int] -> tuple, "origin" stays str
            node_id = tuple(raw_id) if isinstance(raw_id, list) else raw_id

            attrs = {}
            for k, v in entry["attrs"].items():
                if k == "shooter":
                    # restore shooter object from name
                    attrs["shooter"] = shooter_map[v]
                else:
                    attrs[k] = v

            self.process_graph.add_node(node_id, **attrs)
            if isinstance(node_id, tuple) and "success_index" in attrs:
                self.graph_source_map[node_id[0], attrs["success_index"] - 1] = node_id

        for edge in data["edges"]:
            src = tuple(edge["src"]) if isinstance(edge["src"], list) else edge["src"]
            dst = tuple(edge["dst"]) if isinstance(edge["dst"], list) else edge["dst"]
            self.process_graph.add_edge(src, dst)

    def flux_graph(self) -> nx.DiGraph:
        """
        returns a graph of just the fluxer step, for debugging
        """
        G = self.process_graph
        flux_nodes = [n for n in G.nodes if isinstance(n, tuple) and n[0] == self.fluxer.name]
        return G.subgraph(flux_nodes)

    def _input_param(self, key: str) -> str:
        """
        looks up the value of an oxDNA input parameter as it will be seen by the ffs simulations:
        values set via set_input_params override the default ffs input file
        """
        if key in self.input_file_params:
            return str(self.input_file_params[key])
        default = get_default_input(self.ffs_default_input_name)
        default.evaluate()
        return str(default.get_dict()[key])

    def export_results(self, fname: str = "results.csv"):
        export_data = pd.DataFrame(
            columns=["flux", *[shooter.name for shooter in self.shooters], "total"],
            index=["num_attempts", "num_successes", "success_ratio", "rate_per_oxdna_time_unit"]
        )

        # Populate flux column
        # for the flux column, "attempts" are simulation steps and the "ratio" is the
        # flux itself, in successes per step
        flux = self.fluxer.compute_flux()
        n_flux_successes = len(self.fluxer.get_success_confs())
        export_data.loc["num_successes", "flux"] = n_flux_successes
        export_data.loc["num_attempts", "flux"] = n_flux_successes / flux

        # Populate shooter columns
        for shooter in self.shooters:
            export_data.loc["num_attempts", shooter.name] = sum(shooter.attempt_from)
            export_data.loc["num_successes", shooter.name] = shooter.success_count.value

        # Compute success_ratio for all columns except "total"
        cols = ["flux", *[shooter.name for shooter in self.shooters]]
        shooter_cols = cols[1:]
        export_data.loc["success_ratio", shooter_cols] = (
                export_data.loc["num_successes", shooter_cols] / export_data.loc["num_attempts", shooter_cols]
        )
        export_data.loc["success_ratio", "flux"] = flux

        # Compute total (rate constant, in successes per step) as flux * product of shooter probabilities
        export_data.loc["success_ratio", "total"] = np.prod(export_data.loc["success_ratio", cols].values)

        # convert the rates (flux and total) from successes per step to successes per oxDNA
        # time unit and to successes per second. Shooter columns are dimensionless probabilities,
        # so they are left blank in these rows.
        dt = float(self._input_param("dt"))  # oxDNA time units per step
        interaction_type = "rna" if "RNA" in self._input_param("interaction_type").upper() else "dna"
        rate_cols = ("flux", "total")
        for col in rate_cols:
            export_data.loc["rate_per_oxdna_time_unit", col] = export_data.loc["success_ratio", col] / dt

        # physical-time row: pick the smallest SI time unit in which both rates are > 1
        # (a rate per unit time gets larger as the time unit gets larger)
        rates_per_ox = [float(export_data.loc["rate_per_oxdna_time_unit", col]) for col in rate_cols]
        valid_rates = [r for r in rates_per_ox if np.isfinite(r) and r > 0]
        time_unit = "s"  # fallback if no rate is usable
        if valid_rates:
            time_unit = list(SI_TIME_UNITS)[-1]  # fall back to the largest unit
            for unit in SI_TIME_UNITS:
                if all(r / si_units(1., interaction_type, "time", unit) > 1 for r in valid_rates):
                    time_unit = unit
                    break
        row = f"rate_per_{time_unit}"
        export_data.loc[row] = np.nan
        for col, r in zip(rate_cols, rates_per_ox):
            export_data.loc[row, col] = r / si_units(1., interaction_type, "time", time_unit)

        export_data.to_csv(self.root_dir / fname)


    def plot_graph(self):
        G = self.process_graph

        fluxer_name = self.fluxer.name

        # --- DIAGNOSTIC: check flux node parent structure ---
        flux_nodes_all = [n for n, attrs in G.nodes(data=True)
                          if isinstance(n, tuple) and n[0] == fluxer_name]
        non_equil_orphans = []
        for n in flux_nodes_all:
            fs = G.nodes[n].get("fluxer_step", "")
            if fs != "equilibrate":
                preds = list(G.predecessors(n))
                if not any(p != "origin" for p in preds):
                    non_equil_orphans.append((n, fs, preds))
        if non_equil_orphans:
            raise ValueError(
                f"plot_graph: {len(non_equil_orphans)} non-equilibrate flux nodes have no "
                f"non-origin parent. Examples: {non_equil_orphans[:5]}"
            )

        # --- 1. BFS from successful shoot nodes ---
        shooter_names = {s.name for s in self.shooters}
        seed_nodes = [
            n for n, attrs in G.nodes(data=True)
            if isinstance(n, tuple)
               and n[0] in shooter_names
               and attrs.get("success", False)
        ]
        relevant_nodes = set()
        queue_nodes = list(seed_nodes)
        visited = set(queue_nodes)
        while queue_nodes:
            n = queue_nodes.pop()
            relevant_nodes.add(n)
            for pred in G.predecessors(n):
                if pred not in visited:
                    visited.add(pred)
                    queue_nodes.append(pred)
        subgraph = G.subgraph(relevant_nodes)

        # --- 2. Step ordering and labels ---
        step_order = [fluxer_name] + [s.name for s in self.shooters]
        step_labels = {fluxer_name: "Flux", **{s.name: s.name for s in self.shooters}}

        def node_step(n):
            if n == "origin":
                return None
            return fluxer_name if n[0] == fluxer_name else n[0]

        # --- 3. Group nodes by step ---
        step_nodes = defaultdict(list)
        for n in subgraph.nodes:
            step = node_step(n)
            if step is not None:
                step_nodes[step].append(n)

        # --- 4. Flux layout ---
        # Each fluxer_step type gets its own x-column within the flux band.
        # Order left to right mirrors the temporal order of the FFS flux stage.
        FLUX_STEP_ORDER = ["equilibrate", "reset", "to_l-1_fwd", "flux_fwd", "flux_back"]
        FLUX_PRIMARY = {"flux_fwd", "flux_back"}

        # fluxer_step -> whether it appears at all (drives coloring/legend, keyed on the
        # true fluxer_step attribute - unaffected by the success/fail column split below).
        flux_steps_present = []
        for fs in FLUX_STEP_ORDER:
            if any(G.nodes[n].get("fluxer_step") == fs for n in step_nodes[fluxer_name]):
                flux_steps_present.append(fs)

        def get_flux_step(n):
            return G.nodes[n].get("fluxer_step", "flux_fwd")

        def flux_is_primary(n):
            return get_flux_step(n) in FLUX_PRIMARY

        # --- x-column layout key ---
        # lambda_fail is defined as ~lambda_neg1 (flux_generator.py's set_interfaces default) -
        # same order parameter, same threshold value, just the logically-negated comparison. It
        # is NOT a second, separate interface further out - there is no real gap in phase space
        # between "native <= 7" and "native > 7", so they must be drawn as ONE line, not two.
        # That single line separates:
        #   - the native > 7 side: equilibrate, reset, and a FAILED flux_fwd (which re-crossed
        #     lambda_neg1 backward, satisfying lambda_fail again)
        #   - the native <= 7 side: to_l-1_fwd (whose entire job is to reach lambda_neg1, so it
        #     ends up just past it), a SUCCESSFUL flux_fwd (further out, at lambda_0), and
        #     flux_back (later still)
        # flux_fwd's own outcome (the "success" attribute, already recorded via
        # flux_generator.py's flux_report message) decides which side it lands on.
        FLUX_COLUMN_ORDER = ["equilibrate", "reset", "flux_fwd_fail", "to_l-1_fwd", "flux_fwd_success", "flux_back"]

        def flux_col_key(n):
            fs = get_flux_step(n)
            if fs == "flux_fwd":
                return "flux_fwd_success" if G.nodes[n].get("success", False) else "flux_fwd_fail"
            return fs

        flux_columns_present = [
            ck for ck in FLUX_COLUMN_ORDER
            if any(flux_col_key(n) == ck for n in step_nodes[fluxer_name])
        ]

        x_sub_gap = 1.0
        y_spacing = 2.5  # vertical units between stacked nodes — increase to spread out
        x_col_gap = 2.5  # gap between flux step-type columns
        x_step_gap_base = 4.0

        # flux column layout: each column key gets a sub-band wide enough for all
        # process_idxs that appear under it, separated by x_col_gap.
        flux_step_proc_idxs = defaultdict(set)
        for n in step_nodes[fluxer_name]:
            flux_step_proc_idxs[flux_col_key(n)].add(n[1])

        flux_col_x0 = {}  # column key -> x offset of leftmost process_idx in that band
        flux_col_width = {}  # column key -> width of that band
        fc = 0.0
        for ck in flux_columns_present:
            flux_col_x0[ck] = fc
            w = max(len(flux_step_proc_idxs.get(ck, {0})), 1) * x_sub_gap
            flux_col_width[ck] = w
            fc += w + x_col_gap
        flux_total_width = fc - x_col_gap  # trim trailing gap

        # dense rank of process_idx within each column (so gaps don't create whitespace)
        flux_step_proc_rank = {}  # (column key, proc_idx) -> x offset within that column's band
        for ck, proc_idxs in flux_step_proc_idxs.items():
            for rank, proc_idx in enumerate(sorted(proc_idxs)):
                flux_step_proc_rank[(ck, proc_idx)] = rank * x_sub_gap

        # shooter column widths
        step_process_idxs = defaultdict(set)
        for n in subgraph.nodes:
            if isinstance(n, tuple):
                step = node_step(n)
                if step is not None and step != fluxer_name:
                    step_process_idxs[step].add(n[1])

        step_width = {fluxer_name: flux_total_width}
        for step in step_order[1:]:
            step_width[step] = max(len(step_process_idxs.get(step, {0})), 1) * x_sub_gap

        step_x_origin = {}
        cursor = 0.0
        for step in step_order:
            step_x_origin[step] = cursor
            cursor += step_width[step] + x_step_gap_base

        # --- 5. Dense rank ---
        # Flux: shared rank across ALL flux nodes so all columns share the same y-scale.
        # Sort by sim_idx so sequential sims appear in order.
        all_flux_sim_idxs = sorted({n[2] for n in step_nodes[fluxer_name]})
        flux_global_rank = {sim_idx: rank for rank, sim_idx in enumerate(all_flux_sim_idxs)}

        # Shooters: rank per (step, process_idx) as before
        shoot_rank_map: dict[tuple, dict[int, int]] = defaultdict(dict)
        for step in step_order[1:]:
            by_proc: dict[int, list] = defaultdict(list)
            for n in step_nodes[step]:
                by_proc[n[1]].append(n[2])
            for proc_idx, sim_idxs in by_proc.items():
                for rank, sim_idx in enumerate(sorted(sim_idxs)):
                    shoot_rank_map[(step, proc_idx)][sim_idx] = rank

        # Dense rank of process_idx within each shooter step (mirrors flux_step_proc_rank),
        # so a sparse subset of process_idxs (as the BFS filter usually leaves) doesn't spread
        # nodes out to their raw index value - which is what step_width above is NOT sized for.
        step_proc_rank: dict[str, dict[int, int]] = {}
        for step in step_order[1:]:
            step_proc_rank[step] = {
                proc_idx: rank
                for rank, proc_idx in enumerate(sorted(step_process_idxs.get(step, set())))
            }

        # --- 6. Jitter ---
        def _jitter(n, scale=0.35):
            h = int(hashlib.md5(str(n).encode()).hexdigest(), 16)
            return (h % 1000) / 1000.0 * scale - scale / 2

        # --- 7. Build positions ---
        pos = {}
        flux_x0 = step_x_origin[fluxer_name]
        for n in step_nodes[fluxer_name]:
            ck = flux_col_key(n)
            proc_x = flux_step_proc_rank.get((ck, n[1]), 0)
            x = flux_x0 + flux_col_x0.get(ck, 0) + proc_x
            y = -flux_global_rank[n[2]] * y_spacing + _jitter(n)
            pos[n] = (x, y)

        for step in step_order[1:]:
            x0 = step_x_origin[step]
            for n in step_nodes[step]:
                x = x0 + step_proc_rank[step][n[1]] * x_sub_gap
                y = -shoot_rank_map[(step, n[1])][n[2]] * y_spacing + _jitter(n)
                pos[n] = (x, y)

        # --- 8. Per-node color / alpha / size ---
        # Give each flux step type a distinct shade of blue; shooters keep tab10 colors.
        cmap = plt.cm.get_cmap("tab10", len(step_order))
        node_color_map = {step: cmap(i) for i, step in enumerate(step_order)}

        # Flux step-type colors: distinct qualitative color per fluxer_step type.
        # Fixed palette so colors are consistent regardless of which steps are present.
        FLUX_STEP_COLORS = {
            "equilibrate": "#a6cee3",  # light blue
            "reset": "#fb9a99",  # light red/salmon
            "to_l-1_fwd": "#b2df8a",  # light green
            "flux_fwd": "#1f78b4",  # strong blue  (primary)
            "flux_back": "#e31a1c",  # strong red   (primary)
        }
        flux_step_color = {
            fs: FLUX_STEP_COLORS.get(fs, "#999999")
            for fs in flux_steps_present
        }

        draw_nodes = [n for n in subgraph.nodes if node_step(n) is not None]
        draw_subgraph = G.subgraph(draw_nodes)

        y_all = [y for _, y in pos.values()]
        data_height = max(y_all) - min(y_all) if len(y_all) > 1 else 1.0
        fig_w = max(16, 3 * len(step_order))
        fig_h = 22
        # Divide data_height by y_spacing so node size reflects logical density,
        # not the stretched coordinate range.
        logical_height = (data_height / y_spacing) + 2
        pts_per_data_unit = (fig_h * 72) / logical_height
        node_diameter_pts = pts_per_data_unit * y_spacing * 0.8
        base_node_size = max(10, min(node_diameter_pts ** 2, 300))

        node_colors, node_alphas, node_sizes_list = [], [], []
        for n in draw_subgraph.nodes:
            step = node_step(n)
            if step == fluxer_name:
                node_colors.append(flux_step_color[get_flux_step(n)])
                if flux_is_primary(n):
                    node_alphas.append(0.9)
                    node_sizes_list.append(base_node_size)
                else:
                    node_alphas.append(0.35)
                    node_sizes_list.append(base_node_size * 0.5)
            else:
                node_colors.append(node_color_map[step])
                node_alphas.append(0.85)
                node_sizes_list.append(base_node_size)

        # --- 9. Draw ---
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))

        nx.draw_networkx_edges(
            draw_subgraph, pos=pos, ax=ax,
            edge_color="#555555", arrows=True,
            arrowsize=8, width=0.7, alpha=0.5,
        )
        for n, color, alpha, size in zip(
                draw_subgraph.nodes, node_colors, node_alphas, node_sizes_list):
            nx.draw_networkx_nodes(
                draw_subgraph, pos=pos, ax=ax,
                nodelist=[n], node_color=[color],
                node_size=size, alpha=alpha,
            )

        # --- 10. Node labels: success fraction ---
        # For all shooters except the last: derive from child node success attrs in graph.
        # For the last shooter: use attempt_from / success_from multiprocessing arrays,
        # indexed by source_conf_idx — recovered by matching the stored source_conf path
        # against the shooter's starting_confs list.
        final_shooter_obj = self.shooters[-1]
        final_shoot_name = self.shooters[-1].name

        # Build lookup: conf path string -> index in starting_confs for final shooter
        final_conf_path_to_idx = {
            str(conf): i
            for i, conf in enumerate(final_shooter_obj.starting_confs)
        }

        for step in step_order[1:]:
            is_final = (step == final_shoot_name)
            for n in step_nodes[step]:
                if is_final:
                    source_conf_path = G.nodes[n].get("source_conf")
                    if source_conf_path is None:
                        continue
                    conf_idx = final_conf_path_to_idx.get(source_conf_path)
                    if conf_idx is None:
                        continue
                    attempts = final_shooter_obj.attempt_from[conf_idx]
                    successes = final_shooter_obj.success_from[conf_idx]
                    if attempts == 0:
                        continue
                    label_str = f"{successes}/{attempts}"
                else:
                    children = list(G.successors(n))
                    if not children:
                        continue
                    n_success = sum(1 for c in children if G.nodes[c].get("success", False))
                    label_str = f"{n_success}/{len(children)}"
                x, y = pos[n]
                ax.text(x, y, label_str,
                        fontsize=7, ha="center", va="center",
                        color="white", fontweight="bold", zorder=5)

        # flux_fwd nodes with success=True: label with child success ratio
        for n in step_nodes[fluxer_name]:
            if get_flux_step(n) == "flux_fwd" and G.nodes[n].get("success", False):
                children = list(G.successors(n))
                if not children:
                    continue
                n_success = sum(1 for c in children if G.nodes[c].get("success", False))
                x, y = pos[n]
                ax.text(x, y, f"{n_success}/{len(children)}",
                        fontsize=7, ha="center", va="center",
                        color="white", fontweight="bold", zorder=5)

        # --- 11. Vertical dividers ---
        y_max_plot = max(y_all) + 0.5

        # Internal flux divider between step-type columns, labelled with interface names.
        # lambda_fail is the same interface as lambda_neg1 (just the negated comparison, see
        # the FLUX_COLUMN_ORDER comment above) - there is only ONE line here, not a left
        # "lambda_fail" border plus a separate internal "lambda_neg1" one.
        # Layout left->right: equil, reset, flux_fwd(fail) | lambda_neg1 | to_l-1_fwd, flux_fwd(success), flux_back | lambda_0
        # (lambda_0 is drawn separately below, as the flux column's right separator.)
        left_present = [ck for ck in ["equilibrate", "reset", "flux_fwd_fail"] if ck in flux_col_x0]
        right_present = [ck for ck in ["to_l-1_fwd", "flux_fwd_success", "flux_back"] if ck in flux_col_x0]
        if left_present and right_present:
            x_left_edge = flux_x0 + max(flux_col_x0[fs] + flux_col_width[fs] for fs in left_present)
            x_right_edge = flux_x0 + min(flux_col_x0[fs] for fs in right_present)
            x_div = (x_left_edge + x_right_edge) / 2
            ax.axvline(x=x_div, color="black", linestyle=":", linewidth=0.8, alpha=0.4)
            ax.text(x_div, y_max_plot, str(self.fluxer.lambda_neg1),
                    ha="center", va="bottom", fontsize=8, color="black", alpha=0.7, rotation=90)

        # Separators between steps.
        # Each step's OWN lambda_plus1 is the interface its successes cross to seed the next
        # step (e.g. fluxer's flux_fwd successes at lambda_0 are shoot1's starting confs), so
        # the line to the RIGHT of step[i] is labelled step[i].lambda_plus1 - not step[i+1]'s.
        # step_order: [fluxer, shoot1, shoot2, ..., shootN]
        # The flux column's right separator is fluxer.lambda_plus1 (= lambda_0, the first interface to shoot).
        # The final shooter's right separator is its own lambda_plus1 (= lambda_s, overall success).
        step_objects = {fluxer_name: self.fluxer, **{s.name: s for s in self.shooters}}
        for step in step_order:
            if step not in step_nodes:
                continue
            label = str(step_objects[step].lambda_plus1)
            x_right = step_x_origin[step] + step_width[step] + x_step_gap_base / 2
            ax.axvline(x=x_right, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
            ax.text(x_right, y_max_plot, label,
                    ha="center", va="bottom", fontsize=8, color="black", alpha=0.7, rotation=90)

        # --- 13. Step column labels ---
        step_top_y = defaultdict(lambda: 0)
        for n, (x, y) in pos.items():
            step = node_step(n)
            if step:
                step_top_y[step] = max(step_top_y[step], y)

        for step in step_order:
            if step not in step_nodes:
                continue
            x_center = step_x_origin[step] + step_width[step] / 2
            y_top = step_top_y[step] + 1.5
            ax.text(x_center, y_top, step_labels[step],
                    ha="center", va="bottom", fontsize=11, fontweight="bold",
                    color=node_color_map[step])

        # --- 14. Legend ---
        legend_elements = []
        for fs in flux_steps_present:
            legend_elements.append(
                Patch(facecolor=flux_step_color[fs], label=f"flux: {fs}")
            )
        for s in self.shooters:
            if s.name in step_nodes:
                legend_elements.append(
                    Patch(facecolor=node_color_map[s.name], label=s.name)
                )
        ax.legend(handles=legend_elements, loc="lower left", fontsize=8)
        ax.set_title(
            f"FFS Process Graph",
            fontsize=13
        )

        # Set xlim with a small margin around the data
        x_min_all = min(x for x, _ in pos.values()) - x_step_gap_base
        x_max_all = max(x for x, _ in pos.values()) + x_step_gap_base
        ax.set_xlim(x_min_all, x_max_all)
        plt.tight_layout()
        plt.savefig(self.root_dir / "process_graph.svg")
        plt.close(fig)