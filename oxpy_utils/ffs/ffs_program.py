import multiprocessing
import queue
import threading
from pathlib import Path
from time import sleep
from typing import Any, Union
import json

import networkx as nx
import numpy as np
import pandas as pd
from networkx.readwrite import json_graph
import matplotlib.pyplot as plt

from .ffs_interface import FFSInterface
from .ffs_shoot import FFSShooter
from .flux_generator import FFSFluxGenerator
from ..defaults.defaults import default_input_exist
from ..utils.oxlog import OxLogHandler


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

        # use a graph to keep track of process
        self.process_graph = nx.DiGraph()
        # add node for start conf
        self.process_graph.add_node("origin",
                                    path=str(self.fluxer.source_directory))
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
            shooter.oxloghander = self.loghandler
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
        """
        self.fluxer.set_desired_success_count(self.desired_n_successes)
        self.fluxer.input_file_params = self.input_file_params
        assert self.is_ready()
        self.graph_listener = threading.Thread(target=self._graph_updater, daemon=True)
        self.graph_listener.start()

        if self.auto_save:
            self._update_daemon_thread = threading.Thread(target=self._update_daemon, daemon=True)
            self._update_daemon_thread.start()

        for step in [self.fluxer, *self.shooters]:
            if step.success_count.value == 0:
                step.init()
            step.run()

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

        if (self.root_dir/"process_graph.json").is_file():
            self.process_graph = json_graph.node_link_data(self.process_graph)

    def _update_daemon(self):
        while True:
            self.save_graph()
            for shooter in self.shooters:
                if shooter.success_count > 0:
                    shooter.save_success_info()
            # only save data every 5m or so
            sleep(300)

    def _graph_updater(self):
        """
        updates the process graph based on messages from the graph update queue
        1st element of each message is command type
        other elements depend on command type
        1. "TERMINATE" - no other elements, terminates the listener
        2. "CPY_CONF" - source_process_name: str, process_idx: int, sim_idx: int, conf_count: int
        3. "flux" - process_idx: int, sim_idx: int, fluxer_step: str
        4. "shoot" - process_idx: int, sim_idx: int, shoot_name: str, source_conf_idx: int
        5. "shoot_report" - process_idx: int, sim_idx: int, shoot_name: str, status: bool or "undetermined"
        -------------
        """
        # a lookup table that records where a particular configuration was produced, so that later steps (“shoot”
        # nodes) can be connected to the correct parent node in the process graph.
        source_map: dict[tuple[str, int], tuple[str, int, int]] = dict()
        while True:
            try:
                cmd = self.graph_update_queue.get(timeout=5)  # avoid indefinite hang
            except queue.Empty:
                continue  # Timeout — continue listening
            cmd_type, *cmd = cmd # pop first vale
            # terminate signal
            if cmd_type == "TERMINATE":
                break # end of process
            elif cmd_type == "CPY_CONF":
                # unpacck info
                source_process_name, process_idx, sim_idx, conf_count = cmd
                assert -1 < process_idx < self.n_cpus, f"Invalid process idx {process_idx} in graph update"

                # map source process name and conf count to full node id
                source_map[source_process_name, conf_count] = (source_process_name, process_idx, sim_idx)
            else:
                # pop process and sim idxs, which are universal
                process_idx, sim_idx, *cmd = cmd
                assert -1 < process_idx < self.n_cpus, f"Invalid process idx {process_idx} in graph update"
                assert -1 < sim_idx, f"Invalid sim idx {sim_idx} in graph update"
                # difficult to fully validate sim_idx without also knowing process step, since it's unbounded
                # if this is a report on a fluxer step
                if cmd_type == "flux":
                    # pop fluxer step
                    fluxer_step, source_sim = cmd
                    assert fluxer_step in ["equilibrate", "reset", "to_l-1_fwd", "flux_fwd", "flux_back"]
                    # identify nodes with 3-length tuple of program step, process idx, sim idx
                    node_id: tuple[str, int, int] = (self.fluxer.name, process_idx, sim_idx)
                    self.process_graph.add_node(
                        node_id,
                        fluxer_step=fluxer_step,
                        path=str(self.root_dir / f"p{process_idx}" / f"sim{sim_idx}")
                    )
                    assert  source_sim != node_id, "Cannot have self-loop in process graph - source sim cannot be the same node as the new flux node"
                    assert source_sim == "origin" or self.process_graph.nodes[node_id]["fluxer_step"] != self.process_graph.nodes[source_sim]["fluxer_step"], f"Cannot have two nodes with the same fluxer_step in parent-child relationship - check graph structure for errors. Node: {node_id}, source node: {source_sim}, fluxer_step of new node: {self.process_graph.nodes[node_id]['fluxer_step']}, fluxer_step of source node: {self.process_graph.nodes[source_sim]['fluxer_step']}"
                    self.process_graph.add_edge(
                        source_sim,
                        node_id
                    )
                elif cmd_type == "shoot":
                    # shoot name = name of this shooter
                    shoot_name, source_conf_idx = cmd
                    # dirty extract name
                    shoot_idx = next(i for i,s in enumerate(self.shooters) if s.name == shoot_name)
                    # shoot_idx = int(shoot_name[len("shoot"):])-1
                    shooter_obj = self.shooters[shoot_idx]
                    # identify nodes with 3-length tuple of program step, process idx, sim idx
                    node_id: tuple[str, int, int] = (shoot_name, process_idx, sim_idx)
                    self.process_graph.add_node(
                        node_id,
                        shooter=shooter_obj,
                        source_conf=str(shooter_obj.starting_confs[source_conf_idx])
                    )
                    if not shoot_idx: # if this is first shoot (shoot_idx == 0), source is fluxer
                        try:
                            source_sim_node = source_map[(self.fluxer.name, source_conf_idx)]
                        except KeyError as e:
                            raise KeyError(f"Could not find source node for fluxer conf idx {source_conf_idx}") from e
                    else:
                        try:
                            # shoot names index from 1, so this is actually shoot_idx-1+1
                            source_sim_node = source_map[(self.shooters[shoot_idx-1].name, source_conf_idx)]
                        except KeyError as e:
                            raise KeyError(f"Could not find source node for shooter {shoot_idx} conf idx {source_conf_idx}") from e
                    self.process_graph.add_edge(
                        source_sim_node,
                        node_id
                    )
                elif cmd_type == "shoot_report":
                    shoot_name, status = cmd
                    node_id = (shoot_name, process_idx, sim_idx)
                    # queue is FIFO, node should already be in graph
                    self.process_graph.nodes[node_id]["success"] = status # can be true, false, or "undetermined"
                elif cmd_type == "flux_report":
                    status, = cmd
                    node_id = (self.fluxer.name, process_idx, sim_idx)
                    # queue is FIFO, node should already be in graph
                    self.process_graph.nodes[node_id]["success"] = status # can be true, false, or "undetermined"
                else:
                    raise Exception(f"Unknown command type {cmd_type}")


    def save_graph(self):
        G = self.process_graph

        # build a json-safe copy
        data = {
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

        with open(self.root_dir / "process_graph.json", "w") as f:
            json.dump(data, f, indent=4)

    def load_graph(self):
        with open(self.root_dir / "process_graph.json", "r") as f:
            data = json.load(f)

        # build shooter name -> object lookup
        shooter_map = {s.name: s for s in self.shooters}

        self.process_graph = nx.DiGraph()

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

        for edge in data["edges"]:
            src = tuple(edge["src"]) if isinstance(edge["src"], list) else edge["src"]
            dst = tuple(edge["dst"]) if isinstance(edge["dst"], list) else edge["dst"]
            self.process_graph.add_edge(src, dst)

    def flux_graph(self) -> nx.DiGraph:
        """
        returns a graph of just the fluxer step, for debugging
        """
        G = self.process_graph
        flux_nodes = [n for n in G.nodes if n[0] == "flux"]
        return G.subgraph(flux_nodes)

    def export_results(self, fname: str = "results.csv"):
        export_data = pd.DataFrame(
            columns=["flux", *[shooter.name for shooter in self.shooters], "total"],
            index=["num_attempts", "num_successes", "success_ratio"]
        )

        # Populate flux column
        export_data.loc["num_successes", "flux"] = len(self.fluxer.get_success_confs())
        export_data.loc["num_attempts", "flux"] = self.fluxer.total_success_time()

        # Populate shooter columns
        for shooter in self.shooters:
            export_data.loc["num_attempts", shooter.name] = sum(shooter.attempt_from)
            export_data.loc["num_successes", shooter.name] = shooter.success_count.value

        # Compute success_ratio for all columns except "total"
        cols = ["flux", *[shooter.name for shooter in self.shooters]]
        export_data.loc["success_ratio", cols] = (
                export_data.loc["num_successes", cols] / export_data.loc["num_attempts", cols]
        )

        # Compute total as product of all success_ratios
        export_data.loc["success_ratio", "total"] = np.prod(export_data.loc["success_ratio", cols].values)

        export_data.to_csv(self.root_dir / fname)


    def plot_graph(self):
        G = self.process_graph
        import hashlib
        from collections import defaultdict
        from matplotlib.patches import Patch

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
        step_interface_labels = {
            fluxer_name: str(self.fluxer.lambda_plus1),
            **{s.name: str(s.lambda_plus1) for s in self.shooters}
        }

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

        # fluxer_step -> x-column index (only include steps that actually appear)
        flux_steps_present = []
        for fs in FLUX_STEP_ORDER:
            if any(G.nodes[n].get("fluxer_step") == fs for n in step_nodes[fluxer_name]):
                flux_steps_present.append(fs)

        flux_step_col = {fs: i for i, fs in enumerate(flux_steps_present)}

        def get_flux_step(n):
            return G.nodes[n].get("fluxer_step", "flux_fwd")

        def flux_is_primary(n):
            return get_flux_step(n) in FLUX_PRIMARY

        x_sub_gap = 1.0
        y_spacing = 2.5  # vertical units between stacked nodes — increase to spread out
        x_col_gap = 2.5  # gap between flux step-type columns
        x_step_gap_base = 4.0

        # flux column layout: each fluxer_step gets a sub-band wide enough for all
        # process_idxs that appear under that step, separated by x_col_gap.
        flux_step_proc_idxs = defaultdict(set)
        for n in step_nodes[fluxer_name]:
            flux_step_proc_idxs[get_flux_step(n)].add(n[1])

        flux_col_x0 = {}  # fluxer_step -> x offset of leftmost process_idx in that band
        flux_col_width = {}  # fluxer_step -> width of that band
        fc = 0.0
        for fs in flux_steps_present:
            flux_col_x0[fs] = fc
            w = max(len(flux_step_proc_idxs.get(fs, {0})), 1) * x_sub_gap
            flux_col_width[fs] = w
            fc += w + x_col_gap
        flux_total_width = fc - x_col_gap  # trim trailing gap

        # dense rank of process_idx within each fluxer_step (so gaps don't create whitespace)
        flux_step_proc_rank = {}  # (fs, proc_idx) -> x offset within that step's band
        for fs, proc_idxs in flux_step_proc_idxs.items():
            for rank, proc_idx in enumerate(sorted(proc_idxs)):
                flux_step_proc_rank[(fs, proc_idx)] = rank * x_sub_gap

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

        # --- 6. Jitter ---
        def _jitter(n, scale=0.35):
            h = int(hashlib.md5(str(n).encode()).hexdigest(), 16)
            return (h % 1000) / 1000.0 * scale - scale / 2

        # --- 7. Build positions ---
        pos = {}
        flux_x0 = step_x_origin[fluxer_name]
        for n in step_nodes[fluxer_name]:
            fs = get_flux_step(n)
            proc_x = flux_step_proc_rank.get((fs, n[1]), 0)
            x = flux_x0 + flux_col_x0.get(fs, 0) + proc_x
            y = -flux_global_rank[n[2]] * y_spacing + _jitter(n)
            pos[n] = (x, y)

        for step in step_order[1:]:
            x0 = step_x_origin[step]
            for n in step_nodes[step]:
                x = x0 + n[1] * x_sub_gap
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

        # Flux left border = lambda_fail
        x_flux_left = flux_x0 - x_step_gap_base / 2
        ax.axvline(x=x_flux_left, color="black", linestyle="--", linewidth=0.8, alpha=0.4)
        ax.text(x_flux_left, y_max_plot, str(self.fluxer.lambda_fail),
                ha="center", va="bottom", fontsize=8, color="black", alpha=0.7, rotation=90)

        # Internal flux dividers between step-type columns, labelled with interface names
        # Layout left->right:
        # lambda_fail | equil, reset | (visual sep) | to_l-1_fwd | lambda_neg1 | flux_fwd, flux_back | lambda_plus1
        # Single internal flux divider: to_l-1_fwd | lambda_n | flux_fwd, flux_back
        left_present = [fs for fs in ["equilibrate", "reset", "to_l-1_fwd"] if fs in flux_col_x0]
        right_present = [fs for fs in ["flux_fwd", "flux_back"] if fs in flux_col_x0]
        if left_present and right_present:
            x_left_edge = flux_x0 + max(flux_col_x0[fs] + flux_col_width[fs] for fs in left_present)
            x_right_edge = flux_x0 + min(flux_col_x0[fs] for fs in right_present)
            x_div = (x_left_edge + x_right_edge) / 2
            ax.axvline(x=x_div, color="black", linestyle=":", linewidth=0.8, alpha=0.4)
            ax.text(x_div, y_max_plot, str(self.fluxer.lambda_n),
                    ha="center", va="bottom", fontsize=8, color="black", alpha=0.7, rotation=90)

        # Separators between steps.
        # A node in column i has already crossed step[i].lambda_plus1, so that interface
        # belongs to the LEFT of column i (i.e. the right side of column i-1).
        # Concretely: the line between step[i] and step[i+1] is labelled step[i+1].lambda_plus1.
        # The flux column's right separator is shoot1.lambda_plus1 (= the first interface to shoot).
        # The final shooter has no right separator — just success arrows.
        steps_with_nodes = [s for s in step_order if s in step_nodes]
        # build label for the line to the RIGHT of each step (None = no line)
        # step_order: [fluxer, shoot1, shoot2, ..., shootN]
        # line to right of step[i] = lambda_plus1 of step[i+1], except final shooter has none
        step_objects = {fluxer_name: self.fluxer, **{s.name: s for s in self.shooters}}
        for i, step in enumerate(step_order[:-1]):  # all except final shooter
            if step not in step_nodes:
                continue
            next_step = step_order[i + 1]
            label = str(step_objects[next_step].lambda_plus1)
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