import copy
import functools
import itertools
import math
import os
import shutil
import tempfile
import threading
import time
import warnings
from abc import ABC, abstractmethod
from numbers import Number
from pathlib import Path
from typing import Optional, Iterable, Callable, Union
import traceback

import oxpy
import py

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize, LogNorm
from matplotlib.lines import Line2D

from matplotlib.ticker import MaxNLocator
from pandas.errors import EmptyDataError

import json

from oxDNA_analysis_tools.UTILS.boilerplate import PathContext
from .metasimulation import VMMCMetaSimulation
from .vmmc import VirtualMoveMonteCarlo
from .vmmc_replicas import VmmcReplicas, _replica_colors
from ..structure_editor.dna_structure import DNAStructure
from .vmmc_data import VMMCData
from ..utils.order_parameter import OrderParameter

class VmmcWindow(VmmcReplicas):
    """
    Abstract base class for VMMC windows.
    """

    # a state is a tuple of integers representing order parameter valuesg
    state_space_area: set[tuple[int, ...]]

    def __init__(self,
                 sim_dir: Path,
                 n_replicas: int,
                 state_space_area: set[tuple[int, ...]],
                 starting_conf: DNAStructure):
        """
        Initialize the VmmcWindow with specified state space area and starting configuration.

        Parameters:
        state_space_area (set[tuple[int, ...]]): The set of states defining the window.
        starting_conf (DNAStructure): The starting configuration for simulations in this window.
        """
        super().__init__(starting_conf, sim_dir, n_replicas)
        self.state_space_area = state_space_area

    def merge_hist(self) -> pd.DataFrame:
        return functools.reduce(lambda x, y: x.add(y, fill_value=0),
                                [sim.analysis.vmmc_df for sim in self])

    def state_space_of(self, op: int) -> set[int]:
        """
        Get the set of values for a specific order parameter across the window's state space area.
        """
        return set(state[op] for state in self.state_space_area)

    def plot_vmmc_scatter(self, op=0, ax=None):
        """
        Scatter plot of vmmc_df["unwt_occ"] for this window.

        Parameters
        ----------
        op : int
            Order parameter index to use on x-axis
        ax : matplotlib Axes (optional)

        Returns
        -------
        ax
        """

        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 4))

        colors = _replica_colors(len(self.simulations))
        marker_styles = ["o", "s", "^", "v", "D", "P", "X", "*", "<", ">"]

        sim_handles = {}
        marker_handles = {}

        for sim_idx, sim in enumerate(self.simulations):
            df = sim.analysis.vmmc_df
            color = colors[sim_idx]

            n_ops = df.index.nlevels
            other_ops = [i for i in range(n_ops) if i != op]

            if other_ops:
                grouped = df.groupby(level=other_ops)
            else:
                grouped = [((), df)]

            marker_cycle = itertools.cycle(marker_styles)

            for marker, (other_vals, subdf) in zip(marker_cycle, grouped):

                x = subdf.index.get_level_values(op)
                y = subdf["unwt_occ"].values

                ax.scatter(x, y, color=color, marker=marker, alpha=0.8)

                # Collect legend entries (avoid duplicates)
                if sim_idx not in sim_handles:
                    sim_handles[sim_idx] = Line2D(
                        [0], [0], marker='o', color='w',
                        markerfacecolor=color, markersize=8,
                        label=f"Sim {sim_idx}"
                    )

                marker_label = f"Other OP = {other_vals}"
                if marker_label not in marker_handles:
                    marker_handles[marker_label] = Line2D(
                        [0], [0], marker=marker, color='k',
                        linestyle='None', markersize=8,
                        label=marker_label
                    )

        ax.set_xlabel(f"Order Parameter {op}")
        ax.set_ylabel("unwt_occ")
        ax.set_title(self.sim_dir.name)
        ax.grid(alpha=0.2)

        # Combine legends
        handles = list(sim_handles.values()) + list(marker_handles.values())
        labels = [h.get_label() for h in handles]

        ax.legend(
            handles,
            labels,
            fontsize="small",
            loc="upper left",
            bbox_to_anchor=(1.02, 1),
            borderaxespad=0
        )
        return ax

    def create_state_histograms(self, op: OrderParameter, ax: Optional[plt.Axes]=None):
        fig = super().create_state_histograms(op, ax)
        op_idx = self[0].list_order_parameters().index(op)

        # --- show vertical lines for min/max OP value in this window ---
        window_op_vals = sorted(set(state[op_idx] for state in self.state_space_area))
        if window_op_vals:
            min_val = window_op_vals[0]
            max_val = window_op_vals[-1]
            # map values to x positions (indices in x_labels)
            min_pos = min_val
            max_pos = max_val
            # min_pos = x_labels.index(min_val)
            # max_pos = x_labels.index(max_val)

            # determine bar width (bars are centered on integer x positions).
            # try to get exact width from the first patch; fall back to 0.8

            bar_width = 0.8

            # place the lines at the left edge of the min bar and right edge of the max bar
            left_line_x = min_pos - (bar_width / 2.0)
            right_line_x = max_pos + (bar_width / 2.0)
            if ax is not None:
                ax.axvline(x=left_line_x, linestyle='--', color="black", linewidth=1.0, alpha=0.8)
                ax.axvline(x=right_line_x, linestyle='--', color="black", linewidth=1.0, alpha=0.8)
        return fig

class VmmcWindowing(VMMCMetaSimulation):
    """
    A class to handle windowing for VMMC simulations.
    """


    # list of windows. each window is a set of tuples representing states
    # list of starting states for simulations of each window
    window_sim_start_states: dict[int, list[tuple[int, ...]]]

    # number of steps at which to check that literally any moves have been accepted
    # gonna property the first one so it'll auto-set a no_accept function on set
    __no_accept_moves_threshold: Optional[int] = None
    no_accept_moves_callback: Optional[Callable[[VmmcWindow], None]] = None

    def __init__(self, tld_path: Path):
        """
        Initialize the VmmcWindowing with specified window size and overlap.

        Parameters:
        window_size (int): The size of each window.
        overlap (int): The number of overlapping elements between consecutive windows.
        """
        super().__init__(tld_path)

    @property
    def no_accept_moves_threshold(self) -> Optional[int]:
        return self.__no_accept_moves_threshold

    @no_accept_moves_threshold.setter
    def no_accept_moves_threshold(self, value: Number):
        if not isinstance(value, Number):
            raise ValueError("no_accept_moves_threshold must be a number.")
        if value <= 0:
            raise ValueError("no_accept_moves_threshold must be positive.")
        if self.__no_accept_moves_threshold is None:
            def default_callback(w: VmmcWindow):
                raise RuntimeError(f"Window {w.sim_dir.name} has not accpeted accepted moves in {int(value)} steps!")
            self.no_accept_moves_callback = default_callback
        self.__no_accept_moves_threshold = int(value)


    def overlap(self, window_1: int, window_2: int) -> set[tuple[int, ...]]:
        """
        Get the overlapping states between two windows.
        """
        return self[window_1].state_space_area.intersection(self[window_2].state_space_area)

    def add_window(self, window_states: set[tuple[int, ...]], starting_conf: DNAStructure):
        """
        Add a new window with the given states.
        """
        if not all(isinstance(state, tuple) and len(state) == len(self.order_parameters()) for state in window_states):
            raise ValueError("window_states must be a set of tuples of length equal to the order_parameters().")
        assert all([all([0 <= v < len(self.order_parameters()[i]) for  i,v in enumerate(state)])for state in window_states])
        self._subgroups.append(VmmcWindow(
            sim_dir=self.tld / f"window_{len(self)}",
            state_space_area=window_states,
            starting_conf=starting_conf,
            n_replicas=self.n_reps,
        ))

    def get_data(self, window_index: int) -> VMMCData:
        """
        Abstract method to get VMMC data for a specific window.
        """
        raise NotImplementedError("Subclasses must implement get_data method.")

    def load(self, ignore_no_json: bool = False):
        if (self.tld / "setup.json").exists():
            with (self.tld / "setup.json").open("r") as f:
                settings = json.load(f)
            for op_dict in settings["order_parameters"]:
                op = OrderParameter.from_dict(op_dict)
                self.add_order_parameter(op)
            self.n_reps = settings["n_reps"]
            self.extrapolate_hist_Ts = settings["extrapolate_hist_Ts"]
            self._subgroups = []
            for i, window_settings in enumerate(settings["windows"]):
                self._subgroups.append(VmmcWindow(
                    sim_dir=self.tld / f"window_{i}",
                    n_replicas=self.n_reps,
                    state_space_area=set(tuple(state) for state in window_settings["state_space_area"]),
                    starting_conf=None,  # starting conf is not stored; user must provide if needed
                ))
        elif not ignore_no_json:
            raise FileNotFoundError(f"No setup.json found in directory {str(self.tld)}")

        for window in self:
            if window.sim_dir.exists():
                window.init()
                for sim in window:
                    sim.input.read_input()
                    sim.read_order_parameters()
                    sim.load_weights()
                sims = list(window)
                if len(sims) > 1:
                    ref_weights = sims[0].weights
                    mismatched_details = []
                    for i, sim in enumerate(sims[1:], start=1):
                        if not np.allclose(sim.weights, ref_weights, rtol=1e-9, equal_nan=True):
                            diff_idx = np.argwhere(~np.isclose(sim.weights, ref_weights, rtol=1e-9, equal_nan=True))
                            lines = [
                                f"state {tuple(int(x) for x in idx)}: replica_0={ref_weights[tuple(idx)]}, replica_{i}={sim.weights[tuple(idx)]}"
                                for idx in diff_idx[:10]
                            ]
                            if len(diff_idx) > 10:
                                lines.append(f"... and {len(diff_idx) - 10} more differing state(s)")
                            mismatched_details.append(
                                f"  replica {i} ({len(diff_idx)} differing state(s)):\n    " + "\n    ".join(lines)
                            )
                        elif not np.array_equal(sim.weights, ref_weights):
                            diff_idx = np.argwhere(sim.weights != ref_weights)
                            lines = [
                                f"state {tuple(int(x) for x in idx)}: replica_0={repr(ref_weights[tuple(idx)])}, replica_{i}={repr(sim.weights[tuple(idx)])}"
                                for idx in diff_idx[:10]
                            ]
                            if len(diff_idx) > 10:
                                lines.append(f"... and {len(diff_idx) - 10} more")
                            warnings.warn(
                                f"Window {window.sim_dir.name}: replica {i} weights differ from replica 0 "
                                f"within floating-point tolerance ({len(diff_idx)} state(s)):\n    "
                                + "\n    ".join(lines)
                            )
                    if mismatched_details:
                        raise ValueError(
                            f"Window {window.sim_dir.name}: weight mismatch between replicas:\n"
                            + "\n".join(mismatched_details)
                        )

    def check_ready(self):
        covered_states = set()
        for window in self:
            covered_states.update(window.state_space_area)
        # check that all states are covered
        uncovered_states = set(self.legal_state_list) - covered_states
        if uncovered_states:
            raise ValueError("Not all legal states are covered by the defined windows. Uncovered states: "
                             + ",".join([str(state) for state in uncovered_states]))

    def _probe_starting_state(self, sim: VirtualMoveMonteCarlo) -> tuple[int, ...]:
        """
        Run a 1-step probe simulation in a temp dir to read the initial OP state of
        the starting configuration.  Uses uniform weights so that starting outside a
        window's allowed region does not cause a division-by-zero in oxDNA.

        Called during setup() after build_replica() has written the simulation files
        but before build_start_weights() zeros out out-of-window states.
        """
        with tempfile.TemporaryDirectory() as _tmpdir:
            tmpdir = Path(_tmpdir)

            # Copy all flat files from the built sim_dir into the probe directory.
            for src_file in sim.sim_dir.iterdir():
                if src_file.is_file():
                    shutil.copy(src_file, tmpdir / src_file.name)

            # Read the input, patch for a minimal 1-step run, and keep it in-memory.
            with (tmpdir / "input.json").open("r") as fh:
                probe_input = json.load(fh)
            probe_input["steps"] = 1
            probe_input["print_energy_every"] = 1
            probe_input["print_conf_interval"] = 999999
            energy_filename = probe_input.get("energy_file", "energy.dat")

            # Overwrite the weight file with uniform weights.
            # Starting in a zero-weight state (outside the window) would cause
            # W_old = 0 → division by zero in the VMMC acceptance ratio.
            weights_filename = probe_input.get("weights_file", "wfile.txt")
            all_coords = itertools.product(*[range(len(op)) for op in sim.list_order_parameters()])
            with (tmpdir / weights_filename).open("w") as wf:
                for coord in all_coords:
                    wf.write(" ".join(str(c) for c in coord) + " 1.0\n")

            # Run via oxpy (blocking), suppressing stdout/stderr.
            capture = py.io.StdCaptureFD()
            try:
                with PathContext(tmpdir), oxpy.Context():
                    ox_input = oxpy.InputFile()
                    for k, v in probe_input.items():
                        ox_input[k] = str(v)
                    manager = oxpy.OxpyManager(ox_input)
                    manager.run_complete()
                    del manager
            finally:
                capture.reset()

            # Read the first row of the energy file: the OP state after ≤1 accepted move.
            energy_path = tmpdir / energy_filename
            if not energy_path.exists() or energy_path.stat().st_size == 0:
                raise RuntimeError(
                    "Starting-state probe produced no energy output. "
                    "Cannot verify that the starting configuration is inside the window."
                )
            df = pd.read_csv(energy_path, sep=r'\s+', header=None)
            op_names = [op.name for op in sim.list_order_parameters()]
            df.columns = ["time", "U", "p_T", "p_R", "p_V"] + op_names + ["weight"]
            if df.empty:
                raise RuntimeError(
                    "Starting-state probe energy file is unexpectedly empty."
                )
            first_row = df.iloc[0]
            return tuple(int(first_row[name]) for name in op_names)

    def setup(self, windows: Optional[list[int]] = None):
        """
        Docstring for setup
        
        consideration: for each window, each state has the following classifications: legal vs. illegal and in vs. out

        :param self: Description
        :param windows: Description
        :type windows: Optional[list[int]]
        """
        if windows is None:
            windows = list(range(len(self)))
        self.check_ready()
        # find set of all states (incl. illegal)
        all_states = set(itertools.product(*[range(len(op)) for op
                                            in self.order_parameters()]))
        for i_window in windows:
            window = self[i_window]
            window.init()
            window.temperatures = self.extrapolate_hist_Ts
            # construct a weight mask that includes all states (legal or not) outside the window
            weight_mask = set(all_states) - window.state_space_area
            sim: VirtualMoveMonteCarlo # type hint
            for i_sim, sim in enumerate(window):
                for op in self.order_parameters():
                    sim.add_order_parameter(op)
                # build simulation
                self.build_replica(self, sim)

                # Verify the starting configuration's initial OP state is inside
                # the window.  Only probe replica 0 — all replicas share the same
                # starting conf so the state is identical for each.
                if i_sim == 0:
                    initial_state = self._probe_starting_state(sim)
                    if initial_state not in window.state_space_area:
                        raise ValueError(
                            f"Window {i_window}: the starting configuration's initial "
                            f"OP state {initial_state} is outside this window's state "
                            f"space {sorted(window.state_space_area)}. "
                            f"Provide a file_dir whose starting configuration lies "
                            f"within the window's state space."
                        )

                self.build_start_weights(sim, i_window)

                if np.isinf(sim.weights).any():
                    raise ValueError("`build_start_weights` has produced a `inf` value")

                # Get legal-state coordinate list (expected: iterable of tuples)
                legal_state_coords_list = self.filter_legal_states(window.state_space_area)  # ensure list semantics
                if not isinstance(legal_state_coords_list, list):
                    raise ValueError(f"custom `filter_legal_states` function returned illegal type `{type(legal_state_coords_list)}`")

                # Handle empty case quickly: nothing to normalize; zero everything outside
                if len(legal_state_coords_list) == 0:
                    raise ValueError(f"No legal states in window {i_window}")
                legal_state_coords_arr = np.asarray(legal_state_coords_list)

                # Validate coords shape: should be (N, K) where K == sim.weights.ndim
                if legal_state_coords_arr.ndim != 2:
                    raise ValueError(
                        "window.window_legal_state_space must be an iterable of coordinate tuples; "
                        f"got coords.ndim = {legal_state_coords_arr.ndim}, shape = {legal_state_coords_arr.shape}"
                    )

                num_coord_axes = legal_state_coords_arr.shape[1]
                if num_coord_axes != sim.weights.ndim:
                    raise ValueError(
                        "Coordinate tuples length does not match sim.weights dimensionality. "
                        f"coords tuple-length = {num_coord_axes}, sim.weights.ndim = {sim.weights.ndim}"
                    )
                

                # Build tuple-of-index-arrays for fancy indexing of the points
                idx_tuple = tuple(legal_state_coords_arr.T.astype(int))  # (axis0_idxs, axis1_idxs, ...)

                # Select the listed points (returns 1-D array of length N)
                selected_values = sim.weights[idx_tuple]

                # Check for zeros among selected values
                zero_mask = (selected_values == 0)
                if zero_mask.any():
                    zero_coords = legal_state_coords_arr[zero_mask]  # shape (M, K) for M zero points
                    coord_str = ",".join(map(str, map(tuple, zero_coords)))
                    raise ValueError(
                        "Some states in the window have zero weight! States: " + coord_str
                    )

                # Normalize only the selected (legal-window) values by their minimum
                minval = selected_values.min()
                if minval == 0:
                    # Redundant guard (we checked above), but keep safe
                    raise ValueError("Minimum selected weight is zero (unexpected).")
                # Assign back normalized values
                sim.weights[idx_tuple] = selected_values / minval

                # Zero-out everything in weight_mask (the complement)
                if len(weight_mask) > 0:
                    mask_coords = np.asarray(list(weight_mask))
                    if mask_coords.ndim != 2 or mask_coords.shape[1] != sim.weights.ndim:
                        # Something's wrong with weight_mask content
                        raise ValueError(
                            "weight_mask contains coordinates with wrong dimensionality: "
                            f"mask_coords.shape = {mask_coords.shape}, sim.weights.ndim = {sim.weights.ndim}"
                        )
                    idx_mask = tuple(mask_coords.T.astype(int))
                    sim.weights[idx_mask] = 0.

                if np.isnan(sim.weights).any():
                    raise ValueError("Some NaN weights found!!")

                # Continue building files and set seed
                sim.build_vmmc_weight_file()
                sim.build_vmmc_op_file(clear_file=True)
                # todo: does seed need to be better
                sim.input["seed"] = 1234 + i_sim + i_window * 1000
        self.cache_settings()

    def cache_settings(self):
        """
        caches settings in file "setup.json" in tld directory
        """
        with (self.tld / "setup.json").open("w") as f:
            settings = {
                "n_reps": self.n_reps,
                "order_parameters": [op.to_dict() for op in self.order_parameters()],
                "extrapolate_hist_Ts": self.extrapolate_hist_Ts,
                "windows": [
                    {
                        "state_space_area": list(window.state_space_area),
                    } for window in self
                ],
            }
            json.dump(settings, f, indent=4)

    def run(self, windows: Optional[list[int]] = None, join: bool=True):
        if windows is None:
            windows = list(range(len(self)))
        if not all(self[idx].is_set_up() for idx in windows):
            raise ValueError("Some windows have not been set.") # todo better error type
        if (self.__no_accept_moves_threshold is not None) != (self.no_accept_moves_callback is not None):
            raise ValueError("Both no_accept_moves_threshold and no_accept_moves_callback must be set together.")

        for idx in windows:
            for sim in self[idx]:
                sim.oxpy_run()
        observer_threads = []
        if self.__no_accept_moves_threshold is not None:
            # start observers for no-accept-move detection
            for idx in windows:
                window = self[idx]
                observer_thread = threading.Thread(
                    target=self._observe_no_accept_moves,
                    args=(
                        window,
                        self.__no_accept_moves_threshold,
                        self.no_accept_moves_callback,
                    ),
                    name=f"no_accept_observer_window_{idx}",
                    daemon=True,
                )
                observer_threads.append(observer_thread)
                observer_thread.start()
        if join:
            for idx in windows:
                for sim in self[idx]:
                    sim.oxpy_run.process.join()
            for t in observer_threads:
                t.join()

    def plot_window_weights(self,
                            plot_ops=None,
                            const_ops=None,
                            windows: Optional[Union[list[int], int]] = None,
                            use_log: bool = True) -> tuple[plt.Figure, np.ndarray]:
        """
        Plot weights for each window as vertically stacked subplots with a shared x-axis.
        Assumes all replicas within a window share the same weights; uses the first replica.

        :param plot_ops: op identifier(s) passed to VirtualMoveMonteCarlo.plot_weights.
                         Defaults to the first order parameter.
        :param const_ops: optional const_ops dict passed through to plot_weights.
        :param windows: window indices to plot. Defaults to all windows.
        :param use_log: passed through to plot_weights.
        """
        if plot_ops is None:
            plot_ops = 0
        if windows is None:
            windows = list(range(len(self)))
        elif isinstance(windows, int):
            windows = [windows]

        n_windows = len(windows)
        fig, axes = plt.subplots(n_windows, 1, sharex=True, figsize=(8, 3 * n_windows))
        if n_windows == 1:
            axes = np.array([axes])

        is_2d = isinstance(plot_ops, (list, tuple)) and len(plot_ops) == 2

        for i, window_idx in enumerate(windows):
            ax = axes[i]
            self[window_idx][0].plot_weights(plot_ops, const_ops=const_ops, use_log=use_log, ax=ax)
            ax.set_title("")
            if is_2d:
                op2 = self.order_parameters()[plot_ops[1]]
                ax.set_ylabel(op2.name if hasattr(op2, "name") else f"op_{plot_ops[1]}")
            else:
                ax.set_ylabel(f"Window {window_idx}\nWeight")

        # shared title on top subplot only
        op = self.order_parameters()[plot_ops if isinstance(plot_ops, int) else plot_ops[0]]
        axes[0].set_title(f"Weights vs {op.name}")
        op1_idx = plot_ops[0] if is_2d else (plot_ops if isinstance(plot_ops, int) else plot_ops[0])
        max_op1_val = max(
            state[op1_idx]
            for window in self._subgroups
            for state in window.state_space_area
        )
        for ax in axes:
            ax.set_xlim(-0.5, max_op1_val + 0.5)

        if is_2d:
            # y-axis is op2; cap at the global max op2 value across all windows
            op2_idx = plot_ops[1]
            max_op2_val = max(
                state[op2_idx]
                for window in self._subgroups
                for state in window.state_space_area
            )
            for ax in axes:
                ax.set_ylim(-0.5, max_op2_val + 0.5)
        else:
            # standardize y-limits across all subplots
            all_ylims = [ax.get_ylim() for ax in axes]
            global_ymin = min(ylim[0] for ylim in all_ylims)
            global_ymax = max(ylim[1] for ylim in all_ylims)
            for ax in axes:
                ax.set_ylim(global_ymin, global_ymax)

        plt.tight_layout()
        return fig, axes

    def visualize(self, bond_op_index):
        pass

    def _observe_no_accept_moves(
            self,
            window: VmmcWindow,
            threshold: int,
            callback: Callable[[VmmcWindow], None],
            poll_interval: float = 1.0,
    ):
        """
        Observe a window until either:
          - any replica accepts a move (exit silently)
          - all replicas exceed threshold steps without acceptance (call callback)
        """

        # initialize last-known accept steps
        while True:
            all_exceeded = True
            for i, sim in enumerate(window):
                try:
                    sim.analysis.load_energy()
                except AttributeError:
                    all_exceeded = False
                    continue # simfiles.energy can't be set yet b/c energy.txt doesn't exist
                except FileNotFoundError:
                    all_exceeded = False
                    continue # file doesn't exist (alternetive)
                except EmptyDataError:
                    all_exceeded = False
                    continue
                df = sim.analysis.energy_df
                op_names = [op.name for op in sim.list_order_parameters()]
                if (df[op_names] != df[op_names].loc[0]).any().any():
                    print(f"Window {window.sim_dir.name}, replica {i} has accepted a move.")
                    return
                elif df["time"].iloc[-1] < threshold:
                    all_exceeded = False

            if all_exceeded:
                callback(window)

            time.sleep(poll_interval)

    def plot_window_data(self, op: Optional[OrderParameter] = None, plot_w = 10, windows: Optional[Union[list[int], int]] = None):
        """
        plot window histograms
        stacked vertical subplots for each window
        x axis is shared across subplots and is state
        y axis is frequency
        for replicas within each window, plot the total frequency as stacked bars of each replica
        """

        if op is None:
            op = self.order_parameters()[0]
        if windows is None:
            windows = range(len(self))
        elif isinstance(windows, int):
            windows = [windows]
        

        # index of this OP in a state tuple (kept for compatibility with other code)
        op_idx = next(i for i, o in enumerate(self.order_parameters()) if o.name == op.name)

        # collect the full set of possible op values (ensures consistent x-axis across windows)
        op_val_range = sorted(set(state[op_idx] for state in self.legal_state_list))

        if not self._subgroups:
            raise ValueError("No windows defined to plot.")

        n_windows = len(windows)
        # Share x only down each column, not across both columns.
        fig, axes = plt.subplots(n_windows, 2, sharex='col', figsize=(16, 3 * n_windows))

        # ensure axes is an (n_windows, 2) array so indexing axes[i][0]/axes[i][1] always works
        if n_windows == 1:
            axes = np.expand_dims(axes, axis=0)

        # x positions and tick labels
        x_labels = list(op_val_range)
        x_pos = np.arange(len(x_labels))

        # --- compute global max stacked height across all windows ---
        global_max = 0.0

        for idx in windows:
            window = self[idx]
            op_counts_total = np.zeros(len(x_labels), dtype=float)

            for vmmc in window:
                vmmc_data = vmmc.analysis.get_data_over(op).df
                counts_by_val = dict(zip(vmmc_data.index.values,
                                         vmmc_data['count'].values))
                for xi, val in enumerate(x_labels):
                    op_counts_total[xi] += counts_by_val.get(val, 0.0)

            global_max = max(global_max, op_counts_total.max())

        for i, window_idx in enumerate(windows):
            window: VmmcWindow = self[window_idx]
            # window is expected to be an iterable of replicas (vmmc objects)

            ax: plt.Axes = axes[i][0]

            window.create_state_histograms(op, ax)
            ax.set_ylabel(f"Window {window_idx}\nFrequency")
            ax.set_ylim(0, global_max * 1.05)
            ax.set_xticks(x_pos)
            ax.set_xlim(0, max(op_val_range))
            if window.nreplicas > 1 and i == 0:
                ax.legend(loc="upper right", fontsize="small", frameon=False)

            if n_windows > 3:
                ax.tick_params(labelbottom=True)
            ax_r: plt.Axes = axes[i][1]

            # draw bond curves
            window.plot_op_val_curve(op, ax_r, show_legend=i == 0)

            window_op_vals = sorted(set(state[op_idx] for state in window.state_space_area))
            min_val = window_op_vals[0]
            max_val = window_op_vals[-1]

            pad = 0.1  # small visual padding away from integer bin edges

            # upper bound (always draw)
            upper_y = max_val + 0.5 + pad
            ax_r.axhline(y=upper_y, color="black", linestyle='--', linewidth=1.0, alpha=0.8)

            # lower bound (skip if min_val == 0)
            if min_val != 0:
                lower_y = min_val - 0.5 - pad
                ax_r.axhline(y=lower_y, color="black", linestyle='--', linewidth=1.0, alpha=0.8)


        axes[-1][0].set_xticks(x_pos)
        axes[-1][0].set_xticklabels([str(l) for l in x_labels],
                                    rotation=0,
                                    ha="center")
        axes[-1][0].set_xlabel(op.name if hasattr(op, "name") else "state")
        if n_windows > 3:
            for ax, _ in axes:
                ax.set_xticks(x_pos)
                ax.set_xticklabels([str(l) for l in x_labels],
                                   rotation=0,
                                   ha="center")

        plt.tight_layout()
        return fig, axes

    def plot_window_data_2d(self, op1: Optional[Union[int, OrderParameter]] = None,
                            op2: Optional[Union[int, OrderParameter]] = None,
                            windows: Optional[Union[list[int], int]] = None,
                            **kwargs):
        """
        Plot 2D frequency heatmaps for each window over two order parameters.
          - x axis = op1 values
          - y axis = op2 values
          - color   = total unweighted occupancy (summed across replicas)
        """

        if op1 is None:
            op1 = 0
        if isinstance(op1, int):
            op1 = self.order_parameters()[0]
        if op2 is None:
            op2 = 1
        if isinstance(op2, int):
            op2 = self.order_parameters()[op2]
        if windows is None:
            windows = range(len(self._subgroups))
        elif isinstance(windows, int):
            windows = [windows]

        log_scale = kwargs.get('log_scale', False)

        op1_idx = next(i for i, o in enumerate(self.order_parameters()) if o.name == op1.name)
        op2_idx = next(i for i, o in enumerate(self.order_parameters()) if o.name == op2.name)

        op1_name = op1.name
        op2_name = op2.name

        # full sorted value ranges for each OP across all legal states
        op1_vals = sorted(set(state[op1_idx] for state in self.legal_state_list))
        op2_vals = sorted(set(state[op2_idx] for state in self.legal_state_list))

        op1_pos = {v: i for i, v in enumerate(op1_vals)}
        op2_pos = {v: i for i, v in enumerate(op2_vals)}

        n_windows = len(windows)
        ncols = min(n_windows, 4)
        nrows = math.ceil(n_windows / ncols)

        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
        axes = np.array(axes).reshape(nrows, ncols)

        # pre-compute heatmap matrices and global max for consistent colorscale
        global_max = 0.0
        heat_matrices = {}

        for window_idx in windows:
            window = self._subgroups[window_idx]
            mat = np.zeros((len(op2_vals), len(op1_vals)), dtype=float)

            for vmmc in window:
                # accumulate occupancy
                series = vmmc.analysis.vmmc_df["unwt_occ"]
                grouped = series.groupby(level=[op1_name, op2_name]).sum()
                for (v1, v2), occ in grouped.items():
                    xi = op1_pos.get(v1)
                    yi = op2_pos.get(v2)
                    if xi is not None and yi is not None:
                        mat[yi, xi] += occ

            w = window[0].weights
            # sum over all axes that are NOT op1_idx or op2_idx
            axes_to_sum = tuple(i for i in range(w.ndim) if i not in (op1_idx, op2_idx))
            if axes_to_sum:
                w = w.sum(axis=axes_to_sum)
            # ensure layout is (op2, op1) to match mat
            if op1_idx < op2_idx:
                w = w.T

            # index into w using the actual OP values (legal state values may be
            # a subset of the full weight matrix dimensions)
            op1_indices = np.array(op1_vals)
            op2_indices = np.array(op2_vals)
            w_sliced = w[np.ix_(op2_indices, op1_indices)]

            mask = (w_sliced == 0)
            mat_masked = np.ma.masked_where(mask, mat)
            if log_scale:
                mat_display = np.ma.masked_where(mask, np.clip(mat, 1, None))
            else:
                mat_display = mat_masked

            heat_matrices[window_idx] = mat_display

            global_max = max(global_max, mat_masked.compressed().max() if mat_masked.compressed().size > 0 else 0)

        norm = LogNorm(vmin=1, vmax=global_max) if log_scale else Normalize(vmin=0, vmax=global_max)

        max_op2_val = max(state[op2_idx] for window in self._subgroups
                         for state in window.state_space_area)
        max_op2_pixel = op2_pos[max_op2_val]

        for plot_i, window_idx in enumerate(windows):
            row, col = divmod(plot_i, ncols)
            ax: plt.Axes = axes[row, col]
            mat_masked = heat_matrices[window_idx]

            cmap = plt.cm.viridis.copy()
            cmap.set_bad(color='white')

            im = ax.imshow(mat_masked, origin='lower', aspect='auto',
                           norm=norm,
                           cmap=cmap,
                           extent=[-0.5, len(op1_vals) - 0.5,
                                   -0.5, len(op2_vals) - 0.5])
            ax.set_facecolor('white')

            # dashed rectangle showing this window's state space boundary
            window = self._subgroups[window_idx]
            w_op1 = sorted(set(state[op1_idx] for state in window.state_space_area))
            w_op2 = sorted(set(state[op2_idx] for state in window.state_space_area))
            if w_op1 and w_op2:
                x0 = op1_pos[w_op1[0]] - 0.5
                x1 = op1_pos[w_op1[-1]] + 0.5
                y0 = op2_pos[w_op2[0]] - 0.5
                y1 = op2_pos[w_op2[-1]] + 0.5
                rect = plt.Rectangle((x0, y0), x1 - x0, y1 - y0,
                                     linewidth=1.5, edgecolor='white',
                                     linestyle='--', facecolor='none')
                ax.add_patch(rect)

            ax.set_xticks(range(len(op1_vals)))
            ax.set_xticklabels([str(v) for v in op1_vals], rotation=90, fontsize=7)
            ax.set_yticks(range(len(op2_vals)))
            ax.set_yticklabels([str(v) for v in op2_vals], fontsize=7)
            ax.set_xlabel(op1_name)
            ax.set_ylabel(op2_name)
            ax.set_ylim(-0.5, max_op2_pixel + 0.5)
            # todo: mins
            if "op1_max" in kwargs:
                ax.set_xlim(0, kwargs["op1_max"] )
            if "op2_max" in kwargs:
                ax.set_xlim(0, kwargs["op2_max"] )
            ax.set_title(f"Window {window_idx}")
            fig.colorbar(im, ax=ax, label="Unweighted occupancy")

        # hide unused axes
        for plot_i in range(len(windows), nrows * ncols):
            row, col = divmod(plot_i, ncols)
            axes[row, col].set_visible(False)

        plt.tight_layout()
        return fig, axes

    def export_merged_hists(self, fname: str = "merged_hist.dat"):
        """
        merge the histograms of each window and write to a `merged_hist.dat` file
        """
        for window in self._subgroups:
            df = window.merge_hist()
            Ts = self.extrapolate_hist_Ts

            output_path = window.sim_dir / fname

            with open(output_path, "w") as f:
                # Write header line
                # todo: use oxdna units as used in oxdna output
                Ts_str = " ".join(T for T in Ts)
                f.write(f"#t = n/a; extr. Ts: {Ts_str}\n")

                # Write dataframe (space separated, no header/index)
                df.to_csv(
                    f,
                    sep=" ",
                    header=False,
                    float_format="%.6g"
                )

    def wham(self, op: Optional[int]=None) -> dict:
        """
        based on some of Petr's old code
        todo: make not explode
        """

        # weight_files: list of weight files, last_hists: list of last_Hists files
        # column_id: column corresponding to the desired temperature in last_hist_file
        # (default is op_dim + 1)
        # Warning! weight files need to be different files. Otherwise the last_hist files need to be merged together
        count_col = 'unbiased_count' if op is not None else 'unwt_occ'
        weights = []
        last_hists: list[pd.DataFrame] = []
        counters = []


        for window in self._subgroups:
            new_weight = window[0].analysis.weights
            if op is not None:
                # DUBIOUS BEHAVIOR: aggregate weights along op by summing them
                new_weight = new_weight.sum(axis=tuple((i for i in range(new_weight.ndim) if i != op)))
            window_dfs = []
            for i, sim in enumerate(window):
                # NEW BEHAVIOR: if op is specified, get data over that OP; otherwise get the overall vmmc_df (which should have all states and counts)
                if op is not None:
                    df = sim.analysis.get_data_over(self.order_parameters()[op]).df
                else:
                    df = sim.analysis.vmmc_df
                window_dfs.append(df)
            hist_data = functools.reduce(lambda x, y: x.add(y, fill_value=0), window_dfs)
            # datas = [sim.analysis.get_data_over(op) for sim in window]
            # hist_data = sum(datas[1:], datas[0]).df

            total_weight = new_weight.sum()

            weights.append(new_weight)
            last_hists.append(hist_data)
            counters.append(total_weight)


        # initialize result structures
        resulting_hist = {}
        if not last_hists:
            return {}

        if op is not None:
            op_idx = next(i for i, o in enumerate(self.order_parameters()) if o.name == self.order_parameters()[op].name)
            keys = sorted(set(state[op_idx] for state in self.legal_state_list))
        else:
            keys = self.legal_state_list
        for key in keys:
            resulting_hist[key] = 0.0

        expminuslogf = np.ones(len(self._subgroups))
        # copy
        expminuslogfnew = expminuslogf[...]
        rho = {}

        iteration = 0
        # iterate until convergence or max iterations
        while (iteration < 10000 and (iteration == 0 or max(abs(expminuslogf - expminuslogfnew)) > 0.0001)):
            iteration += 1
            expminuslogf = copy.deepcopy(expminuslogfnew)

            for state in keys:
                num = 0.0
                denum = 0.0
                for i, window in enumerate(self):
                    # skip states that aren't represented in this window's histogram
                    if state in last_hists[i].index:
                        weight = weights[i][state]
                        pom = last_hists[i].loc[state][count_col]
                        total_weight = counters[i]
                        if weight == 0.0 and pom > 0.0:
                            raise ValueError(
                                f"State {state} has zero weight in window {i} but was visited "
                                f"{pom} times ({count_col}). This indicates a misconfigured "
                                f"window or a state that escaped its intended sampling region."
                            )
                        num += pom * total_weight
                        # avoid dividing by zero in expminuslogf
                        if expminuslogf[i] != 0.0:
                            denum += weight * total_weight / expminuslogf[i]
                if denum == 0.0:
                    rho[state] = 0.0
                else:
                    rho[state] = float(num) / denum

            total = sum(rho.values())
            if total != 0.0:
                for key in rho.keys():
                    rho[key] /= float(total)
            else:
                # if everything is zero, keep rho zeros to avoid division error
                for key in rho.keys():
                    rho[key] = 0.0

            expminuslogfnew = np.zeros(len(self._subgroups))
            # loop windows
            for i in range(len(self._subgroups)):
                # pull window state space area, but only the values of the OP we're aggregating over (if op is specified)
                window_state_set = sorted(set(state[op_idx] for state in self._subgroups[i].state_space_area)) \
                    if op is not None else set(self[i].state_space_area)
                s = 0.0
                for state in window_state_set:
                    assert (weights[i][state] != 0.0) or (rho[state] == 0)
                    s += weights[i][state] * rho[state]
                expminuslogfnew[i] = s

        print('#Converged after %d iterations' % (iteration))
        return rho

    def plot_merged_hist(self, aggregate_over: Optional[OrderParameter] = None):
        if aggregate_over is None:
            aggregate_over = self.order_parameters()[0]
        agg_idx = next(i for i, o in enumerate(self.order_parameters()) if o.name == aggregate_over.name)
        rhos = self.wham()
        op_range = sorted(set(state[agg_idx] for state in self.legal_state_list))
        bins = [
            sum(rhos[state] for state in rhos.keys() if state[agg_idx] == val) for val in op_range
        ]
        fig, ax = plt.subplots(figsize=(8,5))
        ax.bar(op_range, bins, width=1.0, align='center')
        ax.set_xlabel(aggregate_over.name if hasattr(aggregate_over, "name") else "state")
        ax.set_ylabel("Probability")
        ax.yaxis.set_major_locator(MaxNLocator(integer=True, prune='lower'))
        ax.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.7)
        plt.tight_layout()
        return fig, ax

    def plot_free_energy_profile(self, aggregate_over: Optional[OrderParameter]=None):
        if aggregate_over is None:
            aggregate_over = self.order_parameters()[0]
        agg_idx = next(i for i, o in enumerate(self.order_parameters()) if o.name == aggregate_over.name)
        rhos = self.wham(agg_idx)
        op_range = sorted(set(state[agg_idx] for state in self.legal_state_list))
        bins = np.array([rhos.get(val, 0.0) for val in op_range])
        if np.isnan(bins).all():
            raise ValueError("Was unable to merge histograms with WHAM")
        if (bins == 0).all():
            raise ValueError("All states have zero probability; cannot compute free energy profile.")
        free_energy = -np.log(bins[bins > 0])
        sampled_states = np.array(op_range)[bins>0]
        free_energy -= free_energy[0]

        fig, ax = plt.subplots(figsize=(8,5))
        ax.plot(sampled_states, free_energy, "o-")
        ax.set_xlabel(aggregate_over.name if hasattr(aggregate_over, "name") else "state")
        ax.set_ylabel("Free Energy (kT)")
        ax.yaxis.set_major_locator(MaxNLocator(integer=True, prune='lower'))
        ax.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.7)
        plt.tight_layout()
        return fig, ax

    def get_merged_weights(self) -> np.ndarray:
        """
        Merge the weights from all windows into a single weight array covering the full state space.

        For each window after the first, a scalar scale factor is derived from the geometric mean
        of the ratios between the running merged weights and the new window's weights at their
        overlapping (both-nonzero) states. This aligns the two weight scales before averaging,
        preventing misalignment from naive averaging of independently-normalised windows.
        """
        # Load all window weights
        all_weights = []
        for window in self._subgroups:
            window[0].load_weights()
            all_weights.append(window[0].weights.copy())

        if not all_weights:
            return np.zeros(tuple(len(op) for op in self.order_parameters()))

        # Running sum and count for a weighted average
        merged_sum = all_weights[0].copy()
        merged_count = (merged_sum != 0.).astype(float)

        for w in all_weights[1:]:
            # Current running average as the alignment reference
            merged_avg = np.where(merged_count > 0, merged_sum / merged_count, 0.)

            # Overlapping states: nonzero in both reference and incoming window
            overlap_mask = (merged_avg != 0.) & (w != 0.)

            if overlap_mask.any():
                # Geometric mean of per-state ratios — appropriate for multiplicative weights
                log_ratios = np.log(merged_avg[overlap_mask]) - np.log(w[overlap_mask])
                scale = float(np.exp(np.mean(log_ratios)))
            else:
                scale = 1.0

            w_scaled = w * scale

            # Accumulate into running sum/count
            nonzero_w = w_scaled != 0.
            merged_sum[nonzero_w] += w_scaled[nonzero_w]
            merged_count[nonzero_w] += 1.

        return np.where(merged_count > 0, merged_sum / merged_count, 0.)

    def save_merged_weights(self, skip_val: Optional[float] = 0., fname: str = "merged_weights.txt"):
        """
        combine/average weights across all windows, then save to a file in the same format as the
        individual window weight files (space-separated coordinates + weight).
        """
        assert isinstance(skip_val, (float, int)), f"Invalid value for skip_val: {skip_val}"
        assert isinstance(fname, str), f"Invalid value for fname: {fname}"
        merged_weight = self.get_merged_weights()

        # write
        with (self.tld / fname).open("w") as f:
            for weight_coord in itertools.product(*[range(len(op)) for op in self.order_parameters()]):
                if skip_val is None or merged_weight[weight_coord] != skip_val:
                    weight_file_line = " ".join(str(wc) for wc in weight_coord)
                    f.write(f"{weight_file_line} {merged_weight[weight_coord]}\n")

    def __getitem__(self, item: int) -> VmmcWindow:
        return self._subgroups[item]

    def __iter__(self):
        yield from self._subgroups

    @property
    def windows(self) -> list[VmmcWindow]:
        return self._subgroups
