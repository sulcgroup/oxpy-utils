from __future__ import annotations

from pathlib import Path
import re
import warnings
from typing import Callable, Generator, Optional, Union

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Patch
from matplotlib.gridspec import GridSpec

from .metasimulation import VMMCMetaSimulation
from ..utils.util import generate_distinct_colors
from .vmmc_replicas import VmmcReplicas, _replica_colors

from .vmmc import VirtualMoveMonteCarlo
from ..utils.order_parameter import OrderParameter, possible_states


class VMMCAutoReweight(VMMCMetaSimulation):
    """
    class to iteratively run VMMC simulations and automatically adjust weights based on sampling statistics
    TODO: INTEGRATE WITH VMMCREPLICA CLASS (?)
    """

    # function which will be called
    build_replica: Callable[[VMMCAutoReweight, VirtualMoveMonteCarlo], None]

    start_iter_callback: Optional[Callable[[], None]]
    end_iter_callback: Optional[Callable[[], None]]

    # percent std at which to stop iterating
    max_rel_std: float

    steps_per_iter: float

    reweight_borders: Union[bool, float]

    # directory containing the starting .top/.dat (and optionally op.txt/weights.txt)
    # files used to build every iteration's replicas
    starting_conf: Optional[Path]

    def __init__(self, tld_path: Path, sim_build_func: Optional[Callable] = None):
        """
        :param tld_path: path to top-level directory where iterations will be stored
        """
        super().__init__(tld_path, sim_build_func)
        self.reweight_borders = False
        self.zero_sample_weight_factor = 1e4
        self.steps_per_iter = 1e8
        self.max_iterations = float('inf')
        self.max_rel_std = 5.
        self.starting_conf = None

    def load(self):
        """
        load existing iterations from disk
        """
        iteration_dirs = sorted([d for d in self.tld.iterdir() if d.is_dir()],
                                key=lambda d: int(re.match(r"iteration_(\d+)", d.name).group(1)))
        for iteration_directory in iteration_dirs:
            replicas = [
                VirtualMoveMonteCarlo(replica_directory) for replica_directory
                in sorted(iteration_directory.iterdir()) if replica_directory.is_dir()
            ]
            for replica in replicas:
                replica.input.read_input()
                replica.read_order_parameters()
                replica.load_weights()
                if (replica.sim_dir / replica.input["last_hist_file"]).is_file():
                    replica.sim_files.last_hist = replica.sim_dir / "last_hist.dat"
            self._subgroups.append(VmmcReplicas(iteration_directory, iteration_directory, self.n_reps))
            self._subgroups[-1].simulations = replicas
            if not self.order_parameters():
                self._bond_ops = [*replicas[0].bond_ops()]
                self._dist_op = replicas[0].dist_op()

    def check_result(self, last_it: VmmcReplicas) -> bool:
        """
        Check if sampling criteria are met for accessible states only
        """
        # Create a mask for accessible states. vmmc_df/statistics are indexed by the
        # order parameter values themselves, so idx already *is* the state tuple
        # (or bare scalar, if there's only one order parameter).
        def state_is_desired(idx):
            state_tuple = idx if isinstance(idx, tuple) else (idx,)
            # Ignore states where any distance order parameter > 0
            if any(state_tuple[self.num_bond_ops():]):
                return False
            return state_tuple in self.desired_state_list

        # Check if all desired states were sampled in all simulations
        any_unsampled_state = all([
            (sim.analysis.statistics.loc[
                 [idx for idx in sim.analysis.statistics.index if state_is_desired(idx)]
             ]["sampling_percent"].values > 0).all()
            for sim in last_it
        ])

        if not any_unsampled_state:
            print("Some accessible states were not sampled in this iteration, continuing...")
            return False

        # Get sampling std for accessible states only
        return self.get_sampling_std_filtered(last_it) < self.max_rel_std

    def get_sampling_std_filtered(self, it: VmmcReplicas) -> float:
        """
        Get standard deviation of sampling percent across replicas for accessible states only
        """

        # Create mask for accessible states. `row.name` is the DataFrame index
        # label, i.e. the order parameter value(s) for this row (see state_is_desired).
        def state_is_accessible(row):
            idx = row.name
            state_tuple = idx if isinstance(idx, tuple) else (idx,)
            # Ignore states where any distance order parameter > 0
            if any(state_tuple[self.num_bond_ops():]):
                return False
            return state_tuple in self.legal_state_list

        total_sampling: np.ndarray = np.sum([
            sim.analysis.statistics[sim.analysis.statistics.apply(state_is_accessible, axis=1)][
                "sampling_percent"].values
            for sim in it
        ], axis=0)

        # normalize so it's a percent again
        total_sampling /= len(it)
        # compute standard deviation
        return total_sampling.std()

    def check_ready(self):
        if self.extrapolate_hist_Ts is None:
            raise ValueError("No temperatures specified for histogram extrapolation!")
        if not self._bond_ops:
            raise ValueError("No bond order parameters specified for reweighting!")
        if not self.build_replica:
            raise ValueError("No replica building function specified!")
        if self.starting_conf is None:
            raise ValueError("No starting_conf specified! Set self.starting_conf to a directory "
                              "containing the starting .top/.dat files before calling run().")

    def run(self):
        """

        """
        self.check_ready()
        accept_result = False
        while not accept_result:
            if len(self) >= self.max_iterations:
                print("Reached maximum number of iterations, stopping...")
                break
            self.run_iteration()
            accept_result = self.check_result(self[-1])
        print(f"Standard deviation of sampling percent across last iteration: {self.get_sampling_std_filtered(self[-1])}")

    def run_iteration(self):
        """

        """
        print(f"Beginning iteration {len(self)}")
        # construct new iteration
        # it = [VirtualMoveMonteCarlo(self._tld / f"iteration_{len(self)}" / f"replica_{i}") for i in
        #       range(self.n_reps)]
        it = VmmcReplicas(self.starting_conf,  self.tld / f"iteration_{len(self)}", self.n_reps)
        it.init()
        it.temperatures = self.extrapolate_hist_Ts
        # apply settings
        for vmmc in it:
            for op in self.order_parameters():
                if op is not None:
                    vmmc.add_order_parameter(op)
            self.build_replica(self, vmmc)
        # compute new weights based on last iteration
        if len(self):
            print(f"Computing new weights from iteration {len(self)}....")
            weights = self.compute_next_it_weights(self[-1])
            for vmmc in it:
                vmmc.weights[...] = weights
                vmmc.build_vmmc_weight_file()
        # if this is the first iteration:
        else:
            print("Weighting initial simulations....")
            # loop initial replicas
            for rep_idx, vmmc in enumerate(it):
                # apply initial weighting function
                self.build_start_weights(vmmc)
                # update seed. if seed isn't specified it should default to system clock, but let's be safe
                vmmc.input["seed"] = len(self) * self.n_reps + rep_idx # basically just counting up, will ensure unique
                # label illegal states with weight 1.0, as an indicator. system will never actually visit these states
                vmmc.weights[~self.legal_states_mask] = 1.0
                # modify legal but undesired states to have low weights to discourage sampling
                vmmc.weights[~self.desired_states_mask & self.legal_states_mask] /= self.zero_sample_weight_factor
                # renormalize weights
                vmmc.weights[self.legal_states_mask] /= vmmc.weights[self.legal_states_mask].min()
                # assign extrapolation histogram temperatures
                vmmc.extrapolate_hist = self.extrapolate_hist_Ts
                # build weight and op files
                vmmc.build_vmmc_weight_file()
        for vmmc in it:
            vmmc.input["steps"] = self.steps_per_iter
        # if a start-iteration callback is set, run it immediately before running the oxdna simulations
        self._subgroups.append(it)
        if self.start_iter_callback:
            self.start_iter_callback()
        for vmmc in it:
            vmmc.oxpy_run()
        # wait for all to complete
        for vmmc in it:
            vmmc.oxpy_run.process.join()
        # if an end-iteration callback is set, run it immediately after all simulations complete
        if self.end_iter_callback:
            self.end_iter_callback()
        print(f"Completed iteration {len(self) - 1}")

    def visualize(self, bond_op_index: int = 0, iterations: Optional[list[int]] = None, secondary_bond_op_index: Optional[int] = None):
        """
        Visualize all iterations with weights, bond curves, and pie charts.
        By default, only visualizes the first bond order parameter.

        :param bond_op_index: index of bond order parameter to visualize (default: 0)
        :iterations: list of iteration indices to visualize (default: all)
        """

        if iterations is None:
            iterations = list(range(len(self._subgroups)))

        if bond_op_index >= len(self.bond_ops()):
            raise ValueError(f"bond_op_index {bond_op_index} out of range (only {len(self.bond_ops())} bond ops)")

        # Sort iterations to ensure consistent ordering
        n_iterations = len(iterations)

        # Determine if we're doing 1D or 2D weights based on number of bond ops
        is_2d_weights = secondary_bond_op_index is not None

        # Calculate figure dimensions based on whether weights are 2D
        if is_2d_weights:
            # Get dimensions for square cells
            op1 = self.bond_ops()[bond_op_index]
            op2 = self.bond_ops()[secondary_bond_op_index]
            all_possible_states = possible_states(op1, op2)
            state_set_1, state_set_2 = zip(*all_possible_states)
            max_dim1 = max(state_set_1)
            max_dim2 = max(state_set_2)

            # Calculate width needed for square cells
            cell_size = 0.2  # inches per cell
            heatmap_width = max_dim1 * cell_size
            heatmap_height = max_dim2 * cell_size

            # Total figure width: heatmap + pie charts
            total_width = heatmap_width + 3 * self.n_reps
            total_height = heatmap_height * n_iterations + (n_iterations - 1) * 1

            figsize = (max(16, total_width), max(4 * n_iterations, total_height))
        else:
            figsize = (16, 4 * n_iterations)

        # Create figure with GridSpec for flexible subplot arrangement
        fig = plt.figure(figsize=figsize)
        height_ratios = []
        for _ in range(n_iterations):
            height_ratios.extend([2, 1])  # bond curves get 2x height of pies
        gs = GridSpec(nrows=2 * n_iterations,
                      ncols=self.n_reps + 1,
                      figure=fig,
                      width_ratios=[4] + [1] * self.n_reps,
                      height_ratios=height_ratios,
                      hspace=0.3)

        # Variables to store legend information
        weight_axes = []

        replica_colors = _replica_colors(self.n_reps)

        # Get the bond op we're plotting
        bond_op = self.bond_ops()[bond_op_index]

        # Collect data for shared axis limits
        all_time_values = []
        all_bond_values = []

        for iter_idx, replicas in enumerate(self):
            for sim in replicas:
                all_bond_values.extend(sim.analysis.energy_df[bond_op.name].values)
                all_time_values.extend(sim.analysis.energy_df["time"].values)

        # Determine shared axis limits
        y_min_bonds = 0
        y_max_bonds = len(possible_states(bond_op))
        x_min_time = min(all_time_values)
        x_max_time = max(all_time_values)

        # Create plots for each iteration
        for iter_num, iter_idx in enumerate(iterations):
            # iter_num is the index in the visualization loop
            # iter_idx is the actual iteration number to plot
            replicas = self[iter_idx]
            # Create weights plot spanning both rows in the leftmost column
            ax_weights = fig.add_subplot(gs[2 * iter_num:2 * iter_num + 2, 0])

            if is_2d_weights:
                # Plot weights as 2D heatmap
                order_params_for_plot = tuple(None if i < len(self.bond_ops()) else 0
                                              for i in range(len(replicas[0].list_order_parameters())))
                replicas[0].plot_weights([bond_op_index,secondary_bond_op_index], ax=ax_weights)

                # Force square aspect ratio
                ax_weights.set_aspect('equal', adjustable='box')

                # Add hatching for inaccessible states
                self._add_weight_vis_hatching(ax_weights)
            else:
                # Plot weights as 1D bar chart
                replicas[0].plot_weights(bond_op_index,
                                         ax=ax_weights,
                                         colors=self.get_weights_colors())

            ax_weights.set_ylabel(f"Weights", fontweight='bold')

            # Add iteration number label on the left side
            ax_weights.text(-0.15, 0.95, f"{iter_idx}",
                            transform=ax_weights.transAxes,
                            fontsize=24, fontweight='bold',
                            ha='center', va='center')

            weight_axes.append(ax_weights)

            # Create bond curves using extracted method
            ax_energy = fig.add_subplot(gs[2 * iter_num, 1:])
            self.plot_bond_curves(subgroup_idx=iter_idx,
                                  bond_op_index=bond_op_index,
                                  ax=ax_energy,
                                  replica_colors=replica_colors,
                                  show_legend=iter_num == 0)

            # Override axis limits to match across all iterations
            ax_energy.set_ylim(y_min_bonds, y_max_bonds)
            ax_energy.set_xlim(x_min_time, x_max_time)

            if iter_num == 0:
                ax_energy.set_title("Bond Curves - All Replicas", fontweight='bold')

            # Create pie charts using extracted method
            pie_axes = [fig.add_subplot(gs[2 * iter_num + 1, i + 1]) for i in range(self.n_reps)]
            for rep_idx, ax_pie in enumerate(pie_axes):
                replicas[rep_idx].analysis.plot_sampling_pie_chart(
                    bond_op=self.order_parameters()[bond_op_index],
                    states_to_visualize=self.legal_state_list,
                    colors=self.get_weights_colors(bond_op_index),
                    ax=ax_pie
                )

        # After all pie charts are plotted, generate legend from legal_state_list:
        bond_op_idx = next((i for i, op in enumerate(self.order_parameters())
                            if op.name == bond_op.name), None)

        legal_bond_op_values = sorted(set(state[bond_op_idx] for state in self.legal_state_list))

        pie_colors = self.get_weights_colors(bond_op_index)
        legend_patches = [
            Patch(facecolor=pie_colors[val], label=f"{bond_op.name}={val}")
            for val in legal_bond_op_values
        ]
        fig.legend(handles=legend_patches,
                   loc='center right',
                   title=f'{bond_op.name} States',
                   bbox_to_anchor=(0.99, 0.5),
                   frameon=True)

        # Standardize y-axis across all weight plots
        all_ylims = [ax.get_ylim() for ax in weight_axes]
        global_ymin = min([ylim[0] for ylim in all_ylims])
        global_ymax = max([ylim[1] for ylim in all_ylims])

        for ax in weight_axes:
            ax.set_ylim(global_ymin, global_ymax)

        plt.tight_layout()
        fig.savefig(self._tld / f"autoprofile_analysis_bondop{bond_op_index}.svg", bbox_inches='tight')
        plt.show()

    def plot_iteration_weights(self, iteration: Optional[int] = -1,
                               ax: Optional[plt.Axes] = None) -> Optional[plt.Figure]:
        """
        Plot weights for a given iteration with hatching for inaccessible states.
        Automatically handles 1D (bar chart) or 2D (heatmap) based on number of bond ops.

        :param iteration: iteration index to plot
        :param ax: optional matplotlib axes to plot on. If None, creates new figure
        :return: figure if ax is None, otherwise None
        """
        fig = None
        make_new_fig = ax is None
        if make_new_fig:
            fig, ax = plt.subplots()

        sim = self[iteration][0]

        if len(self._bond_ops) == 1:
            # 1D case - simple bar chart
            sim.plot_weights((None,), ax=ax, colors=self.get_weights_colors())
        elif len(self._bond_ops) == 2:
            # 2D case - heatmap with hatching
            order_params_for_plot = (None, None, 0) if self.has_dist_op() else (None, None)
            sim.plot_weights(order_params_for_plot, ax=ax)

            # Add hatching for inaccessible states
            self._add_weight_vis_hatching(ax)
        else:
            raise ValueError(f"Can only plot weights for 1 or 2 bond order parameters, got {len(self._bond_ops)}")
        if make_new_fig:
            plt.show()
        return fig

    def plot_free_energy_profile(self, op: Optional[OrderParameter] = None,
                                 iteration: int = -1):
        if op is None:
            op = self.bond_ops()[0]

        free_energy_dfs = []

        # wonky index stuff
        # help
        op_vals_idxs = {op.name: np.array(list(set(a)))
                        for op, a
                        in zip(self.order_parameters(), zip(*self.desired_state_list))}
        indexer = op_vals_idxs[op.name]
        fig, ax = plt.subplots()
        colors = _replica_colors(len(self[iteration]))

        for i, sim in enumerate(self[iteration]):
            df = sim.analysis.get_data_over(op).df.copy()
            df = df.iloc[indexer]
            free_energy_dfs.append(df["free_energy"])
            ax.plot(df.index, df["free_energy"], color=colors[i], label=f"Replica {i}")

        ax.set_xlabel(op.name)
        ax.set_ylabel("Free energy")

        ax.grid(True, which="both", linestyle="--", alpha=0.5)

        ax.legend()
        plt.tight_layout()
        return fig

    def compute_next_it_weights(self, last_it: Union[VmmcReplicas, None]=None) -> np.ndarray:
        if last_it is None:
            last_it = self[-1]
        unwt_occ = self.get_overall_unwt_occ(last_it)
        weights = last_it[0].weights.copy()

        # construct reweighting criteria, starting from
        reweight_criteria = unwt_occ.copy()

        # should ensure that impossible states are not reweighted for rest of function
        reweight_criteria[~self.desired_states_mask] = 0.

        # set a maximum reweighting denominator to avoid extreme weight changes
        max_reweight_denom = 1 / weights.max()
        max_reweight_mask = (reweight_criteria > 0) & (reweight_criteria < max_reweight_denom)
        reweight_criteria[max_reweight_mask] = max_reweight_denom

        # divide by overall sampling percent (avoiding division by 0)
        weights[reweight_criteria > 0] /= reweight_criteria[reweight_criteria > 0]

        # renormalize weights, this time with min = 1
        weights[...] /= max(weights[self.legal_states_mask].min(), 1.)

        # where sampling percent is 0, increase
        weights[(reweight_criteria == 0) & self.desired_states_mask] *= self.zero_sample_weight_factor

        # handle border reweighting if enabled
        if self.reweight_borders:
            self._apply_border_reweighting(weights, unwt_occ )

        return weights

    def _apply_border_reweighting(self,
                                  weights: np.ndarray,
                                  unwt_occ: np.ndarray) -> None:
        """
        Reweight border states (legal but not desired) that are adjacent to desired states.

        For border detection, treat any sampled state as "desired" even if not in desired_states_mask.

        If self.reweight_borders is True: set each border state to min of adjacent desired state weights
        If self.reweight_borders is a float: set all border states to that value
        """
        # For border detection: desired states OR any state that was actually sampled
        effective_desired_mask = self.desired_states_mask | (unwt_occ > 0)

        # Border states: legal but not in effective_desired, and adjacent to at least one effective_desired state
        border_mask = self.legal_states_mask & ~effective_desired_mask

        # Get all neighbor offsets for an n-dimensional grid
        ndim = weights.ndim
        neighbor_offsets = []
        for dim in range(ndim):
            for delta in [-1, 1]:
                offset = [0] * ndim
                offset[dim] = delta
                neighbor_offsets.append(tuple(offset))

        # For each border state, check if it's adjacent to any effectively desired state
        border_indices = np.argwhere(border_mask)

        for idx in border_indices:
            idx_tuple = tuple(idx)
            adjacent_desired_weights = []

            # Check all neighbors
            for offset in neighbor_offsets:
                neighbor_idx = tuple(idx[i] + offset[i] for i in range(ndim))

                # Check if neighbor is in bounds
                if all(0 <= neighbor_idx[i] < weights.shape[i] for i in range(ndim)):
                    # Check if neighbor is an effectively desired state
                    if effective_desired_mask[neighbor_idx]:
                        adjacent_desired_weights.append(weights[neighbor_idx])

            # If this state is adjacent to at least one effectively desired state, reweight it
            if adjacent_desired_weights:
                if self.reweight_borders is True:
                    weights[idx_tuple] = min(adjacent_desired_weights)
                else:  # self.reweight_borders is a float
                    weights[idx_tuple] = self.reweight_borders

    def __iter__(self) -> Generator[VmmcReplicas, None, None]:
        """
        design choice: iterating over the VMMCAutoReweight instance yields iterations
        does not actually run the program
        """
        yield from self._subgroups

    def __getitem__(self, item: Union[int, tuple[int,int]]) -> VmmcReplicas:
        if isinstance(item, int):
            return self._subgroups[item]
        elif isinstance(item, tuple) and len(item) == 2 and all(isinstance(i, int) for i in item):
            return self._subgroups[item[0]][item[1]]
        else:
            raise ValueError("Item must be an int or a tuple of two ints")

    def __len__(self):
        return len(self._subgroups)

    def set_primary_bond_op(self, op: OrderParameter):
        """
        set primary bond order parameter
        """
        if isinstance(op, str):
            found_op = next((o for o in self._bond_ops if o.name == op), None)
            if found_op is None:
                raise ValueError(f"No bond order parameter with name {op}")
            self.set_primary_bond_op(found_op)
        else:
            if op not in self._bond_ops:
                self._bond_ops.insert(0, op)
            else:
                self._bond_ops.remove(op)
                self._bond_ops.insert(0, op)

    def set_dist_op(self, op: OrderParameter):
        """
        set distance order parameter for reweighting
        """
        self._dist_op = op


class VMMCGraphReweight(VMMCAutoReweight):
    """
    Variant of VMMCAutoReweight that derives new weights from observed transition
    counts in OP space rather than from unweighted occupancy histograms.

    For each adjacent legal pair (i, j), counts directed transitions c[i→j] and
    c[j→i] from the energy time-series, then solves a least-squares system over
    the adjacency graph to find log-weights satisfying:

        log w[j] - log w[i] = log((c[j→i] + ε) / (c[i→j] + ε))

    When c[i→j] > c[j→i] the system flows too readily from i to j, meaning j
    is over-visited; the formula decreases w[j] relative to w[i] to compensate.
    The pseudo-count ε (graph_pseudo_count) smooths unobserved edges and bounds
    the magnitude of corrections, avoiding the step-changes produced by
    zero_sample_weight_factor in the histogram approach.
    """

    def __init__(self, tld_path: Path, sim_build_func: Optional[Callable] = None):
        super().__init__(tld_path, sim_build_func)
        self.graph_pseudo_count: float = 1.0
        # Trust region: maximum |log-ratio| correction any single edge can contribute
        # per iteration, regardless of what the observed transition data suggests.
        # log(2) means an edge's weight ratio can move at most 2x in one iteration.
        # See the note at the edge-construction loop in compute_next_it_weights for why
        # this is needed even with graph_pseudo_count smoothing and cumulative history.
        self.max_log_weight_step: float = np.log(2.0)
        # Fallback for states flagged by the rare-crossing-event warning: how much of
        # the freshly computed weight to adopt versus keep the previous iteration's
        # weight, in log space (1.0 = fully adopt the new value, i.e. no extra damping;
        # 0.0 = ignore the new evidence entirely). The per-edge cap above bounds how far
        # any *one* edge's target can move the fit, but a sudden large influx of new
        # visits for a previously ~unvisited state can still shift the *global*
        # least-squares solution (and the subsequent min-renormalization) by more than
        # that per-edge cap suggests. This adds a second, targeted damping specifically
        # for states whose evidence just changed dramatically, so one lucky/unlucky
        # iteration doesn't fully dictate the next iteration's bias for that state.
        self.rare_event_damping: float = 0.5

    def compute_next_it_weights(self, last_it: Union[VmmcReplicas, None] = None) -> np.ndarray:
        if last_it is None:
            last_it = self[-1]

        ops = self.order_parameters()
        shape = tuple(len(op) for op in ops)
        ndim = len(ops)
        op_names = [op.name for op in ops]
        legal_states = self.legal_state_list
        legal_set = set(legal_states)

        # Count directed transitions between adjacent legal states, and total
        # visits to each state (needed to normalize transition counts into
        # per-visit rates below — see note at the edge-construction loop).
        #
        # Accumulate over the ENTIRE run so far, not just last_it. TMMC's per-visit
        # rates are only a low-noise estimator once enough samples have piled up;
        # resetting to a single iteration's window every time means a rarely-visited
        # state's rate is dominated by graph_pseudo_count smoothing every iteration,
        # producing large iteration-to-iteration swings instead of settling down as
        # more data comes in. self._subgroups already holds every completed
        # iteration (including ones reloaded via load()), so accumulate over that;
        # fall back to just last_it when there's no recorded history yet (e.g. a
        # bare compute_next_it_weights() call in a unit test).
        history = self._subgroups if self._subgroups else [last_it]

        transition_counts: dict[tuple, int] = {}
        visit_counts: dict[tuple, int] = {}
        latest_visit_counts: dict[tuple, int] = {}
        for it_replicas in history:
            is_latest = it_replicas is history[-1]
            for sim in it_replicas:
                df = sim.analysis.energy_df
                cols = [df[name].astype(int).values for name in op_names]
                state_seq = list(zip(*cols))
                for s in state_seq:
                    visit_counts[s] = visit_counts.get(s, 0) + 1
                    if is_latest:
                        latest_visit_counts[s] = latest_visit_counts.get(s, 0) + 1
                for k in range(len(state_seq) - 1):
                    s_from, s_to = state_seq[k], state_seq[k + 1]
                    if s_from == s_to or s_from not in legal_set or s_to not in legal_set:
                        continue
                    if sum(abs(s_from[d] - s_to[d]) for d in range(ndim)) != 1:
                        continue
                    key = (s_from, s_to)
                    transition_counts[key] = transition_counts.get(key, 0) + 1

        # Warn when a state's evidence base is dominated by a sudden influx of new
        # visits from just this iteration — e.g. VMMC finally crossing into a state it
        # had rarely or never reached before. That's not an instability: it's genuine
        # new evidence, and the resulting weight can legitimately swing hard to reflect
        # it. But without context that swing looks identical to noise, so flag it, and
        # extra-damp those states below (rare_event_damping) so the fresh evidence
        # nudges the weight rather than fully dictating it in one step.
        # Only meaningful once there's real prior history to compare against — skip in
        # the single-iteration fallback (e.g. direct unit-test calls).
        rare_event_states: set[tuple] = set()
        if len(history) > 1:
            for s in legal_set:
                new = latest_visit_counts.get(s, 0)
                prior = visit_counts.get(s, 0) - new
                if new > 0 and new > max(10, 2 * prior):
                    rare_event_states.add(s)
                    warnings.warn(
                        f"State {s} got {new} new visit(s) this iteration vs. only "
                        f"{prior} cumulative visit(s) before it — likely a rare "
                        f"crossing event just happened. Damping this state's weight "
                        f"update to rare_event_damping={self.rare_event_damping} of "
                        f"the freshly computed value instead of adopting it outright."
                    )

        state_to_idx = {s: i for i, s in enumerate(legal_states)}
        n = len(legal_states)
        eps = self.graph_pseudo_count

        # Build one edge per adjacent legal pair (only the +1 direction to avoid duplicates).
        #
        # Note: raw transition counts c[i->j] vs c[j->i] are NOT a valid signal here —
        # VMMC satisfies detailed balance, so those raw counts converge to equal in the
        # long run *regardless* of whether the current weights give flat sampling (it's
        # just conservation of flux across a cut, true for any reversible MC). Comparing
        # them directly makes every edge's correction vanish as sampling grows, collapsing
        # the weights to flat even when the true equilibrium population is very skewed.
        #
        # Instead we use per-visit transition rates (Transition-Matrix-Monte-Carlo style):
        # T(i->j) = c[i->j] / (visits to i). Detailed balance of the underlying dynamics
        # gives p_eq(i) * T(i->j) = p_eq(j) * T(j->i), so T(i->j)/T(j->i) recovers the true
        # (unbiased) equilibrium population ratio p_eq(j)/p_eq(i), independent of whatever
        # bias the current weights already impose. Flat-sampling weights are w ~ 1/p_eq.
        #
        # That estimator is only as good as the number of transition events behind it —
        # an edge crossed ~50 times in each direction still has ~15-20% sampling noise per
        # direction, and exponentiating a noisy log-ratio compounds that multiplicatively.
        # Applying the raw point estimate as this iteration's weight ratio, undamped, can
        # turn a modest, noisy imbalance into a large, overconfident correction that then
        # gets *run* for a full iteration — clip each edge's contribution to at most
        # max_log_weight_step so a single edge's estimate can't move that far in one step,
        # regardless of how large graph_pseudo_count-smoothed counts make it look.
        edges: list[tuple[int, int, float]] = []
        for state in legal_states:
            for dim in range(ndim):
                nbr = list(state)
                nbr[dim] += 1
                nbr = tuple(nbr)
                if not (all(0 <= nbr[d] < shape[d] for d in range(ndim)) and nbr in legal_set):
                    continue
                c_ij = transition_counts.get((state, nbr), 0)
                c_ji = transition_counts.get((nbr, state), 0)
                n_i = visit_counts.get(state, 0)
                n_j = visit_counts.get(nbr, 0)
                t_ij = (c_ij + eps) / (n_i + eps)
                t_ji = (c_ji + eps) / (n_j + eps)
                # t_ij > t_ji → nbr has higher equilibrium population → want w[nbr] < w[state]
                target = np.log(t_ji / t_ij)
                target = np.clip(target, -self.max_log_weight_step, self.max_log_weight_step)
                edges.append((state_to_idx[state], state_to_idx[nbr], target))

        if not edges:
            return last_it[0].weights.copy()

        # Least-squares: for each edge, log_w[j] - log_w[i] = target
        # Plus one normalization row: mean log-weight over desired states = 0
        n_edges = len(edges)
        A = np.zeros((n_edges + 1, n))
        b = np.zeros(n_edges + 1)
        for row, (i, j, target) in enumerate(edges):
            A[row, j] = 1.0
            A[row, i] = -1.0
            b[row] = target
        desired_idxs = [state_to_idx[s] for s in self.desired_state_list if s in legal_set]
        A[-1, desired_idxs] = 1.0 / len(desired_idxs)
        b[-1] = 0.0

        log_w_vec, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

        new_weights = last_it[0].weights.copy()
        for state, idx in state_to_idx.items():
            new_weights[state] = np.exp(log_w_vec[idx])

        # Legal but non-desired states get suppressed
        new_weights[~self.desired_states_mask & self.legal_states_mask] /= self.zero_sample_weight_factor
        # Illegal states keep a sentinel weight of 1
        new_weights[~self.legal_states_mask] = 1.0
        # Renormalize so min legal weight = 1
        legal_min = new_weights[self.legal_states_mask].min()
        if legal_min > 0:
            new_weights[self.legal_states_mask] /= legal_min

        # Damp states flagged above: blend the freshly computed weight with the
        # previous iteration's weight in log space, rather than adopting the new
        # value outright, then renormalize again so min legal weight = 1 still holds.
        if rare_event_states:
            old_weights = last_it[0].weights
            for state in rare_event_states:
                if state not in legal_set:
                    continue
                old_w, candidate_w = old_weights[state], new_weights[state]
                if old_w > 0 and candidate_w > 0:
                    new_weights[state] = old_w * (candidate_w / old_w) ** self.rare_event_damping
            legal_min = new_weights[self.legal_states_mask].min()
            if legal_min > 0:
                new_weights[self.legal_states_mask] /= legal_min

        if self.reweight_borders:
            unwt_occ = self.get_overall_unwt_occ(last_it)
            self._apply_border_reweighting(new_weights, unwt_occ)

        return new_weights

