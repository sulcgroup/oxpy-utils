from __future__ import annotations

import functools
import itertools
import warnings
from pathlib import Path
from typing import Union, Any, Optional

import numpy as np
import pandas as pd
from typing import Optional

from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

from ..replicas.generate_replicas import Replicas, ReplicaGroup
from .vmmc import VirtualMoveMonteCarlo, sigmoid, compute_heat_map
from matplotlib import pyplot as plt
from scipy.stats import sem, t, norm

from ..utils.order_parameter import OrderParameter, possible_states
from ..utils.util import generate_distinct_colors


def _replica_colors(n: int) -> np.ndarray:
    return generate_distinct_colors(n)


class VmmcReplicas(Replicas):
    """
    class to construct vmmc replicas
    runs replica groups a
    """

    # analysis
    prev_num_bins: Union[int, None]
    prev_confidence_interval: Union[float, None]
    replica_historigrams: Union[list, None]
    all_free_energies: Union[list, None]
    sem_free_energy: Any
    sem_histogram: Any

    mean_histogram: np.ndarray
    std_histogram: np.ndarray

    _temperatures: list
    def set_temperatures(self, ts: list[float]):
        self._temperatures = ts
        for sim in self.simulations:
            sim.analysis.temperatures = sorted(ts)

    temperatures = property(fget=lambda self: self._temperatures,
                            fset=lambda self, ts: self.set_temperatures(ts))

    def load(self):
        """
        Load an already-built set of replicas from disk: initializes the
        simulation objects and reads their input files, order parameters, and
        weights, then verifies that all replicas share the same weights.
        """
        if not self.sim_dir.exists():
            return
        self.init()
        for sim in self:
            sim.input.read_input()
            sim.read_order_parameters()
            sim.load_weights()
        sims = list(self)
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
                        f"{self.sim_dir.name}: replica {i} weights differ from replica 0 "
                        f"within floating-point tolerance ({len(diff_idx)} state(s)):\n    "
                        + "\n    ".join(lines)
                    )
            if mismatched_details:
                raise ValueError(
                    f"{self.sim_dir.name}: weight mismatch between replicas:\n"
                    + "\n".join(mismatched_details)
                )

    def __init__(self, conf_source, sim_dir, n_replicas):
        """
        constructor
        """
        super().__init__(VirtualMoveMonteCarlo, conf_source, sim_dir, n_replicas)
        self._temperatures = None
        # initialize analysis funcs for vmmc free energy profiling
        self.stat_funcs = {
            "inverted_finfs_mean": lambda g, name: np.array([sim.analysis.inverted_finfs for sim in g]).mean(
                axis=0),
            "inverted_finfs_sem": lambda g, name: sem(np.array([sim.analysis.inverted_finfs for sim in g]),
                                                      axis=0),
            "inverted_finfs_ci": lambda g, name, confidence_level: t.interval(confidence_level,
                                                                              len(g) - 1,
                                                                              # number of simulations in group minus one
                                                                              loc=g[(name, "inverted_finfs_mean")],
                                                                              scale=g[(name, "inverted_finfs_sem")]),
            "sampling_percent_mean": lambda g, name: np.array(
                [sim.analysis.statistics['sampling_percent'].values for sim in g]).mean(axis=0),
            "sampling_percent_sem": lambda g, name: np.sem(
                np.array([sim.analysis.statistics['sampling_percent'].values for sim in g]), axis=0),
            "y_fit_mean": lambda g, name: np.mean(np.array([sigmoid(sim.analysis.xs(),
                                                                    *sim.analysis.melt_sigmoid_fit_params) for sim in
                                                            g]),
                                                  axis=0),
            "y_fit_sem": lambda g, name: sem(np.array([sigmoid(sim.analysis.xs(),
                                                               *sim.analysis.melt_sigmoid_fit_params) for sim in
                                                       g]),
                                             axis=0),
            "tm_mean": lambda g, name: np.array([sim.analysis.Tm for sim in g]).mean(),
            "tm_sem": lambda g, name: sem(np.array([sim.analysis.Tm for sim in g])),
            "tm_ci": lambda g, name, cinterval: t.interval(cinterval, len(g) - 1, loc=g[(name, "tm_mean")],
                                                           scale=g[(name, "tm_sem")]),
            "wt_prob_sem": lambda g, name: sem(np.array([sim.analysis.statistics['wt_prob'].values for sim in g]), axis=0),
            "wt_prob_mean": lambda g, name: np.array([sim.analysis.statistics['wt_prob'].values for sim in g]).mean(
                axis=0),
            "wt_free_mean": lambda g, name: np.mean(np.array([sim.analysis.statistics['wt_free'].values for sim in g]),
                                                    axis=0),
            "wt_free_sem": lambda g, name: sem(np.array([sim.analysis.statistics['wt_free'].values for sim in g]), axis=0),
            "wt_prob_ci": lambda g, name, clvl: t.interval(clvl, len(g) - 1, loc=g[(name, "wt_prob_mean")],
                                                           scale=g[(name, "wt_prob_sem")]),
            "wt_free_ci": lambda g, name, clvl: t.interval(clvl, len(g) - 1, loc=g[(name, "wt_free_mean")],
                                                           scale=g[(name, "wt_free_sem")]),
            "heat_map": compute_heat_map
        }

        self.prev_num_bins = None
        self.prev_confidence_level = None
        self.replica_histograms = None
        self.all_free_energies = None
        self.sem_free_energy = None

    # def multisystem_replica(self,
    #                         systems: list[tuple[str, Path,Path]],
    #                         n_replicas_per_system: int):
    #     super().multisystem_replica(systems, n_replicas_per_system)
    #     for sim in self.simulations:
    #         sim.analysis.temperatures = self.temperatures

    def plot_mean_free_energy_with_error_bars(self,
                                              num_bins: int = 50,
                                              confidence_level: float = 0.95,
                                              ax: Optional[plt.Axes]=None,
                                              label=None,
                                              errorevery: int = 10):
        """
        Plot the mean free energy landscape with confidence intervals.

        :param num_bins: Number of bins for histogram.
        :param confidence_level: Confidence level for confidence intervals.
        :param ax: Axis on which to plot the graph.
        """
        for sim in self.simulations:
            sim.input.read_input()
        recompute = (
                self.prev_num_bins != num_bins or
                self.prev_confidence_level != confidence_level or
                self.replica_histograms is None or
                self.all_free_energies is None or
                self.sem_free_energy is None
        )

        if recompute:
            self.collect_replica_histograms(num_bins=num_bins)
            self.calculate_individual_free_energies()
            self.calculate_sem_free_energy()

        # Update previous values
        self.prev_num_bins = num_bins
        self.prev_confidence_level = confidence_level
        # Step 4: Calculate Z-score for the given confidence level
        z_score = norm.ppf(1 - (1 - confidence_level) / 2)

        # Step 5: Calculate the confidence interval
        confidence_interval = z_score * self.sem_free_energy

        # Step 6: Plot mean and error
        min_val = float(self.simulations[1].analysis.observables_data['com_distance'].min())
        max_val = float(self.simulations[1].analysis.observables_data['com_distance'].max())
        bin_edges = np.linspace(min_val, max_val, num_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        mean_free_energy = np.nanmean(self.all_free_energies, axis=0)
        if ax is None:
            fig, ax = plt.subplots(dpi=200, figsize=(5.5, 4.5))
        if label is None:
            label = 'VMMC free energy made discrete'
        with plt.style.context(['science', 'no-latex', 'bright']):
            print(bin_centers.shape)
            print(mean_free_energy.shape)
            ax.errorbar(bin_centers * 0.85, mean_free_energy, yerr=confidence_interval, fmt='-', capsize=2.5,
                        capthick=1.2, linewidth=1.5, label=label, errorevery=errorevery)
            # ax.fill_between(bin_centers * 0.85, mean_free_energy - confidence_interval, mean_free_energy + confidence_interval, color='gray', alpha=0.5)
            # ax.set_xlabel('COM Distance')
            # ax.set_ylabel('Free Energy')
            # ax.set_title(f'Mean Free Energy Landscape with {int(confidence_level*100)}% Confidence Intervals')

    def calculate_individual_free_energies(self):
        self.all_free_energies = []
        if self.replica_histograms is None:
            self.collect_replica_histograms()  # use default bin count
        for idx, histogram in enumerate(self.replica_histograms):
            # Check for empty histogram
            if histogram.size == 0:
                print("Empty histogram encountered.")
                continue

            # Check for all zeros
            if np.all(histogram == 0):
                print("Histogram contains only zeros.")
                continue

            # Replace zeros with non-zero minimum
            non_zero_min = histogram[histogram > 0]
            if non_zero_min.size > 0:
                min_val = np.nanmin(non_zero_min)
                histogram[histogram == 0] = min_val
            else:
                # print(histogram)
                # print(idx)
                print("No non-zero minimum value found.")

            # Calculate free energy
            free_energy = -np.log(histogram)

            # Shift so that minimum free energy is zero
            min_free_energy = np.min(free_energy)
            free_energy -= min_free_energy

            self.all_free_energies.append(free_energy)

    def calculate_sem_free_energy(self):
        if self.all_free_energies is None:
            self.calculate_individual_free_energies()
        # Calculate SEM for each bin across all free energy profiles
        self.sem_free_energy = np.nanstd(self.all_free_energies, axis=0) / np.sqrt(len(self.all_free_energies))

    def collect_replica_histograms(self, num_bins=50):
        # Initialize a list to store histograms from each replica
        self.replica_histograms = []

        for sim in self.simulations:
            sim.analysis.read_files()
            try:
                sim.analysis.get_weighted_histogram(num_bins=num_bins)
                self.replica_histograms.append(sim.analysis.weighted_histogram)
            except:
                pass

        # Convert list of arrays to a 2D numpy array for easier analysis
        # self.replica_histograms = np.array(self.replica_histograms)

    def analyze_histogram_convergence_and_errors(self):
        # Calculate the mean, standard deviation, and SEM across replicas
        self.mean_histogram = np.nanmean(self.replica_histograms, axis=0)
        self.std_histogram = np.nanstd(self.replica_histograms, axis=0)
        self.sem_histogram = self.std_histogram / np.sqrt(len(self.simulations))

    def plot_histogram_convergence_and_errors(self, num_bins=50):
        # Calculate bin centers for plotting
        min_val = float(self.simulations[0].analysis.observables_data['com_distance'].min())
        max_val = float(self.simulations[0].analysis.observables_data['com_distance'].max())
        bin_edges = np.linspace(min_val, max_val, num_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Plotting
        plt.figure(figsize=(12, 8))
        plt.plot(bin_centers, self.mean_histogram, label='Mean across replicas')
        plt.fill_between(bin_centers,
                         self.mean_histogram - self.sem_histogram,
                         self.mean_histogram + self.sem_histogram,
                         color='gray', alpha=0.5, label='SEM')
        plt.xlabel('COM Distance')
        plt.ylabel('Weighted Probability')
        plt.title('Histogram Convergence and Error Analysis')
        plt.legend()
        plt.show()

    def statistical_analysis_and_plot(self, confidence_level=0.999):
        """
        Perform statistical analysis over all simulation replicas and plot the results.
        """
        for sim in self.simulations:
            sim.analysis.read_vmmc_op_data()
            sim.analysis.calculate_sampling_and_probabilities()
            sim.analysis.calculate_and_estimate_melting_profiles()


        temp_columns_prob = [[col for col in sim.analysis.statistics.columns if '_prob' in col and 'wt_occ' in col] for sim in self.simulations]
        heat_map = [sim.analysis.statistics[columns].values for sim, columns in zip(self.simulations, temp_columns_prob)]

        x_fit = sim.analysis.x_fit
        y_fit = np.array([sim.analysis.y_fit for sim in self.simulations])
        inverted_finfs = np.array([sim.analysis.inverted_finfs for sim in self.simulations])
        tm = np.array([sim.Tm for sim in self.simulations])
        temperatures = self.simulations[0].analysis.temperatures

        df = len(tm) - 1

        wt_prob_ci = t.interval(confidence_level, df,
                                loc=self.get_analysis_stat_mean("wt_prob"),
                                scale=self.get_analysis_stat_sem("wt_prob"))

        wt_free_sem = sem(self.get_analysis_stat("wt_free"),
                          axis=0)
        # wt_free_ci = t.interval(confidence_level,
        #                         df,
        #                         loc=self.wt_free_mean,
        #                         scale=wt_free_sem)

        # heat_map_mean = np.mean(heat_map, axis=0)

        y_fit_mean = np.mean(y_fit, axis=0)

        inverted_finfs_mean = np.mean(inverted_finfs, axis=0)
        inverted_finfs_sem = sem(inverted_finfs, axis=0)
        inverted_finfs_ci = t.interval(confidence_level,
                                       df,
                                       loc=inverted_finfs_mean,
                                       scale=inverted_finfs_sem)

        tm_mean = np.mean(tm)
        tm_sem = sem(tm)
        tm_ci = t.interval(confidence_level, df, loc=tm_mean, scale=tm_sem)

        self._temperatures = temperatures

        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 10))
        self.plot_wt_prob(axes[0,0], confidence_level)
        self.plot_free_energy_profile(axes[0, 1], confidence_level)
        self.plot_sampling_percent_ci(axes[1, 0])
        self.plot_T_heatmap(axes[1, 1])

        # self.inverted_finfs_mean = inverted_finfs_mean
        # self.inverted_finfs_ci = inverted_finfs_ci
        # self.x_fit = x_fit
        # self.y_fit_mean = y_fit_mean
        # self.tm_mean = tm_mean
        # self.tm_ci = tm_ci

        plt.figure()
        plt.scatter(temperatures, inverted_finfs_mean, marker='o', label='Data Mean')
        plt.plot(x_fit, y_fit_mean, linestyle='--', linewidth=2, label='Sigmoid Fit')
        plt.fill_between(temperatures, inverted_finfs_ci[0], inverted_finfs_ci[1], interpolate=True, color='gray', alpha=0.5)
        plt.axvline(x=tm_mean, color='#D55E00', linestyle='--', linewidth=2, label=f'Tm = {tm_mean:.2f} \u00b1 {tm_ci[1] - tm_mean:.2f} °C')
        plt.xlabel('Temperature (C)')
        plt.ylabel('Fraction of ssDNA')
        plt.title(f'Melting Profile')

        # Set y-axis limits
        plt.ylim(0, 1.1)

        plt.legend()
        plt.grid(True)

    def plot_T_heatmap(self, ax: plt.Axes):
        heat_map_mean = self.get_stat(tuple(), "heat_map")
        extent = [min(self.temperatures),
                  max(self.temperatures),
                  min(self.get_n_bonds()),
                  max(self.get_n_bonds())]
        im = ax.imshow(heat_map_mean, extent=extent, cmap='viridis', aspect='auto')
        ax.set_title('Heatmap of wt_prob across Temperatures')
        ax.set_xlabel('Temperature')
        ax.set_ylabel('Number of Hydrogen Bonds')
        plt.colorbar(im, ax=ax)

    def plot_sampling_percent_ci(self, ax: plt.Axes, plot_ci: Union[bool, float] = False):
        sampling_percent_mean = self.get_analysis_stat_mean("sampling_percent")[self.get_n_bonds()]
        ax.bar(self.get_n_bonds(), sampling_percent_mean)
        if plot_ci:
            sampling_percent_ci = self.get_analysis_stat_ci("sampling_percent", plot_ci)
            ax.fill_between(self.get_n_bonds(), # todo: "range(9)"???
                            sampling_percent_ci[0][self.get_n_bonds()],
                            sampling_percent_ci[1][self.get_n_bonds()],
                            color='gray',
                            alpha=0.5)
        ax.set_xlabel('Number of Hydrogen Bonds')
        ax.set_ylabel('Probability')

    def plot_free_energy_profile(self,
                                 ax: plt.Axes,
                                 confidence: float):

        mean_df = self.get_analysis_stat_mean("wt_free")
        lower_df, upper_df = self.get_analysis_stat_ci("wt_free", confidence)

        x = mean_df.index.to_numpy()
        y = mean_df.squeeze().to_numpy()
        y_lower = lower_df.squeeze().to_numpy()
        y_upper = upper_df.squeeze().to_numpy()

        ax.plot(x, y)
        ax.fill_between(
            x,
            y_lower,
            y_upper,
            interpolate=True,
            color='gray',
            alpha=0.5
        )

        ax.set_xlabel('Number of Hydrogen Bonds')
        ax.set_ylabel('wt_free')

    def plot_wt_prob(self,
                     ax: plt.Axes,
                     confidence: float):
        wt_prob_mean = self.get_analysis_stat_mean("wt_prob")
        ax.plot(self.get_n_bonds(),
                wt_prob_mean[self.get_n_bonds()])
        wt_prob_ci = self.get_analysis_stat_ci("wt_prob", confidence)
        ax.fill_between(self.get_n_bonds(),
                        wt_prob_ci[0][self.get_n_bonds()],
                        wt_prob_ci[1][self.get_n_bonds()],
                        interpolate=True,
                        color='gray',
                        alpha=0.5)
        ax.set_xlabel('Number of Hydrogen Bonds')
        ax.set_ylabel('wt_prob')

    def plot_melting_curve(self, ax=None):

        if ax is None:
            fig, ax = plt.subplots()

        ax.scatter(self.temperatures, self.inverted_finfs_mean, marker='o', label='Data Mean')
        ax.plot(self.x_fit, self.y_fit_mean, linestyle='--', linewidth=2, label='Sigmoid Fit')
        ax.fill_between(self.temperatures, self.inverted_finfs_ci[0], self.inverted_finfs_ci[1], interpolate=True, color='gray', alpha=0.5)
        ax.axvline(x=self.tm_mean, color='#D55E00', linestyle='--', linewidth=2, label=f'Tm = {self.tm_mean:.2f} \u00b1 {self.tm_ci[1] - self.tm_mean:.2f} °C')
        ax.set_xlabel('Temperature (' + u'\N{DEGREE SIGN}'+'C)')
        ax.set_ylabel('Fraction of ssDNA')
        ax.set_title(f'Melting Profile')

        # Set y-axis limits
        ax.set_ylim(0, 1.1)

        ax.legend()
        ax.grid(True)

    def get_order_parameters(self):
        return self.simulations[0].list_order_parameters()

    def get_analysis_stat(self, name: str, op: Optional[OrderParameter] = None) -> pd.DataFrame:
        if op is None:
            op = self.simulations[0].bond_op

        # produce grouped DataFrames
        # todo: deal with how this will be different between stats. we can't sum free energy like this!!!
        assert len(self.get_order_parameters()) == 1 or name != "wt_free", \
            "get_analysis_stat is not designed to handle free energy stats. Use get_analysis_stat_mean/sem/ci instead."
        dfs = [sim.analysis.statistics[name].groupby(op.name).sum() for sim in self.simulations]

        # Give explicit names for the two index levels: simulation id and the grouped index name
        result = pd.concat(dfs, keys=range(len(dfs)), names=['sim', op.name])

        # If your result.columns is a MultiIndex and you also want names for its levels:
        # result.columns.names = ['col_level_1', 'col_level_2']

        return result

    def get_analysis_stat_mean(self, name: str) -> pd.DataFrame:
        """
        Mean across simulations for each value of the grouped index (op.name).
        Returns a DataFrame indexed by op.name with the original columns.
        """
        stat = self.get_analysis_stat(name)  # MultiIndex rows: ('sim', op.name)
        stacked = stat.unstack(level='sim')  # index -> op.name, columns -> (orig_col, sim)
        # average across the 'sim' level of the columns, leaving the original columns
        mean_df = stacked.mean(axis=1)
        return mean_df

    def get_analysis_stat_sem(self, name: str) -> pd.DataFrame:
        """
        Standard error of the mean across simulations for each value of the grouped index (op.name).
        Uses ddof=1 (sample SEM).
        todo: i am very concerned about aggregating this across order parameters
        """
        stat = self.get_analysis_stat(name)
        stacked = stat.unstack(level='sim')
        sem_df = stacked.sem(axis=1, ddof=1)
        return sem_df

    def get_analysis_stat_ci(self, name: str, confidence_level: float) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Two-sided t-based confidence interval across simulations.
        Returns (lower_df, upper_df) DataFrames with the same shape as the mean/sem DataFrames.
        """
        n_sims = len(self.simulations)
        if n_sims < 2:
            raise ValueError("Need at least two simulations to compute SEM / confidence intervals.")

        mean_df = self.get_analysis_stat_mean(name)
        sem_df = self.get_analysis_stat_sem(name)

        df = n_sims - 1
        alpha = 1.0 - float(confidence_level)
        t_mult = t.ppf(1.0 - alpha / 2.0, df)  # scalar multiplier

        lower = mean_df - t_mult * sem_df
        upper = mean_df + t_mult * sem_df

        return lower, upper

    def get_n_bonds(self, op: Optional[OrderParameter] = None) -> list[int]:
        """
        Get the number of bonds from the first simulation's statistics.
        """
        if op is None:
            op = self.simulations[0].bond_op
        return [i for i, in possible_states(op)]

    def create_state_histograms(self, op: OrderParameter, ax: Optional[plt.Axes]=None) -> Optional[plt.Figure]:

        make_new_fig = ax is None
        if ax is None:
            fig, ax = plt.subplots()

        op_idx = self[0].list_order_parameters().index(op)

        max_state = max([state[op_idx] for state in possible_states(*self[0].list_order_parameters())]) + 1
        x_labels = range(max_state)
        x_pos = np.arange(max_state)

        n_replicas = len(self)
        # build an array shape (n_states, n_replicas) filled with zeros
        counts_matrix = np.zeros((max_state, n_replicas), dtype=float)

        replica_labels = []
        for r_idx, vmmc in enumerate(self):
            # get data for this replica over the OP
            vmmc_data = vmmc.analysis.get_data_over(op).df
            # assume index contains op values, and there is a 'count' column
            if 'count' not in vmmc_data.columns:
                raise ValueError("Expected 'count' column in vmmc.analysis.get_data_over(op).df")

            # map counts into the consistent x order (fill 0 for missing op values)
            # convert index values to native python types (if e.g. numpy types)
            replica_op_vals = list(vmmc_data.index.values)
            replica_counts = vmmc_data['count'].values

            # create a dict for fast lookup
            counts_by_val = {val: cnt for val, cnt in zip(replica_op_vals, replica_counts)}

            for xi, val in enumerate(x_labels):
                counts_matrix[xi, r_idx] = counts_by_val.get(val, 0.0)

            # label for legend
            replica_label = getattr(vmmc, "name", f"replica {r_idx}")
            replica_labels.append(replica_label)

        # stacked bar plotting
        bottoms = np.zeros(len(x_labels), dtype=float)
        bar_containers = []
        colors = _replica_colors(n_replicas)
        for r_idx in range(n_replicas):
            heights = counts_matrix[:, r_idx]
            bc = ax.bar(x_pos, heights, bottom=bottoms, label=replica_labels[r_idx], color=colors[r_idx])
            bar_containers.append(bc)
            bottoms = bottoms + heights

        ax.yaxis.set_major_locator(MaxNLocator(integer=True, prune='lower'))
        # title: window index and replica count
        # only show legend if there are multiple replicas

        # grid for readability
        ax.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.7)

        ax.set_xlabel(op.name)
        ax.set_ylabel("Count")

        if make_new_fig:
            fig.show()
            return fig

    def plot_op_val_curve(self,
                          op: OrderParameter,
                          ax: Optional[plt.Axes] = None,
                          replica_colors: Optional[np.ndarray] = None,
                          show_legend: bool = True):

        make_new_fig = ax is None
        if ax is None:
            fig, ax = plt.subplots()

        if len(self) == 0:
            raise ValueError(f"No replicas found")
        # Set up replica colors
        if replica_colors is None:
            replica_colors = _replica_colors(self.nreplicas)

        # Collect data for axis limits
        all_time_values = []
        all_bond_values = []

        for sim in self:
            all_time_values.extend(sim.analysis.energy_df["time"].values)
            all_bond_values.extend(sim.analysis.energy_df[op.name].values)

        # Determine axis limits
        y_min = 0
        y_max = len(possible_states(op))
        x_min = min(all_time_values)
        x_max = max(all_time_values)

        # Plot each replica
        for i, sim in enumerate(self):
            states_data = sim.analysis.energy_df[op.name]
            ax.plot(sim.analysis.energy_df["time"],
                    states_data,
                    color=replica_colors[i],
                    label=f"Replica {i}",
                    linewidth=0.5,
                    alpha=0.7)

        # Set axis properties
        ax.set_ylim(y_min, y_max)
        ax.set_xlim(x_min, x_max)
        ax.set_ylabel(op.name)
        ax.set_xlabel("Time")

        # Set y-axis ticks to integers
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))

        if show_legend:
            ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1), frameon=True)

        if make_new_fig:
            # ax.set_title(f"Bond Curves - Iteration {subgroup_idx if subgroup_idx >= 0 else len(self) + subgroup_idx}")
            plt.tight_layout()
            plt.show()
            return fig


class VmmcReplicasGroup(ReplicaGroup):
    def __init__(self):
        super().__init__(SimulationClass=VirtualMoveMonteCarlo)

    def multisystem_replica(self,
                            systems: list[tuple[str, Path,Path]],
                            n_replicas_per_system: int):
        """
        Create simulation replicas, with across multiple systems with diffrent inital files

        Parameters:
            systems (list): List of strings, where the strings are the name of the directory which will hold the inital files
            n_replicas_per_system (int): number of replicas to make per system
            file_dir_list (list): List of strings with path to intial files
            sim_dir_list (list): List of simulation directory paths
        """

        # todo: at least have the option of passing args as tuple list
        for systemname, sys_file_dir, sys_sim_dir in systems:
            if type(sys_file_dir) == str:
                sys_file_dir = Path(sys_file_dir)
            if type(sys_sim_dir) == str:
                sys_sim_dir = Path(sys_sim_dir)
            if systemname in self.systems:
                systemname = f"{systemname}_n"
            self.systems[systemname] = VmmcReplicas(conf_source=sys_file_dir,
                                                sim_dir=sys_sim_dir,
                                                n_replicas=n_replicas_per_system)
