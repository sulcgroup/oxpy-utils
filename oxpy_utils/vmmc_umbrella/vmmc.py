"""
Virtual-move monte carlo simulation
"""
from __future__ import annotations

import itertools
import os
import shutil
import re
from pathlib import Path
from typing import Generator
import math
import warnings


import pandas as pd

from oxpy_utils.utils.order_parameter import possible_states


from oxpy_utils.utils.util import si_units, generate_distinct_colors
from scipy.optimize import curve_fit

from typing import Optional, Union, List, Dict
import numpy as np
from matplotlib.colors import LogNorm
import matplotlib.cm as cm

import matplotlib.pyplot as plt

from ..utils.order_parameter import OrderParameter
from .vmmc_data import VMMCData, read_vmmc_data
from ..structure_editor.dna_structure import wcbp

# matplotlib.use('TkAgg')

from ..oxdna_simulation import Simulation, OxpyRun, Analysis
from ..structure_editor.dna_structure import load_dna_structure
from ..utils.observable import Observable, ObservableColumn

hist_file_header_re = re.compile(
    r"^#t = (\d+); extr\. Ts: ((?:\d+(?:\.\d+)?\s*)+)$"
)
class VmmcOxpyRun(OxpyRun):
    """Automatically runs a built oxDNA simulation using oxpy within a subprocess"""

    def __init__(self, sim):
        super().__init__(sim)

    def run(self,
            subprocess: bool=True,
            continue_run=False, verbose=True, log=True, join=False, custom_observables=None):
        """ Run oxDNA simulation using oxpy in a subprocess.

        :param subprocess (bool): If false run simulation in parent process (blocks process), if true spawn sim in child process.
            continue_run (number): If False overide previous simulation results. If True continue previous simulation run.
            verbose (bool): If true print directory of simulation when run.
            log (bool): If true print a log file to simulation directory.
            join (bool): If true block main parent process until child process has terminated (simulation finished)
        """
        self.subprocess = subprocess
        self.verbose = verbose
        self.continue_run = continue_run
        self.log = log
        self.join = join
        self.custom_observables = custom_observables

        if continue_run is not False:
            self.sim.input_file({'init_hist_file': self.sim.input.input_dict['last_hist_file']})
        if self.verbose:
            print(f'Running: {str(self.sim_dir).split("/")[-1]}')
        if self.subprocess:
            self.spawn(self.run_complete)
        else:
            self.run_complete()


def compute_heat_map(g) -> np.ndarray:
    """
    TODO: make sim.analysis.statistics a VMMCData object natively
    """
    temp_columns_prob = [
        [col for col in sim.analysis.statistics if '_prob' in col] for sim in
    g]
    assert all(len(col) for col in temp_columns_prob)
    heat_map_3d = np.array([sim.analysis.statistics[columns].values
                     for sim, columns in zip(g, temp_columns_prob)])
    return heat_map_3d


def sigmoid(x, L, x0, k, b):
    return L / (1 + np.exp(-k * (x - x0))) + b


class VmmcAnalysis(Analysis):
    """
    Methods used to interface with oxDNA simulation in jupyter notebook (currently in work)
    """

    # overwriting the base class annotation for type clarity
    sim: VirtualMoveMonteCarlo

    # weights used in oxDNA VMMC

    # ???
    free_energy: float
    # ???
    weighted_histogram: Union[np.ndarray, None]

    # sigmoidal function best-fit params L, x0, k, b (see above)
    melt_sigmoid_fit_params: Union[tuple[float, float, float, float], None]

    # inflection points
    finfs: np.ndarray
    inverted_finfs: np.ndarray

    # temperatures used for
    temperatures: np.ndarray

    # melting temperatures
    Tm: float

    # statistics calculated from raws
    # i kinda hate this one
    _stats_df: Union[pd.DataFrame, None]

    # dataframe to store vmmc data read from last_hist file, indexed by order parameters
    _vmmc_df: Optional[pd.DataFrame]


    def __init__(self, simulation):
        """ Set attributes to know all files in sim_dir and the input_parameters"""
        super().__init__(simulation)
        self.melt_sigmoid_fit_params = None
        self.weighted_histogram = None
        self.Tm = None
        self.temperatures = None
        # set during analysis
        self._stats_df = None
        self._vmmc_df = None

    @property
    def weights(self) -> np.ndarray:
        return self.sim.weights

    def read_files(self):
        """
        reads data from files
        """
        self.read_all_observables()
        self.get_vmmc_weights()

        # todo: pull this up to vmmc
        if self.temperatures is None:
            self.temperatures = np.array(sorted([
                float(tstr.strip()[:-1])
                for tstr in self.sim.input["extrapolate_hist"].split(",")]))

    def get_vmmc_weights(self):
        # if weights are loaded, return them
        if self.weights is not None:
            return self.weights
        # if weights are not loaded but are specified in the vmmc input file
        elif "weights_file" in self.sim.input:
            wfile_path = self.sim.sim_dir / self.sim.input["weights_file"]
            weights_arr = np.loadtxt(wfile_path)
            weights_idx, self.weights = weights_arr.T
            assert np.allclose(weights_idx, np.arange(self.num_op_hbonds() + 1)), \
                "order parameter indices in weight files are not continuous starting from 0" \
                " with length num_op_hbonds()+1!"
            self.weights = pd.DataFrame({'index': weights_idx, 'weight': self.weights})
            return self.weights
        # if no weights are specified, return unweighted array
        else:
            # todo: warning?
            return np.full(shape=(self.num_op_hbonds()), fill_value=1)

    def num_op_hbonds(self) -> int:
        """
        Returns: the number of hydrogen bonds measured by order parameters
        """
        return len(self.weights) - 1

    def get_weighted_histogram(self, num_bins=50, force=False):
        """
        Parameters:
            num_bins the number of bins for the histogram
            force whether to force-recalculate weighted histogram data
        """
        if force or self.weighted_histogram is None:
            # Create an empty histogram
            self.weighted_histogram = np.zeros(num_bins)

            # Ensure min and max are scalar values
            min_val = float(self.observables_data['com_distance'].min())
            max_val = float(self.observables_data['com_distance'].max())

            # Create bin edges
            bin_edges = np.linspace(min_val, max_val, num_bins + 1)

            # Create a mapping of hb_observable to weight for faster lookup
            weight_mapping = self.weights.set_index('index')['weight'].to_dict()
            # Vectorized weight lookup
            weights_vector = self.observables_data['hb_observable'].iloc[:, 0].map(weight_mapping).values.reshape(-1, 1)

            # Vectorized bin index calculation
            bin_indices = np.digitize(self.observables_data['com_distance'].values, bin_edges) - 1

            # Clip bin indices to be within bounds
            bin_indices = np.clip(bin_indices, 0, num_bins - 1)
            # Calculate the weighted histogram
            np.add.at(self.weighted_histogram, bin_indices, 1 / weights_vector)

            # Normalize the histogram
            self.weighted_histogram /= np.sum(self.weighted_histogram)

            # Adding a small constant to avoid log(0)
            epsilon = 1e-15
            self.free_energy = -np.log(self.weighted_histogram + epsilon)

            # Shift the free energy profile so that the minimum value is 0
            min_free_energy = np.min(self.free_energy)
            self.free_energy -= min_free_energy
        return self.weighted_histogram

    def plot_weighted_histogram(self, n_bins=50, label=None, ax=None):
        self.read_files()
        self.get_weighted_histogram(num_bins=n_bins)

        # Create a new figure and axes if not provided
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))

        # Calculate the center of each bin
        min_val = float(self.observables_data['com_distance'].min())
        max_val = float(self.observables_data['com_distance'].max())
        bin_edges = np.linspace(min_val, max_val, n_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Plot on the provided axes
        ax.plot(bin_centers * 0.85, self.free_energy, label=label)

        ax.set_xlabel('COM Distance')
        ax.set_ylabel('Free Energy')
        ax.set_title('Free Energy Landscape')

    def last_hist_analysis(self):
        self.read_vmmc_op_data()
        self.calculate_sampling_and_probabilities()
        self.plot_statistics()
        self.plot_melting_profiles()

    def load_energy(self):
        """
        if order parameters are provided they are outputted in the energy dataframe, so
        include them here
        """
        self.sim_files.parse_current_files()
        if not self.sim.list_order_parameters():
            self.sim.read_order_parameters()
        df = pd.read_csv(self.sim_files.energy,
                         sep='\\s+',
                         header=None)
        df.columns = ['time', "U", 'p_T', 'p_R', 'p_V'] + [op.name for op in self.sim.list_order_parameters()] + ["weight"]

        self._energy_df = df

    def read_vmmc_op_data(self):
        self.sim_files.parse_current_files()
        # todo: eventually make sim_files a dict? for now, just set last_hist path from the input file
        self.sim_files.last_hist = self.sim_dir / self.sim.input["last_hist_file"]
        self.sim.read_order_parameters()

        df, simulation_time = self.read_op_hist_file(self.sim_files.last_hist)
        self.sim.simulation_time = simulation_time
        op_cols = [op.name for op in self.sim.list_order_parameters()]
        self._vmmc_df = df.set_index(op_cols)

    def read_op_hist_file(self, file_name: Path, timestep: Optional[int]=None) -> tuple[pd.DataFrame, int]:
        # Initialize variables to store metadata and data
        simulation_time = None
        temperatures = []
        data = []

        # temperature conversion factor is same in RNA and DNA but this is good practice
        try:
            interaction_type = "RNA" if self.sim.input["interaction_type"] in ["RNA", "RNA2"] else "DNA"
        except KeyError:
            interaction_type = "DNA"

        # Read the file line by line
        with file_name.open('r') as f:
            # search for desired timestep
            simulation_time = -1
            for line in f:
                match = hist_file_header_re.match(line)
                if match:
                    simulation_time = int(match.group(1))
                    if timestep is None or simulation_time == timestep:
                        # found the right timestep, break to read data
                        Ts = list(map(float, re.findall(r"\d+(?:\.\d+)?", match.group(2))))
                        temperatures = [(si_units(temp_unit, interaction_type, "T", "C")) for temp_unit in
                                        Ts]
                        break
            if timestep is not None and simulation_time != timestep:
                raise ValueError(f"Timestep {timestep} not found in histogram file {str(file_name)}. Last timestep is {simulation_time}")

            for i, line in enumerate(f):
                if hist_file_header_re.match(line):
                    break
                # Parse data lines and convert to float
                row = list(map(float, line.strip().split()))
                data.append(row)
        # Create DataFrame from data
        df = pd.DataFrame(data)
        # Rename columns with lowercase names
        # preserve old behavior for bond-only systems
        if len(self.sim.list_order_parameters()) == 1:
            op_cols = ["h_bonds"]
            column_names = op_cols + ["unwt_occ", "wt_occ"]
            column_names += [f"wt_occ_{temp:.1f}C" for temp in temperatures]
        if len(df.columns) == 2 + len(temperatures) + len(self.sim.list_order_parameters()):
            # list_order_parameters will list bond ops then dist ops
            op_cols = [op.name for i, op in enumerate(self.sim.list_order_parameters())]
            column_names = op_cols + ["unwt_occ", "wt_occ"] + [f"wt_occ_{temp:.1f}C" for temp in temperatures]
        else:
            raise Exception(
                f"Unexpected number of columns {len(df.columns)} in last_hist file {str(self.sim_files.last_hist)}. Expected {3 + len(temperatures) + len(self.sim.list_order_parameters())}")
        if np.isnan(df.iloc[-1].values).any():
            warnings.warn("Last timepoint in energy file contains NaN values, probably because of a data race with an ongoing simulation. Removing last row...")
            df = df.iloc[:-1]

        df.columns = column_names
        df[op_cols] = df[op_cols].astype(int)
        return df, simulation_time

    def get_data_over(self, op: OrderParameter, timestep: Optional[int]=None) -> VMMCData:
        """
        :return: vmmc_df data summed over all other order parameters, indexed by the provided op
        """
        if timestep is None:
            df = self.vmmc_df
        else:
            raw_df, _ = self.read_op_hist_file(Path(self.sim_dir) / self.sim.input["traj_hist_file"], timestep)
            op_cols = [order_param.name for order_param in self.sim.list_order_parameters()]
            df = raw_df.set_index(op_cols)
        df = df.rename({"unwt_occ": "count", "wt_occ": "unbiased_count"}, axis="columns")
        # reset index to expose op columns, groupby the target op, then re-index by it
        df_reset = df.reset_index()
        summed_df = df_reset.groupby(op.name)[df.columns.tolist()].sum()
        return VMMCData(
            df=summed_df,
            step=int(self.current_step()),
            op=op
        )

    def partition_traj(self) -> dict[tuple[int, ...], list[int]]:
        """
        todo: better name
        loads the trajectory and classifies timepoints by order parameter values
        """
        self.sim_files.parse_current_files()
        state_timepoints = dict()

        for idx, row in self.energy_df.iterrows():
            state_tuple = tuple(row[op.name] for op in self.sim.list_order_parameters())
            if state_tuple not in state_timepoints:
                state_timepoints[state_tuple] = []
            state_timepoints[state_tuple].append(idx)
        return state_timepoints

    @property
    def statistics(self) -> pd.DataFrame:
        if self._stats_df is None:
            self.calculate_sampling_and_probabilities()
        return self._stats_df

    @property
    def stats_idxd(self) -> pd.DataFrame:
        return self.statistics.set_index(self.vmmc_df.index)

    @property
    def vmmc_df(self) -> pd.DataFrame:
        """
        reads last-step vmmc data, indexed by order parameters
        """
        if self._vmmc_df is None:
            self.read_vmmc_op_data()
        return self._vmmc_df

    def calculate_sampling_and_probabilities(self):
        """
        Calculate the sampling percentage, probability, and -log(probability) for each
        occurrence.
        todo: make this automatically called as-needed
        """

        # create statistics dataframe indexed by order parameters (same as vmmc_df)
        self._stats_df = pd.DataFrame(index=self.vmmc_df.index)

        # Calculate the total unweighted occurrences in the simulation
        total_unwt_occ = self.vmmc_df['unwt_occ'].sum()

        # Calculate the sampling percentage for each state
        self._stats_df['sampling_percent'] = (self.vmmc_df['unwt_occ'] / total_unwt_occ) * 100

        # Calculate the total weighted occurrences in the simulation
        total_wt_occ = self.vmmc_df['wt_occ'].sum()

        # Calculate the probability for each state
        self._stats_df['wt_prob'] = self.vmmc_df['wt_occ'] / total_wt_occ
        # Avoid log(0) by replacing zeros
        epsilon = 1e-15 * self._stats_df['wt_prob'][self._stats_df['wt_prob']>0].min()

        # Calculate the -log(probability) for each state
        self._stats_df['wt_free'] = -np.log(self._stats_df['wt_prob'] + epsilon)
        # Shift the free energy values so that the lowest is zero
        min_wt_free = self.statistics['wt_free'].min()
        self._stats_df['wt_free'] -= min_wt_free

        # Calculate probabilities and -log(probabilities) for extrapolated temperatures
        temp_columns = [col for col in self.vmmc_df.columns if col.startswith("wt_occ_")]
        for col in temp_columns:
            total_temp_occ = self.vmmc_df[col].sum()
            prob_col = f"{col}_prob"
            neglog_prob_col = f"{col}_free"

            self._stats_df[prob_col] = self.vmmc_df[col] / total_temp_occ
            self._stats_df[neglog_prob_col] = -np.log(self._stats_df[prob_col] + epsilon)

            # Shift the free energy values so that the lowest is zero for each temperature
            min_temp_free = self._stats_df[neglog_prob_col].min()
            self._stats_df[neglog_prob_col] -= min_temp_free

    def plot_sampling_pie_chart(self,
                                bond_op: Optional[OrderParameter]=None,
                                states_to_visualize: Optional[list[tuple[int, ...]]]=None,
                                colors: Optional[np.ndarray]=None,  # Array of RGBA colors, one per bond_op value
                                ax: Optional[plt.Axes]=None) -> tuple[Optional[list], Optional[list]]:
        """
        Plot a pie chart showing sampling percentages aggregated by bond_op value.

        :param bond_op: Order parameter to use for categorization
        :param states_to_visualize: List of state tuples that are valid/accessible
        :param colors: Array of RGBA colors, one for each possible value of bond_op
        :param ax: Matplotlib axes to plot on
        :return: Tuple of (legend_labels, legend_colors) if states were sampled, else (None, None)
        """
        if bond_op is None:
            bond_op = self.sim.bond_op
        if states_to_visualize is None:
            states_to_visualize = list(itertools.product([range(len(op)) for op in self.sim.list_order_parameters()]))
        if colors is None:
            colors = generate_distinct_colors(len(bond_op))
        if ax is None:
            fig, ax = plt.subplots()
        # Find the index of the bond_op in the simulation's order parameters
        bond_op_idx = next((i for i, op in enumerate(self.sim.list_order_parameters())
                            if op.name == bond_op.name), None)

        if bond_op_idx is None:
            raise ValueError(f"No order parameter named {bond_op.name}")

        # Build state tuples from the index (order parameters are now the index)
        if isinstance(self.vmmc_df.index, pd.MultiIndex):
            df_tuples = list(self.vmmc_df.index)
        else:
            df_tuples = [(v,) for v in self.vmmc_df.index]

        # Filter to only include states in states_to_visualize
        mask = [tuple_val in states_to_visualize for tuple_val in df_tuples]

        # Get the filtered statistics
        filtered_stats = self.statistics[mask].copy()

        if len(filtered_stats) == 0:
            ax.text(0.5, 0.5, 'No accessible\nstates sampled',
                    ha='center', va='center', transform=ax.transAxes)
            return None, None

        # Add a column for the bond_op value we're aggregating by
        filtered_df_tuples = [t for t, m in zip(df_tuples, mask) if m]
        filtered_stats['bond_op_value'] = [state[bond_op_idx] for state in filtered_df_tuples]

        # Aggregate sampling percentages by bond_op value
        aggregated = filtered_stats.groupby('bond_op_value')['sampling_percent'].sum()

        # Sort by bond_op value to ensure consistent ordering
        aggregated = aggregated.sort_index()

        def autopct_format(pct):
            return f'{pct:.1f}%' if pct >= 1.0 else ''  # Only show if >= 1%

        if len(aggregated) > 0 and aggregated.sum() > 0:
            # Create pie chart
            ax.pie(
                aggregated.values,
                autopct=autopct_format,
                startangle=90,
                pctdistance=1.4,
                colors=[colors[int(val)] for val in aggregated.index]  # Index into colors by bond_op value
            )
        else:
            ax.text(0.5, 0.5, 'No accessible\nstates sampled',
                    ha='center', va='center', transform=ax.transAxes)
        return fig, ax

    def plot_statistics(self):
        """
        plot statistics in a 2x2 figures
        """
        # Create a figure and a grid of subplots
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 10))

        # Line plot for wt_prob
        self.statistics['wt_prob'].plot(ax=axes[0, 0], title='Weighted Probability (wt_prob)', color='#0072B2')
        axes[0, 0].set_xlabel('Index')
        axes[0, 0].set_ylabel('wt_prob')

        # Line plot for wt_free
        self.statistics['wt_free'].plot(ax=axes[0, 1], title='Negative Log Probability (wt_free)', color='#D55E00')
        axes[0, 1].set_xlabel('Index')
        axes[0, 1].set_ylabel('wt_free')

        # Bar plot for sampling_percent
        self.statistics['sampling_percent'].plot(kind='bar', ax=axes[1, 0], title='Sampling Percentage', color='g')
        axes[1, 0].set_xlabel('Index')
        axes[1, 0].set_ylabel('sampling_percent')

        # Heatmap for wt_prob and wt_free across temperatures
        temp_columns_prob = [col for col in self.statistics.columns if '_prob' in col and 'wt_occ' in col]
        temp_columns_free = [col for col in self.statistics.columns if '_free' in col and 'wt_occ' in col]

        im = axes[1, 1].imshow(self.statistics[temp_columns_prob].values, cmap='viridis', aspect='auto')
        axes[1, 1].set_title('Heatmap of wt_prob across Temperatures')
        axes[1, 1].set_xlabel('Temperature')
        axes[1, 1].set_ylabel('Index')
        plt.colorbar(im, ax=axes[1, 1])

        # Show the plot
        plt.tight_layout()
        plt.show()


    def calculate_and_estimate_melting_profiles(self):
        """
        Calculate the melting profiles and estimate the melting temperature (Tm).
        """
        # Initialize an empty DataFrame to store the melting profiles
        self.sim.melting_profiles = pd.DataFrame()

        # Initialize list to store finite-size-effect corrected yields (finfs)
        finfs = []
        temperatures = []  # Initialize list to store temperatures

        # Loop through each temperature column in the vmmc_df DataFrame
        hbond_op_name = self.sim.bond_op.name
        for col in self.vmmc_df.columns:
            if col.startswith("wt_occ_"):
                # Extract temperature from column name
                try:
                    temp = float(col.split('_')[-1].replace('C', ''))
                except ValueError:
                    continue

                # Calculate the ratio of bound to unbound states for this temperature.
                # hbond_op_name is now the index (or part of a MultiIndex).
                if isinstance(self.vmmc_df.index, pd.MultiIndex):
                    hbond_level = self.vmmc_df.index.get_level_values(hbond_op_name)
                else:
                    hbond_level = self.vmmc_df.index
                bound_states = self.vmmc_df[col][hbond_level > 0].sum()
                # weighted occupancy of unbound state(s)
                unbound_states = self.vmmc_df[col][hbond_level == 0].sum()

                # Calculate the melting ratio and finf
                ratio = bound_states / unbound_states if unbound_states != 0 else np.nan
                finf = 1. + 1. / (2. * ratio) - math.sqrt((1. + 1. / (2. * ratio)) ** 2 - 1.)

                # Add this ratio and finf to their respective data structures
                self.sim.melting_profiles[col] = [ratio]
                finfs.append((temp, finf))
                temperatures.append(temp)  # Add temperature to the list

        # Check if finfs is empty
        if not finfs:
            print("Warning: No finite-size-effect corrected yields (finfs) calculated.")
            return

        # Store finfs and temperatures as instance variables
        self.finfs = np.array([f for _, f in sorted(finfs, key=lambda x: x[0])])
        if np.isnan(self.finfs).any():
            bad_coords = np.column_stack(np.nonzero(np.isnan(self.finfs)))
            coord_str = ",".join("(" + ",".join(str(int(x)) for x in coord) + ")" for coord in bad_coords)
            raise ValueError(
                "NaN values found in finite-size-effect corrected yields (finfs). "
                f"Bad coordinates: {coord_str}. Check for zero unbound states or other issues in the data."
            )
        self.temperatures = np.array(sorted(temperatures))

        # Estimate Tm based on finfs
        self.sim.Tm = self._get_Tm(self.temperatures, self.finfs)

        # Invert the finfs to get the fraction of ssDNA
        self.inverted_finfs = np.array([1 - finf for finf in self.finfs])

        # Fit the sigmoid function to the inverted data
        p0 = [max(self.inverted_finfs), np.median(self.temperatures), 1, min(self.inverted_finfs)]  # initial guesses for L, x0, k, b
        self.popt, _ = curve_fit(sigmoid,
                                 self.temperatures,
                                 self.inverted_finfs,
                                 p0,
                                 method='dogbox')

        # Generate fitted data
        self.x_fit = np.linspace(5, 70, 500)
        self.y_fit = sigmoid(self.x_fit, *self.popt)

        idx = np.argmin(np.abs(self.y_fit - 0.5))
        self.sim.Tm = self.x_fit[idx]


    def _get_Tm(self, temps: np.ndarray, finfs: np.ndarray) -> float:
        """
        Helper function to estimate Tm.
        """
        x = finfs.copy()[::-1]
        y = temps.copy()[::-1]
        xin = np.arange(0.1, 1., 0.1)
        f = np.interp(xin, np.array(x), np.array(y))
        return f[4]

    def plot_melting_profiles(self, label=''):
        # Ensure melting profiles are calculated
        self.calculate_and_estimate_melting_profiles()

        # Plotting
        plt.figure(figsize=(10, 6))
        plt.scatter(self.temperatures,
                    self.inverted_finfs,
                    marker='o',
                    label=f'{label}Data')
        plt.plot(self.x_fit,
                 self.y_fit,
                 linestyle='--',
                 linewidth=2,
                 label=f'{label}Sigmoid Fit')

        # Add a vertical line at the melting temperature
        plt.axvline(x=self.sim.Tm,
                    color='#D55E00',
                    linestyle='--',
                    linewidth=2,
                    label=f'{label}Tm = {self.sim.Tm:.2f} °C')

        plt.xlabel('Temperature (°C)')
        plt.ylabel('Fraction of ssDNA')
        plt.title(f'Melting Profile')

        # Set y-axis limits
        plt.ylim(0, 1.1)

        plt.legend()
        plt.grid(True)
        # plt.show()


class VirtualMoveMonteCarlo(Simulation):
    """
    class for a virtual move monte carlo simulation
    """

    _bond_order_parameters: list[OrderParameter]
    _dist_order_parameters: list[OrderParameter]
    _weights = np.ndarray # variable-dimension array to store weights for each pair of order parameters
    _extrapolate_hist: list

    def __init__(self, file_dir: Union[Path, str], sim_dir: Union[Path, str]=None):
        """
        constructor for vmmc simultaion
        """
        super().__init__(file_dir, sim_dir)
        self._analysis = VmmcAnalysis(self)
        self._oxpy_run = VmmcOxpyRun(self)
        self._bond_order_parameters = []
        self._dist_order_parameters = []
        self._weights = None
        # design choice: minimal setting in __init__ - p1 and p2 should be set later
        if "extrapolate_hist" in self.input:
            extr_hist = [t_str.strip() for t_str in self.input["extrapolate_hist"].split(",")]
            # NOTE: FOR SOME REASON EXTRAPOLATE HISTOGRAM TEMPERATURES ARE HARDCODED TO CELSIUS
            try:
                self.extrapolate_hist = sorted([float(T[:-1]) if T.endswith("C") else float(T) for T in extr_hist])
            except ValueError as e:
                print(f"Error parsing extrapolate_hist temperatures. Ensure they are formatted as floats with 'C' suffix (e.g., '25C'). Got {self.input['extrapolate_hist']}")
        else:
            self._extrapolate_hist = None

    @property
    def extrapolate_hist(self) -> list:
        return self._extrapolate_hist

    @extrapolate_hist.setter
    def extrapolate_hist(self, val: list) -> None:
        self._extrapolate_hist = val
        # update input file
        self.input["extrapolate_hist"] = ", ".join([f"{T}" for T in self.extrapolate_hist])

    @property
    def bond_op(self) -> Optional[OrderParameter]:
        if len(self._bond_order_parameters) == 0:
            return None
        return self._bond_order_parameters[0]

    @bond_op.setter
    def bond_op(self, new_op: Union[int, OrderParameter, str]):
        if len(self._bond_order_parameters) == 1:
            self._bond_order_parameters[0] = new_op
        elif len(self._bond_order_parameters) == 0:
            self._bond_order_parameters.append(new_op)
        else:
            raise Exception("Ambiguous how to \"set\" the bond oreder parameter in a simulation with multiple existing bond ops")

    def dist_op(self):
        """
        todo: multiple distance ops?
        """
        return self._dist_order_parameters[0] if self._dist_order_parameters else None

    def set_nucleotides(self, p1: Union[str, list, np.ndarray], p2: Union[str, list, np.ndarray]):
        """
        :param p1: first set of nucleotides for the order parameter
        :param p2: list of complimentary nucleotides for the order parameter
        """
        if len(p1) != len(p2):
            raise ValueError("Mismatch between lengths of nucleotide lists")
        if isinstance(p1, str):
            p1 = np.array([int(nuc) for nuc in p1.split(',')])
        if isinstance(p2, str):
            p2 = np.array([int(nuc) for nuc in p2.split(',')])
        self.add_order_parameter(OrderParameter(
            "native",
            "bond",
            list(zip(p1, p2))
        ))
        # todo: update op file?

    def bond_ops(self) -> Generator[OrderParameter, None, None]:
        yield from self._bond_order_parameters

    def num_bond_ops(self) -> int:
        return len(self._bond_order_parameters)

    def num_ops(self) -> int:
        return len(self._dist_order_parameters) + self.num_bond_ops()

    def get_op(self, name:str) -> OrderParameter:
        """
        find an order parameter by name
        """
        templist = [op for op in self.list_order_parameters() if op.name == name]
        if len(templist) == 1:
            return templist[0]
        elif not templist:
            raise ValueError(f"No order parameter named {name}")
        else:
            raise ValueError(f"Multiple order parameters named {name}, this shouldn't even be *allowed*")

    def veryify_bond_ops(self) -> bool:
        """
        make sure bond ops are watson-crick base paired
        """
        # load dna structure
        dna_structure = load_dna_structure(self.sim_files.top, self.sim_files.dat)
        for op in self.bond_ops():
            for base1, base2 in op.pairs:
                if not wcbp(dna_structure.get_base(base1).base,
                            dna_structure.get_base(base2).base):
                    return False
        return True


    def add_dist_op(self,
                    interfaces: Optional[list[Union[int, float]]]=None,
                    op: Optional[OrderParameter]=None,
                    p1: Optional[list, np.ndarray]=None,
                    p2: Optional[list, np.ndarray]=None):
        """
        adds a mindist order parameter. need to specify in the weights file if you want to sample a system
        biased towards melting or annealing
        we are not allowing strings here
        """
        if p1 is None:
            if op is not None:
                p1, p2 = zip(*op.pairs)
            # if no p1 provided, assume same as bond op
            elif self.bond_op is None:
                raise Exception("No bond order parameter specified!")
            else:
                p1, p2 = zip(*self.bond_op.pairs)
        self.add_order_parameter(OrderParameter(
            "dist",
            "mindistance",
            list(zip(p1, p2)),
            interfaces=interfaces
        ))

    # deprecatimg this as a property bc it implies it is accessable
    # @property
    def list_order_parameters(self) -> list[OrderParameter]:
        """
        Produces a list of order parameters used by the system
        note that this is pass-by-value; changes will not be propegated!!
        :returns: order parameter list, sorted as required by oxDNA
        """
        return self._bond_order_parameters + self._dist_order_parameters

    def add_order_parameter(self, op: OrderParameter):
        if op.order_parameter == "bond":
            self._bond_order_parameters.append(op)
        elif op.order_parameter == "mindistance":
            self._dist_order_parameters.append(op)
        else:
            raise ValueError(f"Unknown order parameter {op.order_parameter}")

    # making a deliberate choice to not write a setter for these

    def get_vmmc_data(self) -> VMMCData:
        return read_vmmc_data(self.sim_dir / self.input["last_hist_file"],
                              self.bond_op
                              )


    def read_order_parameters(self, op_file: Optional[Union[str, Path]]=None):
        """
        reads order parameter data from file
        """
        if op_file is None:
            op_file = self.sim_dir / self.input["op_file"]
        ops = OrderParameter.read_file(op_file)
        self._bond_order_parameters = []
        self._dist_order_parameters = []
        for op in ops:
            if op.order_parameter == "bond":
                self._bond_order_parameters.append(op)
            elif op.order_parameter == "mindistance":
                self._dist_order_parameters.append(op)
            else:
                raise ValueError(f"Unknown order parameter {op.order_parameter}")

    def read_bond_op(self):
        """
        reads bond order parameter from file
        """
        ops = OrderParameter.read_file(self.sim_dir / self.order_params_file)
        if type(ops) != list:
            assert ops.order_parameter == "bond"
            self.bond_op = ops
        else:
            self.bond_op, = [op for op in ops if op.order_parameter == "bond"]

    # why aren't these methods in a BuildSimulation subclass??
    def build(self, clean_build: Union[bool, str] = False):
        """
        Build dat, top, and input files in simulation directory.

        :param clean_build: If True, remove all files in sim_dir before building.
        """
        super().build(clean_build)
        if clean_build == "force":
            if (self.file_dir / "op.txt").exists():
                self.read_order_parameters(self.file_dir / "op.txt")
            if (self.file_dir / "weights.txt").exists():
                self.load_weights(self.file_dir / "weights.txt")

        self.input.swap_default_input("vmmc")
        self.build_vmmc_op_file()
        self.build_vmmc_weight_file()

    def build_vmmc(self, pre_defined_weights: Union[None, list] = None):
        """
        seperate. why?
        """
        self.build_vmmc_op_file()
        self.build_vmmc_weight_file(pre_defined_weights)

    def build_vmmc_op_file(self, clear_file: bool = False):
        """
        constructs order parameter file for vmmc
        :param clear_file: If True, clear the file before writing order parameters
        """
        assert self.bond_op is not None, "No bond order parameter specified!"
        assert "op_file" in self.input, "order parameter file hasn't been named in `input`!"
        op_file_path = self.sim_dir / self.input["op_file"]
        if clear_file and op_file_path.exists():
            op_file_path.unlink()
        for op in self.list_order_parameters():
            op.write(op_file_path)

    @property
    def weights(self) -> Optional[np.ndarray]:
        if self._weights is None:
            self._weights = np.full(shape=self.weights_shape(), fill_value=1.)
        return self._weights

    def weights_shape(self) -> Optional[tuple]:
        if self._weights is not None:
            assert self._weights.shape == tuple([len(op) for op in self.list_order_parameters()])
        return tuple(len(op) for op in self.list_order_parameters())

    def set_weights(self, weights: np.ndarray, op: Optional[str, OrderParameter]=None):
        """
        i can feel myself overenigineering this
        sets weights for vmmc simulation. if op is None, weights must match the shape
        """
        if op is None:
            assert weights.shape == self.weights_shape(), \
                (f"Mismatch between weights shape {weights.shape} and number of order parameters"
                 f" {','.join([str(w) for w in self.weights_shape()])}!")
            self._weights = weights
        else:
            raise Exception("I should implement this")

    def load_weights(self, fp: Optional[Path] = None):
        if not fp:
            self.input.read_input()
            fp = self.sim_dir / self.weights_file
            assert len(self.list_order_parameters())>0, "No order parameters defined!"
        index_cols = [op.name for op in self.list_order_parameters()]

        weights_from_file = pd.read_csv(fp,
                              sep='\\s+',
                              names=index_cols + ['weight'],
                              header=None)
        # Extract index columns and weight column
        indices = tuple(weights_from_file[col].values.astype(int) for col in index_cols)

        self.clear_weights()
        # wipe existing weight matrix, replace with correct shape
        self._weights = np.zeros(shape=self.weights_shape())
        # Assign all weights at once

        self.weights[indices] = weights_from_file['weight'].values


    def plot_weights(self,
                     plot_ops: Union[List[Union[int, str]], int, str],
                     const_ops: Optional[Dict[Union[int, str], Union[int, List[int]]]] = None,
                     use_log: bool = True,
                     ax: Optional[plt.Axes] = None,
                     colors: Optional[np.ndarray] = None,
                     colormap: str = "viridis") -> Optional[plt.Figure]:
        """
        Plot weights as a function of one or two order-parameters, while holding specified
        order-parameters to particular values.

        :plot_ops: single op identifier (int index or str name) or list of two identifiers;
                  identifies which order-parameter(s) should appear on the plot axes.
        :const_ops: mapping from op identifier (index or name) -> int or list[int]; these
                    values are used to filter (hold constant) the weights. If a list is
                    provided, those selected indices are included and summed together.
                    Note: after filtering const_ops values, the axis is collapsed by summation.
        :use_log: for 1D: set y-scale log if positive; for 2D: use LogNorm when possible.
        :ax: optional matplotlib Axes to draw into. If None, a new figure is created and shown.
        :colors: optional color array for 1D bars.
        :colormap: colormap name (fallback if colors is None).
        :returns: If `ax` supplied, returns the Figure containing that Axes. If `ax` is None,
                  the plot is shown and the function returns None.
        """

        # Normalize plot_ops into a list of 1 or 2 op indices
        ops_list = list(self.list_order_parameters())
        n_ops = len(ops_list)

        # Helper to resolve identifier (int or str) -> index
        def resolve_identifier(identifier):
            if isinstance(identifier, int):
                if identifier < 0 or identifier >= n_ops:
                    raise IndexError(f"Order-parameter index {identifier} out of range [0, {n_ops - 1}]")
                return int(identifier)
            if isinstance(identifier, str):
                for i, op in enumerate(ops_list):
                    # accept op.name if available, fallback to str(op)
                    if (hasattr(op, "name") and op.name == identifier) or str(op) == identifier:
                        return i
                raise KeyError(f"No order-parameter named '{identifier}' found")
            raise TypeError(f"Order-parameter identifier must be int or str, got {type(identifier)}")

        # Accept single item or list for plot_ops
        if isinstance(plot_ops, (int, str)):
            plot_op_ids = [plot_ops]
        elif isinstance(plot_ops, (list, tuple)):
            plot_op_ids = list(plot_ops)
        else:
            raise TypeError("plot_ops must be int, str, or a list/tuple of them")

        if len(plot_op_ids) not in (1, 2):
            raise ValueError("plot_ops must identify either 1 or 2 order-parameters to plot")

        # Resolve to indices
        plot_indices = [resolve_identifier(p) for p in plot_op_ids]

        # Build const_ops mapping from index -> list[int]
        const_mapping = {}
        if const_ops is not None:
            if not isinstance(const_ops, dict):
                raise TypeError("const_ops must be a dict mapping op identifier -> int or list[int]")
            for k, v in const_ops.items():
                idx = resolve_identifier(k)
                # normalize to list of ints
                if isinstance(v, int):
                    vals = [int(v)]
                elif isinstance(v, (list, tuple, np.ndarray)):
                    vals = [int(x) for x in v]
                else:
                    raise TypeError("const_ops values must be int or list/tuple/ndarray of ints")
                const_mapping[idx] = vals

        # Ensure no overlap between plot_indices and const_mapping
        overlap = set(plot_indices).intersection(const_mapping.keys())
        if overlap:
            names = ", ".join(str(o) for o in overlap)
            raise ValueError(f"Order-parameters {names} cannot be both plotted and held constant")

        n_plot_axes = len(plot_indices)

        # Start manipulating weights: we will reduce by selecting const_ops and summing over unspecified axes.
        weights = np.asarray(self.weights, dtype=float)

        # We'll process axes from highest index down to 0 so that axis numbers remain valid after reductions.
        for axis in reversed(range(n_ops)):
            if axis in plot_indices:
                # keep axis as-is for plotting
                continue
            elif axis in const_mapping:
                # take only the listed indices, then sum across that axis to collapse it
                selection = const_mapping[axis]
                try:
                    weights = np.take(weights, indices=selection, axis=axis)
                except Exception as e:
                    raise IndexError(f"Failed to select indices {selection} along axis {axis}: {e}")
                weights = np.sum(weights, axis=axis)
            else:
                # axis is neither plotted nor held constant -> aggregate by summation across it
                weights = np.sum(weights, axis=axis)

        # After aggregation, weights shape should match either (N,) for 1D or (N1, N2) for 2D.
        if n_plot_axes == 1:
            # Get the single axis index and expected number of states
            op_idx = plot_indices[0]
            op = ops_list[op_idx]
            # ensure weights is 1D
            if weights.ndim != 1:
                weights = np.squeeze(weights)
                if weights.ndim != 1:
                    raise ValueError("Internal error: after aggregation expected 1D array for plotting")

            # Determine number of bars from possible_states(op)
            state_list = possible_states(op)
            n_bars = len(state_list)

            # Truncate/pad as needed
            if weights.size < n_bars:
                padded = np.full(n_bars, np.nan, dtype=float)
                padded[:weights.size] = weights
                weights_1d = padded
            else:
                weights_1d = weights[:n_bars]

            do_show = ax is None
            if ax is None:
                fig, ax = plt.subplots()
            else:
                fig = ax.figure

            # Colors handling
            if colors is None:
                cmap = cm.get_cmap(colormap)
                cmap_colors = cmap(np.linspace(0, 1, n_bars))

            ax.bar(np.arange(n_bars), weights_1d)
            ax.set_xlabel(op.name if hasattr(op, "name") else f"op_{op_idx}")
            ax.set_ylabel("Weight")
            ax.set_title("VMMC Weights vs " + (op.name if hasattr(op, "name") else f"op_{op_idx}"))
            if use_log:
                if np.nanmax(weights_1d) <= 0:
                    ax.set_yscale("linear")
                else:
                    ax.set_yscale("log")

            if do_show:
                plt.show()
            return fig

        else:
            # 2D heatmap
            idx1, idx2 = plot_indices[0], plot_indices[1]
            op1, op2 = ops_list[idx1], ops_list[idx2]

            arr = np.array(weights)
            if arr.ndim == 1:
                raise ValueError("After aggregation expected 2D array for heatmap plotting but got 1D")
            if arr.ndim > 2:
                arr = np.squeeze(arr)
                if arr.ndim != 2:
                    raise ValueError("After aggregation expected 2D array for heatmap plotting")

            states = possible_states(op1, op2)
            try:
                s1, s2 = zip(*states)
                size1, size2 = max(s1) + 1, max(s2) + 1
            except Exception:
                size1, size2 = arr.shape[0], arr.shape[1]

            if arr.shape == (size1, size2):
                weights_2d = arr
            elif arr.shape == (size2, size1):
                weights_2d = arr.T
            else:
                if arr.size == (size1 * size2):
                    try:
                        weights_2d = arr.reshape((size1, size2))
                    except Exception:
                        raise ValueError("Could not reshape aggregated weights into expected 2D shape")
                else:
                    weights_2d = arr

            do_show = ax is None
            if ax is None:
                fig, ax = plt.subplots()
            else:
                fig = ax.figure

            norm = None
            if use_log:
                positive_mask = np.isfinite(weights_2d) & (weights_2d > 0)
                if np.any(positive_mask):
                    vmin = float(np.nanmin(weights_2d[positive_mask]))
                    vmax = float(np.nanmax(weights_2d[positive_mask]))
                    if vmin > 0 and np.isfinite(vmin) and np.isfinite(vmax):
                        norm = LogNorm(vmin=vmin, vmax=vmax)

            im = ax.imshow(weights_2d.T, aspect='equal', origin='lower', cmap=colormap, norm=norm)
            plt.colorbar(im, ax=ax, label='Weight')
            ax.set_xlabel(op1.name if hasattr(op1, "name") else f"op_{idx1}")
            ax.set_ylabel(op2.name if hasattr(op2, "name") else f"op_{idx2}")
            ax.set_title("VMMC Weights Heatmap")

            if do_show:
                plt.show()
            return fig

    def clear_weights(self):
        """
        explicit method to unset weight matrix, so this does not become problem elsewhere
        """
        self._weights = None

    def build_vmmc_weight_file(self, set_weights: Optional[np.ndarray] = None, skip_val: Optional[float] = None):
        """
        builds a virtual move monte carlo weight file
        :param set_weights: if provided, this weight matrix is used instead of the current self.weights. provided for backwards compatibility
        """
        if set_weights is not None:
            self.weights[...] = set_weights
        # Check for NaNs
        if np.isnan(self.weights).any():
            bad_coords = np.column_stack(np.nonzero(np.isnan(self.weights)))
            coord_str = ",".join(
                "(" + ",".join(str(int(x)) for x in coord) + ")"
                for coord in bad_coords
            )
            raise ValueError(
                "NaN weight values detected! This should never happen. "
                f"Bad weight coordinates: {coord_str}"
            )

        # Check for infinities
        if np.isinf(self.weights).any():
            bad_coords = np.column_stack(np.nonzero(np.isinf(self.weights)))
            coord_str = ",".join(
                "(" + ",".join(str(int(x)) for x in coord) + ")"
                for coord in bad_coords
            )
            raise ValueError(
                "Infinite weight values detected! This should never happen. "
                f"Bad weight coordinates: {coord_str}"
            )
        # write file
        # todo: probably a faster way
        with (self.sim_dir / self.weights_file).open("w") as f:
            for weight_coord in itertools.product(
                *[range(len(op)) for op in self.list_order_parameters()]
            ):
                if skip_val is None or self.weights[weight_coord] != skip_val:
                    weight_file_line = " ".join(str(wc) for wc in weight_coord)
                    f.write(f"{weight_file_line} {self.weights[weight_coord]}\n")
        if skip_val is not None:
            self.input["safe_weights"] = False
            self.input["default_weight"] = skip_val
        elif "default_weight" in self.input:
            del self.input["default_weight"]
            del self.input["safe_weights"] # defaults to true

    def generate_weights(self, increase_factor: float=7., bond_op: Optional[str, OrderParameter]=None) -> np.ndarray:
        """
        generates weights for vmmc simulation by exponentially increasing weight for each
        additional base pair broken
        :param increase_factor: factor by which to increase weight for each additional base pair broken
        :return: list of weights
        """
        if bond_op is not None and isinstance(bond_op, str):
            bond_op = self.get_op(bond_op)
        elif bond_op is None:
            bond_op = self.bond_op
        possible_bonds, = zip(*possible_states(bond_op))
        weights = np.full(shape=len(bond_op), fill_value=1.)
        weights[possible_bonds[1:],] = [
            increase_factor**(len(possible_bonds)-(i+1)) for i in range(1,len(possible_bonds))]
        return weights

    def build_com_hb_observable(self, p1: str, p2: str, print_every: int = 1e3):
        """
        builds observable to track number of H-bonds and distance between H-bonds
        i don't think this is used anywhere??
        """
        com_dist_obs = Observable("com_distance",
                                  print_every,
                                  ObservableColumn(
                                      "distance",
                                      particle_1=p1,
                                      particle_2=p2,
                                      PBC='1'
                                  ))
        hb_list_obs = Observable('hb_observable',
                                print_every,
                                ObservableColumn('hb_list',
                                                only_count=True)
                                 )

        for observable in [com_dist_obs, hb_list_obs]:
            print(observable.file_name)
            self.analysis.observables[observable._file_name] = observable
            self.add_observable(observable)

    @property
    def parallel_tampering(self) -> bool:
        return self.input["sim_type"] == "PT_VMMC"

    @parallel_tampering.setter
    def parallel_tampering(self, new_val: bool):
        if new_val:
            self.input["sim_type"] = "PT_VMMC"
        else:
            self.input["sim_type"] = "VMMC"

    @property
    def oxpy_run(self):
        """
        overwrite property-decorated-function from superclass, to use correct
        VMMC class, and check for PT
        """
        if self.parallel_tampering:
            # todo: more specific exception
            raise Exception("Cannot use oxpy with parallel tampering!")
        if self._oxpy_run is None:
            self._oxpy_run = VmmcOxpyRun(self)
        return self._oxpy_run

    @property
    def analysis(self):
        """
        overwrite property-decorated function from superclass to use correct
        VMMC class
        """
        if self._analysis is None:
            self._analysis = VmmcAnalysis(self)
        return self._analysis

    @property
    def PT_Ts(self) -> list[str]:
        if not self.parallel_tampering:
            raise Exception("Simulation not configured for parallel tempering")
        return [T.strip() for T in self.input["pt_temp_list"].split(",")]

    @PT_Ts.setter
    def PT_Ts(self, Ts: list[str]):
        if not self.parallel_tampering:
            raise Exception("Simulation not configured for parallel tempering")
        self.input["pt_temp_list"] = ",".join(Ts)

    def num_PT_Ts(self):
        return len(self.PT_Ts)

    @property
    def weights_file(self):
        return self.input["weights_file"]

    @weights_file.setter
    def weights_file(self, wfilename: str):
        self.input["weights_file"] = wfilename

    @property
    def order_params_file(self):
        return self.input["op_file"]

    @order_params_file.setter
    def order_params_file(self, ofilename: str):
        self.input["op_file"] = ofilename

    def build_pt_files(self):
        for i, _ in enumerate(self.PT_Ts):
            # copy topology
            thread_top = Path(f"{self.sim_dir / self.sim_files.top}{i}")
            if thread_top.exists():
                thread_top.unlink()
            shutil.copy(
                self.sim_dir / self.sim_files.top,
                thread_top
            )
            # copy initial conf
            thread_dat = Path(f"{self.sim_dir / self.sim_files.dat}{i}")
            if thread_dat.exists():
                thread_dat.unlink()
            shutil.copy(
                self.sim_dir / self.sim_files.dat,
                thread_dat
            )
            # copy weights
            thread_weights = Path(f"{self.sim_dir / self.weights_file}{i}")
            if thread_weights.exists():
                thread_weights.unlink()
            shutil.copy(
                self.sim_dir / self.weights_file,
                thread_weights
            )
            # copy order parameter
            thread_op = Path(f"{str(self.sim_dir / self.order_params_file)}{i}")
            if thread_op.exists():
                thread_op.unlink()
            shutil.copy(
                self.sim_dir / self.order_params_file,
                thread_op
            )

    def cli_run(self, oxDNA_exec_path: Path):
        """
        runs oxDNA from command line
        moslty for mpi / parallel tempering but should be compatible w/ normal
        """
        os.chdir(self.sim_dir)
        if self.parallel_tampering:
            cmd = f"mpirun -np {self.num_PT_Ts()} {str(oxDNA_exec_path)} input "
        else:
            cmd = f"{str(oxDNA_exec_path)} input"
        print(cmd)
        os.system(cmd)

    def split_to_directories(self):
        """
        split a parallel-tempering vmmc simulation into separate directories for each temperature
        """
        assert self.parallel_tampering, "Cannot split vmmc that isn't parallel tempering"
        for file in self.sim_dir.iterdir():
            mpi_prefix_regex = r"mpi_(\d+)_(.+)"
            mpi_suffix_regex = r"(.+)(\d+)$"
            match = re.match(mpi_prefix_regex, file.name)
            if match:
                thread_idx = match.group(1)
                thread_filename = match.group(2)
                thread_dir = self.sim_dir / f"T{self.PT_Ts[int(thread_idx)]}"
                if not thread_dir.exists():
                    thread_dir.mkdir()
                shutil.move(file, thread_dir / thread_filename)
                continue
            match = re.match(mpi_suffix_regex, file.name)
            if match:
                thread_idx = match.group(2)
                thread_filename = match.group(1)
                thread_dir = self.sim_dir / f"T{self.PT_Ts[int(thread_idx)]}"
                if not thread_dir.exists():
                    thread_dir.mkdir()
                shutil.move(file, thread_dir / thread_filename)
                continue

    def pt_dirs_to_sims(self) -> list[VirtualMoveMonteCarlo]:
        """
        splits a parallel-tempering vmmc simulation into separate VirtualMoveMonteCarlo objects for each temperature
        """
        self.split_to_directories()
        vmmc_sims = [None] * self.num_PT_Ts()
        for i, T in enumerate(self.PT_Ts):
            shutil.copy(self.sim_dir / "input", self.sim_dir / f"T{T}" / "input")
            vmmc_sims[i] = VirtualMoveMonteCarlo(self.sim_dir / f"T{T}")
            vmmc_sims[i].input["T"] = T
        return vmmc_sims