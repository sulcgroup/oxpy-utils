from __future__ import annotations

import re
from abc import ABCMeta, abstractmethod, ABC
from functools import cached_property
from pathlib import Path
from typing import Optional, Callable, Any, Union

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator

from ..utils.order_parameter import OrderParameter, possible_states, create_state_mask
from .vmmc import VirtualMoveMonteCarlo
from .vmmc_replicas import VmmcReplicas
from ..utils.util import generate_distinct_colors


class VMMCMetaSimulation(ABC):
    """
    Abstract base class for groups of related VMMC simulations (beyond replicas) for various contexts
    I don't love the name "MetaSimulation" but can't think of a better one right now.
    """

    build_replica: Callable[[Any, VirtualMoveMonteCarlo], None]

    # number of replicas per iteration
    _n_replicas: int

    _tld: Path
    # list of iterations, each iteration is a list of replicas
    _subgroups: list[VmmcReplicas]

    # callback to iteration-0 weights
    build_start_weights: Callable[[VirtualMoveMonteCarlo, Optional[int]], None]
    # callback to filter out states which are illegal
    filter_legal_states: Callable[[list[tuple[int, ...]]], list[tuple[int, ...]]]
    # callback to filter out states which are legal but not intresting for the purposes of reweighting
    filter_desired_states: Callable[[list[tuple[int, ...]]], list[tuple[int, ...]]]

    _dist_op: Optional[OrderParameter]
    _bond_ops: list[OrderParameter]

    extrapolate_hist_Ts: Optional[list[str]] = None


    def __init__(self,
                 tld_path: Path,
                 sim_build_func: Optional[Callable[[Any, VirtualMoveMonteCarlo], None]]=None, # allowed for analysis purposes
                 ):
        """
        :param tld_path: path to top-level directory where iterations will be stored
        :param sim_build_func: function which will be called to build each replica simulation
        """
        self._tld = Path(tld_path)
        self._subgroups = []
        self.build_replica = sim_build_func
        self._n_replicas = 5

        self._dist_op = None
        self._bond_ops = []

        # can overwrite later
        self.build_start_weights = build_start_weights_default
        # default to no filter, overwrite later if needed
        self.filter_legal_states = lambda x:x
        self.filter_desired_states = lambda x:x
        self.start_iter_callback = None
        self.end_iter_callback = None

    def  order_parameters(self) -> list[OrderParameter]:
        """
        :return: list of order parameters used for reweighting
        """
        return  self._bond_ops + [self._dist_op] if self._dist_op is not None else self._bond_ops

    def add_order_parameters(self, ops: list[OrderParameter]):
        """
        add order parameters for reweighting
        """
        for op in ops:
            self.add_order_parameter(op)


    def add_order_parameter(self, op: OrderParameter):
        """
        Docstring for add_order_parameter
        
        :param self: Description
        :param op: Description
        :type op: OrderParameter
        """
        assert not any(op.name == old_op.name for old_op in self.order_parameters()), f"Trying to add an order parameter with existing name {op.name}"
        if op.order_parameter == "mindistance":
            if self._dist_op is not None:
                raise ValueError("Distance order parameter already set")
            self._dist_op = op
        else:
            if op.order_parameter != "bond":
                raise ValueError(f"Unrecognized order parameter type {op.order_parameter}")
            self._bond_ops.append(op)

    @abstractmethod
    def check_ready(self):
        pass

    def setup(self):
        """
        sets up subgroups/iterations for the meta-simulation
        not *required* to be implemented by subclasses, but recommended
        """
        pass

    @abstractmethod
    def run(self, join: bool=True):
        """

        """
        pass

    @abstractmethod
    def visualize(self, *args):
        """
        Visualize all iterations with weights, bond curves, and pie charts.
        By default, only visualizes the first bond order parameter.

        :param bond_op_index: index of bond order parameter to visualize (default: 0)
        """
        pass

    def has_dist_op(self) -> bool:
        return self._dist_op is not None

    def add_bond_op(self, op: OrderParameter):
        """
        add bond order parameter for reweighting
        """
        self._bond_ops.append(op)

    def num_bond_ops(self) -> int:
        """
        :return: number of bond order parameters used for reweighting
        """
        return len(self._bond_ops)

    def bond_ops(self) -> list[OrderParameter]:
        """
        :return: list of bond order parameters used for reweighting
        """
        return self._bond_ops

    @property
    def n_reps(self):
        return self._n_replicas

    @n_reps.setter
    def n_reps(self, value):
        if value < 1:
            raise ValueError("Number of replicas per iteration must be at least 1")
        if self._subgroups:
            raise ValueError("Cannot change number of replicas per iteration after loading iterations")
        self._n_replicas = value

    @cached_property
    def legal_state_list(self) -> list[tuple[int, ...]]:
        """
        cached property.
        """
        return self.filter_legal_states(possible_states(*self.order_parameters()))

    @cached_property
    def desired_state_list(self) -> list[tuple[int, ...]]:
        return self.filter_desired_states(self.legal_state_list)

    @property
    def tld(self):
        """
        accessor for tld.
        note: tld not settable publically
        :return: top-level directory path
        """
        return self._tld

    @cached_property
    def desired_states_mask(self) -> np.ndarray:
        return create_state_mask(
            *self.order_parameters(),
            accessible_states=self.desired_state_list
        )

    @cached_property
    def legal_states_mask(self) -> np.ndarray:
        """
        generate n-dimensional array mask where each (n, m, ...) index is True if
        the state is accessable, False otherwise
        """
        return create_state_mask(
            *self.order_parameters(),
            accessible_states=self.legal_state_list
        )

    def plot_bond_curves(self,
                         subgroup_idx: int = -1,
                         bond_op_index: int = 0,
                         ax: Optional[plt.Axes] = None,
                         replica_colors: Optional[np.ndarray] = None,
                         show_legend: bool = True) -> Optional[plt.Figure]:
        """
        Plot bond order parameter values vs. time for a single iteration.

        :param subgroup_idx: iteration index to plot (default: -1 for last iteration)
        :param bond_op_index: index of bond order parameter to plot (default: 0)
        :param ax: optional matplotlib axes to plot on. If None, creates new figure
        :param replica_colors: optional array of colors for replicas. If None, uses the Okabe-Ito palette
        :param show_legend: whether to show legend (default: True)
        :return: figure if ax is None, otherwise None
        """
        fig = None
        make_new_fig = ax is None
        if make_new_fig:
            fig, ax = plt.subplots(figsize=(12, 4))

        # Get replicas for this iteration
        replicas = self[subgroup_idx]

        # Bounds-check before indexing bond_ops(), not after: the raw IndexError from an
        # out-of-range index would otherwise fire first and this check would never run
        if bond_op_index >= len(self.bond_ops()):
            raise ValueError(f"bond_op_index {bond_op_index} out of range (only {len(self.bond_ops())} bond ops)")
        bond_op = self.bond_ops()[bond_op_index]
        replicas.plot_op_val_curve(bond_op, ax, replica_colors, show_legend)

        return fig

    def _add_weight_vis_hatching(self, ax: plt.Axes):
        """
        Add visual indicators for illegal and undesired states on a 2D heatmap.
        - Illegal states: gray hatching (original style)
        - Legal but undesired states: white/light border outline
        """

        op1 = self._bond_ops[0]
        op2 = self._bond_ops[1]
        all_possible_states = possible_states(op1, op2)
        state_set_1, state_set_2 = zip(*all_possible_states)

        max_dim1 = max(state_set_1)
        max_dim2 = max(state_set_2)

        # Get masks
        legal_mask = self.legal_states_mask[:max_dim1, :max_dim2, 0].T
        desired_mask = self.desired_states_mask[:max_dim1, :max_dim2, 0].T

        illegal_mask = ~legal_mask
        undesired_mask = legal_mask & ~desired_mask

        # Add gray hatching for illegal states (original style)
        add_hatching(ax, illegal_mask)

        # Add white/light border to undesired states
        undesired_rows, undesired_cols = np.where(undesired_mask)
        for row, col in zip(undesired_rows, undesired_cols):
            rect = Rectangle((col - 0.5, row - 0.5), 1, 1,
                            fill=False,
                            edgecolor='white',  # or try 'lightgray', '#CCCCCC'
                            linewidth=2.5)
            ax.add_patch(rect)

    def get_weights_colors(self, op: Optional[Union[str, int, OrderParameter]]=None) -> np.ndarray:
        if op is None:
            op = self.bond_ops()[0]
        elif isinstance(op, str):
            op = next((o for o in self.bond_ops() if o.name == op), None)
            if op is None:
                raise ValueError(f"No bond order parameter with name {op}")
        elif isinstance(op, int):
            op = self.bond_ops()[op]

        try:
            op_idx = next(i for i, _op in enumerate(self.order_parameters()) if _op.name == op.name)
        except StopIteration:
            raise ValueError(f"Order parameter {op.name} not found in order parameters")

        all_states = possible_states(*self.order_parameters())
        unique_values = sorted(set(state[op_idx] for state in all_states))
        n_values = len(unique_values)

        return generate_distinct_colors(n_values)

    def get_overall_unwt_occ(self, vmmc_from: VmmcReplicas):
        """
        get overall unweighted occupancies from given iteration of VMMC simulations
        """
        # initialize overall unweighted occupancy array
        overall_unwt_occ = np.zeros(shape=[len(op) for op in self.order_parameters()])
        # get bond order parameter names

       # iterate over replicas
        for i, sim in enumerate(vmmc_from):
            # load simulation input information
            sim.input.read_input()
            # use unweighted state occupancies to estimate sampling
            filtered_df: pd.DataFrame = sim.analysis.vmmc_df

            # Extract the bond order parameter indices and unwt_occ values. vmmc_df is
            # indexed by the order parameter value(s) (see read_vmmc_op_data) -- idx *is*
            # the state already, not something to look up a row column for. A single order
            # parameter produces a plain (non-Multi) Index, so idx arrives as a bare scalar
            # in that case.
            for idx, row in filtered_df.iterrows():
                indices = idx if isinstance(idx, tuple) else (idx,)
                # Add the unweighted occupancy to the correct position
                overall_unwt_occ[indices] += row['unwt_occ']

        # divide by most visited state to normalize
        overall_unwt_occ /= overall_unwt_occ.max()
        # if we are using a distance order parameter
        # if self._dist_op:
        #     expand to include distance order parameter dimension
            # overall_unwt_occ = np.broadcast_to(overall_unwt_occ[..., np.newaxis], [len(op) for op in self.order_parameters()])
        return overall_unwt_occ

    def __getitem__(self, item: int) -> VmmcReplicas:
        return self._subgroups[item]

    def __iter__(self):
        yield from self._subgroups

    def __len__(self) -> int:
        return len(self._subgroups)


def build_start_weights_default(vmmc: VirtualMoveMonteCarlo):
    vmmc.weights[:] = vmmc.generate_weights(7.)


def add_hatching(ax: plt.Axes,
                 inaccessible_mask:np.ndarray):
    """
    Add hatching pattern to inaccessible cells on a heatmap

    :param ax: matplotlib axes to add hatching to
    :param inaccessible_mask: 2D boolean array where True indicates inaccessible cells
    """
    hatch_pattern = '///'
    edgecolor = 'white'
    linewidth = 0
    for i in range(inaccessible_mask.shape[0]):
        for j in range(inaccessible_mask.shape[1]):
            if inaccessible_mask[i, j]:
                rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                     fill=False, hatch=hatch_pattern,
                                     edgecolor=edgecolor, linewidth=linewidth)
                ax.add_patch(rect)
