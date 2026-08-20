"""
Standardized chaining of VmmcWindowing (splitting a system's order-parameter space into
windows) with VMMCAutoReweight (iteratively adjusting weights within a single window from
observed sampling statistics), so each window is run to convergence instead of once with a
single static weight matrix.
"""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Optional, Type, Union

import numpy as np

from ..oxdna_simulation import BuildSimulationFromStructure
from .auto_reweight import VMMCAutoReweight
from .vmmc import VirtualMoveMonteCarlo
from .windowing import VmmcWindowing


class VmmcWindowedAutoReweight:
    """
    Runs one VMMCAutoReweight (or subclass, e.g. VMMCGraphReweight) per window of an
    already-configured VmmcWindowing, each confined to that window's state_space_area, then
    splices each window's final, converged iteration back into the parent VmmcWindowing so
    its existing analysis (get_merged_weights, wham, plot_merged_hist, plot_window_data, ...)
    keeps working on the reweighted data.

    Usage:
        windowing = VmmcWindowing(tld)
        windowing.add_order_parameters([...])
        windowing.add_window(state_space_area, starting_conf)
        ...  # as many windows as needed
        windowing.extrapolate_hist_Ts = [...]

        chained = VmmcWindowedAutoReweight(windowing)   # writes to windowing.tld / "reweighted"
        chained.build_replica = my_build_replica          # (reweighter, sim) -> None
        chained.steps_per_iter = 5e7
        chained.max_iterations = 10
        chained.run()

        # windowing's own analysis now reflects the reweighted windows:
        windowing.wham()

    This splices the in-memory result back into `windowing`'s `_subgroups` immediately after
    each window finishes, so a caller can inspect intermediate progress in the same process.
    In a later session, `VmmcWindowing.load(splice_reweighted=True)` reconstructs the same
    spliced state from disk (see VmmcWindowing._load_reweighted_windows) -- provided
    `tld_path` was left at its default (`windowing.tld / "reweighted"`) or the same path is
    passed to `load()` as `reweight_tld`.
    """

    def __init__(self,
                 windowing: VmmcWindowing,
                 tld_path: Optional[Path] = None,
                 reweighter_cls: Type[VMMCAutoReweight] = VMMCAutoReweight):
        self.windowing = windowing
        self.tld = Path(tld_path) if tld_path is not None else windowing.tld / "reweighted"
        self.reweighter_cls = reweighter_cls

        # applied to each replica after the window's starting structure is wired in;
        # same signature as VMMCAutoReweight.build_replica: (reweighter, sim) -> None
        self.build_replica: Optional[Callable[[VMMCAutoReweight, VirtualMoveMonteCarlo], None]] = None
        # same signature as VMMCAutoReweight.build_start_weights: (sim) -> None
        # left None: for a single-bond-op windowing, falls back to a window-scoped default
        # (see _window_scoped_start_weights); for multi-op windowing, falls back to
        # reweighter_cls's own (global, unscoped) default -- provide your own here in that case
        self.build_start_weights: Optional[Callable[[VirtualMoveMonteCarlo], None]] = None
        # increase_factor used by the window-scoped default start-weights function, when used
        # (see build_start_weights above). Same meaning as
        # VirtualMoveMonteCarlo.generate_weights()'s own increase_factor.
        self.start_weights_increase_factor: float = 7.

        # forwarded onto each window's reweighter
        self.steps_per_iter: float = 1e8
        self.max_iterations: float = float('inf')
        self.max_rel_std: float = 5.
        self.zero_sample_weight_factor: float = 1e4
        self.reweight_borders: Union[bool, float] = False

        self._reweighters: dict[int, VMMCAutoReweight] = {}

    def build_window_reweighter(self, window_idx: int) -> VMMCAutoReweight:
        """
        Construct (but do not run) the VMMCAutoReweight for a single window, confined to
        that window's state_space_area.
        """
        window = self.windowing[window_idx]
        starting_conf = window.file_dir
        if starting_conf is None:
            raise ValueError(
                f"Window {window_idx} has no starting configuration attached (state was "
                "probably reconstructed via VmmcWindowing.load(), which does not restore "
                "it). Use a windowing instance whose windows were built with add_window() "
                "in this session."
            )

        ar = self.reweighter_cls(self.tld / f"window_{window_idx}")
        ar.add_order_parameters(self.windowing.order_parameters())
        ar.extrapolate_hist_Ts = self.windowing.extrapolate_hist_Ts
        ar.n_reps = self.windowing.n_reps
        ar.steps_per_iter = self.steps_per_iter
        ar.max_iterations = self.max_iterations
        ar.max_rel_std = self.max_rel_std
        ar.zero_sample_weight_factor = self.zero_sample_weight_factor
        ar.reweight_borders = self.reweight_borders
        # out-of-window states must be a hard exclusion, not the base classes' soft "illegal"
        # sentinel (1.0) -- unlike physically-impossible states, out-of-window states are
        # perfectly reachable and are often thermodynamically favorable (e.g. a window that
        # excludes the fully-bonded state), so a soft sentinel does not keep the walker inside
        # the window. Matches VmmcWindowing.setup()'s own out-of-window weight of 0.
        ar.illegal_state_weight = 0.0

        window_states = window.state_space_area
        global_filter = self.windowing.filter_legal_states
        # legal within a window: globally legal AND inside this window's slice of state space
        ar.filter_legal_states = lambda states: [s for s in global_filter(states) if s in window_states]
        # every legal (in-window) state is the flat-sampling target -- that's the point of a window
        ar.filter_desired_states = lambda states: list(states)

        outer_build_replica = self.build_replica

        def build_replica(reweighter, sim):
            sim.set_builder(BuildSimulationFromStructure(sim, starting_conf))
            if outer_build_replica is not None:
                outer_build_replica(reweighter, sim)

        ar.build_replica = build_replica

        if self.build_start_weights is not None:
            ar.build_start_weights = self.build_start_weights
        elif len(self.windowing.order_parameters()) == 1:
            ar.build_start_weights = self._window_scoped_start_weights(window_states)
        # else: multi-op windowing with no explicit build_start_weights -- falls back to
        # reweighter_cls's own default (generate_weights(), which is itself only meaningful
        # for a single bond op), same as before this method existed. Provide your own
        # build_start_weights for multi-op windowing.

        return ar

    def _window_scoped_start_weights(self,
                                     window_states: set[tuple[int, ...]]
                                     ) -> Callable[[VirtualMoveMonteCarlo], None]:
        """
        Returns a build_start_weights(sim) callable equivalent to
        VirtualMoveMonteCarlo.generate_weights(), but with the exponent sized to this
        window's own span of states instead of the order parameter's full global range.

        generate_weights() sizes its exponent from the *global* number of possible states, so
        a narrow window inherits a starting gradient far steeper than the window itself needs
        (e.g. a 6-state window out of a 13-state global range gets weights spanning up to
        7**11, not 7**5) -- badly mismatching neighboring in-window states and giving the
        iterative reweighter a much larger, and much more unstable, gap to close.

        Only meaningful for single-bond-op windowing (matching generate_weights()'s own
        scope, which cannot even be assigned into a multi-dimensional weights array).
        """
        # states in this window, sorted ascending along the (single) bond op's axis
        local_values = sorted(state[0] for state in window_states)
        n_local = len(local_values)
        increase_factor = self.start_weights_increase_factor

        def build_start_weights(sim: VirtualMoveMonteCarlo):
            bond_op = sim.bond_op
            weights = np.full(shape=len(bond_op), fill_value=1.)
            for rank, state_val in enumerate(local_values):
                # rank 0 = fewest bonds in this window -> highest weight (favor melting,
                # matching generate_weights()'s own convention); last rank -> baseline (1.0)
                weights[state_val] = increase_factor ** (n_local - 1 - rank)
            sim.weights[...] = weights

        return build_start_weights

    def run(self, windows: Optional[list[int]] = None) -> "VmmcWindowedAutoReweight":
        """
        Run auto-reweighting for each window in turn (sequentially), splicing each window's
        final iteration back into the parent VmmcWindowing as it completes.
        """
        self.windowing.check_ready()
        # write setup.json now (not just on VmmcWindowing.setup(), which this bypasses) so
        # VmmcWindowing.load(splice_reweighted=True) can reconstruct the windows later, even
        # if this call is interrupted partway through
        self.windowing.cache_settings()
        if windows is None:
            windows = list(range(len(self.windowing)))
        for idx in windows:
            print(f"=== Auto-reweighting window {idx} ({len(windows)} total) ===")
            ar = self.build_window_reweighter(idx)
            ar.run()
            self._reweighters[idx] = ar
            self._splice_into_windowing(idx, ar)
        return self

    def _splice_into_windowing(self, window_idx: int, ar: VMMCAutoReweight):
        """
        Replace the window at `window_idx` in the parent VmmcWindowing with one wrapping the
        reweighter's final, converged iteration.
        """
        final_it = ar[-1]
        self.windowing._splice_window(window_idx, final_it.sim_dir, list(final_it))

    def __getitem__(self, window_idx: int) -> VMMCAutoReweight:
        return self._reweighters[window_idx]

    def __iter__(self):
        yield from self._reweighters.values()

    def __len__(self):
        return len(self._reweighters)
