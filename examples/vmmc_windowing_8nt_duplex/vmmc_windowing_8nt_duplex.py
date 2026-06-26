# =============================================================================
# Tutorial: VMMC Windowing — Free Energy Profile of an 8-nt DNA Duplex
# =============================================================================
#
# Virtual-Move Monte Carlo (VMMC) is a cluster-move algorithm that is highly
# efficient for simulating nucleic acid folding and unfolding. Unlike molecular
# dynamics, VMMC accepts or rejects whole rigid-body moves of nucleotide
# clusters, dramatically reducing the time needed to cross free-energy barriers.
#
# Windowing divides the reaction-coordinate state space into overlapping
# regions ("windows") and runs an independent VMMC simulation inside each one.
# This lets every region of the free-energy landscape be sampled on equal
# footing, even those that are rarely visited in an unbiased simulation.
# The per-window histograms are then combined using WHAM to reconstruct the
# global free-energy profile.
#
# This tutorial uses the same 8-nt DNA duplex system as the other tutorials in
# this repository. The reaction coordinate is the number of native hydrogen
# bonds (0 = fully melted, 8 = fully formed).
#
# System:  strand 1 (nucleotides 0–7) paired with strand 2 (15–8)
# Windows: two overlapping windows covering the full bonding range 0–8
#   Window 0: 5–8 bonds  (folded end)
#   Window 1: 0–5 bonds  (melted end)
#   Overlap:  both windows include state 5, which is required for WHAM
#
# =============================================================================

import shutil
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from oxpy_utils.utils.order_parameter import OrderParameter
from oxpy_utils.vmmc_umbrella.windowing import VmmcWindowing


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

HERE = Path(__file__).parent

# oxDNA topology + starting configurations
OXDNA_FILES = HERE / "oxdna_files"

# Standard fully-formed starting conf for window 0 (8 bonds)
FOLDED_START = OXDNA_FILES           # VmmcWindowing accepts a directory

# The default init.dat starts at 8 bonds — outside window 1's state space.
# We provide a partially-melted conf (≤7 bonds) so window 1 initialises
# inside its own state space.  See oxdna_files/melted_start.dat.
MELTED_START_DIR = OXDNA_FILES / "melted_start_dir"   # built below


def prepare_melted_start_dir():
    """
    Build a source directory for window 1 that shares the same topology but
    uses a partially-melted starting configuration.
    """
    MELTED_START_DIR.mkdir(exist_ok=True)
    shutil.copy(OXDNA_FILES / "duplex_box_30.top", MELTED_START_DIR)
    shutil.copy(
        OXDNA_FILES / "melted_start.dat",
        MELTED_START_DIR / "duplex_box_30.dat",
    )
    return MELTED_START_DIR


# ---------------------------------------------------------------------------
# Reaction-coordinate order parameter
# ---------------------------------------------------------------------------
# The bond order parameter counts Watson–Crick hydrogen bonds between the two
# strands.  Strand 1 runs 5'→3' as nucleotides 0–7; strand 2 runs 3'→5' as
# nucleotides 8–15 (so pair 0 is 0↔15, pair 1 is 1↔14, etc.).

BONDS = list(zip(
    [0, 1, 2, 3, 4, 5, 6, 7],
    reversed([8, 9, 10, 11, 12, 13, 14, 15]),
))
NATIVE_OP = OrderParameter("native", "bond", BONDS)

# With 8 pairs the OP ranges from 0 (no bonds) to 8 (all bonds).
# len(NATIVE_OP) == 9 (pairs + 1), so valid states are 0 through 8.


# ---------------------------------------------------------------------------
# Window state spaces
# ---------------------------------------------------------------------------
# Each state is a 1-tuple because we have a single order parameter.
# The two windows must overlap so that WHAM can stitch the histograms together.

WINDOW_0_STATES = {(s,) for s in range(5, 9)}   # 5, 6, 7, 8 bonds
WINDOW_1_STATES = {(s,) for s in range(0, 6)}   # 0, 1, 2, 3, 4, 5 bonds


# ---------------------------------------------------------------------------
# Build and configure the VmmcWindowing object
# ---------------------------------------------------------------------------

def make_windowing(output_dir: Path, n_replicas: int = 2) -> VmmcWindowing:
    """
    Construct and configure a VmmcWindowing object for the 8-nt duplex.

    Parameters
    ----------
    output_dir : Path
        Directory where simulation subdirectories will be written.
    n_replicas : int
        Number of independent replicas per window.  More replicas reduce
        statistical noise but increase runtime.
    """
    w = VmmcWindowing(output_dir)
    w.add_order_parameter(NATIVE_OP)
    w.n_reps = n_replicas

    # Temperatures at which the unbiased histograms will be extrapolated.
    # These must be expressible in the same string format as the oxDNA input.
    w.extrapolate_hist_Ts = ["30C", "37C", "40C", "46C", "55C"]

    # -----------------------------------------------------------------------
    # Quirk 1: filter_legal_states must return a list, not a set.
    # The default lambda passes the set through unchanged, but setup() later
    # calls list operations on the result — override it with sorted().
    # -----------------------------------------------------------------------
    w.filter_legal_states = lambda states: sorted(states)

    # -----------------------------------------------------------------------
    # build_replica: called once per simulation before it is run.
    # Customise the oxDNA input parameters here.
    # -----------------------------------------------------------------------
    def build_replica(windowing, sim):
        sim.build(clean_build="force")
        sim.input.swap_default_input("vmmc")

        # --- Simulation parameters ---
        # steps: 1e4 is fast enough for testing; production runs should use
        # at least 1e7–1e8 steps to achieve converged free-energy estimates.
        sim.input["steps"] = int(1e4)         # increase for production
        sim.input["T"] = "40C"
        sim.input["salt_concentration"] = 1.0  # 1 M NaCl
        sim.input["interaction_type"] = "DNA2"
        sim.input["print_energy_every"] = int(1e3)
        sim.input["print_conf_interval"] = int(1e3)

    w.build_replica = build_replica

    # -----------------------------------------------------------------------
    # Quirk 2: build_start_weights is called as (sim, window_idx).
    # The library default helper only accepts one argument — provide a
    # two-argument replacement.
    # -----------------------------------------------------------------------
    def build_start_weights(sim, window_idx):
        # generate_weights(T_scale) returns weights inversely proportional to
        # the Boltzmann factor, giving roughly flat sampling across all states.
        sim.weights[...] = sim.generate_weights(7.0)

    w.build_start_weights = build_start_weights

    # -----------------------------------------------------------------------
    # Add windows.  Each window needs a source directory containing the
    # .top and .dat files for its starting configuration.
    # -----------------------------------------------------------------------
    melted_src = prepare_melted_start_dir()
    w.add_window(WINDOW_0_STATES, FOLDED_START)   # window 0: folded end
    w.add_window(WINDOW_1_STATES, melted_src)     # window 1: melted end

    return w


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

def run(output_dir: Path, n_replicas: int = 2):
    w = make_windowing(output_dir, n_replicas=n_replicas)

    print("Setting up windows…")
    w.setup()

    print("Running VMMC simulations…")
    # join=True blocks until all simulations complete.  Set join=False and
    # call w.run(join=False) if you want to do other work while they run.
    w.run(join=True)

    print("Reading VMMC data…")
    for window in w:
        for sim in window:
            sim.analysis.read_vmmc_op_data()
            sim.analysis.calculate_sampling_and_probabilities()

    return w


# ---------------------------------------------------------------------------
# Analyse and visualise
# ---------------------------------------------------------------------------

def analyse(w: VmmcWindowing):
    """Run standard analyses and produce all standard plots."""

    # Per-window sampling statistics
    print("\n--- Per-window sampling statistics ---")
    for i, window in enumerate(w):
        for j, sim in enumerate(window):
            stats = sim.analysis.statistics
            print(f"  Window {i}, replica {j}:")
            print(stats[["sampling_percent", "wt_prob"]].to_string(index=True))

    # Merged weights: combine the per-window weight arrays into a single
    # global weight profile (using geometric-mean rescaling at overlaps)
    merged_weights = w.get_merged_weights()
    print(f"\nMerged weights shape: {merged_weights.shape}")
    print(f"Merged weights: {merged_weights}")

    # WHAM free-energy profile
    # wham(op=0) uses get_data_over() which constructs a VMMCData with the
    # 'unbiased_count' column that WHAM requires.
    rho = w.wham(op=0)
    print(f"\nWHAM probability distribution (n_bonds → probability):")
    for state, prob in sorted(rho.items()):
        print(f"  {state} bonds: {prob:.4f}")

    # Visualisation
    print("\nGenerating plots…")

    fig_weights, axes_weights = w.plot_window_weights()
    fig_weights.suptitle("Per-window VMMC weights")
    fig_weights.tight_layout()
    fig_weights.savefig(w.tld / "window_weights.png", dpi=150)
    plt.close(fig_weights)

    fig_data, _ = w.plot_window_data(NATIVE_OP)
    fig_data.suptitle("Per-window sampling histograms")
    fig_data.tight_layout()
    fig_data.savefig(w.tld / "window_data.png", dpi=150)
    plt.close(fig_data)

    fig_fe, ax_fe = w.plot_free_energy_profile()
    ax_fe.set_title("Free-energy profile — 8-nt duplex at 40 °C")
    ax_fe.set_xlabel("Number of native bonds")
    ax_fe.set_ylabel(r"$\Delta G$ (simulation units)")
    fig_fe.tight_layout()
    fig_fe.savefig(w.tld / "free_energy_profile.png", dpi=150)
    plt.close(fig_fe)

    # Per-window state histograms
    for i, window in enumerate(w):
        fig_hist = window.create_state_histograms(NATIVE_OP)
        if fig_hist is not None:
            fig_hist.suptitle(f"Window {i} state histograms")
            fig_hist.tight_layout()
            fig_hist.savefig(w.tld / f"window_{i}_state_histograms.png", dpi=150)
            plt.close(fig_hist)

    print(f"\nPlots saved to {w.tld}")
    return rho


# ---------------------------------------------------------------------------
# Resuming from a saved run
# ---------------------------------------------------------------------------

def resume(output_dir: Path) -> VmmcWindowing:
    """
    Reload a previously completed run without re-running simulations.

    VmmcWindowing.setup() writes a cache file (setup.json) so the object can
    be reconstructed in a fresh Python session:

        w = VmmcWindowing(output_dir)
        w.load()
        # re-read analysis data
        for window in w:
            for sim in window:
                sim.analysis.read_vmmc_op_data()
                sim.analysis.calculate_sampling_and_probabilities()
    """
    w = VmmcWindowing(output_dir)
    w.load()
    for window in w:
        for sim in window:
            sim.analysis.read_vmmc_op_data()
            sim.analysis.calculate_sampling_and_probabilities()
    return w


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="VMMC windowing tutorial for the 8-nt DNA duplex"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=HERE / "vmmc_windowing_output",
        help="Directory where simulation files are written (default: ./vmmc_windowing_output)"
    )
    parser.add_argument(
        "--n-replicas", type=int, default=2,
        help="Number of independent replicas per window (default: 2)"
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Load an existing run from --output-dir instead of running new simulations"
    )
    args = parser.parse_args()

    if args.resume:
        w = resume(args.output_dir)
    else:
        w = run(args.output_dir, n_replicas=args.n_replicas)

    rho = analyse(w)
    print("\nDone.")