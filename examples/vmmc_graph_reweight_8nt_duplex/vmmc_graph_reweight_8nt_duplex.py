# =============================================================================
# Tutorial: VMMC Graph Reweighting — Free Energy Profile of an 8-nt DNA Duplex
# =============================================================================
#
# VMMCGraphReweight runs VMMC simulations iteratively, adjusting the flat-
# histogram weights each iteration based on observed transition counts in
# order-parameter (OP) space rather than occupancy histograms.
#
# For each adjacent pair of legal states (i, j), it counts directed transitions
# c[i→j] and c[j→i] from the energy time-series, then solves a least-squares
# system over the OP-space adjacency graph to find new log-weights satisfying:
#
#     log w[j] − log w[i] = log((c[j→i] + ε) / (c[i→j] + ε))
#
# When the system flows too readily from i to j (c[i→j] > c[j→i]), the
# formula decreases w[j] relative to w[i] to discourage further flow.
# The pseudo-count ε (graph_pseudo_count) gives a smooth, bounded correction
# for unobserved edges — avoiding the large step-changes that arise when a
# state is simply marked "unsampled" in the histogram approach.
#
# This tutorial runs the same 8-nt DNA duplex used in the windowing tutorial,
# but without dividing the state space into windows.  The reweighter
# automatically finds weights that produce flat sampling across all bond states.
#
# System:    strand 1 (nucleotides 0–7) paired with strand 2 (15–8)
# OP:        number of native Watson–Crick hydrogen bonds (0 = fully melted,
#            8 = fully formed)
# Reweight:  graph-based, ε = 1.0 (default)
#
# =============================================================================

from pathlib import Path

import matplotlib.pyplot as plt

from oxpy_utils.oxdna_simulation import BuildSimulationFromStructure
from oxpy_utils.structure_editor.dna_structure import load_dna_structure
from oxpy_utils.utils.order_parameter import OrderParameter
from oxpy_utils.vmmc_umbrella.auto_reweight import VMMCGraphReweight

# ---------------------------------------------------------------------------
# Paths — reuse the oxdna_files from the vmmc_windowing tutorial
# ---------------------------------------------------------------------------

HERE = Path(__file__).parent
OXDNA_FILES = HERE.parent / "vmmc_windowing_8nt_duplex" / "oxdna_files"
STARTING_STRUCTURE = load_dna_structure(
    OXDNA_FILES / "duplex_box_30.top", OXDNA_FILES / "duplex_box_30.dat"
)


# ---------------------------------------------------------------------------
# Reaction-coordinate order parameter
# ---------------------------------------------------------------------------
# 8 Watson–Crick pairs; OP ranges over 0 (no bonds) to 8 (all bonds formed).

BONDS = list(zip(
    [0, 1, 2, 3, 4, 5, 6, 7],
    reversed([8, 9, 10, 11, 12, 13, 14, 15]),
))
NATIVE_OP = OrderParameter("native", "bond", BONDS)


# ---------------------------------------------------------------------------
# Build and configure the reweighter
# ---------------------------------------------------------------------------

def make_reweighter(output_dir: Path, n_replicas: int = 3) -> VMMCGraphReweight:
    """
    Construct and configure a VMMCGraphReweight for the 8-nt duplex.

    Parameters
    ----------
    output_dir :
        Directory where per-iteration simulation subdirectories are written.
    n_replicas :
        Independent replicas run per iteration.  More replicas give better
        transition statistics and reduce sampling variance, at proportional
        runtime cost.
    """
    ar = VMMCGraphReweight(output_dir)
    ar.add_order_parameter(NATIVE_OP)
    ar.n_reps = n_replicas

    # Temperatures at which the unbiased histograms are extrapolated after
    # each iteration.  Must match the format accepted by the oxDNA input.
    ar.extrapolate_hist_Ts = ["30C", "37C", "40C", "46C", "55C"]

    # -----------------------------------------------------------------------
    # Convergence criteria
    # -----------------------------------------------------------------------
    # Stop when the standard deviation of per-state sampling percentages
    # across the desired states drops below this threshold.
    ar.max_rel_std = 5.0          # percent; tighten to ~2 for production

    # Hard cap on the number of iterations (failsafe).
    ar.max_iterations = 10

    # Steps per replica per iteration.  1e6 is enough to smoke-test the pipeline (no
    # crashes, sane-looking numbers) but not enough data to actually reweight against;
    # 1e7 turned out to still be too short for reliable re-nucleation statistics once
    # melted (see the state-0 investigation: one replica can spend nearly an entire
    # iteration's budget stuck in the fully-melted state while the others never reach it
    # at all, giving wildly replica-dependent occupancy).  Bumped to 5e7 to give
    # nucleation more chances per iteration.  Production runs typically need 5e7-1e8.
    ar.steps_per_iter = int(5e7)

    # -----------------------------------------------------------------------
    # Pseudo-count: controls how aggressively unobserved edges are corrected.
    # Larger ε → milder correction → more iterations but smoother convergence.
    # -----------------------------------------------------------------------
    ar.graph_pseudo_count = 1.0

    # -----------------------------------------------------------------------
    # Legal states: all bond counts 0–8 are physically reachable.
    # Desired states: all legal states (we want uniform sampling everywhere).
    # -----------------------------------------------------------------------
    # filter_legal_states receives the full list of possible OP states and
    # returns the subset the system is allowed to visit.
    ar.filter_legal_states = lambda states: [s for s in states]

    # filter_desired_states receives the legal state list and returns the
    # subset whose sampling we are trying to flatten.  States outside this
    # set are still legal but will receive suppressed weights.
    ar.filter_desired_states = lambda states: [s for s in states]

    # -----------------------------------------------------------------------
    # build_replica: called once per simulation before it runs.
    # Set all oxDNA input parameters here.
    # -----------------------------------------------------------------------
    def build_replica(reweighter, sim):
        sim.set_builder(BuildSimulationFromStructure(sim, STARTING_STRUCTURE))
        sim.build(clean_build="force")
        sim.input.swap_default_input("vmmc")

        sim.input["T"] = "40C"
        sim.input["salt_concentration"] = 1.0   # 1 M NaCl
        sim.input["interaction_type"] = "DNA2"

        # Record state every 1 000 steps so transition counts are dense enough
        # for the graph reweighter to estimate edge flow reliably.
        sim.input["print_energy_every"] = int(1e3)
        sim.input["print_conf_interval"] = int(1e3)

    ar.build_replica = build_replica

    # -----------------------------------------------------------------------
    # build_start_weights: called for each replica in iteration 0 only.
    # generate_weights(T) returns exponentially increasing weights for lower
    # bond counts, giving a reasonable starting point for flat sampling.
    # The reweighter's parent class provides a sensible default; override
    # here only if you need custom initialisation.
    # -----------------------------------------------------------------------
    def build_start_weights(sim):
        sim.weights[...] = sim.generate_weights(7.0)

    ar.build_start_weights = build_start_weights

    # -----------------------------------------------------------------------
    # end_iter_callback: called once per iteration, right after all its replicas
    # finish running (see VMMCAutoReweight.run_iteration). Show the weights that
    # produced this iteration alongside the resulting state-space (bond-count-vs-
    # time) curve, so you can watch it converge (or not) iteration by iteration
    # instead of only inspecting the final result.
    # -----------------------------------------------------------------------
    def show_iteration_progress():
        iteration_idx = len(ar) - 1
        fig, (ax_weights, ax_curve) = plt.subplots(1, 2, figsize=(14, 4))

        ar.plot_iteration_weights(iteration=iteration_idx, ax=ax_weights)
        ax_weights.set_title(f"Iteration {iteration_idx} weights")

        ar.plot_bond_curves(subgroup_idx=iteration_idx, bond_op_index=0, ax=ax_curve)
        ax_curve.set_title(f"Iteration {iteration_idx} state-space curve")

        plt.tight_layout()
        fig.savefig(ar.tld / f"iteration_{iteration_idx}_progress.png", dpi=150)
        plt.show()

    ar.end_iter_callback = show_iteration_progress

    return ar


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

def run(output_dir: Path, n_replicas: int = 3) -> VMMCGraphReweight:
    ar = make_reweighter(output_dir, n_replicas=n_replicas)

    print("Running graph-reweight iterations…")
    # run() loops until convergence (max_rel_std) or max_iterations is reached.
    # Each iteration: builds replicas → computes weights from transition graph
    # → runs oxDNA → checks convergence criterion.
    ar.run()

    print(f"Completed {len(ar)} iteration(s).")
    return ar


# ---------------------------------------------------------------------------
# Analyse and visualise
# ---------------------------------------------------------------------------

def analyse(ar: VMMCGraphReweight):
    """Load analysis data and produce standard plots."""

    # Read per-replica sampling statistics for the last iteration
    last_it = ar[-1]
    for sim in last_it:
        sim.analysis.read_vmmc_op_data()
        sim.analysis.calculate_sampling_and_probabilities()

    # Print per-replica sampling percentages
    print("\n--- Sampling statistics (last iteration) ---")
    for i, sim in enumerate(last_it):
        print(f"  Replica {i}:")
        print(sim.analysis.statistics[["sampling_percent", "wt_prob"]].to_string())

    # Sampling std across the desired states — the convergence metric
    std = ar.get_sampling_std_filtered(last_it)
    print(f"\nSampling std across desired states: {std:.2f}%  (target < {ar.max_rel_std}%)")

    # Weight evolution across iterations
    print(f"\nVisualising all {len(ar)} iteration(s)…")
    ar.visualize(bond_op_index=0)

    # Free energy profile from the last iteration
    fig_fe = ar.plot_free_energy_profile(op=ar.bond_ops()[0], iteration=-1)
    if fig_fe is not None:
        fig_fe.suptitle("Free-energy profile — 8-nt duplex at 40 °C (last iteration)")
        fig_fe.tight_layout()
        fig_fe.savefig(ar.tld / "free_energy_profile.png", dpi=150)
        plt.close(fig_fe)
        print(f"Free-energy plot saved to {ar.tld / 'free_energy_profile.png'}")


# ---------------------------------------------------------------------------
# Resuming from a saved run
# ---------------------------------------------------------------------------

def resume(output_dir: Path) -> VMMCGraphReweight:
    """
    Reload a previously completed run without re-running simulations.

    VMMCGraphReweight.load() walks the output directory, reconstructs each
    iteration's VmmcReplicas, and reloads the order parameters and weights
    from disk.  Call this in a fresh Python session to access analysis data
    from a run that has already finished.

    Example
    -------
    ar = resume(Path("vmmc_graph_reweight_output"))
    analyse(ar)
    """
    ar = VMMCGraphReweight(output_dir)
    ar.add_order_parameter(NATIVE_OP)
    ar.extrapolate_hist_Ts = ["30C", "37C", "40C", "46C", "55C"]
    ar.load()
    return ar


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="VMMC graph-reweight tutorial for the 8-nt DNA duplex"
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=HERE / "vmmc_graph_reweight_output",
        help="Directory where simulation files are written (default: ./vmmc_graph_reweight_output)"
    )
    parser.add_argument(
        "--n-replicas", type=int, default=3,
        help="Independent replicas per iteration (default: 3)"
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Load an existing run from --output-dir instead of running new simulations"
    )
    args = parser.parse_args()

    if args.resume:
        ar = resume(args.output_dir)
    else:
        ar = run(args.output_dir, n_replicas=args.n_replicas)

    analyse(ar)
    print("\nDone.")