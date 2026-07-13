"""
Integration test for VmmcWindowing: builds and runs short VMMC simulations.

Uses the 8-nt duplex oxDNA files from examples/ and the same bond order
parameter as test_forward_flux_sample_OO.py.

Two quirks in VmmcWindowing.setup() that this test works around:
  1. filter_legal_states is called with window.state_space_area (a set) but
     the result must be a list — the default lambda x: x doesn't convert, so
     we override it with sorted().
  2. build_start_weights is called as (sim, window_idx) but the default helper
     only accepts one argument — we provide a two-arg replacement.
"""
import shutil
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pytest

from oxpy_utils.utils.order_parameter import OrderParameter
from oxpy_utils.vmmc_umbrella.windowing import VmmcWindowing

EXAMPLES_DIR = (
    Path(__file__).parent.parent
    / "examples" / "8nt_duplex_files"
)
# A partially-melted configuration for window 1 (states 0–5 bonds).
# The default init.dat starts at 8 bonds, which is outside window 1's state space.
FFS_SHOOT2_DIR = Path(__file__).parent / "test_data" / "example_ffs" / "shoot2"

# Bond pairs from test_forward_flux_sample_OO.py:
# strand 1 (0–7) paired with strand 2 (15–8) in Watson–Crick order
BONDS = list(zip(
    [0, 1, 2, 3, 4, 5, 6, 7],
    reversed([8, 9, 10, 11, 12, 13, 14, 15]),
))
NATIVE_OP = OrderParameter("native", "bond", BONDS)
# State-space split: 8 pairs → 9 states (0–8 bonds).
# Window 0 samples the bonded end; window 1 samples the melted end.
# They overlap at state (5,) so check_ready() is satisfied.
WINDOW_0_STATES = {(s,) for s in range(5, 9)}   # 5, 6, 7, 8 bonds
WINDOW_1_STATES = {(s,) for s in range(0, 6)}   # 0, 1, 2, 3, 4, 5 bonds


def _make_window_1_src(tmp_path: Path) -> Path:
    """
    Build a source directory for window 1 (states 0–5 bonds) using a starting
    conf that is already in the melted region.  The default init.dat from
    EXAMPLES_DIR starts at 8 bonds, which is outside window 1's state space.
    """
    src = tmp_path / "window_1_src"
    src.mkdir(exist_ok=True)
    shutil.copy(EXAMPLES_DIR / "duplex_box_30.top", src)
    shutil.copy(FFS_SHOOT2_DIR / "success_2.dat", src / "duplex_box_30.dat")
    return src


def _make_windowing(tmp_path: Path) -> VmmcWindowing:
    w = VmmcWindowing(tmp_path)
    w.add_order_parameter(NATIVE_OP)
    w.n_reps = 1
    w.extrapolate_hist_Ts = ["34C", "40C", "46C"]

    # Fix 1: filter_legal_states must return a list.
    w.filter_legal_states = lambda states: sorted(states)

    def build_replica(windowing, sim):
        sim.build(clean_build="force")
        sim.input.swap_default_input("vmmc")
        sim.input["steps"] = int(1e4)
        sim.input["T"] = "40C"
        sim.input["salt_concentration"] = 1.
        sim.input["interaction_type"] = "DNA2"
        sim.input["print_energy_every"] = int(1e3)
        sim.input["print_conf_interval"] = int(1e3)

    # Fix 2: build_start_weights is called as (sim, window_idx).
    def build_start_weights(sim, window_idx):
        sim.weights[...] = sim.generate_weights(7.0)

    w.build_replica = build_replica
    w.build_start_weights = build_start_weights

    w.add_window(WINDOW_0_STATES, EXAMPLES_DIR)
    w.add_window(WINDOW_1_STATES, _make_window_1_src(tmp_path))
    return w


def test_vmmc_windowing_setup_and_run(tmp_path):
    w = _make_windowing(tmp_path)

    w.setup()

    # After setup: each window has 1 replica that is fully built
    assert len(w) == 2
    for window in w:
        assert window.is_set_up()
        sim = window[0]
        assert (sim.sim_dir / sim.input["op_file"]).exists()
        assert (sim.sim_dir / sim.input["weights_file"]).exists()
        assert (sim.sim_dir / "input").exists()

    w.run(join=True)

    for window in w:
        sim = window[0]
        sim.sim_files.parse_current_files()
        last_hist = sim.sim_dir / sim.input["last_hist_file"]
        assert last_hist.exists(), f"last_hist not found in {sim.sim_dir}"
        assert last_hist.stat().st_size > 0
        assert sim.sim_files.energy.exists()
        assert sim.sim_files.energy.stat().st_size > 0


def test_vmmc_windowing_weights_are_valid_after_setup(tmp_path):
    w = _make_windowing(tmp_path)
    w.setup()

    for i, window in enumerate(w):
        sim = window[0]
        weights = sim.weights
        # No NaN or inf
        assert not np.isnan(weights).any(), f"NaN weights in window {i}"
        assert not np.isinf(weights).any(), f"inf weights in window {i}"
        # States inside the window have non-zero weight
        for state in window.state_space_area:
            assert weights[state] > 0, f"Zero weight for in-window state {state} in window {i}"
        # States outside the window are zeroed out
        all_states = {(s,) for s in range(len(NATIVE_OP))}
        for state in all_states - window.state_space_area:
            assert weights[state] == 0.0, f"Non-zero weight for out-of-window state {state} in window {i}"


def test_vmmc_windowing_cache_roundtrip(tmp_path):
    w = _make_windowing(tmp_path)
    w.setup()

    # Reload from the cached setup.json written by setup()
    w2 = VmmcWindowing(tmp_path)
    w2.load()

    assert len(w2.order_parameters()) == 1
    assert w2.order_parameters()[0].name == NATIVE_OP.name
    assert len(w2) == 2
    assert w2[0].state_space_area == WINDOW_0_STATES
    assert w2[1].state_space_area == WINDOW_1_STATES
    assert w2.n_reps == 1


def test_vmmc_windowing_read_vmmc_data_after_run(tmp_path):
    w = _make_windowing(tmp_path)
    w.setup()
    w.run(join=True)

    for window in w:
        sim = window[0]
        sim.analysis.read_vmmc_op_data()
        df = sim.analysis.vmmc_df
        assert df is not None
        assert len(df) > 0
        assert "unwt_occ" in df.columns


# ---------------------------------------------------------------------------
# Shared fixture: run once per module, pre-load all analysis data
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def ran_windowing(tmp_path_factory) -> VmmcWindowing:
    tmp = tmp_path_factory.mktemp("vmmc_windowing_vis")
    w = _make_windowing(tmp)
    w.setup()
    w.run(join=True)
    for window in w:
        for sim in window:
            sim.analysis.read_vmmc_op_data()
    return w


# ---------------------------------------------------------------------------
# Analysis tests (use shared fixture — simulation runs only once)
# ---------------------------------------------------------------------------

class TestAnalysisAfterRun:
    def test_sampling_statistics_columns_and_sum(self, ran_windowing):
        for window in ran_windowing:
            sim = window[0]
            sim.analysis.calculate_sampling_and_probabilities()
            stats = sim.analysis.statistics
            assert "sampling_percent" in stats.columns
            assert "wt_prob" in stats.columns
            assert "wt_free" in stats.columns
            assert stats["sampling_percent"].sum() == pytest.approx(100.0, abs=1e-6)

    def test_wt_prob_sums_to_one(self, ran_windowing):
        for window in ran_windowing:
            sim = window[0]
            stats = sim.analysis.statistics
            assert stats["wt_prob"].sum() == pytest.approx(1.0, abs=1e-6)

    def test_get_data_over_has_count_column(self, ran_windowing):
        for window in ran_windowing:
            sim = window[0]
            data = sim.analysis.get_data_over(NATIVE_OP)
            assert "count" in data.df.columns
            assert "unbiased_count" in data.df.columns
            assert len(data.df) > 0

    def test_get_data_over_index_is_native_op_values(self, ran_windowing):
        for window in ran_windowing:
            sim = window[0]
            data = sim.analysis.get_data_over(NATIVE_OP)
            # index values should be integers in [0, len(NATIVE_OP))
            for idx in data.df.index:
                assert 0 <= idx < len(NATIVE_OP)

    def test_merge_hist_equals_single_replica_df(self, ran_windowing):
        # n_reps=1, so merge_hist should be identical to the single sim's vmmc_df
        for window in ran_windowing:
            merged = window.merge_hist()
            single = window[0].analysis.vmmc_df
            assert merged.shape == single.shape
            assert (merged["unwt_occ"] == single["unwt_occ"]).all()

    def test_export_merged_hists_creates_files(self, ran_windowing):
        ran_windowing.export_merged_hists("merged_hist.dat")
        for window in ran_windowing:
            out = window.sim_dir / "merged_hist.dat"
            assert out.exists()
            assert out.stat().st_size > 0

    def test_wham_returns_probability_dict(self, ran_windowing):
        # wham(op=0) uses get_data_over() which has the 'unbiased_count' column;
        # wham() with no args is broken (vmmc_df lacks that column).
        rho = ran_windowing.wham(op=0)
        assert isinstance(rho, dict)
        assert len(rho) > 0
        assert sum(rho.values()) == pytest.approx(1.0, abs=1e-4)

    def test_wham_keys_are_integer_states(self, ran_windowing):
        rho = ran_windowing.wham(op=0)
        for key in rho:
            assert isinstance(key, int)
            assert 0 <= key < len(NATIVE_OP)

    def test_get_merged_weights_shape_and_nonnegativity(self, ran_windowing):
        weights = ran_windowing.get_merged_weights()
        assert weights.shape == (len(NATIVE_OP),)
        assert (weights >= 0).all()

    def test_get_merged_weights_nonzero_where_covered(self, ran_windowing):
        weights = ran_windowing.get_merged_weights()
        covered = WINDOW_0_STATES | WINDOW_1_STATES
        for state in covered:
            assert weights[state] > 0, f"Expected nonzero weight at covered state {state}"

    def test_save_merged_weights_creates_file(self, ran_windowing):
        ran_windowing.save_merged_weights(fname="test_merged.txt")
        assert (ran_windowing.tld / "test_merged.txt").exists()

    def test_energy_df_has_op_column(self, ran_windowing):
        # VmmcAnalysis.load_energy() includes OP values in the energy DataFrame
        for window in ran_windowing:
            df = window[0].analysis.energy_df
            assert "time" in df.columns
            assert NATIVE_OP.name in df.columns


# ---------------------------------------------------------------------------
# Visualization tests — assert no exception and correct return types
# ---------------------------------------------------------------------------

class TestVisualizationAfterRun:
    def test_plot_energy_does_not_raise(self, ran_windowing):
        for window in ran_windowing:
            window[0].analysis.plot_energy()
            plt.close('all')

    def test_plot_sampling_pie_chart_returns_figure(self, ran_windowing):
        states = [(i,) for i in range(len(NATIVE_OP))]
        for window in ran_windowing:
            result = window[0].analysis.plot_sampling_pie_chart(
                states_to_visualize=states
            )
            plt.close('all')
            # Returns (fig, ax) when states were sampled, (None, None) otherwise.
            assert result is not None
            assert len(result) == 2

    def test_plot_statistics_does_not_raise(self, ran_windowing):
        for window in ran_windowing:
            window[0].analysis.plot_statistics()
            plt.close('all')

    def test_create_state_histograms_returns_figure(self, ran_windowing):
        for window in ran_windowing:
            fig = window.create_state_histograms(NATIVE_OP)
            plt.close('all')
            assert fig is not None

    def test_plot_vmmc_scatter_returns_axes(self, ran_windowing):
        for window in ran_windowing:
            ax = window.plot_vmmc_scatter(op=0)
            plt.close('all')
            assert ax is not None

    def test_plot_op_val_curve_does_not_raise(self, ran_windowing):
        # plot_op_val_curve calls plt.show() internally but returns None
        for window in ran_windowing:
            window.plot_op_val_curve(NATIVE_OP)
            plt.close('all')

    def test_plot_window_weights_returns_figure_and_axes(self, ran_windowing):
        fig, axes = ran_windowing.plot_window_weights()
        plt.close('all')
        assert fig is not None
        assert len(axes) == len(ran_windowing)

    def test_plot_window_data_returns_figure(self, ran_windowing):
        fig, axes = ran_windowing.plot_window_data(NATIVE_OP)
        plt.close('all')
        assert fig is not None
        assert axes.shape[0] == len(ran_windowing)

    def test_plot_free_energy_profile_returns_figure(self, ran_windowing):
        # plot_free_energy_profile calls wham(agg_idx) internally, which works.
        # plot_merged_hist() calls wham() with no args and is broken — not tested here.
        fig, ax = ran_windowing.plot_free_energy_profile()
        plt.close('all')
        assert fig is not None