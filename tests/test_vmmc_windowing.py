"""
Tests for VmmcWindow and VmmcWindowing in vmmc_umbrella/windowing.py,
and the supporting order-parameter utilities in utils/order_parameter.py.

Construction note: Replicas.__init__ stores references only — simulations are not
created until init() is called.  VmmcWindow(starting_conf=None) is therefore cheap
and safe to use in unit tests.
"""
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from oxpy_utils.utils.order_parameter import (
    OrderParameter, possible_states, create_state_mask,
)
from oxpy_utils.vmmc_umbrella.windowing import VmmcWindow, VmmcWindowing


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def bond_op() -> OrderParameter:
    """2-pair bond OP: states 0, 1, 2 bonds."""
    return OrderParameter("bonds", "bond", [(0, 1), (2, 3)])


@pytest.fixture
def dist_op() -> OrderParameter:
    return OrderParameter("dist", "mindistance", [(0, 5)])


def _make_window(tmp_path: Path, name: str, states: set) -> VmmcWindow:
    return VmmcWindow(
        sim_dir=tmp_path / name,
        n_replicas=2,
        state_space_area=states,
        starting_conf=None,
    )


def _window_with_weights(tmp_path: Path, name: str, states: set,
                         weights: np.ndarray) -> VmmcWindow:
    """Window whose first replica has a mock weights array."""
    window = _make_window(tmp_path, name, states)
    mock_sim = MagicMock()
    mock_sim.weights = weights.copy()
    mock_sim.load_weights.return_value = None
    window.simulations = [mock_sim]
    return window


# ---------------------------------------------------------------------------
# possible_states
# ---------------------------------------------------------------------------

class TestPossibleStates:
    def test_single_op_all_states(self, bond_op):
        states = possible_states(bond_op)
        # 2 pairs, 4 unique nucleotides — 0, 1, 2 bonds all valid (2*2 <= 4)
        assert sorted(states) == [(0,), (1,), (2,)]

    def test_single_op_filters_impossible_states(self):
        # 1 pair, 2 unique nucleotides — max 1 bond (2*2 > 2 means 2 bonds impossible)
        op = OrderParameter("b", "bond", [(0, 1)])
        states = possible_states(op)
        assert (2,) not in states
        assert (0,) in states and (1,) in states

    def test_two_ops_cartesian(self, bond_op, dist_op):
        # dist op is mindistance, all states always included
        states = possible_states(bond_op, dist_op)
        bond_vals = {s[0] for s in states}
        dist_vals = {s[1] for s in states}
        assert bond_vals == {0, 1, 2}
        # dist_op has 1 pair → len(dist_op) = 2 (states 0 and 1)
        assert dist_vals == {0, 1}

    def test_no_ops_raises(self):
        with pytest.raises(ValueError):
            possible_states()


# ---------------------------------------------------------------------------
# create_state_mask
# ---------------------------------------------------------------------------

class TestCreateStateMask:
    def test_shape_matches_op_lengths(self, bond_op):
        mask = create_state_mask(bond_op)
        assert mask.shape == (len(bond_op),)

    def test_accessible_states_are_true(self, bond_op):
        states = possible_states(bond_op)
        mask = create_state_mask(bond_op)
        for s in states:
            assert mask[s], f"State {s} should be accessible"

    def test_inaccessible_states_are_false(self):
        # Pairs (0,1) and (0,2) share nucleotide 0 → 3 unique nucleotides, len=3.
        # State 2: 2*2=4 > 3 unique nucleotides → impossible, so mask[(2,)] must be False.
        op = OrderParameter("b", "bond", [(0, 1), (0, 2)])
        mask = create_state_mask(op)
        assert mask.shape == (3,)
        assert mask[(0,)]
        assert mask[(1,)]
        assert not mask[(2,)]

    def test_custom_accessible_states(self, bond_op):
        mask = create_state_mask(bond_op, accessible_states=[(0,), (2,)])
        assert mask[(0,)]
        assert not mask[(1,)]
        assert mask[(2,)]


# ---------------------------------------------------------------------------
# VmmcWindow
# ---------------------------------------------------------------------------

class TestVmmcWindow:
    def test_construction(self, tmp_path):
        window = _make_window(tmp_path, "w0", {(0,), (1,), (2,)})
        assert window.state_space_area == {(0,), (1,), (2,)}
        assert window.nreplicas == 2

    def test_state_space_of_first_op(self, tmp_path):
        states = {(0, 0), (0, 1), (1, 0), (2, 1)}
        window = _make_window(tmp_path, "w0", states)
        assert window.state_space_of(0) == {0, 1, 2}

    def test_state_space_of_second_op(self, tmp_path):
        states = {(0, 0), (0, 1), (1, 0), (2, 1)}
        window = _make_window(tmp_path, "w0", states)
        assert window.state_space_of(1) == {0, 1}

    def test_merge_hist_sums_dataframes(self, tmp_path):
        df1 = pd.DataFrame({"unwt_occ": [10.0, 20.0]}, index=pd.Index([0, 1]))
        df2 = pd.DataFrame({"unwt_occ": [5.0, 15.0]}, index=pd.Index([0, 1]))

        window = _make_window(tmp_path, "w0", {(0,), (1,)})
        for df in (df1, df2):
            mock = MagicMock()
            mock.analysis.vmmc_df = df
            window.simulations.append(mock)

        result = window.merge_hist()
        assert result["unwt_occ"].tolist() == [15.0, 35.0]

    def test_merge_hist_fill_value_for_missing_states(self, tmp_path):
        df1 = pd.DataFrame({"unwt_occ": [10.0]}, index=pd.Index([0]))
        df2 = pd.DataFrame({"unwt_occ": [5.0]}, index=pd.Index([1]))

        window = _make_window(tmp_path, "w0", {(0,), (1,)})
        for df in (df1, df2):
            mock = MagicMock()
            mock.analysis.vmmc_df = df
            window.simulations.append(mock)

        result = window.merge_hist()
        assert result.loc[0, "unwt_occ"] == 10.0
        assert result.loc[1, "unwt_occ"] == 5.0


# ---------------------------------------------------------------------------
# VmmcWindowing — construction and order parameter management
# ---------------------------------------------------------------------------

class TestVmmcWindowingConstruction:
    def test_construct_with_path(self, tmp_path):
        w = VmmcWindowing(tmp_path)
        assert w.tld == tmp_path
        assert w.order_parameters() == []

    def test_add_bond_op(self, tmp_path, bond_op):
        w = VmmcWindowing(tmp_path)
        w.add_order_parameter(bond_op)
        assert bond_op in w.order_parameters()
        assert w._dist_op is None

    def test_add_dist_op(self, tmp_path, dist_op):
        w = VmmcWindowing(tmp_path)
        w.add_order_parameter(dist_op)
        assert w._dist_op is dist_op

    def test_dist_op_appears_last_in_order_parameters(self, tmp_path, bond_op, dist_op):
        w = VmmcWindowing(tmp_path)
        w.add_order_parameter(bond_op)
        w.add_order_parameter(dist_op)
        ops = w.order_parameters()
        assert ops[-1] is dist_op
        assert ops[0] is bond_op

    def test_add_duplicate_name_raises(self, tmp_path, bond_op):
        w = VmmcWindowing(tmp_path)
        w.add_order_parameter(bond_op)
        duplicate = OrderParameter("bonds", "bond", [(4, 5)])
        with pytest.raises(AssertionError):
            w.add_order_parameter(duplicate)

    def test_add_second_dist_op_raises(self, tmp_path, dist_op):
        w = VmmcWindowing(tmp_path)
        w.add_order_parameter(dist_op)
        with pytest.raises(ValueError):
            w.add_order_parameter(OrderParameter("dist2", "mindistance", [(1, 6)]))

    def test_n_reps_setter(self, tmp_path):
        w = VmmcWindowing(tmp_path)
        w.n_reps = 3
        assert w.n_reps == 3

    def test_n_reps_zero_raises(self, tmp_path):
        w = VmmcWindowing(tmp_path)
        with pytest.raises(ValueError):
            w.n_reps = 0

    def test_n_reps_after_subgroups_raises(self, tmp_path):
        w = VmmcWindowing(tmp_path)
        w._subgroups.append(MagicMock())
        with pytest.raises(ValueError):
            w.n_reps = 4


# ---------------------------------------------------------------------------
# VmmcWindowing — threshold setter
# ---------------------------------------------------------------------------

class TestNoAcceptThreshold:
    def test_set_threshold_creates_callback(self, tmp_path):
        w = VmmcWindowing(tmp_path)
        w.no_accept_moves_threshold = 1000
        assert w.no_accept_moves_threshold == 1000
        assert w.no_accept_moves_callback is not None

    def test_callback_raises_runtime_error(self, tmp_path):
        w = VmmcWindowing(tmp_path)
        w.no_accept_moves_threshold = 1000
        mock_window = MagicMock()
        mock_window.sim_dir.name = "window_0"
        with pytest.raises(RuntimeError):
            w.no_accept_moves_callback(mock_window)

    def test_non_positive_raises(self, tmp_path):
        w = VmmcWindowing(tmp_path)
        with pytest.raises(ValueError):
            w.no_accept_moves_threshold = 0
        with pytest.raises(ValueError):
            w.no_accept_moves_threshold = -10

    def test_stored_as_int(self, tmp_path):
        w = VmmcWindowing(tmp_path)
        w.no_accept_moves_threshold = 500.9
        assert isinstance(w.no_accept_moves_threshold, int)
        assert w.no_accept_moves_threshold == 500


# ---------------------------------------------------------------------------
# VmmcWindowing — overlap and check_ready
# ---------------------------------------------------------------------------

class TestWindowingLogic:
    def test_overlap_returns_intersection(self, tmp_path):
        w = VmmcWindowing(tmp_path)
        w._subgroups = [
            _make_window(tmp_path, "w0", {(0,), (1,), (2,)}),
            _make_window(tmp_path, "w1", {(2,), (3,), (4,)}),
        ]
        assert w.overlap(0, 1) == {(2,)}

    def test_overlap_empty_when_no_shared_states(self, tmp_path):
        w = VmmcWindowing(tmp_path)
        w._subgroups = [
            _make_window(tmp_path, "w0", {(0,), (1,)}),
            _make_window(tmp_path, "w1", {(2,), (3,)}),
        ]
        assert w.overlap(0, 1) == set()

    def test_check_ready_passes_when_fully_covered(self, tmp_path, bond_op):
        w = VmmcWindowing(tmp_path)
        w.add_order_parameter(bond_op)
        # legal states: (0,), (1,), (2,)
        w._subgroups = [
            _make_window(tmp_path, "w0", {(0,), (1,)}),
            _make_window(tmp_path, "w1", {(1,), (2,)}),
        ]
        w.check_ready()  # should not raise

    def test_check_ready_raises_when_state_uncovered(self, tmp_path, bond_op):
        w = VmmcWindowing(tmp_path)
        w.add_order_parameter(bond_op)
        # Only covers (0,) and (1,) — (2,) is uncovered
        w._subgroups = [
            _make_window(tmp_path, "w0", {(0,), (1,)}),
        ]
        with pytest.raises(ValueError, match="Uncovered"):
            w.check_ready()


# ---------------------------------------------------------------------------
# VmmcWindowing — JSON settings roundtrip
# ---------------------------------------------------------------------------

class TestCacheAndLoadSettings:
    def test_roundtrip_restores_order_parameters(self, tmp_path, bond_op):
        tmp_path.mkdir(exist_ok=True)
        w = VmmcWindowing(tmp_path)
        w.add_order_parameter(bond_op)
        w.extrapolate_hist_Ts = ["30C", "40C"]
        w._subgroups = [
            _make_window(tmp_path, "window_0", {(0,), (1,)}),
            _make_window(tmp_path, "window_1", {(1,), (2,)}),
        ]
        w.cache_settings()

        assert (tmp_path / "setup.json").exists()

        w2 = VmmcWindowing(tmp_path)
        w2.load()

        ops = w2.order_parameters()
        assert len(ops) == 1
        assert ops[0].name == bond_op.name
        assert ops[0].order_parameter == bond_op.order_parameter
        assert ops[0].pairs == bond_op.pairs

    def test_roundtrip_restores_window_count(self, tmp_path, bond_op):
        tmp_path.mkdir(exist_ok=True)
        w = VmmcWindowing(tmp_path)
        w.add_order_parameter(bond_op)
        w.extrapolate_hist_Ts = ["30C"]
        w._subgroups = [
            _make_window(tmp_path, "window_0", {(0,), (1,)}),
            _make_window(tmp_path, "window_1", {(1,), (2,)}),
        ]
        w.cache_settings()

        w2 = VmmcWindowing(tmp_path)
        w2.load()
        assert len(w2._subgroups) == 2

    def test_roundtrip_restores_state_space_areas(self, tmp_path, bond_op):
        tmp_path.mkdir(exist_ok=True)
        w = VmmcWindowing(tmp_path)
        w.add_order_parameter(bond_op)
        w.extrapolate_hist_Ts = ["30C"]
        areas = [{(0,), (1,)}, {(1,), (2,)}]
        for i, area in enumerate(areas):
            w._subgroups.append(_make_window(tmp_path, f"window_{i}", area))
        w.cache_settings()

        w2 = VmmcWindowing(tmp_path)
        w2.load()
        for i, area in enumerate(areas):
            assert w2._subgroups[i].state_space_area == area

    def test_roundtrip_restores_n_reps(self, tmp_path, bond_op):
        tmp_path.mkdir(exist_ok=True)
        w = VmmcWindowing(tmp_path)
        w.add_order_parameter(bond_op)
        w.n_reps = 3
        w.extrapolate_hist_Ts = ["30C"]
        w._subgroups = [_make_window(tmp_path, "window_0", {(0,)})]
        w.cache_settings()

        w2 = VmmcWindowing(tmp_path)
        w2.load()
        assert w2.n_reps == 3


# ---------------------------------------------------------------------------
# VmmcWindowing._splice_window
# ---------------------------------------------------------------------------

class TestSpliceWindow:
    def test_replaces_subgroup_preserving_state_and_conf(self, tmp_path, bond_op):
        w = VmmcWindowing(tmp_path)
        w.add_order_parameter(bond_op)
        w.extrapolate_hist_Ts = ["30C"]
        w._subgroups = [_make_window(tmp_path, "window_0", {(0,), (1,)})]
        w[0].file_dir = "some_starting_conf"

        fake_sim = MagicMock()
        w._splice_window(0, tmp_path / "new_sim_dir", [fake_sim, fake_sim])

        new_window = w[0]
        assert new_window.sim_dir == tmp_path / "new_sim_dir"
        assert new_window.state_space_area == {(0,), (1,)}
        assert new_window.file_dir == "some_starting_conf"
        assert new_window.simulations == [fake_sim, fake_sim]


# ---------------------------------------------------------------------------
# VmmcWindowing.load(splice_reweighted=...)
# ---------------------------------------------------------------------------

class TestLoadSpliceReweighted:
    def _cached_windowing(self, tmp_path, bond_op, areas) -> VmmcWindowing:
        tmp_path.mkdir(exist_ok=True)
        w = VmmcWindowing(tmp_path)
        w.add_order_parameter(bond_op)
        w.extrapolate_hist_Ts = ["30C"]
        w._subgroups = [_make_window(tmp_path, f"window_{i}", area) for i, area in enumerate(areas)]
        w.cache_settings()
        return w

    def test_flag_off_by_default_leaves_raw_windows(self, tmp_path, bond_op):
        self._cached_windowing(tmp_path, bond_op, [{(0,), (1,), (2,)}])
        w2 = VmmcWindowing(tmp_path)
        w2.load()
        assert w2[0].sim_dir == tmp_path / "window_0"

    @patch("oxpy_utils.vmmc_umbrella.windowing.VMMCAutoReweight")
    def test_splices_when_reweighted_dir_present(self, mock_ar_cls, tmp_path, bond_op):
        self._cached_windowing(tmp_path, bond_op, [{(0,), (1,), (2,)}])
        (tmp_path / "reweighted" / "window_0").mkdir(parents=True)

        fake_sim = MagicMock()
        fake_final_it = MagicMock()
        fake_final_it.sim_dir = tmp_path / "reweighted" / "window_0" / "iteration_2"
        fake_final_it.__iter__ = MagicMock(return_value=iter([fake_sim]))
        fake_final_it.__len__ = MagicMock(return_value=1)
        fake_ar = MagicMock()
        fake_ar.__len__ = MagicMock(return_value=3)
        fake_ar.__getitem__ = MagicMock(side_effect=lambda i: fake_final_it if i == -1 else None)
        mock_ar_cls.return_value = fake_ar

        w2 = VmmcWindowing(tmp_path)
        w2.load(splice_reweighted=True)

        mock_ar_cls.assert_called_once_with(tmp_path / "reweighted" / "window_0")
        fake_ar.load.assert_called_once()
        assert w2[0].sim_dir == fake_final_it.sim_dir
        assert w2[0].simulations == [fake_sim]

    @patch("oxpy_utils.vmmc_umbrella.windowing.VMMCAutoReweight")
    def test_skips_window_without_reweighted_dir(self, mock_ar_cls, tmp_path, bond_op):
        self._cached_windowing(tmp_path, bond_op, [{(0,), (1,), (2,)}])
        w2 = VmmcWindowing(tmp_path)
        w2.load(splice_reweighted=True)

        mock_ar_cls.assert_not_called()
        assert w2[0].sim_dir == tmp_path / "window_0"

    @patch("oxpy_utils.vmmc_umbrella.windowing.VMMCAutoReweight")
    def test_skips_window_with_no_iterations(self, mock_ar_cls, tmp_path, bond_op):
        self._cached_windowing(tmp_path, bond_op, [{(0,), (1,), (2,)}])
        (tmp_path / "reweighted" / "window_0").mkdir(parents=True)
        fake_ar = MagicMock()
        fake_ar.__len__ = MagicMock(return_value=0)
        mock_ar_cls.return_value = fake_ar

        w2 = VmmcWindowing(tmp_path)
        w2.load(splice_reweighted=True)

        assert w2[0].sim_dir == tmp_path / "window_0"

    @patch("oxpy_utils.vmmc_umbrella.windowing.VMMCAutoReweight")
    def test_respects_explicit_reweight_tld(self, mock_ar_cls, tmp_path, bond_op):
        self._cached_windowing(tmp_path, bond_op, [{(0,), (1,), (2,)}])
        custom_dir = tmp_path / "custom_reweight_loc"
        (custom_dir / "window_0").mkdir(parents=True)

        fake_sim = MagicMock()
        fake_final_it = MagicMock()
        fake_final_it.sim_dir = custom_dir / "window_0" / "iteration_0"
        fake_final_it.__iter__ = MagicMock(return_value=iter([fake_sim]))
        fake_final_it.__len__ = MagicMock(return_value=1)
        fake_ar = MagicMock()
        fake_ar.__len__ = MagicMock(return_value=1)
        fake_ar.__getitem__ = MagicMock(side_effect=lambda i: fake_final_it if i == -1 else None)
        mock_ar_cls.return_value = fake_ar

        w2 = VmmcWindowing(tmp_path)
        w2.load(splice_reweighted=True, reweight_tld=custom_dir)

        mock_ar_cls.assert_called_once_with(custom_dir / "window_0")


# ---------------------------------------------------------------------------
# VmmcWindowing — get_merged_weights and save_merged_weights
# ---------------------------------------------------------------------------

class TestMergedWeights:
    def _windowing_with_weights(self, tmp_path, bond_op,
                                w1_weights, w2_weights) -> VmmcWindowing:
        tmp_path.mkdir(exist_ok=True)
        w = VmmcWindowing(tmp_path)
        w.add_order_parameter(bond_op)
        w._subgroups = [
            _window_with_weights(tmp_path, "window_0", {(0,), (1,), (2,)}, w1_weights),
            _window_with_weights(tmp_path, "window_1", {(0,), (1,), (2,)}, w2_weights),
        ]
        return w

    def test_single_window_returns_its_weights(self, tmp_path, bond_op):
        tmp_path.mkdir(exist_ok=True)
        w = VmmcWindowing(tmp_path)
        w.add_order_parameter(bond_op)
        weights = np.array([1.0, 2.0, 3.0])
        w._subgroups = [_window_with_weights(tmp_path, "window_0",
                                             {(0,), (1,), (2,)}, weights)]
        result = w.get_merged_weights()
        np.testing.assert_array_almost_equal(result, weights)

    def test_two_identical_windows_average_to_same(self, tmp_path, bond_op):
        weights = np.array([1.0, 2.0, 3.0])
        w = self._windowing_with_weights(tmp_path, bond_op, weights, weights)
        result = w.get_merged_weights()
        np.testing.assert_array_almost_equal(result, weights)

    def test_scale_factor_aligns_overlapping_windows(self, tmp_path, bond_op):
        # Window 1 covers states 0-2, Window 2 is scaled 2x relative to Window 1
        w1 = np.array([1.0, 2.0, 3.0])
        w2 = np.array([2.0, 4.0, 6.0])  # 2x scale: overlap ratio = 0.5
        w = self._windowing_with_weights(tmp_path, bond_op, w1, w2)
        result = w.get_merged_weights()
        # After scaling w2 down by 0.5, both windows agree — mean equals w1
        np.testing.assert_array_almost_equal(result, w1)

    def test_non_overlapping_windows_filled(self, tmp_path, bond_op):
        w1 = np.array([1.0, 2.0, 0.0])
        w2 = np.array([0.0, 0.0, 3.0])
        w = self._windowing_with_weights(tmp_path, bond_op, w1, w2)
        result = w.get_merged_weights()
        assert result[0] == pytest.approx(1.0)
        assert result[1] == pytest.approx(2.0)
        assert result[2] == pytest.approx(3.0)

    def test_save_merged_weights_creates_file(self, tmp_path, bond_op):
        weights = np.array([1.0, 2.0, 3.0])
        w = VmmcWindowing(tmp_path)
        w.add_order_parameter(bond_op)
        w._subgroups = [_window_with_weights(tmp_path, "window_0",
                                             {(0,), (1,), (2,)}, weights)]
        w.save_merged_weights("merged_weights.txt")
        assert (tmp_path / "merged_weights.txt").exists()

    def test_save_merged_weights_skips_zero_by_default(self, tmp_path, bond_op):
        weights = np.array([1.0, 0.0, 3.0])
        w = VmmcWindowing(tmp_path)
        w.add_order_parameter(bond_op)
        w._subgroups = [_window_with_weights(tmp_path, "window_0",
                                             {(0,), (1,), (2,)}, weights)]
        w.save_merged_weights("merged_weights.txt")
        content = (tmp_path / "merged_weights.txt").read_text()
        lines = [l for l in content.strip().splitlines()]
        # State 1 has weight 0 and should be skipped
        assert all("1 " not in l.split()[:-1] for l in lines)

    def test_save_merged_weights_skip_val_none_includes_zeros(self, tmp_path, bond_op):
        weights = np.array([1.0, 0.0, 3.0])
        w = VmmcWindowing(tmp_path)
        w.add_order_parameter(bond_op)
        w._subgroups = [_window_with_weights(tmp_path, "window_0",
                                             {(0,), (1,), (2,)}, weights)]
        w.save_merged_weights(fname="merged_weights.txt", skip_val=None)
        content = (tmp_path / "merged_weights.txt").read_text()
        assert len(content.strip().splitlines()) == 3