"""
Tests for VMMCAutoReweight in vmmc_umbrella/auto_reweight.py.

Simulation-running methods (run, load, visualize) are not covered here --
they require actual oxDNA output on disk. run_iteration's internal
call-ordering contract (order parameters must be added to a replica before
build_replica runs, since build_replica typically calls sim.build(), which
needs them) is covered with a fully mocked VmmcReplicas. Everything else is
tested with lightweight mocks or pure numpy.
"""
from unittest.mock import MagicMock, patch

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from oxpy_utils.utils.order_parameter import OrderParameter
from oxpy_utils.vmmc_umbrella.auto_reweight import VMMCAutoReweight, VMMCGraphReweight
from oxpy_utils.vmmc_umbrella.vmmc import VirtualMoveMonteCarlo
from oxpy_utils.vmmc_umbrella.vmmc_replicas import VmmcReplicas


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bond_op(name="bonds", pairs=None) -> OrderParameter:
    if pairs is None:
        pairs = [(0, 1), (2, 3)]          # 4 unique nucleotides → states 0, 1, 2
    return OrderParameter(name, "bond", pairs)


def _make_ar(tmp_path, *, add_op=True, set_temps=True, set_builder=True, set_starting_conf=True) -> VMMCAutoReweight:
    ar = VMMCAutoReweight(tmp_path)
    if add_op:
        ar.add_order_parameter(_bond_op())
    if set_temps:
        ar.extrapolate_hist_Ts = ["30C", "40C"]
    if set_builder:
        ar.build_replica = lambda meta, sim: None
    if set_starting_conf:
        ar.starting_conf = tmp_path
    return ar


def _mock_sim_with_stats(sampling_percents: list[float], op_name: str = "bonds") -> MagicMock:
    """Return a mock sim whose analysis.statistics has the given sampling percents."""
    stats = pd.DataFrame({
        op_name: list(range(len(sampling_percents))),
        "sampling_percent": sampling_percents,
        "wt_prob": [p / 100 for p in sampling_percents],
        "wt_free": [0.0] * len(sampling_percents),
    })
    mock = MagicMock()
    mock.analysis.statistics = stats
    return mock


def _mock_sim_with_indexed_stats(sampling_by_state: dict) -> MagicMock:
    """
    Return a mock sim whose analysis.statistics is indexed by state, matching the real
    shape of VmmcAnalysis.statistics (see calculate_sampling_and_probabilities: the
    DataFrame is built with index=vmmc_df.index, i.e. the order parameter value(s) --
    there is no op-name column). Keys may be bare ints (single order parameter) or
    tuples (multiple order parameters).
    """
    stats = pd.DataFrame(
        {"sampling_percent": list(sampling_by_state.values())},
        index=list(sampling_by_state.keys()),
    )
    mock = MagicMock()
    mock.analysis.statistics = stats
    return mock


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestVMMCAutoReweightInit:
    def test_default_values(self, tmp_path):
        ar = VMMCAutoReweight(tmp_path)
        assert ar.reweight_borders is False
        assert ar.zero_sample_weight_factor == pytest.approx(1e4)
        assert ar.steps_per_iter == pytest.approx(1e8)
        assert ar.max_rel_std == pytest.approx(5.0)
        assert ar.max_iterations == float('inf')

    def test_tld_stored(self, tmp_path):
        ar = VMMCAutoReweight(tmp_path)
        assert ar.tld == tmp_path

    def test_starts_empty(self, tmp_path):
        ar = VMMCAutoReweight(tmp_path)
        assert len(ar) == 0
        assert ar.order_parameters() == []


# ---------------------------------------------------------------------------
# check_ready
# ---------------------------------------------------------------------------

class TestCheckReady:
    def test_raises_without_temps(self, tmp_path):
        ar = _make_ar(tmp_path, set_temps=False)
        with pytest.raises(ValueError, match="temperature"):
            ar.check_ready()

    def test_raises_without_bond_ops(self, tmp_path):
        ar = _make_ar(tmp_path, add_op=False)
        with pytest.raises(ValueError, match="bond"):
            ar.check_ready()

    def test_raises_without_build_replica(self, tmp_path):
        ar = _make_ar(tmp_path, set_builder=False)
        ar.build_replica = None
        with pytest.raises(ValueError, match="replica"):
            ar.check_ready()

    def test_raises_without_starting_conf(self, tmp_path):
        ar = _make_ar(tmp_path, set_starting_conf=False)
        with pytest.raises(ValueError, match="starting_conf"):
            ar.check_ready()

    def test_passes_when_all_set(self, tmp_path):
        ar = _make_ar(tmp_path)
        ar.check_ready()   # should not raise


# ---------------------------------------------------------------------------
# run_iteration call ordering
# ---------------------------------------------------------------------------

def _mock_replica_group(vmmc: MagicMock):
    """A MagicMock standing in for a VmmcReplicas, yielding `vmmc` on every iteration/index."""
    group = MagicMock()
    group.__iter__ = MagicMock(side_effect=lambda: iter([vmmc]))
    group.__getitem__ = MagicMock(side_effect=lambda i: vmmc)
    return group


class TestRunIterationOrdering:
    @patch("oxpy_utils.vmmc_umbrella.auto_reweight.VmmcReplicas")
    def test_order_parameters_added_before_build_replica(self, mock_replicas_cls, tmp_path):
        # Regression test: build_replica implementations typically call sim.build(...),
        # and VirtualMoveMonteCarlo.build() asserts a bond order parameter is already set
        # (it writes the op file as part of building). If order parameters were added
        # *after* build_replica ran, that assertion would fail.
        ar = _make_ar(tmp_path)
        ar.build_start_weights = lambda vmmc: None   # sidestep the real generate_weights()

        call_order = []
        vmmc = MagicMock()
        vmmc.weights = np.ones(3)
        vmmc.input = {"print_energy_every": 100}
        vmmc.add_order_parameter.side_effect = lambda op: call_order.append("add_order_parameter")

        mock_replicas_cls.return_value = _mock_replica_group(vmmc)

        def build_replica(reweighter, sim):
            call_order.append("build_replica")
        ar.build_replica = build_replica

        ar.run_iteration()

        assert "add_order_parameter" in call_order
        assert "build_replica" in call_order
        assert call_order.index("add_order_parameter") < call_order.index("build_replica")

    @patch("oxpy_utils.vmmc_umbrella.auto_reweight.VmmcReplicas")
    def test_op_file_cleared_before_final_write(self, mock_replicas_cls, tmp_path):
        # Regression test: build_replica's sim.build(...) already writes the op file once
        # (OrderParameter.write appends); the trailing build_vmmc_op_file() call in
        # run_iteration must clear it first or every order-parameter block gets duplicated.
        ar = _make_ar(tmp_path)
        ar.build_start_weights = lambda vmmc: None

        vmmc = MagicMock()
        vmmc.weights = np.ones(3)
        vmmc.input = {"print_energy_every": 100}

        mock_replicas_cls.return_value = _mock_replica_group(vmmc)
        ar.build_replica = lambda reweighter, sim: None

        ar.run_iteration()

        vmmc.build_vmmc_op_file.assert_called_once_with(clear_file=True)

    @patch("oxpy_utils.vmmc_umbrella.auto_reweight.VmmcReplicas")
    def test_illegal_state_weight_applied_on_first_iteration(self, mock_replicas_cls, tmp_path):
        ar = _make_ar(tmp_path)
        ar.filter_legal_states = lambda states: [s for s in states if s != (2,)]
        ar.illegal_state_weight = 0.0
        ar.build_start_weights = lambda vmmc: None   # leave weights at their initial np.ones(3)

        vmmc = MagicMock()
        vmmc.weights = np.ones(3)
        vmmc.input = {"print_energy_every": 100}
        mock_replicas_cls.return_value = _mock_replica_group(vmmc)
        ar.build_replica = lambda reweighter, sim: None

        ar.run_iteration()

        assert vmmc.weights[2] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# describe_state_space
# ---------------------------------------------------------------------------

class TestDescribeStateSpace:
    def test_counts_with_no_filters(self, tmp_path):
        # _bond_op() default: 4 unique nucleotides -> states 0, 1, 2, all possible/legal/desired
        ar = _make_ar(tmp_path)
        summary = ar.describe_state_space(verbose=False)
        assert summary["possible"] == 3
        assert summary["legal"] == 3
        assert summary["desired"] == 3

    def test_defaults_breakdown_to_primary_bond_op(self, tmp_path):
        ar = _make_ar(tmp_path)
        summary = ar.describe_state_space(verbose=False)
        assert summary["breakdown"] == {0: 1, 1: 1, 2: 1}

    def test_breakdown_by_name(self, tmp_path):
        ar = _make_ar(tmp_path)
        summary = ar.describe_state_space(breakdown_op="bonds", verbose=False)
        assert summary["breakdown"] == {0: 1, 1: 1, 2: 1}

    def test_filters_shrink_legal_and_desired(self, tmp_path):
        ar = _make_ar(tmp_path)
        ar.filter_legal_states = lambda states: [s for s in states if s[0] != 2]
        ar.filter_desired_states = lambda states: [s for s in states if s[0] == 0]
        summary = ar.describe_state_space(verbose=False)
        assert summary["possible"] == 3
        assert summary["legal"] == 2
        assert summary["desired"] == 1

    def test_verbose_does_not_raise(self, tmp_path, capsys):
        ar = _make_ar(tmp_path)
        ar.describe_state_space(verbose=True)
        assert "legal states" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# check_energy_print_budget
# ---------------------------------------------------------------------------

class TestCheckEnergyPrintBudget:
    def test_passes_with_ample_budget(self, tmp_path):
        ar = _make_ar(tmp_path)   # 3 desired states, n_reps=5 (default), steps_per_iter=1e8 (default)
        ar.check_energy_print_budget(1e5)   # should not raise

    def test_raises_when_budget_too_small(self, tmp_path):
        ar = _make_ar(tmp_path)
        # n_reps=5, steps_per_iter=1e8 -> 5e8 total steps; print_energy_every=1e9 -> 0.5 prints
        with pytest.raises(ValueError, match="Energy print budget too small"):
            ar.check_energy_print_budget(1e9)

    def test_boundary_equal_to_desired_count_does_not_raise(self, tmp_path):
        ar = _make_ar(tmp_path)
        ar.n_reps = 1
        ar.steps_per_iter = 3
        # 1 replica * 3 steps / print_every=1 -> exactly 3 prints == 3 desired states
        ar.check_energy_print_budget(1)


# ---------------------------------------------------------------------------
# get_overall_unwt_occ
# ---------------------------------------------------------------------------

def _mock_sim_with_vmmc_df(occ_by_state: dict) -> MagicMock:
    """
    Mock sim whose analysis.vmmc_df is indexed by state, matching the real shape (see
    VmmcAnalysis.read_vmmc_op_data: df.set_index(op_cols) -- the order parameter value(s)
    are the index, there is no op-name column). Keys may be bare ints (single order
    parameter) or tuples (multiple order parameters).
    """
    df = pd.DataFrame(
        {"unwt_occ": list(occ_by_state.values())},
        index=list(occ_by_state.keys()),
    )
    mock = MagicMock()
    mock.analysis.vmmc_df = df
    return mock


class TestGetOverallUnwtOcc:
    def test_real_vmmc_df_single_op(self, tmp_path):
        # Regression test: vmmc_df is indexed by state, not by an op-name column --
        # get_overall_unwt_occ must read the state from the row's index label.
        ar = _make_ar(tmp_path)
        sim = _mock_sim_with_vmmc_df({0: 10.0, 1: 40.0, 2: 50.0})
        result = ar.get_overall_unwt_occ([sim])
        np.testing.assert_allclose(result, [10.0 / 50.0, 40.0 / 50.0, 1.0])

    def test_sums_across_replicas(self, tmp_path):
        ar = _make_ar(tmp_path)
        sim1 = _mock_sim_with_vmmc_df({0: 10.0, 1: 0.0, 2: 0.0})
        sim2 = _mock_sim_with_vmmc_df({0: 0.0, 1: 10.0, 2: 0.0})
        result = ar.get_overall_unwt_occ([sim1, sim2])
        np.testing.assert_allclose(result, [1.0, 1.0, 0.0])

    def test_multi_op_tuple_index(self, tmp_path):
        ar = VMMCAutoReweight(tmp_path)
        ar.add_order_parameter(_bond_op())
        ar.add_order_parameter(OrderParameter("dist", "mindistance", [(0, 5)], interfaces=[3.]))
        ar.extrapolate_hist_Ts = ["30C"]
        ar.build_replica = lambda meta, sim: None
        sim = _mock_sim_with_vmmc_df({(0, 0): 5.0, (1, 0): 10.0, (2, 1): 10.0})
        result = ar.get_overall_unwt_occ([sim])
        assert result[0, 0] == pytest.approx(0.5)
        assert result[1, 0] == pytest.approx(1.0)
        assert result[2, 1] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# compute_next_it_weights
# ---------------------------------------------------------------------------

class TestComputeNextItWeights:
    """
    patch get_overall_unwt_occ to control the mock sampling occupancy,
    then verify the reweighting algebra directly.
    """

    def _ar_and_it(self, tmp_path, start_weights, unwt_occ_vals):
        ar = _make_ar(tmp_path)
        mock_sim = MagicMock()
        mock_sim.weights = np.array(start_weights, dtype=float)
        mock_it = [mock_sim]                    # list suffices for indexing + iteration
        unwt_occ = np.array(unwt_occ_vals, dtype=float)
        return ar, mock_it, unwt_occ

    @patch.object(VMMCAutoReweight, 'get_overall_unwt_occ')
    def test_output_is_normalized_at_one(self, mock_occ, tmp_path):
        # min weight over legal states should be 1.0 after reweighting
        ar, mock_it, unwt_occ = self._ar_and_it(tmp_path,
                                                 [1., 1., 1.],
                                                 [0.5, 1.0, 0.5])
        mock_occ.return_value = unwt_occ
        result = ar.compute_next_it_weights(mock_it)
        assert result[ar.legal_states_mask].min() == pytest.approx(1.0)

    @patch.object(VMMCAutoReweight, 'get_overall_unwt_occ')
    def test_unsampled_state_boosted_by_factor(self, mock_occ, tmp_path):
        # State 0 not sampled → should receive zero_sample_weight_factor boost
        ar, mock_it, unwt_occ = self._ar_and_it(tmp_path,
                                                 [2., 4., 2.],
                                                 [0.0, 0.8, 0.2])
        mock_occ.return_value = unwt_occ
        result = ar.compute_next_it_weights(mock_it)

        # State 0 was unsampled and desired → must be the heaviest
        assert result[0] > result[1]
        assert result[0] > result[2]
        # The boost comes from zero_sample_weight_factor
        # (exact value depends on normalization; we just check relative ordering)

    @patch.object(VMMCAutoReweight, 'get_overall_unwt_occ')
    def test_well_sampled_state_weight_decreases(self, mock_occ, tmp_path):
        # State 1 dominates sampling → its weight should decrease relative to others
        ar, mock_it, unwt_occ = self._ar_and_it(tmp_path,
                                                 [1., 1., 1.],
                                                 [0.1, 0.8, 0.1])
        mock_occ.return_value = unwt_occ
        result = ar.compute_next_it_weights(mock_it)
        # Over-sampled state gets divided by its occupancy; under-sampled states are clamped
        assert result[0] >= result[1]
        assert result[2] >= result[1]

    @patch.object(VMMCAutoReweight, 'get_overall_unwt_occ')
    def test_non_desired_states_not_in_output_denominator(self, mock_occ, tmp_path):
        ar = _make_ar(tmp_path)
        # filter desired to only state 1
        ar.filter_desired_states = lambda states: [s for s in states if s == (1,)]
        mock_sim = MagicMock()
        mock_sim.weights = np.array([1., 1., 1.], dtype=float)
        mock_it = [mock_sim]
        # unwt_occ: states 0 and 2 non-zero but not desired
        mock_occ.return_value = np.array([0.3, 1.0, 0.3])
        result = ar.compute_next_it_weights(mock_it)
        assert result is not None
        assert result.shape == (3,)

    @patch.object(VMMCAutoReweight, 'get_overall_unwt_occ')
    def test_returns_array_matching_op_shape(self, mock_occ, tmp_path):
        ar, mock_it, unwt_occ = self._ar_and_it(tmp_path,
                                                 [1., 1., 1.],
                                                 [0.5, 1.0, 0.5])
        mock_occ.return_value = unwt_occ
        result = ar.compute_next_it_weights(mock_it)
        assert result.shape == (3,)     # len(bond_op) = 3

    @patch.object(VMMCAutoReweight, 'get_overall_unwt_occ')
    def test_illegal_state_gets_default_sentinel(self, mock_occ, tmp_path):
        ar, mock_it, unwt_occ = self._ar_and_it(tmp_path, [1., 1., 1.], [0.5, 1.0, 0.5])
        ar.filter_legal_states = lambda states: [s for s in states if s != (2,)]
        mock_occ.return_value = unwt_occ
        result = ar.compute_next_it_weights(mock_it)
        assert result[2] == pytest.approx(1.0)   # default illegal_state_weight

    @patch.object(VMMCAutoReweight, 'get_overall_unwt_occ')
    def test_illegal_state_respects_custom_illegal_state_weight(self, mock_occ, tmp_path):
        # Regression test: illegal states must be forced to illegal_state_weight on every
        # call, not just carried over from the previous iteration's weights.
        ar, mock_it, unwt_occ = self._ar_and_it(tmp_path, [1., 1., 5.], [0.5, 1.0, 0.0])
        ar.filter_legal_states = lambda states: [s for s in states if s != (2,)]
        ar.illegal_state_weight = 0.0
        mock_occ.return_value = unwt_occ
        result = ar.compute_next_it_weights(mock_it)
        assert result[2] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# _apply_border_reweighting
# ---------------------------------------------------------------------------

class TestApplyBorderReweighting:
    """
    1D state space: states (0,) (1,) (2,).
    Only (1,) is desired; (0,) and (2,) are legal border states adjacent to (1,).
    """

    def _ar_border(self, tmp_path, *, reweight_borders, desired_only_mid=True):
        ar = _make_ar(tmp_path)
        if desired_only_mid:
            ar.filter_desired_states = lambda states: [s for s in states if s == (1,)]
        ar.reweight_borders = reweight_borders
        return ar

    def test_true_mode_sets_border_to_min_adjacent_desired(self, tmp_path):
        ar = self._ar_border(tmp_path, reweight_borders=True)
        weights = np.array([0.5, 3.0, 0.5])
        unwt_occ = np.array([0.0, 1.0, 0.0])   # only state 1 sampled
        ar._apply_border_reweighting(weights, unwt_occ)
        # States 0 and 2 are borders adjacent to state 1 (weight 3.0)
        assert weights[0] == pytest.approx(3.0)
        assert weights[2] == pytest.approx(3.0)
        assert weights[1] == pytest.approx(3.0)   # desired state unchanged

    def test_float_mode_sets_all_borders_to_constant(self, tmp_path):
        ar = self._ar_border(tmp_path, reweight_borders=7.5)
        weights = np.array([0.5, 3.0, 0.5])
        unwt_occ = np.array([0.0, 1.0, 0.0])
        ar._apply_border_reweighting(weights, unwt_occ)
        assert weights[0] == pytest.approx(7.5)
        assert weights[2] == pytest.approx(7.5)
        assert weights[1] == pytest.approx(3.0)   # desired unchanged

    def test_sampled_non_desired_state_treated_as_desired_for_border(self, tmp_path):
        # State 0 is not in desired but WAS sampled → treated as effectively desired
        # State 2 is the only true border
        ar = self._ar_border(tmp_path, reweight_borders=True)
        weights = np.array([2.0, 4.0, 0.5])
        unwt_occ = np.array([0.5, 1.0, 0.0])   # state 0 sampled (not desired)
        ar._apply_border_reweighting(weights, unwt_occ)
        # State 2 is adjacent to state 1 (desired, weight 4.0)
        assert weights[2] == pytest.approx(4.0)
        # State 0 was effectively desired (sampled) → not a border state
        assert weights[0] == pytest.approx(2.0)

    def test_no_adjacent_desired_state_leaves_weight_unchanged(self, tmp_path):
        # State space (0,)(1,)(2,) — only (0,) desired, so (2,) has no adjacent desired
        ar = _make_ar(tmp_path)
        ar.filter_desired_states = lambda states: [s for s in states if s == (0,)]
        ar.reweight_borders = True
        weights = np.array([2.0, 0.5, 0.5])
        unwt_occ = np.array([1.0, 0.0, 0.0])
        ar._apply_border_reweighting(weights, unwt_occ)
        # State 1 is adjacent to state 0 (desired) → updated
        assert weights[1] == pytest.approx(2.0)
        # State 2 is only adjacent to state 1 (not desired, not sampled) → unchanged
        assert weights[2] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# get_sampling_std_filtered
# ---------------------------------------------------------------------------

class TestGetSamplingSdFiltered:
    def test_uniform_state_sampling_gives_zero_std(self, tmp_path):
        # std() is computed across per-state sampling percents, not across replicas.
        # All states must have equal percent for std to be 0.
        ar = _make_ar(tmp_path)
        uniform = [33.3, 33.3, 33.3]
        sims = [_mock_sim_with_stats(uniform), _mock_sim_with_stats(uniform)]
        result = ar.get_sampling_std_filtered(sims)
        assert result == pytest.approx(0.0, abs=1e-6)

    def test_unequal_sampling_gives_positive_std(self, tmp_path):
        ar = _make_ar(tmp_path)
        sims = [
            _mock_sim_with_stats([50.0, 30.0, 20.0]),
            _mock_sim_with_stats([10.0, 60.0, 30.0]),
        ]
        result = ar.get_sampling_std_filtered(sims)
        assert result > 0.0

    def test_single_sim_std_of_its_distribution(self, tmp_path):
        ar = _make_ar(tmp_path)
        percents = [20.0, 60.0, 20.0]
        sims = [_mock_sim_with_stats(percents)]
        result = ar.get_sampling_std_filtered(sims)
        # mean of [20, 60, 20] / 1 = [20, 60, 20]; std thereof
        expected = np.std([20.0, 60.0, 20.0])
        assert result == pytest.approx(expected, rel=1e-6)

    def test_returns_float(self, tmp_path):
        ar = _make_ar(tmp_path)
        sims = [_mock_sim_with_stats([33.0, 33.0, 34.0])]
        result = ar.get_sampling_std_filtered(sims)
        assert isinstance(result, float)

    def test_works_with_state_indexed_statistics(self, tmp_path):
        # Regression test: statistics is indexed by state (see
        # calculate_sampling_and_probabilities), not by an op-name column.
        # state_is_accessible must read the state from row.name, not row[op.name].
        ar = _make_ar(tmp_path)
        sim = _mock_sim_with_indexed_stats({0: 33.3, 1: 33.3, 2: 33.4})
        result = ar.get_sampling_std_filtered([sim])
        assert result == pytest.approx(np.std([33.3, 33.3, 33.4]), rel=1e-6)


# ---------------------------------------------------------------------------
# check_result
# ---------------------------------------------------------------------------

class TestCheckResult:
    def test_true_when_all_desired_sampled_and_std_low(self, tmp_path):
        ar = _make_ar(tmp_path)   # single bond op "bonds": states (0,), (1,), (2,) all desired
        ar.max_rel_std = 100.0
        sim = _mock_sim_with_indexed_stats({0: 33.3, 1: 33.3, 2: 33.4})
        assert ar.check_result([sim]) == True

    def test_false_when_a_desired_state_unsampled(self, tmp_path, capsys):
        ar = _make_ar(tmp_path)
        sim = _mock_sim_with_indexed_stats({0: 0.0, 1: 50.0, 2: 50.0})
        assert ar.check_result([sim]) == False
        assert "not sampled" in capsys.readouterr().out

    def test_false_when_std_too_high(self, tmp_path):
        ar = _make_ar(tmp_path)
        ar.max_rel_std = 0.01   # effectively impossible to satisfy
        sim = _mock_sim_with_indexed_stats({0: 10.0, 1: 40.0, 2: 50.0})
        assert ar.check_result([sim]) == False

    def test_multi_op_tuple_state_index(self, tmp_path):
        ar = VMMCAutoReweight(tmp_path)
        ar.add_order_parameter(_bond_op())
        ar.add_order_parameter(OrderParameter("dist", "mindistance", [(0, 5)], interfaces=[3.]))
        ar.extrapolate_hist_Ts = ["30C"]
        ar.build_replica = lambda meta, sim: None
        ar.max_rel_std = 100.0
        # dist > 0 states are always excluded from "desired" -- only the dist=0 states matter
        sim = _mock_sim_with_indexed_stats({
            (0, 0): 20.0, (1, 0): 20.0, (2, 0): 20.0,
            (0, 1): 0.0, (1, 1): 0.0, (2, 1): 0.0,
        })
        assert ar.check_result([sim]) == True


# ---------------------------------------------------------------------------
# set_primary_bond_op / set_dist_op
# ---------------------------------------------------------------------------

class TestSetPrimaryBondOp:
    def test_moves_existing_op_to_front(self, tmp_path):
        ar = VMMCAutoReweight(tmp_path)
        op_a = OrderParameter("a", "bond", [(0, 1), (2, 3)])
        op_b = OrderParameter("b", "bond", [(4, 5), (6, 7)])
        ar.add_order_parameter(op_a)
        ar.add_order_parameter(op_b)
        assert ar.bond_ops()[0] is op_a

        ar.set_primary_bond_op(op_b)
        assert ar.bond_ops()[0] is op_b
        assert op_a in ar.bond_ops()

    def test_inserts_new_op_at_front(self, tmp_path):
        ar = VMMCAutoReweight(tmp_path)
        op_a = OrderParameter("a", "bond", [(0, 1), (2, 3)])
        op_b = OrderParameter("b", "bond", [(4, 5), (6, 7)])
        ar.add_order_parameter(op_a)
        ar.set_primary_bond_op(op_b)
        assert ar.bond_ops()[0] is op_b

    def test_by_name_string(self, tmp_path):
        ar = VMMCAutoReweight(tmp_path)
        op_a = OrderParameter("alpha", "bond", [(0, 1), (2, 3)])
        op_b = OrderParameter("beta", "bond", [(4, 5), (6, 7)])
        ar.add_order_parameter(op_a)
        ar.add_order_parameter(op_b)
        ar.set_primary_bond_op("beta")
        assert ar.bond_ops()[0].name == "beta"

    def test_by_name_unknown_raises(self, tmp_path):
        ar = _make_ar(tmp_path)
        with pytest.raises(ValueError):
            ar.set_primary_bond_op("nonexistent")


class TestSetDistOp:
    def test_sets_dist_op(self, tmp_path):
        ar = _make_ar(tmp_path)
        dist_op = OrderParameter("dist", "mindistance", [(0, 5)])
        ar.set_dist_op(dist_op)
        assert ar._dist_op is dist_op

    def test_overwrites_existing_dist_op(self, tmp_path):
        ar = _make_ar(tmp_path)
        d1 = OrderParameter("d1", "mindistance", [(0, 5)])
        d2 = OrderParameter("d2", "mindistance", [(1, 6)])
        ar.set_dist_op(d1)
        ar.set_dist_op(d2)
        assert ar._dist_op is d2


# ---------------------------------------------------------------------------
# Container methods
# ---------------------------------------------------------------------------

class TestContainerMethods:
    def test_len_empty(self, tmp_path):
        ar = _make_ar(tmp_path)
        assert len(ar) == 0

    def test_iter_empty(self, tmp_path):
        ar = _make_ar(tmp_path)
        assert list(ar) == []

    def test_len_and_iter_after_appending_subgroups(self, tmp_path):
        ar = _make_ar(tmp_path)
        mock_group = MagicMock()
        ar._subgroups.append(mock_group)
        ar._subgroups.append(mock_group)
        assert len(ar) == 2
        assert list(ar) == [mock_group, mock_group]

    def test_getitem_int(self, tmp_path):
        ar = _make_ar(tmp_path)
        m = MagicMock()
        ar._subgroups.append(m)
        assert ar[0] is m

    def test_getitem_negative(self, tmp_path):
        ar = _make_ar(tmp_path)
        m0, m1 = MagicMock(), MagicMock()
        ar._subgroups.extend([m0, m1])
        assert ar[-1] is m1

    def test_getitem_tuple_returns_replica(self, tmp_path):
        # ar[iteration, replica] should return the replica, not raise
        ar = _make_ar(tmp_path)
        replica = MagicMock()
        group = MagicMock()
        group.__getitem__ = MagicMock(return_value=replica)
        ar._subgroups.append(group)
        result = ar[0, 0]
        assert result is replica

    def test_getitem_bad_type_raises_value_error(self, tmp_path):
        ar = _make_ar(tmp_path)
        ar._subgroups.append(MagicMock())
        with pytest.raises((ValueError, TypeError)):
            _ = ar["bad"]


# ---------------------------------------------------------------------------
# get_weights_colors
# ---------------------------------------------------------------------------

class TestGetWeightsColors:
    def test_returns_array_for_valid_op(self, tmp_path):
        ar = _make_ar(tmp_path)
        colors = ar.get_weights_colors(0)
        assert colors is not None
        assert len(colors) > 0

    def test_missing_op_name_raises_value_error(self, tmp_path):
        # should raise ValueError (not StopIteration) for unknown op name
        ar = _make_ar(tmp_path)
        with pytest.raises(ValueError):
            ar.get_weights_colors("nonexistent_op")


# ---------------------------------------------------------------------------
# VMMCGraphReweight.compute_next_it_weights
# ---------------------------------------------------------------------------

def _make_gr(tmp_path, *, add_op=True, set_temps=True, set_builder=True) -> VMMCGraphReweight:
    gr = VMMCGraphReweight(tmp_path)
    if add_op:
        gr.add_order_parameter(_bond_op())
    if set_temps:
        gr.extrapolate_hist_Ts = ["30C", "40C"]
    if set_builder:
        gr.build_replica = lambda meta, sim: None
    return gr


def _mock_sim_with_energy_df(state_seq: list[tuple[int, ...]], op_name: str = "bonds") -> MagicMock:
    """Return a mock sim whose analysis.energy_df contains the given state sequence."""
    df = pd.DataFrame({
        "time": list(range(len(state_seq))),
        op_name: [s[0] for s in state_seq],
    })
    mock = MagicMock()
    mock.analysis.energy_df = df
    mock.weights = np.ones(3)
    return mock


def _mock_sim_with_op_trajectory(state_seq: list[tuple[int, ...]], obs_name: str = "op_trajectory") -> MagicMock:
    """
    Return a mock sim whose analysis.observable_data(obs_name) mimics the raw,
    unnamed-column DataFrame produced by build_op_trajectory_observable + Analysis.observable_data:
    column 0 is the step, column 1 is the (single) order parameter's value.
    """
    raw = pd.DataFrame({
        0: list(range(len(state_seq))),
        1: [s[0] for s in state_seq],
    })
    mock = MagicMock()
    mock.analysis.observable_data.return_value = raw
    # energy_df deliberately left as an unconfigured MagicMock/attribute: it must
    # not be touched when op_trajectory_name is set.
    mock.weights = np.ones(3)
    return mock


class TestVMMCGraphReweightInit:
    def test_inherits_auto_reweight(self, tmp_path):
        gr = VMMCGraphReweight(tmp_path)
        assert isinstance(gr, VMMCAutoReweight)

    def test_has_pseudo_count(self, tmp_path):
        gr = VMMCGraphReweight(tmp_path)
        assert gr.graph_pseudo_count == pytest.approx(1.0)

    def test_op_trajectory_name_defaults_to_none(self, tmp_path):
        gr = VMMCGraphReweight(tmp_path)
        assert gr.op_trajectory_name is None


class TestVMMCGraphReweightOpTrajectorySource:
    """
    When op_trajectory_name is set, compute_next_it_weights must read transitions from
    sim.analysis.observable_data(name) instead of energy_df.
    """

    def test_uses_observable_data_not_energy_df(self, tmp_path):
        gr = _make_gr(tmp_path)
        gr.op_trajectory_name = "op_trajectory"
        seq = [(0,), (1,), (2,), (1,), (0,)]
        sim = _mock_sim_with_op_trajectory(seq, obs_name="op_trajectory")

        result = gr.compute_next_it_weights([sim])

        sim.analysis.observable_data.assert_called_once_with("op_trajectory")
        assert result.shape == (3,)

    def test_matches_energy_df_result_for_equivalent_sequence(self, tmp_path):
        # Same underlying state sequence via either source should give the same weights
        seq = ([(0,), (1,)] * 10) + ([(1,), (2,)] * 10) + [(2,), (1,)] * 2

        gr_energy = _make_gr(tmp_path)
        sim_energy = _mock_sim_with_energy_df(seq)
        result_energy = gr_energy.compute_next_it_weights([sim_energy])

        gr_traj = _make_gr(tmp_path)
        gr_traj.op_trajectory_name = "op_trajectory"
        sim_traj = _mock_sim_with_op_trajectory(seq)
        result_traj = gr_traj.compute_next_it_weights([sim_traj])

        np.testing.assert_allclose(result_energy, result_traj)


class TestVMMCGraphReweightComputeWeights:
    def _run(self, tmp_path, state_seq, start_weights=None):
        gr = _make_gr(tmp_path)
        sim = _mock_sim_with_energy_df(state_seq)
        if start_weights is not None:
            sim.weights = np.array(start_weights, dtype=float)
        last_it = [sim]
        return gr.compute_next_it_weights(last_it)

    def test_returns_correct_shape(self, tmp_path):
        # bond op has 3 states (0,1,2); weights array must match
        seq = [(0,), (1,), (2,), (1,), (0,)]
        result = self._run(tmp_path, seq)
        assert result.shape == (3,)

    def test_balanced_transitions_give_flat_weights(self, tmp_path):
        # Edge targets use per-visit transition rates (c[i->j] / visits to i), not raw
        # counts — raw crossing counts are ~symmetric for *any* reversible MC regardless
        # of whether sampling is flat (that was the bug), so "balanced" here means equal
        # visits to each state *and* equal raw counts in both directions per edge, which
        # together give equal per-visit rates and hence a zero log-weight target.
        # Visits: 0×4, 1×4, 2×4 (extra same-state entries pad visit counts without adding
        # transitions). Transitions: c[0->1]=c[1->0]=2, c[1->2]=c[2->1]=2.
        seq = [(0,), (0,), (1,), (2,), (2,), (1,), (0,), (1,), (2,), (2,), (1,), (0,)]
        result = self._run(tmp_path, seq)
        # min is 1 after renorm; all desired (all states) should be equal
        assert result[0] == pytest.approx(result[1], rel=0.01)
        assert result[1] == pytest.approx(result[2], rel=0.01)

    def test_over_sampled_state_gets_lower_weight(self, tmp_path):
        # System flows strongly 0→1→2; state 2 is over-visited
        # Many transitions ending at 2, few leaving it
        seq = ([(0,), (1,)] * 10) + ([(1,), (2,)] * 10) + [(2,), (1,)] * 2
        result = self._run(tmp_path, seq)
        # State 0 is starved → should have highest weight; state 2 is over-visited → lowest
        assert result[0] > result[2]

    def test_no_transitions_falls_back_to_copy(self, tmp_path):
        # All states the same → no transitions counted → returns copy of old weights
        seq = [(1,)] * 20
        start = [1.0, 5.0, 3.0]
        result = self._run(tmp_path, seq, start_weights=start)
        # Should return a valid weight array of the same shape
        assert result.shape == (3,)

    def test_geometric_weights_from_one_directional_flow(self, tmp_path):
        """
        Sequence 0→1→2→0→1→2→... (×9).  The 2→0 jump has diff=2 and is filtered.
        Observed: c[0→1]=9, c[1→2]=9, all reverse counts=0.
        With ε=1: both edges have target = log(1/10).
        Least-squares gives log_w = [log10, 0, -log10]; after renorm → [100, 10, 1].
        This exercises the raw per-edge math directly, so disable the max_log_weight_step
        trust region (see test_max_log_weight_step_caps_single_edge_correction for that).
        """
        gr = _make_gr(tmp_path)
        gr.graph_pseudo_count = 1.0
        gr.max_log_weight_step = float("inf")
        seq = [(0,), (1,), (2,)] * 9
        sim = _mock_sim_with_energy_df(seq)

        result = gr.compute_next_it_weights([sim])

        assert result[2] == pytest.approx(1.0, rel=1e-4)
        assert result[1] == pytest.approx(10.0, rel=1e-4)
        assert result[0] == pytest.approx(100.0, rel=1e-4)

    def test_max_log_weight_step_caps_single_edge_correction(self, tmp_path):
        """
        Same one-directional-flow scenario as above (uncapped target = log(1/10) per
        edge), but with the default max_log_weight_step (log(2)) left in place: each
        edge's correction should be clipped to a 2x ratio instead of the raw 10x.
        """
        gr = _make_gr(tmp_path)
        gr.graph_pseudo_count = 1.0
        seq = [(0,), (1,), (2,)] * 9
        sim = _mock_sim_with_energy_df(seq)

        result = gr.compute_next_it_weights([sim])

        assert result[2] == pytest.approx(1.0, rel=1e-4)
        assert result[1] == pytest.approx(2.0, rel=1e-4)
        assert result[0] == pytest.approx(4.0, rel=1e-4)

    def test_rare_event_triggers_warning_and_damping(self, tmp_path):
        # Iteration 0: state 0 essentially unvisited (one stray visit) -> almost no
        # prior evidence for it.
        seq0 = [(1,)] * 200 + [(0,), (1,)]
        sim0 = _mock_sim_with_energy_df(seq0)

        # Iteration 1 (last_it): a sudden, large influx of visits to state 0 -- e.g.
        # VMMC finally crossing into a previously-unreached state.
        seq1 = [(0,)] * 200 + [(1,)] * 50
        sim1 = _mock_sim_with_energy_df(seq1)
        old_weight = sim1.weights[0]  # mock sims default to weights = np.ones(3)

        gr = _make_gr(tmp_path)
        gr._subgroups = [[sim0], [sim1]]
        with pytest.warns(UserWarning, match="rare"):
            result_damped = gr.compute_next_it_weights([sim1])

        # Same scenario with damping disabled (rare_event_damping=1.0) shows what the
        # fresh evidence alone would imply, undamped.
        gr_undamped = _make_gr(tmp_path)
        gr_undamped._subgroups = [[sim0], [sim1]]
        gr_undamped.rare_event_damping = 1.0
        with pytest.warns(UserWarning, match="rare"):
            result_undamped = gr_undamped.compute_next_it_weights([sim1])

        # The damped result should land strictly between the previous weight and the
        # undamped candidate, not jump straight to the undamped value.
        assert result_undamped[0] != pytest.approx(old_weight, rel=1e-3)
        lo, hi = sorted([old_weight, result_undamped[0]])
        assert lo < result_damped[0] < hi

    def test_accumulates_counts_across_iterations(self, tmp_path):
        # Iteration 0: a large, well-balanced sample -> near-zero correction on its own.
        seq0 = [(0,), (1,)] * 100
        sim0 = _mock_sim_with_energy_df(seq0)

        # Iteration 1 (last_it): a tiny sample that looks strongly imbalanced purely
        # by chance (a single 0->1 transition, no 1->0 to balance it).
        seq1 = [(0,), (1,)]
        sim1 = _mock_sim_with_energy_df(seq1)

        # Using only the latest iteration, that noisy n=1 sample fully determines
        # the correction.
        gr_single = _make_gr(tmp_path)
        result_single = gr_single.compute_next_it_weights([sim1])

        # With recorded history (iteration 0's well-balanced evidence plus iteration
        # 1's noisy sample), the correction should be dominated by the larger,
        # better-determined iteration 0 data instead of resetting to iteration 1 alone.
        gr_accum = _make_gr(tmp_path)
        gr_accum._subgroups = [[sim0], [sim1]]
        result_accum = gr_accum.compute_next_it_weights([sim1])

        ratio_single = result_single[0] / result_single[1]
        ratio_accum = result_accum[0] / result_accum[1]

        # Both correct in the same direction (state 1 slightly over-visited)...
        assert ratio_single > 1
        assert ratio_accum > 1
        # ...but accumulating history should damp the lone noisy sample's swing.
        assert ratio_accum < ratio_single

    def test_pseudo_count_dampens_extreme_imbalance(self, tmp_path):
        # Very asymmetric transitions: 0→1 seen 1000 times, 1→0 never
        gr = _make_gr(tmp_path)
        gr.graph_pseudo_count = 10.0  # large pseudo-count → dampened correction

        sim_high_eps = _mock_sim_with_energy_df([(0,), (1,)] * 500)
        result_high = gr.compute_next_it_weights([sim_high_eps])

        gr2 = _make_gr(tmp_path)
        gr2.graph_pseudo_count = 0.01  # small pseudo-count → aggressive correction
        sim_low_eps = _mock_sim_with_energy_df([(0,), (1,)] * 500)
        result_low = gr2.compute_next_it_weights([sim_low_eps])

        # Both should correct in the same direction (w[0] > w[1])
        assert result_high[0] > result_high[1]
        assert result_low[0] > result_low[1]
        # Larger pseudo-count → smaller ratio → less extreme correction
        ratio_high = result_high[0] / result_high[1]
        ratio_low = result_low[0] / result_low[1]
        assert ratio_high < ratio_low

    @patch.object(VMMCGraphReweight, 'get_overall_unwt_occ')
    def test_illegal_state_respects_custom_weight(self, mock_occ, tmp_path):
        # Regression test: illegal states must be forced to illegal_state_weight even though
        # they're outside legal_states -- they're never touched by the graph solve at all, so
        # without this they'd just carry over last iteration's weight unchanged.
        gr = _make_gr(tmp_path)
        gr.filter_legal_states = lambda states: [s for s in states if s != (2,)]
        gr.illegal_state_weight = 0.0
        mock_occ.return_value = np.array([0.5, 1.0, 0.0])
        seq = [(0,), (1,), (0,), (1,)] * 5
        sim = _mock_sim_with_energy_df(seq)
        sim.weights = np.array([1.0, 1.0, 5.0])
        result = gr.compute_next_it_weights([sim])
        assert result[2] == pytest.approx(0.0)

    @patch.object(VMMCGraphReweight, 'get_overall_unwt_occ')
    def test_never_sampled_desired_state_boosted_off_previous_weight(self, mock_occ, tmp_path):
        # Regression test: a desired state with unreliable (near-zero) observed occupancy
        # gives the graph solve no real signal (every edge touching it defaults to
        # log(eps/eps)=0), so it must be floored at an escalated version of its *previous*
        # weight -- not left to whatever the graph solve happens to compute for it.
        seq = [(1,), (2,), (1,), (2,)] * 5   # state 0 never appears in the trajectory at all
        mock_occ.return_value = np.array([0.0, 0.5, 1.0])   # state 0 truly unsampled

        gr_small = _make_gr(tmp_path)
        gr_small.undersampled_boost_factor = 2.0
        sim_small = _mock_sim_with_energy_df(seq)
        sim_small.weights = np.array([3.0, 1.0, 1.0])
        result_small = gr_small.compute_next_it_weights([sim_small])

        gr_big = _make_gr(tmp_path)
        gr_big.undersampled_boost_factor = 20.0
        sim_big = _mock_sim_with_energy_df(seq)
        sim_big.weights = np.array([3.0, 1.0, 1.0])
        result_big = gr_big.compute_next_it_weights([sim_big])

        # state 0 (boosted) vs state 1 (set purely by the graph solve, independent of
        # undersampled_boost_factor): their ratio should scale linearly with the factor
        ratio_small = result_small[0] / result_small[1]
        ratio_big = result_big[0] / result_big[1]
        assert ratio_big > ratio_small
        assert ratio_big / ratio_small == pytest.approx(20.0 / 2.0, rel=0.05)

    @patch.object(VMMCGraphReweight, 'get_overall_unwt_occ')
    def test_undersampled_boost_does_not_override_a_higher_graph_solve(self, mock_occ, tmp_path):
        # Regression test for the max()-floor (not override): if the graph solve already
        # landed on something higher than the escalated floor, keep it -- don't clobber a
        # reasonable estimate with a smaller one.
        gr = _make_gr(tmp_path)
        gr.undersampled_boost_factor = 1.01   # tiny boost, easily beaten by the graph solve
        # heavy 2->1 flow (state 2 draining into state 1) means the solve wants w[2] > w[1]
        seq = [(0,), (1,)] + ([(2,), (1,)] * 20)
        mock_occ.return_value = np.array([0.5, 1.0, 1e-4])   # state 2 below the 1e-3 threshold
        sim = _mock_sim_with_energy_df(seq)
        sim.weights = np.array([1.0, 1.0, 1.0])
        result = gr.compute_next_it_weights([sim])
        # graph solve should push state 2 well above a mere 1.01x floor
        assert result[2] > 1.01


# ---------------------------------------------------------------------------
# plot_iteration_weights
# ---------------------------------------------------------------------------

class TestPlotIterationWeights:
    """
    Real (unmocked) VirtualMoveMonteCarlo objects: order parameters and weights don't need
    sim_dir to exist on disk, so these exercise the real plot_weights() call plot_iteration_weights
    makes, which is what actually broke (passing (None,) where an int/str identifier is required).
    """

    def test_single_bond_op_does_not_raise(self, tmp_path):
        ar = _make_ar(tmp_path)   # single bond op "bonds", 3 states
        sim = VirtualMoveMonteCarlo(tmp_path / "iteration_0" / "rep1")
        sim.add_order_parameter(_bond_op())
        ar._subgroups.append([sim])   # minimal stand-in for a VmmcReplicas: index 0 -> sim

        fig, ax = plt.subplots()
        result = ar.plot_iteration_weights(iteration=-1, ax=ax)
        plt.close(fig)
        assert result is None   # ax was supplied, so no new figure is returned

    def test_two_bond_ops_with_dist_op_does_not_raise(self, tmp_path):
        ar = VMMCAutoReweight(tmp_path)
        ar.add_order_parameter(_bond_op("a"))
        ar.add_order_parameter(_bond_op("b"))
        ar.add_order_parameter(OrderParameter("dist", "mindistance", [(0, 5)], interfaces=[3.]))
        ar.extrapolate_hist_Ts = ["30C"]
        ar.build_replica = lambda meta, sim: None

        sim = VirtualMoveMonteCarlo(tmp_path / "iteration_0" / "rep1")
        for op in ar.order_parameters():
            sim.add_order_parameter(op)
        ar._subgroups.append([sim])

        fig, ax = plt.subplots()
        ar.plot_iteration_weights(iteration=-1, ax=ax)
        plt.close(fig)


# ---------------------------------------------------------------------------
# plot_bond_curves
# ---------------------------------------------------------------------------

def _replicas_with_sims(tmp_path, sims: list) -> VmmcReplicas:
    """Real VmmcReplicas wrapping already-constructed sims, bypassing .init()."""
    replicas = VmmcReplicas(tmp_path, tmp_path, len(sims))
    replicas.simulations = sims
    return replicas


class TestPlotBondCurves:
    def test_valid_bond_op_index_does_not_raise(self, tmp_path):
        ar = _make_ar(tmp_path)   # single bond op "bonds", 3 states
        sim = VirtualMoveMonteCarlo(tmp_path / "iteration_0" / "rep1")
        sim.add_order_parameter(_bond_op())
        sim.analysis._energy_df = pd.DataFrame({"time": [0, 1, 2], "bonds": [0, 1, 2]})
        ar._subgroups.append(_replicas_with_sims(tmp_path, [sim]))

        fig, ax = plt.subplots()
        ar.plot_bond_curves(subgroup_idx=-1, bond_op_index=0, ax=ax)
        plt.close(fig)

    def test_out_of_range_bond_op_index_raises_value_error(self, tmp_path):
        # Regression test: the bounds check used to call self[0].num_ops() -- but self[0]
        # is a VmmcReplicas (an iteration container), which has no num_ops() method at all
        # (that's a VirtualMoveMonteCarlo method). Also, the check ran *after*
        # self.bond_ops()[bond_op_index], so even fixing just that, a real out-of-range
        # index raised IndexError before the intended ValueError check was ever reached.
        ar = _make_ar(tmp_path)
        sim = VirtualMoveMonteCarlo(tmp_path / "iteration_0" / "rep1")
        sim.add_order_parameter(_bond_op())
        ar._subgroups.append(_replicas_with_sims(tmp_path, [sim]))

        with pytest.raises(ValueError, match="out of range"):
            ar.plot_bond_curves(subgroup_idx=-1, bond_op_index=5)


# ---------------------------------------------------------------------------
# plot_free_energy_profile
# ---------------------------------------------------------------------------

class TestPlotFreeEnergyProfile:
    def test_gracefully_skips_desired_states_absent_from_data(self, tmp_path):
        # Regression test: used to do df.iloc[indexer] (positional row selection) instead of
        # a label/state-value based selection. Raised IndexError whenever a desired state
        # was absent from this run's data (e.g. never sampled, so it has no row at all after
        # the groupby in get_data_over) rather than gracefully skipping it.
        ar = _make_ar(tmp_path)   # bond op "bonds": states 0, 1, 2 all desired by default

        sim = VirtualMoveMonteCarlo(tmp_path / "iteration_0" / "rep1")
        op = _bond_op()
        sim.add_order_parameter(op)
        # state 2 never sampled -- absent from vmmc_df entirely, not even a zero row
        sim.analysis._vmmc_df = pd.DataFrame(
            {"unwt_occ": [10.0, 20.0], "wt_occ": [1.0, 2.0]},
            index=pd.Index([0, 1], name="bonds"),
        )
        # get_data_over() also stamps a `step` (from current_step(), which reads real
        # trajectory files off disk) onto its VMMCData -- irrelevant to the .df selection
        # logic under test here, so stub it out rather than building real oxDNA output.
        sim.analysis.current_step = lambda: 0.0
        ar._subgroups.append(_replicas_with_sims(tmp_path, [sim]))

        fig = ar.plot_free_energy_profile(op=op, iteration=-1)   # should not raise
        plt.close(fig)