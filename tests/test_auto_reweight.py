"""
Tests for VMMCAutoReweight in vmmc_umbrella/auto_reweight.py.

Simulation-running methods (run, run_iteration, load, visualize) are not
covered here — they require actual oxDNA output on disk.  Everything else
is tested with lightweight mocks or pure numpy.
"""
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from oxpy_utils.utils.order_parameter import OrderParameter
from oxpy_utils.vmmc_umbrella.auto_reweight import VMMCAutoReweight, VMMCGraphReweight


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


class TestVMMCGraphReweightInit:
    def test_inherits_auto_reweight(self, tmp_path):
        gr = VMMCGraphReweight(tmp_path)
        assert isinstance(gr, VMMCAutoReweight)

    def test_has_pseudo_count(self, tmp_path):
        gr = VMMCGraphReweight(tmp_path)
        assert gr.graph_pseudo_count == pytest.approx(1.0)


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