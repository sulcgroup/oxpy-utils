"""Tests for vmmc_umbrella/vmmc_data.py: VMMCData, read_vmmc_data, average_vmmc_data."""
import pytest
import numpy as np
import pandas as pd

from oxpy_utils.utils.order_parameter import OrderParameter
from oxpy_utils.vmmc_umbrella.vmmc_data import VMMCData, average_vmmc_data, read_vmmc_data


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bond_op(n_pairs=2) -> OrderParameter:
    pairs = [(i * 2, i * 2 + 1) for i in range(n_pairs)]
    return OrderParameter("bonds", "bond", pairs)


def _make_df(counts, unbiased, temps=None) -> pd.DataFrame:
    """Build a minimal DataFrame for VMMCData with h_bonds as the index."""
    n = len(counts)
    d: dict = {
        "count": counts,
        "unbiased_count": unbiased,
    }
    if temps:
        for T, vals in temps.items():
            d[f"unbiased_count_{T}"] = vals
    return pd.DataFrame(d, index=range(n))


def _make_vmmc(counts=None, unbiased=None, op=None, step=100) -> VMMCData:
    if counts is None:
        counts = [10, 5, 3]
    if unbiased is None:
        unbiased = [8, 4, 2]
    if op is None:
        op = _bond_op(n_pairs=2)
    return VMMCData(op=op, df=_make_df(counts, unbiased), step=step)


# ---------------------------------------------------------------------------
# Construction / properties
# ---------------------------------------------------------------------------

class TestVMMCDataConstruction:
    def test_counts_property(self):
        vd = _make_vmmc(counts=[10, 5, 3])
        assert list(vd.counts) == [10, 5, 3]

    def test_unbiased_counts_property(self):
        vd = _make_vmmc(unbiased=[8, 4, 2])
        assert list(vd.unbiased_counts) == [8, 4, 2]

    def test_total_count(self):
        vd = _make_vmmc(counts=[10, 5, 3])
        assert vd.total_count == 18

    def test_total_unbiased_count(self):
        vd = _make_vmmc(unbiased=[8, 4, 2])
        assert vd.total_unbiased_count == 14

    def test_sampling_prob_sums_to_one(self):
        vd = _make_vmmc()
        assert vd.sampling_prob.sum() == pytest.approx(1.0)

    def test_unbiased_sampling_prob_sums_to_one(self):
        vd = _make_vmmc()
        assert vd.unbiased_sampling_prob.sum() == pytest.approx(1.0)

    def test_free_energy_finite(self):
        vd = _make_vmmc()
        assert np.isfinite(vd.free_energy.values).all()

    def test_all_zero_unbiased_raises(self):
        with pytest.raises(ValueError):
            _make_vmmc(unbiased=[0, 0, 0])

    def test_temperatures_extracted_from_columns(self):
        df = _make_df(
            counts=[10, 5, 3],
            unbiased=[8, 4, 2],
            temps={25.0: [7, 3, 2], 37.0: [1, 1, 0]},
        )
        vd = VMMCData(op=_bond_op(), df=df, step=100)
        assert 25.0 in vd.temperatures
        assert 37.0 in vd.temperatures

    def test_no_temperature_columns_gives_empty_list(self):
        vd = _make_vmmc()
        assert vd.temperatures == []

    def test_h_bonds_is_index(self):
        vd = _make_vmmc()
        assert list(vd.h_bonds) == [0, 1, 2]


# ---------------------------------------------------------------------------
# __getitem__
# ---------------------------------------------------------------------------

class TestVMMCDataGetItem:
    def test_int_index_returns_row(self):
        vd = _make_vmmc()
        row = vd[0]
        assert row["count"] == 10

    def test_float_index_returns_temperature_column(self):
        df = _make_df(
            [10, 5, 3], [8, 4, 2],
            temps={30.0: [7, 3, 2]},
        )
        vd = VMMCData(op=_bond_op(), df=df, step=0)
        col = vd[30.0]
        assert list(col) == [7, 3, 2]

    def test_invalid_index_raises(self):
        vd = _make_vmmc()
        with pytest.raises(Exception):
            vd[object()]  # type: ignore[index]


# ---------------------------------------------------------------------------
# __add__
# ---------------------------------------------------------------------------

class TestVMMCDataAdd:
    def test_counts_are_summed(self):
        vd1 = _make_vmmc(counts=[10, 5, 3], unbiased=[8, 4, 2], step=100)
        vd2 = _make_vmmc(counts=[2, 2, 2], unbiased=[1, 1, 1], step=200)
        result = vd1 + vd2
        assert list(result.counts) == [12, 7, 5]

    def test_unbiased_counts_are_summed(self):
        vd1 = _make_vmmc(counts=[10, 5, 3], unbiased=[8, 4, 2], step=100)
        vd2 = _make_vmmc(counts=[2, 2, 2], unbiased=[1, 1, 1], step=200)
        result = vd1 + vd2
        assert list(result.unbiased_counts) == [9, 5, 3]

    def test_step_is_average(self):
        vd1 = _make_vmmc(step=100)
        vd2 = _make_vmmc(step=200)
        result = vd1 + vd2
        assert result.step == 150

    def test_op_preserved(self):
        op = _bond_op()
        vd1 = _make_vmmc(op=op)
        vd2 = _make_vmmc(op=op)
        result = vd1 + vd2
        assert result.op is op

    def test_different_op_raises(self):
        op1 = OrderParameter("a", "bond", [(0, 1), (2, 3)])
        op2 = OrderParameter("b", "bond", [(4, 5), (6, 7)])
        vd1 = _make_vmmc(op=op1)
        vd2 = _make_vmmc(op=op2)
        with pytest.raises(ValueError):
            vd1 + vd2

    def test_non_vmmc_raises(self):
        vd = _make_vmmc()
        with pytest.raises(ValueError):
            vd + 5  # type: ignore[operator]


# ---------------------------------------------------------------------------
# average_vmmc_data
# ---------------------------------------------------------------------------

class TestAverageVMMCData:
    def test_average_counts_equal_sum(self):
        vd1 = _make_vmmc(counts=[10, 5, 3], unbiased=[8, 4, 2])
        vd2 = _make_vmmc(counts=[2, 2, 2], unbiased=[1, 1, 1])
        result = average_vmmc_data([vd1, vd2])
        assert list(result.counts) == [12, 7, 5]

    def test_average_step_is_mean(self):
        vd1 = _make_vmmc(step=100)
        vd2 = _make_vmmc(step=300)
        result = average_vmmc_data([vd1, vd2])
        assert result.step == pytest.approx(200.0)

    def test_single_item_raises(self):
        with pytest.raises(AssertionError):
            average_vmmc_data([_make_vmmc()])

    def test_three_items(self):
        vds = [_make_vmmc(counts=[6, 3, 1], unbiased=[5, 2, 1]) for _ in range(3)]
        result = average_vmmc_data(vds)
        assert list(result.counts) == [18, 9, 3]


# ---------------------------------------------------------------------------
# read_vmmc_data
# ---------------------------------------------------------------------------

class TestReadVMMCData:
    def _write_vmmc_file(self, path, op, temps=(0.09334,), rows=None):
        """Write a minimal valid VMMC data file."""
        temp_str = " ".join(str(t) for t in temps)
        with path.open("w") as f:
            f.write(f"t = 1000000; temperatures: {temp_str}\n")
            if rows is None:
                # one row per state (pairs+1 states)
                rows = [[i, 10 - i * 3, 8 - i * 2] + [max(1, 7 - i * 2)] * len(temps)
                        for i in range(len(op.pairs) + 1)]
            for row in rows:
                f.write(" ".join(str(x) for x in row) + "\n")
        return path

    def test_reads_simulation_time(self, tmp_path):
        op = _bond_op(n_pairs=2)
        fp = self._write_vmmc_file(tmp_path / "data.txt", op)
        vd = read_vmmc_data(fp, op)
        assert vd.step == 1000000

    def test_row_count_matches_op_pairs(self, tmp_path):
        op = _bond_op(n_pairs=2)  # 2 pairs → 3 states (0, 1, 2)
        fp = self._write_vmmc_file(tmp_path / "data.txt", op)
        vd = read_vmmc_data(fp, op)
        assert len(vd.df) == 3

    def test_temperature_columns_created(self, tmp_path):
        op = _bond_op(n_pairs=2)
        fp = self._write_vmmc_file(tmp_path / "data.txt", op, temps=(0.09334, 0.10334))
        vd = read_vmmc_data(fp, op)
        assert len(vd.temperatures) == 2

    def test_wrong_op_type_raises(self, tmp_path):
        op_dist = OrderParameter("d", "mindistance", [(0, 1)])
        fp = tmp_path / "data.txt"
        fp.write_text("t = 0; temperatures: 0.1\n0 10 8 7\n1 5 4 3\n")
        with pytest.raises(AssertionError):
            read_vmmc_data(fp, op_dist)

    def test_missing_file_raises(self, tmp_path):
        op = _bond_op()
        with pytest.raises(AssertionError):
            read_vmmc_data(tmp_path / "nonexistent.txt", op)