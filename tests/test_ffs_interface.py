"""Tests for ffs/ffs_interface.py: Comparison, FFSInterface, Condition, order_params."""
import pytest

from oxpy_utils.ffs.ffs_interface import (
    Comparison,
    Condition,
    FFSInterface,
    order_params,
    write_order_params,
)
from oxpy_utils.utils.order_parameter import OrderParameter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _op(name="hb", pairs=None) -> OrderParameter:
    if pairs is None:
        pairs = [(0, 1), (2, 3)]
    return OrderParameter(name, "bond", pairs)


def _iface(op=None, val=3, compare=Comparison.GEQ) -> FFSInterface:
    return FFSInterface(op=op or _op(), val=val, compare=compare)


# ---------------------------------------------------------------------------
# Comparison enum
# ---------------------------------------------------------------------------

class TestComparison:
    def test_values(self):
        assert Comparison.LT.value == "<"
        assert Comparison.GT.value == ">"
        assert Comparison.LEQ.value == "<="
        assert Comparison.GEQ.value == ">="

    def test_four_members(self):
        assert len(Comparison) == 4


# ---------------------------------------------------------------------------
# FFSInterface.test
# ---------------------------------------------------------------------------

class TestFFSInterfaceTest:
    def test_lt_true(self):
        iface = _iface(val=5.0, compare=Comparison.LT)
        assert iface.test(4.9) is True

    def test_lt_false(self):
        iface = _iface(val=5.0, compare=Comparison.LT)
        assert iface.test(5.0) is False

    def test_gt_true(self):
        iface = _iface(val=2.0, compare=Comparison.GT)
        assert iface.test(2.1) is True

    def test_gt_false(self):
        iface = _iface(val=2.0, compare=Comparison.GT)
        assert iface.test(2.0) is False

    def test_leq_at_boundary(self):
        iface = _iface(val=3.0, compare=Comparison.LEQ)
        assert iface.test(3.0) is True

    def test_geq_at_boundary(self):
        iface = _iface(val=3.0, compare=Comparison.GEQ)
        assert iface.test(3.0) is True

    def test_geq_below_boundary(self):
        iface = _iface(val=3.0, compare=Comparison.GEQ)
        assert iface.test(2.9) is False


# ---------------------------------------------------------------------------
# FFSInterface.__invert__
# ---------------------------------------------------------------------------

class TestFFSInterfaceInvert:
    @pytest.mark.parametrize("orig,expected", [
        (Comparison.LT,  Comparison.GEQ),
        (Comparison.GT,  Comparison.LEQ),
        (Comparison.LEQ, Comparison.GT),
        (Comparison.GEQ, Comparison.LT),
    ])
    def test_invert_comparison(self, orig, expected):
        iface = _iface(compare=orig)
        inverted = ~iface
        assert inverted.compare == expected

    def test_invert_preserves_op_and_val(self):
        op = _op()
        iface = FFSInterface(op=op, val=7.0, compare=Comparison.GT)
        inverted = ~iface
        assert inverted.op is op
        assert inverted.val == pytest.approx(7.0)

    def test_double_invert_is_original(self):
        iface = _iface(compare=Comparison.LT)
        assert (~(~iface)).compare == Comparison.LT


# ---------------------------------------------------------------------------
# FFSInterface.flip
# ---------------------------------------------------------------------------

class TestFFSInterfaceFlip:
    @pytest.mark.parametrize("orig,expected", [
        (Comparison.LT,  Comparison.GT),
        (Comparison.GT,  Comparison.LT),
        (Comparison.LEQ, Comparison.GEQ),
        (Comparison.GEQ, Comparison.LEQ),
    ])
    def test_flip_comparison(self, orig, expected):
        iface = _iface(compare=orig)
        flipped = iface.flip()
        assert flipped.compare == expected

    def test_flip_preserves_val(self):
        iface = _iface(val=4.0, compare=Comparison.GT)
        assert iface.flip().val == pytest.approx(4.0)

    def test_flip_differs_from_invert(self):
        iface = _iface(compare=Comparison.GT)
        assert iface.flip().compare != (~iface).compare


# ---------------------------------------------------------------------------
# FFSInterface.__str__
# ---------------------------------------------------------------------------

class TestFFSInterfaceStr:
    def test_str_format(self):
        op = _op(name="mybonds")
        iface = FFSInterface(op=op, val=5, compare=Comparison.GEQ)
        assert str(iface) == "mybonds >= 5"

    def test_str_lt(self):
        op = _op(name="dist")
        iface = FFSInterface(op=op, val=2.5, compare=Comparison.LT)
        assert str(iface) == "dist < 2.5"


# ---------------------------------------------------------------------------
# FFSInterface frozen
# ---------------------------------------------------------------------------

class TestFFSInterfaceFrozen:
    def test_is_frozen(self):
        iface = _iface()
        with pytest.raises((AttributeError, TypeError)):
            iface.val = 99  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Condition
# ---------------------------------------------------------------------------

class TestCondition:
    def test_default_condition_type_or(self):
        cond = Condition("stop", [_iface()])
        assert cond.condition_type == "or"

    def test_and_condition_type(self):
        cond = Condition("stop", [_iface()], condition_type="and")
        assert cond.condition_type == "and"

    def test_invalid_condition_type_raises(self):
        with pytest.raises(AssertionError):
            Condition("stop", [_iface()], condition_type="xor")

    def test_file_name(self):
        cond = Condition("my_cond", [_iface()])
        assert cond.file_name() == "my_cond.txt"

    def test_write_creates_file(self, tmp_path):
        cond = Condition("stopcond", [_iface(val=3, compare=Comparison.GEQ)])
        cond.write(tmp_path)
        out = (tmp_path / "stopcond.txt").read_text()
        assert "action = stop_or" in out
        assert "condition1" in out

    def test_write_and_condition(self, tmp_path):
        cond = Condition("stopcond", [_iface()], condition_type="and")
        cond.write(tmp_path)
        out = (tmp_path / "stopcond.txt").read_text()
        assert "action = stop_and" in out

    def test_write_multiple_interfaces(self, tmp_path):
        iface1 = _iface(val=2, compare=Comparison.GEQ)
        iface2 = _iface(val=6, compare=Comparison.LEQ)
        cond = Condition("multi", [iface1, iface2])
        cond.write(tmp_path)
        out = (tmp_path / "multi.txt").read_text()
        assert "condition1" in out
        assert "condition2" in out

    def test_get_order_params_deduplicates(self):
        op = _op(name="shared")
        iface1 = FFSInterface(op=op, val=2, compare=Comparison.GEQ)
        iface2 = FFSInterface(op=op, val=4, compare=Comparison.LEQ)
        cond = Condition("stop", [iface1, iface2])
        ops = cond.get_order_params()
        assert len(ops) == 1
        assert ops[0] is op


# ---------------------------------------------------------------------------
# order_params
# ---------------------------------------------------------------------------

class TestOrderParams:
    def test_single_interface(self):
        op = _op()
        iface = _iface(op=op)
        result = order_params(iface)
        assert result == [op]

    def test_deduplicates_same_name(self):
        op = _op(name="shared")
        iface1 = FFSInterface(op=op, val=1, compare=Comparison.GEQ)
        iface2 = FFSInterface(op=op, val=3, compare=Comparison.LEQ)
        result = order_params(iface1, iface2)
        assert len(result) == 1

    def test_distinct_ops_both_included(self):
        op_a = OrderParameter("a", "bond", [(0, 1)])
        op_b = OrderParameter("b", "bond", [(2, 3)])
        iface_a = FFSInterface(op=op_a, val=1, compare=Comparison.GEQ)
        iface_b = FFSInterface(op=op_b, val=1, compare=Comparison.GEQ)
        result = order_params(iface_a, iface_b)
        assert len(result) == 2
        names = {op.name for op in result}
        assert names == {"a", "b"}

    def test_empty_returns_empty(self):
        assert order_params() == []


# ---------------------------------------------------------------------------
# write_order_params
# ---------------------------------------------------------------------------

class TestWriteOrderParams:
    def test_writes_to_file(self, tmp_path):
        op = OrderParameter("op1", "bond", [(0, 1)])
        fp = tmp_path / "ops.txt"
        write_order_params(fp, op)
        content = fp.read_text()
        assert "order_parameter = bond" in content
        assert "name = op1" in content

    def test_appends_multiple_ops(self, tmp_path):
        op1 = OrderParameter("a", "bond", [(0, 1)])
        op2 = OrderParameter("b", "mindistance", [(2, 3)])
        fp = tmp_path / "ops.txt"
        write_order_params(fp, op1, op2)
        content = fp.read_text()
        assert content.count("order_parameter") == 2