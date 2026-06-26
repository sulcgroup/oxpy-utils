"""Tests for utils/order_parameter.py: OrderParameter dataclass, possible_states, create_state_mask."""
import json
from pathlib import Path

import numpy as np
import pytest

from oxpy_utils.utils.order_parameter import (
    OrderParameter,
    create_state_mask,
    possible_states,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bond_op(name="bonds", pairs=None) -> OrderParameter:
    if pairs is None:
        pairs = [(0, 1), (2, 3)]
    return OrderParameter(name, "bond", pairs)


def _bond_op_with_ifaces(pairs=None, interfaces=None) -> OrderParameter:
    if pairs is None:
        pairs = [(0, 1), (2, 3)]
    if interfaces is None:
        interfaces = [0.5, 1.5]
    return OrderParameter("iface_op", "bond", pairs, interfaces=interfaces)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestOrderParameterConstruction:
    def test_valid_bond(self):
        op = _bond_op()
        assert op.name == "bonds"
        assert op.order_parameter == "bond"
        assert op.pairs == [(0, 1), (2, 3)]

    def test_valid_mindistance(self):
        op = OrderParameter("dist", "mindistance", [(0, 5)])
        assert op.order_parameter == "mindistance"

    def test_invalid_type_raises(self):
        with pytest.raises(Exception):
            OrderParameter("bad", "hbond", [(0, 1)])

    def test_frozen(self):
        op = _bond_op()
        with pytest.raises((AttributeError, TypeError)):
            op.name = "new_name"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# __len__
# ---------------------------------------------------------------------------

class TestLen:
    def test_without_interfaces_is_pairs_plus_one(self):
        op = OrderParameter("op", "bond", [(0, 1), (2, 3), (4, 5)])
        assert len(op) == 4   # 3 pairs + 1

    def test_single_pair_len_two(self):
        op = OrderParameter("op", "bond", [(0, 1)])
        assert len(op) == 2

    def test_with_interfaces_is_interfaces_plus_one(self):
        op = _bond_op_with_ifaces(interfaces=[0.5, 1.5, 2.5])
        assert len(op) == 4   # 3 interfaces + 1


# ---------------------------------------------------------------------------
# nucleotides property
# ---------------------------------------------------------------------------

class TestNucleotides:
    def test_unique_indices_collected(self):
        op = OrderParameter("op", "bond", [(0, 1), (2, 3)])
        assert op.nucleotides == {0, 1, 2, 3}

    def test_shared_nucleotide_counted_once(self):
        op = OrderParameter("op", "bond", [(0, 1), (0, 2)])
        assert op.nucleotides == {0, 1, 2}


# ---------------------------------------------------------------------------
# index / __getitem__
# ---------------------------------------------------------------------------

class TestIndex:
    def test_forward_lookup(self):
        op = OrderParameter("op", "bond", [(0, 1), (2, 3)])
        assert op.index(0, 1) == 0
        assert op.index(2, 3) == 1

    def test_reversed_lookup(self):
        op = OrderParameter("op", "bond", [(0, 1), (2, 3)])
        assert op.index(1, 0) == 0

    def test_missing_pair_raises(self):
        op = OrderParameter("op", "bond", [(0, 1)])
        with pytest.raises(IndexError):
            op.index(9, 9)

    def test_getitem_tuple(self):
        op = OrderParameter("op", "bond", [(4, 5), (6, 7)])
        assert op[(4, 5)] == 0
        assert op[(6, 7)] == 1

    def test_getitem_bad_type_raises(self):
        op = _bond_op()
        with pytest.raises(TypeError):
            op["bad"]  # type: ignore[index]


# ---------------------------------------------------------------------------
# write / from_string roundtrip
# ---------------------------------------------------------------------------

class TestWriteFromString:
    def test_roundtrip_no_interfaces(self, tmp_path):
        op = OrderParameter("hb", "bond", [(0, 1), (2, 3)])
        fp = tmp_path / "op.txt"
        op.write(fp)
        block = fp.read_text()
        # Strip surrounding braces for from_string
        inner = block.strip().lstrip("{").rstrip("}")
        parsed = OrderParameter.from_string(inner)
        assert parsed.name == op.name
        assert parsed.order_parameter == op.order_parameter
        assert parsed.pairs == op.pairs

    def test_roundtrip_with_interfaces(self, tmp_path):
        op = OrderParameter("hb", "bond", [(0, 1)], interfaces=[0.5])
        fp = tmp_path / "op.txt"
        op.write(fp)
        block = fp.read_text()
        inner = block.strip().lstrip("{").rstrip("}")
        parsed = OrderParameter.from_string(inner)
        assert parsed.interfaces == pytest.approx([0.5])

    def test_read_file_multiple_ops(self, tmp_path):
        op1 = OrderParameter("a", "bond", [(0, 1)])
        op2 = OrderParameter("b", "mindistance", [(2, 3)])
        fp = tmp_path / "ops.txt"
        op1.write(fp)
        op2.write(fp)
        ops = OrderParameter.read_file(fp)
        assert len(ops) == 2
        assert ops[0].name == "a"
        assert ops[1].name == "b"


# ---------------------------------------------------------------------------
# to_dict / from_dict roundtrip
# ---------------------------------------------------------------------------

class TestDictRoundtrip:
    def test_no_interfaces(self):
        op = OrderParameter("x", "bond", [(1, 2), (3, 4)])
        d = op.to_dict()
        assert d["name"] == "x"
        assert d["order_parameter"] == "bond"
        assert d["pairs"] == [[1, 2], [3, 4]]
        assert "interfaces" not in d
        restored = OrderParameter.from_dict(d)
        assert restored == op

    def test_with_interfaces(self):
        op = OrderParameter("y", "bond", [(0, 1)], interfaces=[1.0, 2.0])
        restored = OrderParameter.from_dict(op.to_dict())
        assert restored.interfaces == pytest.approx([1.0, 2.0])

    def test_write_json_from_dict(self, tmp_path):
        op = OrderParameter("z", "mindistance", [(5, 6)])
        fp = tmp_path / "op.json"
        op.write_json(fp)
        raw = json.loads(fp.read_text())
        assert "z" in raw
        # from_dict uses the flat format produced by to_dict
        restored = OrderParameter.from_dict(op.to_dict())
        assert restored == op


# ---------------------------------------------------------------------------
# possible_states
# ---------------------------------------------------------------------------

class TestPossibleStates:
    def test_single_op_no_impossible_states(self):
        # 2 pairs with 4 unique nucleotides → all states 0..2 possible
        op = OrderParameter("op", "bond", [(0, 1), (2, 3)])
        states = possible_states(op)
        assert (0,) in states
        assert (1,) in states
        assert (2,) in states

    def test_single_op_filters_impossible(self):
        # 2 pairs sharing nucleotide 0 → 3 unique nucleotides
        # state (2,) requires 2×2=4 bonds > 3 nucleotides → impossible
        op = OrderParameter("op", "bond", [(0, 1), (0, 2)])
        states = possible_states(op)
        assert (2,) not in states
        assert (0,) in states
        assert (1,) in states

    def test_two_ops_cartesian(self):
        op1 = OrderParameter("a", "bond", [(0, 1)])  # states 0, 1
        op2 = OrderParameter("b", "bond", [(2, 3)])  # states 0, 1
        states = possible_states(op1, op2)
        assert (0, 0) in states
        assert (0, 1) in states
        assert (1, 0) in states
        assert (1, 1) in states

    def test_mindistance_op_all_states_included(self):
        # mindistance has no physical bond constraint so all states are possible
        op = OrderParameter("d", "mindistance", [(0, 5)])
        states = possible_states(op)
        assert len(states) == len(op)

    def test_no_ops_raises(self):
        with pytest.raises(ValueError):
            possible_states()


# ---------------------------------------------------------------------------
# create_state_mask
# ---------------------------------------------------------------------------

class TestCreateStateMask:
    def test_shape_matches_op_lengths(self):
        op1 = OrderParameter("a", "bond", [(0, 1), (2, 3)])  # len 3
        op2 = OrderParameter("b", "bond", [(4, 5)])           # len 2
        mask = create_state_mask(op1, op2)
        assert mask.shape == (3, 2)

    def test_accessible_states_are_true(self):
        op = OrderParameter("op", "bond", [(0, 1), (2, 3)])
        mask = create_state_mask(op)
        for state in [(0,), (1,), (2,)]:
            assert mask[state] is np.bool_(True)

    def test_inaccessible_states_are_false(self):
        # 2 pairs sharing nucleotide → state (2,) is impossible
        op = OrderParameter("op", "bond", [(0, 1), (0, 2)])
        mask = create_state_mask(op)
        assert mask[(2,)] is np.bool_(False)

    def test_custom_accessible_states(self):
        op = OrderParameter("op", "bond", [(0, 1), (2, 3)])
        # Manually restrict to only state (1,)
        mask = create_state_mask(op, accessible_states=[(1,)])
        assert mask[(0,)] is np.bool_(False)
        assert mask[(1,)] is np.bool_(True)
        assert mask[(2,)] is np.bool_(False)

    def test_all_false_when_no_accessible_states(self):
        op = OrderParameter("op", "bond", [(0, 1)])
        mask = create_state_mask(op, accessible_states=[])
        assert not mask.any()