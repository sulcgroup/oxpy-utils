"""
Interface for forward flux sampling
"""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Union

from oxpy_utils.utils.order_parameter import OrderParameter
from ..oxdna_simulation import Simulation


class Comparison(Enum):
    LT = "<"
    GT = ">"
    LEQ = "<="
    GEQ = ">="
    # no equals


def write_order_params(op_file_name: Path, *args):
    for op in args:
        op.write(op_file_name)


@dataclass(frozen=True)
class FFSInterface:
    """
    Interface for forward flux sampling
    An interface is defined by some order parameter having a defined relation to a value
    A simulation passes through an interface simulation.orderparameter [compare] val
    changes from False to True

    todo: allow for use of an observable as per Lorenzo's ANNaMo work
    https://github.com/lorenzo-rovigatti/ANNaMo_files/tree/main/examples/FFS
    """

    # name of parameter which is used to define this interface
    op: OrderParameter = field()
    # value to compare it to
    val: Any = field()
    # comparison type
    compare: Comparison = field()

    def __invert__(self) -> FFSInterface:
        """
        Returns a copy of this interface, but with an inverted comparison operator
        """
        if self.compare == Comparison.LT:
            newop = Comparison.GEQ
        elif self.compare == Comparison.GT:
            newop = Comparison.LEQ
        elif self.compare == Comparison.LEQ:
            newop = Comparison.GT
        elif self.compare == Comparison.GEQ:
            newop = Comparison.LT
        else:
            raise Exception(f"unrecognized operator {self.compare}")

        return FFSInterface(self.op, self.val, newop)

    def flip(self) -> FFSInterface:
        """
        similar to __invert__ but instead of the logical opposite it reverses
        the direction of the boundry in phase-space. if that makes any sense
        """

        if self.compare == Comparison.LT:
            newop = Comparison.GT
        elif self.compare == Comparison.GT:
            newop = Comparison.LT
        elif self.compare == Comparison.LEQ:
            newop = Comparison.GEQ
        elif self.compare == Comparison.GEQ:
            newop = Comparison.LEQ
        else:
            raise Exception(f"unrecognized operator {self.compare}")

        return FFSInterface(self.op, self.val, newop)

    def test(self, val: Union[float, Simulation]) -> bool:
        if isinstance(val, float):
            if self.compare == Comparison.LT:
                return val < self.val
            elif self.compare == Comparison.GT:
                return val > self.val
            elif self.compare == Comparison.LEQ:
                return val <= self.val
            elif self.compare == Comparison.GEQ:
                return val >= self.val
            else:
                raise Exception(f"unrecognized operator {self.compare}")
        else:
            return self.test(self.op.compute_value(val))

    def __str__(self):
        return f"{self.op.name} {self.compare.value} {self.val}"

@dataclass(frozen=True)
class Condition:
    """
    stores information about a forward-flux-sampling condition
    the condition consists of a set of interfaces
    """

    # condition name, for writing a file
    condition_name: str = field()

    # or-deliniated interfaces
    interfaces: list[FFSInterface] = field()
    condition_type: str = field(default="or")

    def __post_init__(self):
        assert self.condition_type in ["or", "and"], f"Invalid condition type {self.condition_type}"

    def write(self, write_dir: Path):
        with (write_dir / self.file_name()).open("w") as f:
            f.write(f"action = stop_{self.condition_type}\n")
            for n, interface in enumerate(self.interfaces):
                f.write(f"condition{n + 1} = " + "{\n" +
                        f"{interface.op.name} {interface.compare.value} {interface.val}" +
                        "\n}\n")

    def file_name(self) -> str:
        return f"{self.condition_name}.txt"

    def get_order_params(self) -> list[OrderParameter]:
        return order_params(*self.interfaces)


def order_params(*args: FFSInterface) -> list[OrderParameter]:
    """
    lists all order parameters used in the interfaces passed as params
    """

    ops = []
    op_names = set()  # use name set to avoid pass-by-value bullshit
    for interface in args:
        if interface.op.name not in op_names:
            op_names.add(interface.op.name)
            ops.append(interface.op)
    return ops
