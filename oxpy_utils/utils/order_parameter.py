from __future__ import annotations

import itertools
import json
from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Union, Iterable, Optional, Any

import numpy as np

# as of right now (Dec. 2024) there are only two allowed order parameters:
# "bond" and "mindistance". they are hardcoded in `oxDNA/src/Utilites/OrderParameters.h`
ALLOWED_ORDER_PARAMETERS = [
    "mindistance",
    "bond",
]


@dataclass(frozen=True)
class OrderParameter:
    """
    an order parameter for forward-flux-sampling
    """
    name: str = field()
    order_parameter: str = field()  # specific set of options in oxdna
    pairs: list[tuple[int, int]] = field()  # list of pairs of residue indices
    # partition the order parameter into interfaces. each item is a value for the order parameter
    interfaces: Optional[list[Union[int, float]]] = field(default=None)

    def __post_init__(self):
        if self.order_parameter not in ALLOWED_ORDER_PARAMETERS:
            raise Exception(f"Invalid order parameter {self.order_parameter}")
        # todo: type checking for `pairs`

    def index(self, p1: int, p2: int) -> int:
        """
        """
        try:
            return self.pairs.index((p1, p2))
        except ValueError:
            # maybe order is reversed?
            try:
                return self.pairs.index((p2, p1))
            except ValueError:
                raise IndexError(f"Order parameter name={self.name} (type {self.order_parameter} has no pair {p1}, {p2}")

    def __getitem__(self, item: Union[int, tuple[int, int]]) -> Union[int, tuple[int,int]]:
        """

        """
        if type(item) == int:
            # assume indexing pairs
            return self[item[0] ]
        elif type(item) == tuple and len(item) == 2:
            return self.index(*item)
        else:
            raise TypeError(f"Trying to index OrderParameter object with... {item}????")

    def write(self, fp: Path):
        """
        writes the order parameters to the file at the given file path, appending any exiisting content
        :param fp: file path at which to write order parameter
        """
        with fp.open("a+") as f:
            f.write("{\n")
            f.write(f"\torder_parameter = {self.order_parameter}\n")
            f.write(f"\tname = {self.name}\n")
            assert len(self.pairs) > 0
            for (n, (base1, base2)) in enumerate(self.pairs):
                f.write(f"\tpair{n + 1} = {base1}, {base2}\n")
            if self.interfaces is not None:
                f.write(f"\tinterfaces = {','.join([str(iface) for iface in self.interfaces])}\n")
            f.write("}\n")

    def write_json(self, fp: Path):
        """

        """
        with fp.open("w") as f:
            json.dump({
                self.name: {
                    "order_parameter": self.order_parameter,
                    "pairs": [list(p) for p in self.pairs],
                }
            }, f)

    def __len__(self):
        """
        returns for the number of pairs in the order parameter
        """
        if self.interfaces is None:
            return len(self.pairs) + 1
        else:
            return len(self.interfaces) + 1

    @property
    def nucleotides(self) -> set[int]:
        return set(itertools.chain(*self.pairs))

    @classmethod
    def from_string(cls, block: str):
        """
        Parses a block of text and returns an OrderParameter instance.
        """
        order_parameter_match = re.search(r"order_parameter = (\w+)", block)
        name_match = re.search(r"name = (\w+)", block)
        pairs_matches = re.findall(r"pair\d+ = (\d+), (\d+)", block)

        if not order_parameter_match or not name_match:
            raise ValueError("Block is not in the expected format")
        interfaces_match = re.search(r"interfaces = ([\d\.]+(?:,[\d\.]+)*)", block)
        # Extract the order parameter and name
        order_parameter = order_parameter_match.group(1)
        name = name_match.group(1)

        # Extract pairs
        pairs = [(int(base1), int(base2)) for base1, base2 in pairs_matches]

        # Create an instance of OrderParameter
        if interfaces_match:
            interfaces = [float(interface) for interface in interfaces_match.group(1).split(",")]
            return cls(name=name,
                       order_parameter=order_parameter,
                       pairs=pairs,
                       interfaces=interfaces)
        else:
            return cls(name=name, order_parameter=order_parameter, pairs=pairs)

    @classmethod
    def read_file(cls, fp: Path):
        """
        Reads a file containing multiple order parameters and returns a list of OrderParameter instances.
        """
        with fp.open("r") as f:
            content = f.read()

        # Split content into blocks, assuming each block is surrounded by { }
        blocks = re.findall(r"\{(.*?)\}", content, re.DOTALL)

        # Parse each block and create an OrderParameter object
        order_parameters = [cls.from_string(block) for block in blocks]
        return order_parameters
    
    def to_dict(self) -> dict[str, Any]:
        """
        Serialize this OrderParameter into a JSON-serializable dict.
        Format:
        {
            "name": <str>,
            "order_parameter": <str>,
            "pairs": [[i, j], ...],
            "interfaces": [ ... ]   # optional
        }
        """
        d = {
            "name": self.name,
            "order_parameter": self.order_parameter,
            "pairs": [list(p) for p in self.pairs],
        }
        if self.interfaces is not None:
            d["interfaces"] = list(self.interfaces)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> OrderParameter:
        """
        Create an OrderParameter from a dictionary produced by to_dict().
        Accepts `pairs` as lists/tuples and `interfaces` as optional.
        """
        name = d["name"]
        order_parameter = d["order_parameter"]
        pairs_raw = d.get("pairs", [])
        pairs = [(int(a), int(b)) for a, b in pairs_raw]
        interfaces = None
        if "interfaces" in d and d["interfaces"] is not None:
            # keep them as floats if any contain a dot, else ints
            interfaces = [float(x) for x in d["interfaces"]]
        return cls(name=name, order_parameter=order_parameter, pairs=pairs, interfaces=interfaces)

    def write_json(self, fp: Path):
        """
        Write THIS OrderParameter to `fp` as a single JSON object, overwriting existing content.
        The file will contain a dict with the name as the top-level key, matching your previous format:
        {
            "<name>": {
               "order_parameter": ...,
               "pairs": [[...], ...],
               "interfaces": [...]
            }
        }
        """
        obj = {
            self.name: {
                "order_parameter": self.order_parameter,
                "pairs": [list(p) for p in self.pairs],
            }
        }
        if self.interfaces is not None:
            obj[self.name]["interfaces"] = list(self.interfaces)
        with fp.open("w") as f:
            json.dump(obj, f, indent=2)

def possible_states(*args: OrderParameter) -> list[tuple[int, ...]]:
    """
    Given multiple OrderParameter instances, returns a list of all possible states
    :args: OrderParameter instances
    :return: list of tuples representing accessible states
    """
    if len(args) == 0:
        raise ValueError("At least one OrderParameter must be provided")
    accessable_state_list = []
    for state in itertools.product(*[range(len(op)) for op in args]):
        for op, op_state in zip(args, state):
            # ignore distance order parameters
            is_possible = True
            if op.order_parameter == "bond":
                # if the number of bonds is greater than twice the number of unique bases in the order parameter, the
                # state is impossible
                if 2 * op_state > len(op.nucleotides):
                    is_possible = False
                    break
        # TODO: INTERACTIONS BETWEEN ORDER PARAMETERS

        if is_possible:
            accessable_state_list.append(state)
    return accessable_state_list


def create_state_mask(*args: OrderParameter, accessible_states=None) -> np.ndarray:
    """
    Creates a boolean mask for n-dimensional numpy array indexing based on possible states.

    :args: OrderParameter instances (same as passed to possible_states)
    :param accessible_states: optional list of accessible states to use instead of calculating all possible states
    :return: n-dimensional boolean numpy array where True indicates accessible states
    """
    if accessible_states is None:
        accessible_states = possible_states(*args)

    # Get the shape of the full state space
    shape = tuple(len(op) for op in args)

    # Initialize mask with all False
    mask = np.zeros(shape, dtype=bool)

    # Get all possible states

    # Set accessible states to True
    for state in accessible_states:
        mask[state] = True

    return mask
