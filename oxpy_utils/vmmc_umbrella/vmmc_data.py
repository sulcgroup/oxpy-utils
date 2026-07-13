from  __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd

from ..utils.order_parameter import OrderParameter


@dataclass(frozen=True)
class VMMCData:
    """
    dataclass wrapper for a Pandas dataframe containing VMMC data
    todo: better class name?
    Note that indices in order parameter correspond to base pair indices, while
    the index in our dataframe here is number of bonds formed (w/ no regard to which bases form them)
    Attributes:
        - op: Reference to an OrderParameter object containing the order parameter data.
        - df: A pandas DataFrame storing the data, with the number of hydrogen bonds as the index.
        - step: The simulation time step.
    """
    # Attributes
    op: OrderParameter
    df: pd.DataFrame
    step: int # why is this here?

    _temperatures: list[float] = None
    _total_count: int = None
    _total_unbiased_count: int = None

    def __post_init__(self):
        # Compute and store the values for the properties
        object.__setattr__(self, '_total_count', self.counts.sum())
        self.df["sampling_prob"] = self.counts / self._total_count
        object.__setattr__(self, '_total_unbiased_count', self.unbiased_counts.sum())
        self.df["unbiased_sampling_prob"] = self.unbiased_counts / self._total_unbiased_count

        # Precompute free energy
        if not (self.df["unbiased_sampling_prob"].values > 0).any():
            # todo: custom exception for easier catch
            raise ValueError("All unbiased sampling probabilities are zero, cannot compute free energy!")
        epsilon = 1e-15 * self.df["unbiased_sampling_prob"][self.df["unbiased_sampling_prob"].values>0].min()

        self.df["free_energy"] = -np.log(self.df["unbiased_sampling_prob"] + epsilon)

        # Precompute temperatures
        temperatures = [float(col.replace("unbiased_count_", "")) for col in self.df.columns if
                        col.startswith("unbiased_count_")]
        object.__setattr__(self, '_temperatures', temperatures)

        for T in temperatures:
            self.df[f"unbiased_count_{T}_prob"] = self.df[f"unbiased_count_{T}"] / self.df[f"unbiased_count_{T}"].sum()
            self.df[f"unbiased_count_{T}_free"] = -np.log(self.df[f"unbiased_count_{T}_prob"] + epsilon)

        # a few properties to quick-access count and unbiased count
        # counts of timepoints simulation spent in each state (bond count)

    h_bonds = property(lambda self: self.df.index)
    counts = property(lambda self: self.df['count'])
    # unweighted percent that the system spends in each state
    sampling_prob = property(lambda self: self.df["sampling_prob"])
    # unbiased count (counts accounting for weighting)
    unbiased_counts = property(lambda self: self.df['unbiased_count'])
    # noramlized unbiased count
    unbiased_sampling_prob = property(lambda self: self.df["unbiased_sampling_prob"])
    # free energy
    free_energy = property(lambda self: self.df["free_energy"])

    # counts
    total_count = property(lambda self: self._total_count)
    total_unbiased_count = property(lambda self: self._total_unbiased_count)

    @property
    def temperatures(self):
        return self._temperatures

    def __getitem__(self, item: Union[str, int, tuple[int, int]]):
        """
        quick accessor for dataframe data
        """
        if type(item) == tuple:
            assert len(item) == 2, "Invalid accessor!"
            if type(item) == tuple:
                identifier, T = item
                # accessing by...  temperature, order param bond???
                assert type(T) == float
                assert type(identifier) == int
                return self[T][identifier]
        elif type(item) == str:
            return self.df[item[0]]
        # shorthand for row indexing
        elif type(item) == int:
            return self.df.loc[item]
        elif type(item) == float:
            # accessing by temperature
            return self.df[f"unbiased_count_{item}"]

        else:
            raise Exception(f"You passed... {item} to VMMCData.[] ???")

    def __add__(self, other: VMMCData) -> VMMCData:
        if not isinstance(other, VMMCData):
            raise ValueError("Can only sum VMMCData with another VMMCData!")
        if self.op != other.op:
            raise ValueError("Can only sum VMMCData with same order parameter!")
        counts = self.counts + other.counts
        unbiased_counts = self.unbiased_counts + other.unbiased_counts
        return VMMCData(df=pd.DataFrame({
                "h_bonds": self.h_bonds,
                "count": counts,
                "unbiased_count": unbiased_counts
            }),
            op=self.op,
            step=(self.step + other.step) // 2)

def read_vmmc_data(file: Path, op: OrderParameter, T_precision: int = 1) -> VMMCData:
    """
    Class method to read VMMC data from a file and return an instance of VMMCData.
    """
    assert file.is_file(), f"Path {str(file)} does not point to a file"
    assert op.order_parameter is not None and op.order_parameter == "bond", "Cannot construct VMMC with order parameter other than num Hbonds"

    # Open file and read header
    with file.open("r") as f:
        # Parse metadata line
        metadata_parts = f.readline().strip().split(";")
        simulation_time = int(metadata_parts[0].split("=")[1].strip())
        # Temperatures are in oxDNA simulation units
        sim_temp_units = list(map(float, metadata_parts[1].split(":")[1].strip().split()))
        # Convert simulation temperature units to Celsius
        # avoid floating point errors by replacing them with rounding errors
        temperatures = [round((temp_unit * 3000) - 273.15, T_precision) for temp_unit in sim_temp_units]

    # Read the rest of the file as a pandas dataframe
    df = pd.read_csv(file, skiprows=1, header=None, sep=r'\s+')
    assert len(op.pairs) == len(df.index) - 1,\
        "Number of order parameter pairs are incompatible with data histogram shape!"
    df.columns = ["h_bonds", "count", "unbiased_count", *[f"unbiased_count_{T}" for T in temperatures]]

    # Ensure correct data types
    df["h_bonds"] = df["h_bonds"].astype(int)
    df["count"] = df["count"].astype(int)  # Handles large numbers
    df.set_index("h_bonds", inplace=True)

    return VMMCData(
        op=op,
        df=df,
        step=simulation_time
    )

def average_vmmc_data(data_hists: list[VMMCData]) -> VMMCData:
    """
    averages together a group of histograms
    """
    assert len(data_hists) > 1
    counts = np.sum([data.counts for data in data_hists], axis=0)
    unbiased_counts = np.sum([data.unbiased_counts for data in data_hists], axis=0)
    return VMMCData(df=pd.DataFrame({
        "h_bonds": data_hists[0].h_bonds,
        "count": counts,
        "unbiased_count": unbiased_counts
    }),
    op=data_hists[0].op,
    step=np.average([data.step for data in data_hists]))
