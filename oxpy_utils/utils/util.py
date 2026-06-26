import contextlib
import os
from pathlib import Path
from typing import Union, Optional
import math


import numpy as np
import matplotlib.pyplot as plt

# general-purpose utility functions

def rotation_matrix(axis: np.ndarray, theta: float) -> np.ndarray:
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)
    theta = np.asarray(theta)
    axis = axis / math.sqrt(np.dot(axis, axis))
    a = math.cos(theta / 2)
    b, c, d = -axis * math.sin(theta / 2)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])

def process_path(p: Union[Path, str], prepend: Union[Path, None] = None) -> Path:
    if isinstance(p, str):
        p = Path(p)
    if not p.is_absolute():
        if p.parts[0] == "~":
            p = p.expanduser()
        elif p.parts[0] not in (".", "..") and prepend is not None:
            p = prepend / p
    return p

# todo: more precision
# source
# oxDNA units
# oxRNA units https://dna.physics.ox.ac.uk/index.php?title=RNA_model_introduction
NEWTONS_PER_UNIT = {
    "rna": 4.93e-11,
    "dna": 4.863e-11
}
METERS_PER_UNIT = {
    "rna": 8.4e-10,
    "dna": 8.518e-10
}
# todo: more units
DEGREES_K_PER_UNIT = {
    "rna": 3000,
    "dna": 3000
}
SECONDS_PER_UNIT = {
    "rna": 3.06e-12,
    "dna": 3.03e-12
}
KG_PER_UNIT = {
    "rna": 5.34e-25,
    "dna": 5.25e-25
}
JOULES_PER_UNIT = {
    "rna": 4.142e-20,
    "dna": 4.142e-20
}

# Okabe-Ito colorblind-friendly palette (RGBA, 8 colors)
# Distinguishable under deuteranopia, protanopia, and tritanopia.
# Reference: Okabe & Ito (2008), "Color Universal Design"
OKABE_ITO = np.array([
    [0.000, 0.447, 0.698, 1.],  # Blue       #0072B2
    [0.902, 0.624, 0.000, 1.],  # Orange     #E69F00
    [0.000, 0.620, 0.451, 1.],  # Bluish green #009E73
    [0.835, 0.369, 0.000, 1.],  # Vermillion  #D55E00
    [0.337, 0.706, 0.914, 1.],  # Sky blue   #56B4E9
    [0.941, 0.894, 0.259, 1.],  # Yellow     #F0E442
    [0.800, 0.475, 0.655, 1.],  # Reddish purple #CC79A7
    [0.000, 0.000, 0.000, 1.],  # Black      #000000
])


def generate_distinct_colors(n: int) -> np.ndarray:
    """
    Return n colorblind-friendly RGBA colors.

    Uses the Okabe-Ito palette for n ≤ 8; cycles through it with lightness
    variation for larger n.
    """
    if n <= len(OKABE_ITO):
        return OKABE_ITO[:n]

    # For n > 8: cycle through Okabe-Ito, alternating lighter/darker variants
    base = OKABE_ITO[:, :3]  # RGB only
    colors = []
    for i in range(n):
        rgb = base[i % len(base)]
        # alternate between original and a lightened version
        if (i // len(base)) % 2 == 1:
            rgb = np.clip(rgb + 0.35 * (1.0 - rgb), 0.0, 1.0)
        colors.append(np.append(rgb, 1.0))
    return np.array(colors)

def si_units(measurement: Union[str, float, np.array], interaction_type: str, measurement_type, to: Optional[str] = None) -> float:
    interaction_type = interaction_type.lower()
    if measurement_type == "T" or measurement_type.lower() == "temperature":
        if to is not None and to == "C":
            return si_units(measurement, interaction_type, "T", "K") - 273.15
        elif to == "K":
            return DEGREES_K_PER_UNIT[interaction_type] * measurement
        else:
            raise ValueError(f"Unsupported temperature conversion to {to}")
    elif measurement_type == "distance" or measurement_type == "d" or measurement_type == "du":
        return METERS_PER_UNIT[interaction_type] * measurement


def ox_units(measurement: Union[str, float], interaction_type: str, units: Optional[str] = None) -> float:
    """
    generic function to convert a measurement to oxDNA or oxRNA units
    for hybrid model, we incorrectly assume units are the same
    :param interaction_type: "dna" or "rna", because the two use slightly different units
    :param units: if measurement is passed as a float or a string with no units, look here
    """
    interaction_type = interaction_type.lower()
    if units is None:
        units = "".join([c for i,c in enumerate(measurement) if not c.isdigit()]).strip()
        if not units:
            raise ValueError(f"Measurement was provided as `{measurement}` but no units were provided!")
        measurement = measurement[:-len(units)]

    measurement = float(measurement)
    # SI Units
    if units == "N":
        return measurement / NEWTONS_PER_UNIT[interaction_type]
    elif units == "s":
        return measurement / SECONDS_PER_UNIT[interaction_type]
    elif units == "K":
        return measurement / DEGREES_K_PER_UNIT[interaction_type]
    elif units == "J":
        return measurement / JOULES_PER_UNIT[interaction_type]
    elif units == "kg":
        return measurement / KG_PER_UNIT[interaction_type]
    elif units == "m":
        return measurement / METERS_PER_UNIT[interaction_type]
    # prefixes (only common ones at scale)
    # force
    elif units == "nN":
        # convert 1000x the measurement from pN
        return ox_units(measurement * 1e-9, interaction_type, "N")
    elif units == "pN":
        # 1 force sim unit s.u  = 48.6 piconewtons units
        return ox_units(measurement * 1e-12, interaction_type, "N")
    # distance
    elif units == "nm":
        return ox_units(measurement * 1e-9, interaction_type, "m")
    elif units == "um" or units == "μm": # allow greek letter mu or latin "u"
        return ox_units(measurement * 1e-6, interaction_type, "m")
    elif units == "pm": # is this even relevant?
        return ox_units(measurement * 1e-12, interaction_type, "m")
    ## temperature
    elif units == "C":
        return ox_units(measurement + 273.15, interaction_type, "K")
    elif units == "F":
        raise ValueError("I said a REAL unit of measurement")
    # time
    elif units == "ns":
        return ox_units(measurement * 1e-9, interaction_type, "s")
    elif units == "ps":
        return ox_units(measurement * 1e-12, interaction_type, "s")
    # energy, good god
    elif units == "nJ":
        return ox_units(measurement * 1e-9, interaction_type, "J")
    elif units == "pJ":
        return ox_units(measurement * 1e-12, interaction_type, "J")
    elif units == "fJ":
        return ox_units(measurement * 1e-15, interaction_type, "J")
    elif units == "aJ":
        return ox_units(measurement * 1e-18, interaction_type, "J")
    elif units == "zJ":
        return ox_units(measurement * 1e-21, interaction_type, "J")

    else:
        raise ValueError(f"Invalid or unsupported unit {units}")