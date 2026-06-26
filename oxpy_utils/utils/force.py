from __future__ import annotations

import collections
import copy
import json
from collections import namedtuple
from enum import Enum
from pathlib import Path
from typing import Union, Any, Iterable, Generator

import numpy as np

"""
Currently implemented external forces for this oxDNA wrapper.
these functions are from Matt's code and are preserved for backwards compatibility
"""

ForceTypeInfo = namedtuple("ForceTypeInfo", ["type_name",
                                             "type_req_params",
                                             "type_param_defaults"], defaults=[[]])

# todo: check args
class ForceType(Enum):
    MORSE = ForceTypeInfo("morse",
                          ["ref_particle", "a", "D", "r0", "PBC"])
    SKEW_TRAP = ForceTypeInfo("skew_trap",
                              ["ref_particle", "stdev", "r0", "shape", "PBC"])
    CENTER_OF_MASS = ForceTypeInfo("com",
                                   ["ref_list", "stiff", "r0", "PBC", "rate", "com_list"],
                                   [1.])
    MUTUAL_TRAP = ForceTypeInfo("mutual_trap",
                                ["ref_particle", "stiff", "r0", "PBC"],
                                [0.9, 1.2, True])
    STRING = ForceTypeInfo("string",
                           ["F0",  "dir", "rate"],
                           [0.])
    HARMONIC_TRAP = ForceTypeInfo("trap",
                                  ["pos0", "stiff", "rate", "dir"],
                                  [0., np.array([0.,0.,0.])])
    ROT_HARMONIC_TRAP = ForceTypeInfo("twist",
                                      ["stiff", "rate", "base", "pos0", "center", "axis", "mask"])
    REPULSION_PLANE = ForceTypeInfo("repulsion_plane",
                                    ["stiff", "direction", "position"])
    REPULSION_SPHERE = ForceTypeInfo("sphere",
                                     ["center", "stiff", "r0", "rate"],
                                     [1.])

def get_force_type(type_name: str) -> ForceType:
    for force_type in ForceType:
        if force_type.value.type_name == type_name:
            return force_type
    raise Exception(f"No force type named {type_name}")

class Force(collections.abc.Mapping):
    force_type: ForceType
    particle: int
    force_params: dict[str, Any]

    def __init__(self, **kwargs: Any):
        if "force_type" in kwargs:
            force_type = kwargs.pop("force_type")
        elif "type" in kwargs:
            force_type = kwargs.pop("type")
        else:
            raise Exception("No arguement specifiying force type")
        if isinstance(force_type, str):
            force_type = get_force_type(force_type)
        self.force_type = force_type
        if "particle" in kwargs:
            self.particle = kwargs["particle"]
            del kwargs["particle"]
        else:
            self.particle = -1
        force_params = kwargs
        for k, v in zip(force_type.value.type_req_params[-len(force_type.value.type_param_defaults):],
                        force_type.value.type_param_defaults):
            if k not in force_params:
                force_params[k] = v
        if not all(p in force_params for p in self.force_type.value.type_req_params):
            raise ValueError(f"Missing some parameters required for force {self.force_type.name}")
        if any(p not in self.force_type.value.type_req_params for p in force_params):
            raise ValueError(f"Excess parameters provided for force {self.force_type.name}")
        # todo: typing
        self.force_params = force_params

    def spacial_transform(self, translation: np.ndarray = np.zeros(shape=(3,)), rotation: np.ndarray = np.identity(3)) -> Force:
        """
        applies a spacial transform to the force, returns the transformed force.
        """
        f2 = copy.deepcopy(self)
        if "pos0" in f2.force_params:
            f2.force_params["pos0"] = rotation * f2.force_params["pos0"] + translation
        if "dir" in f2.force_params:
            f2.force_params["dir"] = rotation * f2.force_params["dir"]
        if "center" in f2.force_params:
            f2.force_params["center"] = rotation * f2.force_params["center"] + translation
        if "axis" in f2.force_params:
            f2.force_params["axis"] = rotation * f2.force_params["axis"]
        if "mask" in f2.force_params:
            f2.force_params["mask"] = rotation * f2.force_params["mask"] # translate this?? idk quite what it is
        return f2

    # impl these next three so we can use dict unpacking to convert to json
    def __len__(self):
        return len(self.force_params) + 2

    def __iter__(self):
        yield "type"
        yield "particle"
        yield from self.force_params

    def __getitem__(self, key:str):
        if key == "type":
            return self.force_type.value.type_name
        elif key == "particle":
            return str(self.particle)
        elif key in self.force_params:
            if not isinstance(self.force_params[key], np.ndarray):
                return str(self.force_params[key])
            else:
                assert self.force_params[key].shape == (3,) # should always be 3-coords?
                return ",".join([str(v) for v in self.force_params[key]])

        else:
            raise KeyError(f"No parameter {key} for force type {self.force_type.value.type_name}")

    def __setitem__(self, key: str, value):
        # shouldn't allow setting type
        if key == "particle":
            #  dunno why you'd do this
            self.particle = value
        elif key in self.force_params:
            self.force_params[key] = value
        else:
            raise KeyError(f"No parameter {key} for force type {self.force_type.value.type_name}")

def zip_strands(strand1: Iterable[int], strand2: Iterable[int]) -> Generator[Force, None, None]:
    """
    creates forces to bind two strands together
    """
    for b1, b2 in zip(strand1, strand2):
        # use pbc=false to avoid wonky pbc issues
        yield Force(force_type=ForceType.MUTUAL_TRAP, particle=b1, ref_particle=b2, PBC=False)
        yield Force(force_type=ForceType.MUTUAL_TRAP, particle=b2, ref_particle=b1, PBC=False)

def morse(particle=None, ref_particle=None, a=None, D=None, r0=None, PBC=None):
    """Morse potential"""
    return ({"force": {
        "type": ForceType.MORSE.value.type_name,
        "particle": f'{particle}',
        "ref_particle": f'{ref_particle}',
        "a": f'{a}',
        "D": f'{D}',
        "r0": f'{r0}',
        "PBC": f'{PBC}',
    }
    })

def skew_force(particle=None, ref_particle=None, stdev=None, r0=None, shape=None, PBC=None):
    """Skewed Gaussian potential"""
    return ({"force": {
        "type": ForceType.SKEW_TRAP.value.type_name,
        "particle": f'{particle}',
        "ref_particle": f'{ref_particle}',
        "stdev": f'{stdev}',
        "r0": f'{r0}',
        "shape": f'{shape}',
        "PBC": f'{PBC}'
    }
    })

def com_force(com_list=None, ref_list=None, stiff=None, r0=None, PBC=None, rate=None):
    """Harmonic trap between two groups"""
    return ({"force": {
        "type": ForceType.CENTER_OF_MASS.value.type_name,
        "com_list": f'{com_list}',
        "ref_list": f'{ref_list}',
        "stiff": f'{stiff}',
        "r0": f'{r0}',
        "PBC": f'{PBC}',
        "rate": f'{rate}'
    }
    })

def mutual_trap(particle=None, ref_particle=None, stiff=None, r0=None, PBC=None):
    """
    A spring force that pulls a particle towards the position of another particle

    Parameters:
        particle (int): the particle that the force acts upon
        ref_particle (int): the particle that the particle will be pulled towards
        stiff (float): the force constant of the spring (in simulation units)
        r0 (float): the equlibrium distance of the spring
        PBC (bool): does the force calculation take PBC into account (almost always 1)
    """
    return ({"force": {
        "type": ForceType.MUTUAL_TRAP.value.type_name,
        "particle": particle,
        "ref_particle": ref_particle,
        "stiff": stiff,
        "r0": r0,
        "PBC": PBC
    }
    })

def string(particle, f0, rate, direction):
    """
    A linear force along a vector

    Parameters:
        particle (int): the particle that the force acts upon
        f0 (float): the initial strength of the force at t=0 (in simulation units)
        rate (float or SN string): growing rate of the force (simulation units/timestep)
        dir ([float, float, float]): the direction of the force
    """
    return ({"force": {
        "type": ForceType.STRING.value.type_name,
        "particle": particle,
        "f0": f0,
        "rate": rate,
        "dir": direction
    }})

def harmonic_trap(particle: int,
                  pos0: Union[list[float], np.ndarray],
                  stiff: float,
                  rate: Union[float, None]=None,
                  direction: Union[list[float], np.ndarray, None]=None):
    """
    A linear potential well that traps a particle

    Parameters:
        particle (int): the particle that the force acts upon
        pos0 ([float, float, float]): the position of the trap at t=0
        stiff (float): the stiffness of the trap (force = stiff * dx)
        rate (float): the rate of movement of the trap position (simulation units/time step)
        direction ([float, float, float]): the direction of movement of the trap
    """
    assert len(pos0) == 3
    if isinstance(pos0, np.ndarray):
        pos0 = list(pos0)

    force_dict = {
        "type": ForceType.HARMONIC_TRAP.value.type_name,
        "particle": particle,
        "stiff": stiff,
        "pos0": pos0
    }
    # `direction` and `force` are optional params only used for traps which move over time
    if direction is not None:
        assert len(direction) == 3
        if isinstance(direction, np.ndarray):
            direction = list(direction)
        force_dict["rate"] = rate
        force_dict["dir"] = direction
    return {"force": force_dict}

def rotating_harmonic_trap(particle: int, stiff: float, rate: float, base: float, pos0: list[float], center, axis, mask):
    """
    A harmonic trap that rotates in space with constant angular velocity

    Parameters:
        particle (int): the particle that the force acts upon
        pos0 ([float, float, float]): the position of the trap at t=0
        stiff (float): the stiffness of the trap (force = stiff * dx)
        rate (float): the angular velocity of the trap (simulation units/time step)
        base (float): initial phase of the trap
        axis ([float, float, float]): the rotation axis of the trap
        mask([float, float, float]): the masking vector of the trap (force vector is element-wise multiplied by mask)
    """
    return {"force": {
        "type": ForceType.ROT_HARMONIC_TRAP.value.type_name,
        "particle": particle,
        "pos0": pos0,
        "stiff": stiff,
        "rate": rate,
        "base": base,
        "center": center,
        "axis": axis,
        "mask": mask
    }}

def repulsion_plane(particle: int, stiff: float, direction: list[float], position: float):
    """
    A plane that forces the affected particle to stay on one side.

    Parameters:
        particle (int): the particle that the force acts upon.  -1 will act on whole system.
        stiff (float): the stiffness of the trap (force = stiff * distance below plane)
        dir ([float, float, float]): the normal vecor to the plane
        position(float): position of the plane (plane is d0*x + d1*y + d2*z + position = 0)
    """
    return ({"force": {
        "type": ForceType.REPULSION_PLANE.value.type_name,
        "particle": particle,
        "stiff": stiff,
        "dir": direction,
        "position": position
    }})

def repulsion_sphere(particle, center, stiff, r0, rate=1):
    """
    A sphere that encloses the particle
    todo: should particle param be used?
    Parameters:
        particle (int): the particle that the force acts upon
        center ([float, float, float]): the center of the sphere
        stiff (float): stiffness of trap
        r0 (float): radius of sphere at t=0
        rate (float): the sphere's radius changes to r = r0 + rate*t
    """
    return ({"force": {
        "type": ForceType.REPULSION_SPHERE.value.type_name,
        "center": center,
        "stiff": stiff,
        "r0": r0,
        "rate": rate
    }})

def load_forces_from_txt(fp: str | Path) -> list[Force]:
    """
    todo: would it be better to have a generator?
    loads a list of forces from a txt file
    :param fp: file path to load from
    :return: list of Force objects
    """
    forces = []

    if isinstance(fp, str):
        fp = Path(fp)
    assert fp.suffix == ".txt"
    with fp.open("r") as f:
        force_dict = dict()
        for line in f:
            # can actually skip open braces
            if line.strip() not in ["", "{"]:
                if "}" not in line.strip():  # no force types have nested objects.... right?
                    key, val = line.strip().split("=")
                    key = key.strip()
                    val = val.strip()
                    force_dict[key] = val
                else:
                    # add force and create new force dict
                    # todo: names??
                    forces.append(Force(**force_dict))
                    force_dict.clear()
    return forces

def load_forces_from_json(fp: str | Path) -> list[Force]:
    """
    TODO: TEST THIS!!!!
    loads a list of forces from a json file
    :param fp: file path to load from
    :return: list of Force objects
    """
    if isinstance(fp, str):
        fp = Path(fp)
    with fp.open("r") as f:
        data = json.load(f)
    forces: list[Force] = []
    for force_name, force_data in data.items():
        forces.append(Force(**force_data))
    return forces