"""
File created 29 March 2024 by josh to replace Observable class
"""
from __future__ import annotations

import copy
from typing import Union, Generator, Any


# TODO: observable class!!

def simulation_time(units=None, print_every=None, name=None):
    """
    Print the current simulation time
    """
    return Observable(name,
                      print_every,
                      ObservableColumn(
                          "step",
                          units=units
                      )).export()

def distance(particle_1=None, particle_2=None, PBC=None, print_every=None, name=None):
    """
    Calculate the distance between two (groups) of particles
    """
    return Observable(name,
                      print_every,
                      ObservableColumn(
                          "distance",
                          particle_1=particle_1,
                          particle_2=particle_2,
                          PBC=PBC
                      )).export()


def hb_list(print_every: Union[None, int] = None,
            name=None,
            only_count=None) -> dict:
    """
    Compute the number of hydrogen bonds between the specified particles
    """
    return Observable(name,
                      print_every,
                      ObservableColumn(
                          "hb_list",
                          order_parameters_file="hb_list.txt",
                          only_count=only_count
                      )).export()


def particle_position(particle_id=None, orientation=None, absolute=None, print_every=None, name=None)->dict:
    """
    Return the x,y,z postions of specified particles
    """
    return Observable(name,
                      print_every,
                      ObservableColumn(
                          "particle_position",
                          particle_id=particle_id,
                          orientation=orientation,
                          absolute=absolute
                      )).export()


def potential_energy(print_every=None, split=None, name=None, precision=6, general_format='true') -> dict:
    """
    Return the potential energy
    """
    return Observable(name,
                      print_every,
                      ObservableColumn(
                          "potential_energy",
                          split=f"{split}",
                          precision=precision,
                          general_format=general_format
                      )).export()


def force_energy(print_every: Union[None, int] = None,
                 name: Union[None, str] = None,
                 print_group=None, 
                 precision=6, 
                 general_format=True) -> dict:
    """
    Return the energy exerted by external forces
    """
    if print_group is not None:
        col = ObservableColumn("force_energy", print_group=f"{print_group}",
                               precision=precision,
                               general_format=general_format)
    else:
        col = ObservableColumn("force_energy",
                               precision=precision,
                               general_format=general_format)
    return Observable(name, print_every, col).export()



def kinetic_energy(print_every=None, name=None) -> dict:
    """
    Return the kinetic energy
    """
    return Observable(name, print_every, ObservableColumn("kinetic_energy")).export()


def pair_energy(print_every=None, name=None, particle1_id=None, particle2_id=None, precision=6, general_format='true') -> dict:
    """
    Return the pair energy
    """
    if (particle1_id is not None) and (particle2_id is not None):
        col = ObservableColumn("pair_energy",
                                particle1_id=particle1_id,
                                particle2_id=particle2_id,
                                precision=precision,
                                general_format=general_format)
    elif (particle1_id is not None) and (particle2_id is None):
        raise ValueError("particle2_id must be specified if particle1_id is specified")
    else:
        col = ObservableColumn("pair_energy",
                                precision=precision,
                                general_format=general_format)
    
    return Observable(name, print_every, col).export()


class Observable:
    """
    class for observable methods

    """


    # deprecated: methods to create observable objects
    # going fwd pls call methods directly
    simulation_time = simulation_time
    
    distance = distance

    hb_list = hb_list

    particle_position = particle_position

    potential_energy = potential_energy

    force_energy = force_energy

    kinetic_energy = kinetic_energy

    # TODO: multitype observables?

    # class member vars
    # all observables will have these characteristics
    _file_name: str  # name of file to print data to
    _print_every: int  # interval at which to print
    _cols: list[ObservableColumn]
    _obs_kwargs: dict[str, Any]

    def __init__(self, name: str, print_every: Union[int, str, float],
                 *args: Union[ObservableColumn, dict[str, Any]],
                 **kwargs):
        self._file_name = name
        self._print_every = int(float(print_every))  # round float if you find one
        self._cols = [col if isinstance(col, ObservableColumn) else ObservableColumn(**col)
                      for col in args]
        self._obs_kwargs = kwargs

    def get_file_name(self) -> str:
        return self._file_name

    def set_file_name(self, newname: str):
        assert isinstance(newname, str)
        self._file_name = newname

    def get_print_every(self) -> int:
        return self._print_every

    def set_print_every(self, newval: int):
        assert isinstance(newval, int)
        self._print_every = newval

    def get_cols(self) -> Generator[ObservableColumn, None, None]:
        for col in self._cols:
            yield copy.deepcopy(col)

    def add_col(self, col: ObservableColumn, prepend: bool = True):
        if prepend:
            self._cols.insert(0, col)
        else:
            self._cols.append(col)

    def __len__(self):
        return len(self._cols)

    def __add__(self, other: ObservableColumn):
        return Observable(self.file_name, self.print_every, *self.cols)

    def export(self) -> dict:
        return {
            "output": self.to_dict()
        }

    def to_dict(self) -> dict:
        return {
            # needs to be str for frustrating reasons
            "print_every": f"{self.print_every}",
            "name": self.file_name,
            "cols": [
                col.export() for col in self.cols
            ],
            **self._obs_kwargs
        }

    def __repr__(self):
        """
        :return: string in oxDNA format representing the observable, with line breaks
        """
        obsstr = "{\n" + \
                f"\tprint_every = {self.print_every}\n" + \
                f"\tname = {self.file_name}\n"
        for icol, col in enumerate(self.cols):
            obsstr += f"\tcol_{icol+1} = " + "{\n"
            for key, value in col.export().items():
                obsstr += f"\t\t{key} = {value}\n"
            obsstr += "\t}\n"
        obsstr = obsstr + "}"
        return obsstr

    file_name = property(get_file_name, set_file_name)
    print_every = property(get_print_every, set_print_every)
    cols = property(get_cols)

    # TODO: make callable on simulation or something?


class ObservableColumn:
    _type_name: str  # name of ovservable type (e.g. "distance", "hb_list", "PatchyBonds")
    col_attrs: dict[str, str]

    def __init__(self, name: str, **kwargs: str):
        self._type_name = name
        self.col_attrs = kwargs

    def get_type_name(self) -> str:
        return self._type_name

    def export(self) -> dict:
        return {
            "type": self.type_name,
            # convert all column vals to strings for some reason
            **{key: f"{self.col_attrs[key]}" for key in self.col_attrs}
        }

    type_name = property(get_type_name)  # type name should not be settable
