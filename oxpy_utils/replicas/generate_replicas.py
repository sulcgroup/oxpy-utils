from __future__ import annotations

import itertools
from pathlib import Path
from os.path import exists
from typing import Any, Union, Callable, Type, Generator

from oxpy_utils.oxdna_simulation import Simulation


class Replicas:
    """
    set of replicas of a single simulation
    """
    prefix: str  # prefix which will be applied to replica directories
    file_dir: Path  # source files for replicas
    nreplicas: int # number of replicas

    simulations: list[Simulation]
    sim_dir: Path
    SimulationClass: Type

    # mapping of simulation names to sets of associated derived analytial stuff
    __derived_statistics: dict[tuple[Any, ...], Any]
    #
    stat_funcs: dict[str, Callable[[Replicas, ...], Any]]

    def __init__(self, SimulationClass: Type, conf_source: Path, sim_dir: Path, nreplicas: int, prefix: str = "rep",
                 ):
        self.SimulationClass = SimulationClass
        if sim_dir is None:
            raise ValueError("sim_dir cannot be None")
        self.sim_dir = sim_dir
        self.prefix = prefix
        self.file_dir = conf_source
        self.nreplicas = nreplicas

        self.__derived_statistics = dict()
        self.stat_funcs = dict()

        # todo: switch to i? to not index from 1
        self.simulations = []

    def init(self):
        for i in range(self.nreplicas):
            if exists(f"{self.sim_dir}_{i}"):
                self.simulations.append(self.SimulationClass(self.file_dir,
                                                             f"{self.sim_dir}_{i}"))
            else:
                self.simulations.append(self.SimulationClass(self.file_dir,
                                                             self.sim_dir / f"{self.prefix}{i+1}") )

    def __getitem__(self, item: int) -> Simulation:
        return self.simulations[item]

    def __len__(self) -> int:
        return len(self.simulations)

    def __iter__(self) -> Generator[Simulation, None, None]:
        yield from self.simulations

    def construct_replicas(self, callback: Callable[[int, Simulation], None] = lambda i, s: None):
        """
        constructs replicas
        optional callback function will be applied to each simulation before calling  Simulation.build()
        """
        self.sim_dir.mkdir(exist_ok=True)
        for i, sim in enumerate(self.simulations):
            # envoke optional callback
            callback(i, sim)
            sim.build()

    def is_set_up(self):
        return len(self.simulations) == self.nreplicas

    def concat_trajs(self, concat_dir: Union[Path, str] = 'concat_dir'):
        """
        Concatenate the trajectory of multiple replicas and writes them to a directory
        """

        if type(concat_dir) == str:
            concat_dir = Path(concat_dir)
        if not concat_dir.is_absolute():
            concat_dir = self.sim_dir / concat_dir

        # construct simulation (which we will not run) in concat d0r
        concat_sim = self.SimulationClass(self.file_dir, concat_dir)
        concat_sim.input(trajectory_file="contat_traj.dat")
        concat_sim.build()

        with (concat_dir / "contat_traj.dat").open("wb") as outfile:
            for sim in self.simulations:
                # todo: dynamic naming for trajectory.bin file
                with (self.sim_dir / "trajectory.bin").open('rb') as infile:
                    outfile.write(infile.read())

    def get_stat(self, params: tuple[Any, ...], func: str):
        """
        compute statistic using one of the functions in self.stat_funcs
        """
        assert type(params) == tuple
        assert func in self.stat_funcs, f"Invalid statistic {func}"
        if params not in self.__derived_statistics:
            self.__derived_statistics[params] = dict()
        if func not in self.__derived_statistics[params]:
            self.__derived_statistics[params][func] = self.stat_funcs[func](self, *params)
        return self.__derived_statistics[params][func]

    def pickle_cache(self):
        pass # todo

    def unpickle_cache(self):
        pass # todo

    def clear_cache(self):
        self.__derived_statistics = {}

class ReplicaGroup:
    """
    Methods to generate multisystem replicas
    Each system is identified by a name
    """

    # distinct systems that we want to run multiple replicas of
    systems: dict[str, Replicas]
    SimulationClass: Type = Simulation


    def __init__(self, SimulationClass: Type = Simulation):
        self.systems = dict()
        self.SimulationClass = SimulationClass
        self.stat_funcs = {}
        self.__derived_statistics = {}

    def build(self):
        for system in self.systems.values():
            system.init()

    names = property(lambda self: self.systems.keys())

    def set_input_val(self, key: str, val):
        """
        batch function to set an input file value for all simulations
        """
        for sim in self.sim_list:
            sim.input[key] = val

    def get_sims_list(self) -> list[Simulation]:
        """
        lists all simulations in the replicas within this group of replicas
        """
        return list(itertools.chain.from_iterable([reps.simulations for reps in self.systems.values()]))

    def get_sims(self, name: str) -> list[Simulation]:
        return self.systems[name].simulations

    sim_list: list[Simulation] = property(get_sims_list)


    def multisystem_replica(self,
                            systems: list[tuple[str, Path,Path]],
                            n_replicas_per_system: int):
        """
        Create simulation replicas, with across multiple systems with diffrent inital files

        Parameters:
            systems (list): List of strings, where the strings are the name of the directory which will hold the inital files
            n_replicas_per_system (int): number of replicas to make per system
            file_dir_list (list): List of strings with path to intial files
            sim_dir_list (list): List of simulation directory paths
        """

        # todo: at least have the option of passing args as tuple list
        for systemname, sys_file_dir, sys_sim_dir in systems:
            if type(sys_file_dir) == str:
                sys_file_dir = Path(sys_file_dir)
            if type(sys_sim_dir) == str:
                sys_sim_dir = Path(sys_sim_dir)
            if systemname in self.systems:
                systemname = f"{systemname}_n"
            self.systems[systemname] = Replicas(self.SimulationClass,
                                                sys_sim_dir,
                                                n_replicas_per_system,
                                                f"{systemname}_n",
                                                sys_file_dir)

    def check_all_complete(self):
        for system in self.systems.values():
            if not system.is_complete():
                return False
        return True

    def concat_all_system_traj(self):
        "Concatenate the trajectory of multiple replicas for each system"
        for system in self.systems.values():
            system.concat_trajs()



    def __getitem__(self, item: Union[str, tuple]) -> Union[Any, list[Simulation], list[Any]]:
        """
        what the fuck
        """
        # if single string is provided
        if isinstance(item, str):
            # is the `item` provided the name of one of our groups?
            if item in self.systems:
                # return simulations relating to that group. ignore statistics
                return self.systems[item].simulations
            # is `item` the name of a statistic?
            elif item in self.__derived_statistics:
                # return the data for the statistic for all groups
                return [
                    self.get_stat(name, (), item)
                    for name in self.names
                ]
            else:
                raise Exception(f"{item} not simulation group indicator or analysis func")
        elif isinstance(item, tuple):
            # todo: replace with exception
            assert len(item) == 2, "Invalid invoker for getitem"
            name, second = item
            # if we passed a pair of items, and the second item is a string, that means
            # the second item in the pair should be the name of a statistic
            if type(second) == str:
                # if the first item is a tuple, we have passed `(name of a group, parameters for analysis function)`
                if type(name) == tuple:
                    # split tuple
                    name, params = name
                    # allow for passing single parameter as naked value, for convenience
                    if type(params) != tuple:
                        params = params, # am i crazy
                    return self.get_stat(name, params, second)
                else:
                    # todo: replace assertion w/ exception
                    assert type(name) == str, "Invalid invoker for getitem"
                    # if the first item is a tuple, we have our most standard indexer:
                    # the name of a statistic and the name of a group
                    # we can assume/pray that analysis function for stat has no additional params
                    return self.get_stat(name, (), second)
            else:
                assert type(second) == tuple
                # get f(name, *params) for every f
                return [self.get_stat(name, second, func) for func in self.stat_funcs.keys()]
        else:
            raise Exception(f"Invalid invoker for getitem {type(item)}")