from __future__ import annotations

import abc
import errno
import pickle
import sys
from abc import ABC, abstractmethod
import os
import shutil
from json import dumps, loads, dump, load
import signal
from shutil import SameFileError
from typing import Optional, Callable
import warnings

import pandas as pd
import oxpy
import multiprocessing as mp
import py
import ipywidgets as widgets
from IPython.display import display
import matplotlib.pyplot as plt
from time import sleep
import nvidia_smi
import timeit
import subprocess as sp
import traceback
import queue
import json

import numpy as np

from oxDNA_analysis_tools.UTILS.boilerplate import PathContext
from oxDNA_analysis_tools.UTILS.data_structures import TrajInfo, TopInfo
from oxDNA_analysis_tools.UTILS.get_confs import Configuration
from oxDNA_analysis_tools.UTILS.oxview import oxdna_conf
from oxDNA_analysis_tools.UTILS.RyeReader import describe, get_confs, write_conf, inbox
from oxDNA_analysis_tools.align import align as _oat_align
from oxDNA_analysis_tools.mean import mean as _oat_mean
from oxDNA_analysis_tools.deviations import deviations as _oat_deviations
from oxDNA_analysis_tools.centroid import centroid as _oat_centroid
from oxDNA_analysis_tools.pca import pca as _oat_pca
from oxDNA_analysis_tools.decimate import decimate as _oat_decimate
from oxDNA_analysis_tools.minify import minify as _oat_minify
from oxDNA_analysis_tools.output_bonds import output_bonds as _oat_output_bonds
from oxDNA_analysis_tools.rg import rg as _oat_rg

from .defaults.defaults import DefaultInput, SEQ_DEP_PARAMS, NA_PARAMETERS, RNA_PARAMETERS, get_default_input
from .structure_editor.dna_structure import DNAStructure, load_dna_structure
from .utils.force import *
from .utils.observable import Observable

# import cupy

class SimDirInfo(abc.ABC):
    """
    interface for file_dir and sim_dir methods
    Inheriting classes can either define protected variables to store
    values for file_dir and sim_dir or they can refer to other class member vars
    """

    @abstractmethod
    def get_file_dir(self) -> Path:
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def set_file_dir(self, p: Union[Path, str]):
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def get_sim_dir(self) -> Path:
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def set_sim_dir(self, p: Union[Path, str]):
        raise NotImplementedError("Subclasses must implement this method")

    # Properties for file_dir and sim_dir
    # all inheriting classes must include the following:
    '''
    file_dir = property(get_file_dir, set_file_dir)
    sim_dir = property(get_sim_dir, set_sim_dir)
    '''


class Simulation(SimDirInfo):
    # directory containing source files for simulation
    file_dir: Path
    # directory where simulations are run
    sim_dir: Path

    _sim_files: Union[None, SimFiles]
    _build_sim: Union[None, BuildSimulation]
    _input: Union[None, Input]
    _analysis: Union[None, Analysis]
    _protein: Union[None, Protein]
    _oxpy_run: Union[None, OxpyRun]
    _sequence_dependant: Union[None, SequenceDependant]
    _slurm_run: Union[None, SlurmRun]
    _forces: Union[None, SimForces]

    # # list of observables
    # # if i was consitant with Matt's design choices I would make this a subclass
    # # but I don't want to
    # _observables: dict[str, Observable]

    """
    Used to interactively interface and run an oxDNA simulation.

    Parameters:
        file_dir (str): Path to directory containing initial oxDNA dat and top files. Alternatively: a DNAStructure

        sim_dir (str): Path to directory where a simulation will be run using initial files.
    """

    def __init__(self,
                 file_dir: Union[str, Path, Simulation],
                 sim_dir: Union[str, Path, None] = None,
                 input_file_params: dict = {}):
        """
        Instance lower level class objects used to compose the Simulation class features.
        """

        # handle alternate param types for file_dir
        if isinstance(file_dir, Path):
            self.file_dir = file_dir
            self._build_sim = None
        elif isinstance(file_dir, str):
            self.file_dir = Path(file_dir)
            self._build_sim = None
        elif isinstance(file_dir, DNAStructure):
            assert sim_dir is not None
            self.file_dir = sim_dir
            self._build_sim = BuildSimulationFromStructure(self, file_dir)
        elif isinstance(file_dir, Simulation):
            self.file_dir = file_dir.sim_dir
            self._build_sim = None
        elif file_dir is None:
            warnings.warn("File dir set to `None`, hopefully you know what you're doing.")
        else:
            raise ValueError(f"Invalid type {type(file_dir)} for parameter file_dir")

        # handle alternate param types for sim_dir
        if sim_dir is None:  # if no sim dir is provided, use file dir
            self.sim_dir = self.file_dir
        elif isinstance(sim_dir, str):
            self.sim_dir = Path(sim_dir)
        elif isinstance(sim_dir, Path):
            self.sim_dir = sim_dir
        else:
            raise ValueError(f"Invalid type {type(sim_dir)} for parameter sim_dir")
        # tolerate sim_dir not existing, we can create it later

        # initialize all members as None,
        self._sim_files = None
        self._input = None
        self._analysis = None
        self._protein = None
        self._oxpy_run = None
        self._sequence_dependant = None
        self._slurm_run = None
        self._forces = None  # i feel so dirty

    @property
    def sim_files(self):
        if self._sim_files is None:
            self._sim_files = SimFiles(self)
        return self._sim_files

    @property
    def build_sim(self):
        if self._build_sim is None:
            self._build_sim = BuildSimulation(self)
        return self._build_sim

    def set_builder(self, builder: BuildSimulation):
        """
        assigns BuildSimulation object to construct simulation files
        this method is important for compatibilty with pypatchy methods
        """
        self._build_sim = builder

    @property
    def input(self):
        if self._input is None:
            self._input = Input(self)
        return self._input

    @property
    def analysis(self):
        if self._analysis is None:
            self._analysis = Analysis(self)
        return self._analysis

    @property
    def protein(self):
        if self._protein is None:
            self._protein = Protein(self)
        return self._protein

    @property
    def oxpy_run(self):
        if self._oxpy_run is None:
            self._oxpy_run = OxpyRun(self)
        return self._oxpy_run

    @property
    def oat(self) -> Analysis:
        return self.analysis

    @property
    def sequence_dependant(self):
        if self._sequence_dependant is None:
            self._sequence_dependant = SequenceDependant(self)
        return self._sequence_dependant

    @property
    def forces(self):
        if self._forces is None:
            # set input in forces
            self.input["external_forces"] = '1'
            self.input["external_forces_file"] = "forces.json"
            self.input["external_forces_as_JSON"] = "1"
            self._forces = SimForces(self)
        return self._forces

    def add_forces(self, load_from: str="forces.json", force_file_name:str="forces.json"):

        self.input["external_forces"] = '1'
        self.input["external_forces_file"] = "forces.json"
        self.input["external_forces_as_JSON"] = "1"
        if self._forces is None:
            self._forces = SimForces(self, load_from=load_from, force_file_name=force_file_name)
        self.build_sim.build_force()

    def clear_forces(self):
        self._forces = None
        del self.input["external_forces"]
        del self.input["external_forces_file"]
        del self.input["external_forces_as_json"]

    def has_sim_files(self):
        return self._sim_files is not None

    def has_build_sim(self):
        return self._build_sim is not None

    def has_input(self):
        return self._input is not None

    def has_analysis(self):
        return self._analysis is not None

    def has_protein(self):
        return self._protein is not None

    def has_oxpy_run(self):
        return self._oxpy_run is not None

    def has_forces(self):
        return self._forces is not None

    def get_conf(self, conf_idx: int) -> DNAStructure:
        """
        gets a conf in this simulation
        """
        # the trajectory.dat file doesn't include the initial configuration at index 0
        if conf_idx == 0:
            return load_dna_structure(
                self.sim_dir / self.input["topology"],
                self.sim_dir / self.input["conf_file"]
            )
        else:
            return load_dna_structure(self.sim_dir / self.input["topology"],
                                  self.sim_dir / self.input["trajectory_file"],
                                # conf at idx 0 in the traj is the simulation conf idx 1
                                  conf_idx-1)

    def last_conf_structure(self) -> DNAStructure:
        return load_dna_structure(self.sim_dir / self.input["topology"],
                                  self.sim_dir / self.input["lastconf_file"])

    def build(self, clean_build: Union[str, bool] = False):
        """
        Build dat, top, and input files in simulation directory.
        Is there a reason this is here and not in BuildSimulation

        :param clean_build: If sim_dir already exists, remove it and then rebuild sim_dir
        """
        if self.sim_dir.exists():
            if clean_build == 'force':
                # if we're not working with seperate file and sim dirs:
                if self.file_dir == self.sim_dir:
                    for file in self.sim_dir.iterdir():
                        # skip removing .top and init.dat files
                        if file != self.sim_files.dat and file != self.sim_files.top:
                            file.unlink()

                else:
                    # default clean-build behavior: remove all existing files from simulation directory
                    shutil.rmtree(self.sim_dir)
                    self.build_sim.force_cache = None
            elif clean_build:
                answer = input('Are you sure you want to delete all simulation files? '
                               'Type y/yes to continue or anything else to return (use clean_build="force" '
                               'to skip this message)')
                if (answer == 'y') or (answer == 'yes'):
                    self.input.input_dict = {}
                    shutil.rmtree(f'{self.sim_dir}/')
                    self.build_sim.force_cache = None
                else:
                    print('Remove optional argument clean_build and rerun to continue')
                    return None
            elif not clean_build:
                print(
                    'The simulation directory already exists, if you wish to write'
                    ' over the directory set clean_build=force')
                return None
            else:
                raise ValueError(f"Invalid clean_build value `{clean_build}`")
        self.build_sim.build_sim_dir()
        self.build_sim.build_dat_top()
        self.build_sim.build_input()

        self.sim_files.parse_current_files()

        # call callback, if it exists
        if self.build_sim.build_callback:
            self.build_sim.build_callback(self)

    def input_file(self, parameters: dict[str, Union[str, int, float, bool]]):
        """
        Modify the parameters of the oxDNA input file, all parameters are available at
         https://lorenzo-rovigatti.github.io/oxDNA/input.html

        Parameters:
            parameters (dict): dictionary of oxDNA input file parameters
        """
        self.input.modify_input(parameters)

    def add_protein_par(self):
        """
        Add a parfile from file_dir to sim_dir and add file name to input file
        """
        self.build_sim.build_par()
        self.protein.par_input()

    def add_force_txt_file(self):
        """
        Reads forces.txt file from
        In Matt's code, forces.txt are those forces which are present in the initial simulation files,
        while .json files storee forces added later through the ipy_oxdna interface, but non-Umbrella-Sampling
        applications may not make this distinction. The method is incluyded here for compatibility purposes
        """
        self.build_sim.get_force_txt_file()
        self.build_sim.build_force_from_file()
        shutil.copy(self.build_sim.force_file, self.sim_dir / self.build_sim.force_file.name)
        self.input["external_forces"] = 1

    def copy_forces(self):
        """
        generalized method to copy forces from file dir to sim dir
        """
        self.build_sim.get_general_force_file()
        self.build_sim.build_force_from_file()
        self.input["external_forces"] = 1

    def add_force(self, force_js: Union[dict, Force]):
        """
        Add an external force to the simulation.
        :param force_js: a force, in either dict or Force object form
        Parameters:
            force_js (ipy_oxdna.force.Force): A force  object, essentially a dictionary,
             specifying the external force parameters. (alternatively: in new parlance, a Force object)
        """
        if isinstance(force_js, dict):
            self.add_force(Force(**force_js))
        else:
            self.forces.add_force(force_js)
            # if not (self.sim_dir / "forces.json").is_file():
            #     self.input["external_forces"] = '1'
            #     self.input["external_forces_file"] = "forces.json"
            #     self.input["external_forces_as_JSON"] = "1"
        self.build_sim.build_force()

    def add_observable(self, observable_js: Union[Observable, dict[str, Any]]):
        """
        Add an observable that will be saved as a text file to the simulation.

        :param observable_js: (ipy_oxdna.observable.Observable) An observable object, essentially a dictionary, specifying the observable parameters.
        """
        if isinstance(observable_js, dict):
            if not os.path.exists(os.path.join(self.sim_dir, "observables.json")):
                self.input_file({'observables_file': 'observables.json'})
            self.build_sim.build_observable(observable_js)
        else:
            self.add_observable(observable_js.export())

    def slurm_run(self, run_file, job_name='oxDNA'):
        """
        Write a provided sbatch run file to the simulation directory.

        Parameters:
            run_file (str): Path to the provided sbatch run file.
            job_name (str): Name of the sbatch job.
        """
        self.sim_files.run_file = os.path.abspath(os.path.join(self.sim_dir, run_file))
        self._slurm_run = SlurmRun(self.sim_dir, run_file, job_name)

    def make_sequence_dependant(self):
        """ Add a sequence dependant file to simulation directory and modify input file to use it."""
        self.sequence_dependant.make_sim_sequence_dependant()

    def get_file_dir(self) -> Path:
        return self.file_dir

    def set_file_dir(self, p: Union[Path, str]):
        if isinstance(p, Path):
            # assert p.exists(), f"Cannot make a simulation from non-existing file dir {str(p)}"
            self.file_dir = p
        else:
            self.set_file_dir(Path(p))

    def get_sim_dir(self) -> Path:
        return self.sim_dir

    def set_sim_dir(self, p: Union[Path, str]):
        if isinstance(p, Path):
            self.sim_dir = p
        else:
            self.set_sim_dir(Path(p))

    @classmethod
    def from_pickle(cls, filename):
        """ Read a pickled simulation object from a file."""
        with open(filename, 'rb') as f:
            sim = pickle.load(f)
        return sim

    def pickle_sim(self):
        """ Pickle the simulation object to a file."""
        with open(f'{self.sim_dir}/sim.pkl', 'wb') as f:
            pickle.dump(self, f)


class SimulationComponent(SimDirInfo, ABC):
    """
    abstract class for a component of a simulation object
    """
    sim: Simulation

    def __init__(self, sim: Simulation):
        self.sim = sim

    # override methods from SimDirInfo to invoke Simulation object
    # to mimimize potential issues w/ same thing stored in different place
    def get_file_dir(self) -> Path:
        return self.sim.file_dir

    def set_file_dir(self, p: Union[Path, str]):
        if isinstance(p, Path):
            self.sim.file_dir = p
        else:
            self.set_file_dir(Path(p))

    def get_sim_dir(self) -> Path:
        return self.sim.sim_dir

    def set_sim_dir(self, p: Union[Path, str]):
        if isinstance(p, Path):
            self.sim.sim_dir = p
        else:
            self.set_sim_dir(Path(p))

    file_dir = property(get_file_dir, set_file_dir)
    sim_dir = property(get_sim_dir, set_sim_dir)


class Protein(SimulationComponent):
    """
    Methods used to enable anm simulations with proteins
    """

    def par_input(self):
        self.sim.input_file({
            'interaction_type': 'DNANM',
            'parfile': self.sim.build_sim.par
        })

class SimForces(SimulationComponent):
    """
    methods for tracking simulation forces
    """

    _forces: dict[str, Force]

    def __init__(self, sim: Simulation, force_file_name: str = "forces.json", load_from: Optional[str] = None):
        """
        default constructor. will automatically load forces from file_dir unless load_from is set to None
        """
        super().__init__(sim)
        self._forces = dict()
        self._force_file = self.sim_dir / force_file_name
        if load_from is not None:
            if (self.file_dir / load_from).is_file():
                self.load_from(self.file_dir / load_from)
            elif load_from != "forces.json":
                # only raise the exception if specific force file name was provided
                raise Exception(f"No such file as {str(self.file_dir/load_from)}")

    def fold_forces(self, fold_str: str):
        """
        Adds forces to fold the structure.
        :param fold_str: string showing folding of strand, in dot-bracket notation
        """
        assert fold_str.count("(") == fold_str.count(")"), "Mismatch between close and open parentheses"
        stack = []
        pairs: list[tuple[int, int]] = []
        for i, c in enumerate(fold_str):
            if c == "(":
                stack.append(i)
            elif c == ")":
                if not stack:
                    raise ValueError("Unmatched closing parenthesis at position {}".format(i))
                start = stack.pop()
                pairs.append((start, i))
        if stack:
            raise ValueError("Unmatched opening parenthesis at positions {}".format(stack))
        for force in zip_strands(*zip(*pairs)):
            self.add_force(force)

    def load_from(self, file: Path):
        """
        loads forces from a json file
        todo: compatibilty with whatever the other format is called
        """
        if file.suffix == ".json":
            # todo: use load_forces_from_json method
            with file.open("r") as f:
                j = json.load(f)
                for force_name, force in j.items():
                    self[force_name] = force
        else:
            for force in load_forces_from_txt(file):
                self.add_force(force)


    def clear(self):
        self._forces.clear()

    def save(self, file_name: str = "forces.json"):
        with open(self.sim_dir / file_name, "w") as f:
            json.dump({
                force_name: {**force} for force_name, force in self._forces.items()
            }, f)

    def __len__(self):
        return len(self._forces)

    def add_force(self, force: Force, name: str = None):
        """
        adds a force to the simulation's force list
        """
        if name is None:
            i = len(self._forces)
            # my god there has got to be a better way to write a do-while loop
            while True:
                name = f"force_{len(self._forces)}"
                if name not in self._forces:
                    break
        self._forces[name] = force

    def __getitem__(self, item: str):
        return self._forces[item]

    def __setitem__(self, key:str, value:Union[Force, dict]):
        if isinstance(key, dict):
            value = Force(**value)
        self._forces[key] = value

    def __iter__(self) -> Iterable[tuple[str, Force]]:
        yield from self._forces.items()

    def dump_old_fmt(self, file: str = "forces.txt"):
        with (self.sim_dir / file).open("w") as f:
            for force in self._forces.values():
                f.write("{\n")
                for key, value in force.items():
                    if type(value) in [str, int, float]:
                        f.write(f"{key} = {value}\n")
                    else:
                        f.write(f"{key} = {', '.join(value)}\n")
                f.write("}\n")

    # static methods that mirror old Force class
    @staticmethod
    def morse(**kwargs):
        return morse(**kwargs)

    @staticmethod
    def skew_force(**kwargs):
        return skew_force(**kwargs)

    @staticmethod
    def com_force(**kwargs):
        return skew_force(**kwargs)

    @staticmethod
    def mutual_trap(**kwargs):
        return mutual_trap(**kwargs)

    @staticmethod
    def string(**kwargs):
        return string(**kwargs)

    @staticmethod
    def harmonic_force(**kwargs):
        return harmonic_trap(**kwargs)

    @staticmethod
    def rotating_harmonic_trap(**kwargs):
        return harmonic_trap(**kwargs)

    @staticmethod
    def repulsion_plane(**kwargs):
        return repulsion_plane(**kwargs)

    @staticmethod
    def repulsion_sphere(**kwargs):
        return repulsion_sphere(**kwargs)


class BuildSimulation(SimulationComponent):
    # force_cache: Union[dict, None] # dict of forces in simulation
    par: Any
    force_file: Path
    is_file: bool

    # names of top and conf file in file directory
    top_file_name: Optional[str]
    conf_file_name: Optional[str]

    # dict which maps names of file in file directory to sim directory
    name_mapper: dict[str, str]

    build_callback: Optional[Callable[[Simulation], None]] = None

    """ Methods used to create/build oxDNA simulations."""

    def __init__(self, sim: Simulation):
        """ Initalize access to simulation information"""
        SimulationComponent.__init__(self, sim)
        self.force_cache = None

        self.name_mapper = {
            "conf_file": "init.dat"  # default-case: rename last_conf to init
        }
        self.top_file_name = None
        self.conf_file_name = None

    def build_sim_dir(self):
        """Make the simulation directory"""
        if not self.sim.sim_dir.exists():
            os.makedirs(self.sim.sim_dir)

    def build_dat_top(self):
        """
        Write intial conf and toplogy to simulation directory
        """
        self.find_starting_top_dat()

        # copy dat file to sim directory
        if not (self.file_dir / self.conf_file_name).exists():
            raise FileNotFoundError(f"Initial conf file `{str(self.file_dir/self.conf_file_name)}` not found")
        try:
            shutil.copy(self.file_dir / self.conf_file_name,
                        self.sim.sim_dir)
        except SameFileError:
            pass # if we're using same directory for file dir and sim dir this may happen, can ignore
        shutil.move(self.sim.sim_dir / self.conf_file_name,
                    self.sim.sim_dir / self.sim.input.get_conf_file())

        # copy top file to sim directory
        if not (self.file_dir / self.top_file_name):
            raise FileNotFoundError(f"Topology file {str(self.file_dir/self.top_file_name)} not found")
        try:
            shutil.copy(self.file_dir / self.top_file_name,
                        self.sim.sim_dir)
        except SameFileError:
            pass # if we're using same directory for file dir and sim dir this may happen, can ignore
        shutil.move(self.sim.sim_dir / self.top_file_name,
                    self.sim.sim_dir / self.sim.input.get_top_file())

    def copy_traj_conf(self, t: int, traj_file: str = "trajectory.dat"):
        """
        extracts a conf from the trajectory file in the file directory,
         and writes it to the conf file in the sim directory
        :param t: time step of trajectory to extract
        :param traj_file: name of trajectory file in file directory
        """
        if traj_file.startswith("/"):
            traj_path = Path(traj_file)
        else:
            traj_path = self.sim.file_dir / traj_file
        if not traj_path.exists():
            raise FileNotFoundError(f"Trajectory file {traj_path} does not exist")
        top_info, traj_info = describe(str(self.sim.file_dir / self.top_file_name),
                                       str(traj_path))
        conf = get_confs(str(traj_path), traj_info, t)[0]
        write_conf(self.sim.input.get_conf_file(), conf, include_vel=True) # sure

    def find_starting_top_dat(self):
        # find file-directory top and dat
        if self.top_file_name is None:
            self.top_file_name = find_top_file(self.file_dir, self.sim).name
        if self.conf_file_name is None:
            self.conf_file_name = find_conf_file(self.file_dir, self.sim).name
        if "conf_file" in self.name_mapper:
            self.sim.input.set_conf_file(self.name_mapper["conf_file"])
        else:
            self.sim.input.set_conf_file(self.conf_file_name)
        if "topology" in self.name_mapper:
            self.sim.input.set_top_file(self.name_mapper["topology"])
        else:
            self.sim.input.set_top_file(self.top_file_name)
        return self.file_dir / self.top_file_name, self.file_dir /  self.conf_file_name

    def list_file_dir(self) -> list[str]:
        """
        :return: a list of files in the data source directory
        """
        return os.listdir(self.sim.file_dir)

    def build_input(self, production=False):
        """
        Calls a methods from the Input class which writes a oxDNA input file in plain text and json
        :param production: todo
        """
        self.sim.input.initalize_input()
        self.sim.input.write_input(production=production)

    def copy_input(self):
        """
        copies the values from input file in the file directory
        to the input file in the simulation directory
        """
        # ignore the json file
        input_file_data = load_input_file(self.file_dir / "input")
        self.sim.input.modify_input(input_file_data)

    def get_par(self):
        """
        searches for an anisotropic network parameter (par) file
        """
        files = self.list_file_dir()
        self.par = [file for file in files if file.endswith('.par')][0]

    def build_par(self):
        """
        builds an anisotropic network parameter (par) file
        """
        self.get_par()
        shutil.copy(os.path.join(self.sim.file_dir, self.par), self.sim.sim_dir)

    def get_force_txt_file(self):
        """
        Searches the file directory for a file called "force.txt" or "forces.txt" and sets self.force_file
        to the path of that file.
        :raise SimBuildMissingFileException: if no force.txt or forces.txt file is found in the file directory
        """
        files = self.list_file_dir()
        force_file = next((file for file in files if file.endswith('force.txt') or file.endswith("forces.txt")), None)
        if force_file is None:
            raise SimBuildMissingFileException(self.sim, "force.txt")
        self.force_file = self.file_dir / force_file

    def get_general_force_file(self):
        """
        generalized version of get_force_txt_file
        finds any file in the file directory that contains force data
        :raise SimBuildMissingFileException: if no force.txt, forces.txt, force.json, or forces.json file is found in the file directory
        """
        try:
            self.get_force_txt_file()
        except SimBuildMissingFileException as e:
            files = self.list_file_dir()
            force_file = next((file for file in files if file.endswith("forces.json") or file.endswith("force.json")),
                              None)
            if force_file is None:
                raise SimBuildMissingFileException(self.sim, "forces.json") from e


    def build_force_from_file(self, clean: bool=False):
        """
        reads a forces.txt or forces.json file, populates self.sim.forces with forces from the file, and then
        saves the forces in forces.json
        if clean=True, first delete all existing forces
        """

        if clean:
            self.sim.forces.clear()
        self.sim.forces.load_from(self.force_file)

        self.sim.forces.save()

    # def copy_forces(self):
    #     """
    #     copies forces.json file from file directory to simulation directory
    #     todo: better implementation of this feature
    #     """
    #     assert (self.file_dir / "forces.json").exists(), f"File directory `{str(self.file_dir)} does not contain a forces.json file"
    #     self.sim.forces.load_from(self.file_dir/"forces.json")
    #     self.build_force()
    #     # self.sim.add_force_file()
    #     # shutil.copy(self.file_dir / "forces.json", self.sim_dir)

    def build_force(self, force_js: dict=None):
        """
        # force_js is deprecated
        """
        if force_js:
            self.sim.forces.add_force(Force(**force_js))
        self.sim.forces.save()
        # force_file_path = self.sim_dir / "forces.json"
        #
        # # Initialize the cache and create the file if it doesn't exist
        # if self.force_cache is None:
        #     # if a force file does not already exist
        #     if not force_file_path.is_file():
        #         # force cache = pre- existing forces
        #         # if there's no force file existing already this is empty
        #         self.force_cache = {}
        #         # write empty force dict to file
        #         with force_file_path.open("w") as f:
        #             json.dump(self.force_cache, f, indent=4)
        #         self.is_empty = True  # Set the flag to True for a new file
        #     else:
        #         # read existing forces
        #         with force_file_path.open("r") as f:
        #             self.force_cache = json.load(f)
        #         self.is_empty = not bool(self.force_cache)  # Set the flag based on the cache
        #
        # # Check for duplicates in the cache
        # for force in list(self.force_cache.values()):
        #     # TODO: CLEAN UP THIS LINE
        #     if list(force.values())[1] == list(list(force_js.values())[0].values())[1]:
        #         return
        #
        # # Add the new force to the cache
        # new_key = f'force_{len(self.force_cache)}'
        # self.force_cache[new_key] = force_js['force']
        #
        # # Append the new force to the existing JSON file
        # self.append_to_json_file(str(force_file_path),
        #                          new_key,
        #                          force_js['force'],
        #                          self.is_empty)
        #
        # self.is_empty = False  # Update the flag

    def append_to_json_file(self,
                            file_path: str,
                            new_entry_key: str,
                            new_entry_value: Any,
                            is_empty: bool):
        with open(file_path, 'rb+') as f:
            f.seek(-1, os.SEEK_END)  # Go to the last character of the file
            f.truncate()  # Remove the last character (should be the closing brace)

            if not is_empty:
                f.write(b',\n')  # Only add a comma if the JSON object is not empty

            new_entry_str = f'    "{new_entry_key}": {json.dumps(new_entry_value, indent=4)}\n}}'
            f.write(new_entry_str.encode('utf-8'))

    def build_observable(self, observable_js: dict, one_out_file=False):
        """
        Write observable file is one does not exist. If a observable file exists add additional observables to the file.
        
        :param observable_js: observable dictornary obtained from the Observable class methods
        """

        if not (self.sim.sim_dir / "observables.json").exists():
            with open(os.path.join(self.sim.sim_dir, "observables.json"), 'w') as f:
                f.write(dumps(observable_js, indent=4))
        else:
            with open(os.path.join(self.sim.sim_dir, "observables.json"), 'r') as f:
                read_observable_js = loads(f.read())
                multi_col = False
                for observable in list(read_observable_js.values()):
                    if list(observable.values())[1] == list(list(observable_js.values())[0].values())[1]:
                        read_observable_js['output']['cols'].append(observable_js['output']['cols'][0])
                        multi_col = True
                if not multi_col:
                    read_observable_js[f'output_{len(list(read_observable_js.keys()))}'] = read_observable_js['output']
                    del read_observable_js['output']
                    read_observable_js.update(observable_js.items())
                with open(os.path.join(self.sim.sim_dir, "observables.json"), 'w') as f:
                    f.write(dumps(read_observable_js, indent=4))

    def build_hb_list_file(self, p1: str, p2: str):
        self.sim.sim_files.parse_current_files()
        column_names = ['strand', 'nucleotide', '3_prime', '5_prime']

        try:
            top = pd.read_csv(self.sim.sim_files.top, sep=' ', names=column_names).iloc[1:, :].reset_index(drop=True)
            top['index'] = top.index
            p1 = p1.split(',')
            p2 = p2.split(',')
            i = 1
            with open(os.path.join(self.sim.sim_dir, "hb_list.txt"), 'w') as f:
                f.write("{\norder_parameter = bond\nname = all_native_bonds\n")
            complement = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}
            for nuc1 in p1:
                nuc1_data = top.iloc[int(nuc1)]
                nuc1_complement = complement[nuc1_data['nucleotide']]
                for nuc2 in p2:
                    nuc2_data = top.iloc[int(nuc2)]
                    if nuc2_data['nucleotide'] == nuc1_complement:
                        with open(os.path.join(self.sim.sim_dir, "hb_list.txt"), 'a') as f:
                            f.write(f'pair{i} = {nuc1}, {nuc2}\n')
                        i += 1
            with open(os.path.join(self.sim.sim_dir, "hb_list.txt"), 'a') as f:
                f.write("}\n")
            return None

        except:
            with open(self.sim.sim_files.force, 'r') as f:
                lines = f.readlines()
                lines = [int(line.strip().split()[1].replace('"', '')[:-1]) for line in lines if 'particle' in line]
                line_sets = [(lines[i], lines[i + 1]) for i in range(0, len(lines), 2)]
                line_sets = {tuple(sorted(t)) for t in line_sets}
            with open(os.path.join(self.sim.sim_dir, "hb_list.txt"), 'w') as f:
                f.write("{\norder_parameter = bond\nname = all_native_bonds\n")
                for idx, line_set in enumerate(line_sets):
                    f.write(f'pair{idx} = {line_set[0]}, {line_set[1]}\n')
                f.write("}\n")

            return None

class BuildSimulationFromStructure(BuildSimulation):
    """
    build Simulation from DNA sturcture
    """
    structure: DNAStructure
    def __init__(self, sim: Simulation, structure: DNAStructure):
        super().__init__(sim)
        self.structure = structure

    def build_dat_top(self):
        """
        Write intial conf and toplogy to simulation directory
        """
        # apply default top and conf file names if not set
        if self.sim.input.get_conf_file() is None:
            self.sim.input.set_conf_file("init.dat")
        if self.sim.input.get_top_file() is None:
            self.sim.input.set_top_file("topology.top")

        self.structure.export_top_conf(
            self.sim.sim_dir / self.sim.input.get_top_file(),
            self.sim.sim_dir / self.sim.input.get_conf_file()
        )

class OxpyRun(SimulationComponent):
    # setup params
    sim_dir: str
    # run params
    subprocess: bool
    verbose: bool
    continue_run: bool
    log: Union[False, str]  # name of log file, or False if log is off
    join: bool
    custom_observables: bool
    sim_output: str
    sim_err: str
    process: Optional[mp.Process]

    error_message: Optional[str]

    """Automatically runs a built oxDNA simulation using oxpy within a subprocess"""

    def __init__(self, sim: Simulation):
        """ Initalize access to simulation inforamtion."""
        SimulationComponent.__init__(self, sim)
        self.process = None
        self.my_obs = {}

    def run(self,
            subprocess=True,
            continue_run=False,
            verbose=True,
            log: Union[str, bool] = True,
            join=False,
            custom_observables=None):
        """ Run oxDNA simulation using oxpy in a subprocess.
        
        Parameters:
            subprocess (bool): If false run simulation in parent process (blocks process), if true spawn sim in child process.
            continue_run (number): If False overide previous simulation results. If True continue previous simulation run.
            verbose (bool): If true print directory of simulation when run.
            log (bool): If not False, print a log file to simulation directory. If True, the file will be auto-named to "log.log". otherwise it will be given the provided name
            join (bool): If true block main parent process until child process has terminated (simulation finished)
        """
        # why are these class members?
        self.subprocess = subprocess
        self.verbose = verbose
        self.continue_run = continue_run
        if log is True:
            self.log = "log.log"
        else:
            self.log = log
        self._join = join
        self.custom_observables = custom_observables

        if self.verbose:
            if type(self.sim.sim_dir) == str:
                print(f'Running: {self.sim.sim_dir}')
            else: # should never fire
                print(f'Running: {str(self.sim.sim_dir)}')

        if self.subprocess:
            self.spawn(self.run_complete)
        else:
            self.run_complete()


    __call__ = run

    def join(self):
        """Join the subprocess and refresh sim_files to reflect new output."""
        if self.process is not None:
            self.process.join()
            self.sim.sim_files.parse_current_files()

    def spawn(self, f, args=()):
        """Spawn subprocess"""
        p = mp.Process(target=f, args=args)
        p.start()
        self.process = p
        if self._join:
            self.join()

    def run_complete(self):
        """Run an oxDNA simulation"""
        self.error_message = None
        tic = timeit.default_timer()
        # capture outputs
        capture = py.io.StdCaptureFD()
        if self.continue_run is not False:
            self.sim.input_file({"conf_file": self.sim.sim_files.last_conf,
                                 "refresh_vel": "0",
                                 "restart_step_counter": "0",
                                 "steps": f'{self.continue_run}'})
        with PathContext(self.sim.sim_dir):
            with open('input.json', 'r') as f:
                my_input = loads(f.read())
            try:
                with oxpy.Context():
                    ox_input = oxpy.InputFile()
                    for k, v in my_input.items():
                        # todo: error-handling for vals that don't stringify nicely
                        ox_input[k] = str(v)
                    try:
                        # todo: there's a beautiful myriad of ways this can crash silently
                        # out of CUDA memory being a notable one
                        manager = oxpy.OxpyManager(ox_input)
                        if hasattr(self.sim.sim_files, 'run_time_custom_observable'):
                            with open(self.sim.sim_files.run_time_custom_observable, 'r') as f:
                                self.my_obs = load(f)
                            for key, value in self.my_obs.items():
                                my_obs = [eval(observable_string, {"self": self}) for observable_string in value['observables']]
                                manager.add_output(key, print_every=value['print_every'], observables=my_obs)
                        manager.run_complete()
                        del manager
                    except oxpy.OxDNAError as e:
                        self.error_message = traceback.format_exc()
                        raise e
                    except KeyboardInterrupt as e:
                        self.error_message = 'KeyboardInterrupt'
                    except Exception as e:
                        print('Unexpected python error')
                        self.error_message = traceback.format_exc()
            finally:
                # Always restore the captured stdout/stderr, even when an OxDNAError
                # propagates out of the try above. Without this, a raised OxDNAError skips
                # straight past capture.reset(), leaving stdout/stderr redirected into the
                # (now-abandoned) StdCaptureFD buffer — so the exception's own traceback,
                # printed by Python's default handler, vanishes instead of being visible.
                self.sim_output, self.sim_err = capture.reset()

            # grab captured err and outputs
            toc = timeit.default_timer()
            if self.verbose:
                print(f'Run time: {toc - tic}')
                if self.error_message is not None:
                    print(
                        f'Exception encountered in {self.sim.sim_dir}:\n{type(self.error_message).__name__}: {self.error_message}')
                else:
                    if type(self.sim.sim_dir) == str:
                        print(f'Finished: {self.sim.sim_dir.parent}')
                    else:
                        print(f'Finished: {self.sim.sim_dir}')
            if self.log:
                with open('log.log', 'w') as f:
                    f.write(self.sim_output)  # write output log
                    f.write(self.sim_err)  # write error log
                    f.write(f'Run time: {toc - tic}')  # write runtime
                    if self.error_message is not None:
                        f.write(f'Exception: {self.error_message}')
        self.sim.sim_files.parse_current_files()

    def cms_obs(self, *args, name=None, print_every=None):
        self.my_obs[name] = {'print_every': print_every, 'observables': []}
        for particle_indexes in args:
            self.my_obs[name]['observables'].append(f'self.cms_observables({particle_indexes})()')

        self.write_custom_observable()

    def write_custom_observable(self):
        with open(os.path.join(self.sim.sim_dir, "run_time_custom_observable.json"), 'w') as f:
            dump(self.my_obs, f, indent=4)

    def cms_observables(self, particle_indexes):
        class ComPositionObservable(oxpy.observables.BaseObservable):
            def get_output_string(self, curr_step):
                output_string = ''

                np_idx = np.array(list(map(int, particle_indexes[0].split(','))), dtype=int)

                observable_energy = np.zeros((1,))

                particles = np.array(self.config_info.particles(), dtype='object')

                my_particles = particles[np_idx]

                for p_1 in my_particles:

                    for p_2 in my_particles:
                        if p_2.index != p_1.index:
                            
                            observable_energy[0] += self.config_info.interaction.pair_interaction(p_1, p_2, compute_r=True)
                            observable_energy[1] += self.config_info.interaction.pair_interaction_term(2, p_1, p_2)

                positions = np.array(self.config_info.flattened_conf.positions)
                
                
                
                step = int(self.config_info.current_step)
                box = np.array(self.config_info.box_sides)
                energy = np.array([-1, 0.5, -0.5])
                a1s = np.array(self.config_info.flattened_conf.a1s)
                a3s = np.array(self.config_info.flattened_conf.a3s)
                
                conf = inbox(Configuration(step, box, energy, positions, a1s, a3s), center=True)
                in_pos = conf.positions
                # print(in_pos.shape)
                
                # all_np_idxes = [
                #     np.fromstring(item[0], sep=',', dtype=int)
                #     for item in particle_indexes[1].values()
                # ]
                # reference_points = [np.mean(in_pos[idx], axis=0) for idx in all_np_idxes]
                
                euler_ref_idxes = particle_indexes[1]
                m0 = self.indexes_to_array(euler_ref_idxes['m0'])
                m0_0 = self.indexes_to_array(euler_ref_idxes['m0_0'])
                m0_1 = self.indexes_to_array(euler_ref_idxes['m0_1'])
                
                m1 = self.indexes_to_array(euler_ref_idxes['m1'])
                m1_0 = self.indexes_to_array(euler_ref_idxes['m1_0'])
                m1_1 = self.indexes_to_array(euler_ref_idxes['m1_1'])
                
                com_m0 = np.mean(in_pos[m0, :], axis=0)
                
                in_pos  -= com_m0
                
                com_m0_centered = np.mean(in_pos[m0, :], axis=0)
                
                com_m0_0 = np.mean(in_pos[m0_0, :], axis=0)
                com_m0_1 = np.mean(in_pos[m0_1, :], axis=0)
                
                com_m1 = np.mean(in_pos[m1, :], axis=0)
                com_m1_0 = np.mean(in_pos[m1_0, :], axis=0)
                com_m1_1 = np.mean(in_pos[m1_1, :], axis=0)
                
                coms = [com_m0_centered, com_m0_0, com_m0_1, com_m1, com_m1_0, com_m1_1]
                com_string = ''
                for com in coms:
                    com_string += f'{com[0]} {com[1]} {com[2]} '

                output_string = f'{step} {observable_energy[0]} {observable_energy[1]} {com_string}'

                # args = (positions, m0, m0_0, m0_1, m1, m1_0, m1_1)
                # euler_angles_set, omega_rad = self.euler_angle_and_omega(*args)
                
                # output_string = f'{observable_energy[0]} {observable_energy[1]} {euler_angles_set[0]} {euler_angles_set[1]} {euler_angles_set[2]} {omega_rad}'            
                return output_string

        return ComPositionObservable

    def hb_contact_observables(self, particle_indexes):
        class ComPositionObservable(oxpy.observables.BaseObservable):
            def get_output_string(self, curr_step):
                output_string = ''
                np_idx = [list(map(int, particle_idx.split(','))) for particle_idx in particle_indexes]
                particles = np.array(self.config_info.particles())
                indexed_particles = [particles[idx] for idx in np_idx]
                cupy_array = np.array(
                    [np.array([particle.pos for particle in particle_list]) for particle_list in indexed_particles],
                    dtype=np.float64)

                box_length = np.float64(self.config_info.box_sides[0])

                pos = np.zeros((cupy_array.shape[1], cupy_array.shape[2]), dtype=np.float64)
                np.subtract(cupy_array[0], cupy_array[1], out=pos, dtype=np.float64)

                pos = pos - box_length * np.round(pos / box_length)

                new_pos = np.linalg.norm(pos, axis=1)
                r0 = np.full(new_pos.shape, 1.2)
                gamma = 58.7
                shape = 1.2

                final = np.sum(1 / (1 + np.exp((new_pos - r0 * shape) * gamma))) / np.float64(new_pos.shape[0])

                output_string += f'{final} '
                return output_string

        return ComPositionObservable

    # def cms_observables(self, particle_indexes):
    #         class ComPositionObservable(oxpy.observables.BaseObservable):
    #             def get_output_string(self, curr_step):
    #                 output_string = ''
    #                 np_idx = [list(map(int, particle_idx.split(','))) for particle_idx in particle_indexes]
    #                 particles = np.array(self.config_info.particles())
    #                 indexed_particles = [particles[idx] for idx in np_idx]
    #                 cupy_array = np.array([np.array([particle.pos for particle in particle_list]) for particle_list in indexed_particles], dtype=object)
    #                 for array in cupy_array:
    #                     pos = np.mean(array, axis=0)
    #                     output_string += f'{pos[0]},{pos[1]},{pos[2]} '
    #                 return output_string
    #         return ComPositionObservable


class SlurmRun:
    """Using a user provided slurm run file, setup a slurm job to be run"""

    def __init__(self, sim_dir, run_file, job_name):
        self.sim_dir = sim_dir
        self.run_file = run_file
        self.job_name = job_name
        self.write_run_file()

    def write_run_file(self):
        """ Write a run file to simulation directory."""
        with open(self.run_file, 'r') as f:
            lines = f.readlines()
            with open(os.path.join(self.sim_dir, 'run.sh'), 'w') as r:
                for line in lines:
                    if 'job-name' in line:
                        r.write(f'#SBATCH --job-name="{self.job_name}"\n')
                    else:
                        r.write(line)

    def sbatch(self):
        """ Submit sbatch run file."""
        # TODO: better pls
        os.chdir(self.sim_dir)
        os.system("sbatch run.sh")


def bytes_to_megabytes(byte):
    return byte / 1048576


class SigTermException(Exception):
    pass


class SimulationManager:
    """
    todo: explaination
    """
    manager: mp.Manager
    # todo: replace w/ generator?
    sim_queue: queue.Queue[Simulation]
    process_queue: queue.Queue
    gpu_memory_queue: queue.Queue
    terminate_queue: queue.Queue
    worker_process_list: list[mp.Process]

    warnings.filterwarnings(
        "ignore",
        "os.fork\\(\\) was called\\. os\\.fork\\(\\) is incompatible with multithreaded code, and JAX is multithreaded, so this will likely lead to a deadlock\\.",
        RuntimeWarning
    )

    """ In conjunction with nvidia-cuda-mps-control, allocate simulations to avalible cpus and gpus."""

    def __init__(self, n_processes=None, sleep_time=5.0):
        """
        Initalize the multiprocessing queues used to manage simulation allocation.
        
        The sim_queue utilizes a single process to store all queued simulations and allocates simulations to cpus.
        The process_queue manages the number of processes/cpus avalible to be sent to gpu memory.
        gpu_memory_queue is used to block the process_queue from sending simulations to gpu memory if memoy is near full.
        
        Parameters:
            n_processes (int): number of processes/cpus avalible to run oxDNA simulations in parallel.
        """
        if n_processes is None:
            self.n_processes = self.get_number_of_processes()
        else:
            if type(n_processes) is not int:
                raise ValueError('n_processes must be an integer')
            self.n_processes = n_processes
        self.manager = mp.Manager()
        self.sim_queue = self.manager.Queue()
        self.process_queue = self.manager.Queue(self.n_processes)
        self.gpu_memory_queue = self.manager.Queue(1)
        self.terminate_queue = self.manager.Queue(1)
        self.worker_process_list = []
        self.sleep_time = sleep_time
        
        signal.signal(signal.SIGTERM, self._sigterm_handler)


    def _sigterm_handler(self, signum, frame):
        raise SigTermException("SIGTERM received")


    def get_number_of_processes(self):
        try:
            # Try using os.sched_getaffinity() available on some Unix systems
            if sys.platform.startswith('linux'):
                return len(os.sched_getaffinity(0))
            else:
                # Fallback to multiprocessing.cpu_count() which works cross-platform
                return mp.cpu_count()
        except Exception as e:
            # Handle possible exceptions (e.g., no access to CPU info)
            print(f"Failed to determine the number of CPUs: {e}")
            return 1  # Safe fallback if number of CPUs can't be determined

    def gpu_resources(self) -> tuple[float, int]:
        """
        Method to probe the number and current available memory of gpus.
        :return: tuple where the first element is the amount of free memory on the
         GPU with the most free memory, and the second element is the index of that GPU
        """
        avalible_memory = []
        num_gpus = self.get_num_gpus()
        # loop available gpus
        for i in range(num_gpus):
            # no idea how this code works
            handle = nvidia_smi.nvmlDeviceGetHandleByIndex(i)
            info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
            avalible_memory.append(bytes_to_megabytes(info.total) - bytes_to_megabytes(info.used))
        # find gpu with most memory
        gpu_most_aval_mem_free = max(avalible_memory)
        gpu_most_aval_mem_free_idx = avalible_memory.index(gpu_most_aval_mem_free)
        return float(np.round(gpu_most_aval_mem_free, 2)), gpu_most_aval_mem_free_idx

    def get_num_gpus(self):
        """
        find the number of GPUs available on this system using nvidia-smi
        """
        try:
            nvidia_smi.nvmlInit()
        except nvidia_smi.NVMLError as e:
            raise Exception('nvidia-smi not avalible, ensure you have a cuda enabled GPU') from e
        return nvidia_smi.nvmlDeviceGetCount()

    def _bytes_to_megabytes(self, byte):
        # TODO: make this not a class method?
        return byte / 1048576

    def get_sim_mem(self, sim: Simulation, gpu_idx):
        """
        Returns the amount of simulation memory requried to run an oxDNA simulation.
        Note: A process running a simulation will need more memory then just required for the simulation.
              Most likely overhead from nvidia-cuda-mps-server

        Parameters:
            sim (Simulation): Simulation object to probe the required memory of.
            gpu_idx: depreciated
        """
        steps = sim.input.input_dict['steps']
        last_conf_file = sim.input.input_dict['lastconf_file']
        sim.input_file({'lastconf_file': os.devnull, 'steps': '0'})
        sim.oxpy_run.run(subprocess=False, verbose=False, log=False)
        sim.input_file({'lastconf_file': f'{last_conf_file}', 'steps': f'{steps}'})

        err_split = sim.oxpy_run.sim_output[1].split()
        try:
            mem = err_split.index('memory:')
            sim_mem = err_split[mem + 1]
        except Exception as e: # todo: more specific exception
            print("Unable to determine CUDA memory usage")
            print(traceback.format_exc())
            raise e

        return float(sim_mem)

    def queue_sim(self, sim: Simulation, continue_run=False):
        """ 
        Add simulation object to the queue of all simulations.
        
        Parameters:
            sim (Simulation): Simulation to be queued.
            continue_run (bool): If true, continue previously run oxDNA simulation
        """
        if continue_run is not False:
            sim.input_file({"conf_file": sim.sim_files.last_conf, "refresh_vel": "0",
                            "restart_step_counter": "0", "steps": f"{continue_run}"})
        self.sim_queue.put(sim)

    def worker_manager(self, gpu_mem_block=False, custom_observables=None, run_when_failed=False, cpu_run=False):
        """
        Head process in charge of allocating queued simulations to processes and gpu memory.
        """
        tic = timeit.default_timer()
        if cpu_run is True:
            gpu_mem_block = False
        self.custom_observables = custom_observables
        # as long as there are simulations in the queue
        try:
            # or any(p.is_alive() for p in self.worker_process_list): # Keep running if sims queued OR workers active
            while not self.sim_queue.empty():
                # --- Check for simulation errors ---
                if not self.terminate_queue.empty():
                    if run_when_failed is False:
                        simulation_error_message = self.terminate_queue.get()
                        self.handle_death(exception=oxpy.OxDNAError, message=simulation_error_message)
                        # handle_death should exit or raise, otherwise this loop continues
                        break # Exit loop after handling death if handle_death returns

                # --- Try to launch a new worker if queue has sims AND we have capacity ---
                # todo: isn't this if-statement redundant because of the outer while-loop?
                if not self.sim_queue.empty():

                    # --- If process_queue if full block, otherwise start a new process ---
                    self.process_queue.put('Simulation worker started')
                    sleep(0.1)

                    sim = self.sim_queue.get()
                    gpu_idx = -1 # Default for CPU run or if GPU query fails
                    free_gpu_memory = 0.0 # in Mystery Units

                    # --- Find the GPU with the most free memory ---
                    if not cpu_run:
                        # Clear CUDA_device from input so SimulationManager can re-assign it
                        if 'CUDA_device' in sim.input.input_dict:
                            del sim.input.input_dict['CUDA_device']

                        free_gpu_memory, gpu_idx = self.gpu_resources()
                        if gpu_idx == -1:
                             print("Warning: Could not get GPU resources, check nvidia-smi. Running on CPU instead?")
                        else:
                            sim.input_file({'CUDA_device': str(gpu_idx)})


                    # --- GPU Memory Blocking Logic ---
                    if gpu_mem_block and not cpu_run and gpu_idx != -1:
                        # Estimate memory needed *before* starting process
                        sim_mem = self.get_sim_mem(sim, gpu_idx)
                        if free_gpu_memory < (2 * sim_mem): # Check before potentially blocking
                            print(f"Waiting for GPU memory. Need ~{2*sim_mem:.2f} MB, have {free_gpu_memory:.2f} MB.")
                            wait_for_gpu_memory = True
                            while wait_for_gpu_memory:
                                #holding on with the change for testing
                                wait_for_gpu_memory = False
                            print('gpu memory freed')

                    # --- Start the worker process ---
                    p = mp.Process(target=self.worker_job, args=(sim, gpu_idx,), kwargs={'gpu_mem_block': gpu_mem_block})
                    p.start()
                    self.worker_process_list.append(p)
                    
                
                if not cpu_run: # Only sleep if not CPU and not already blocked by memory check
                    sleep(self.sleep_time)
                elif cpu_run:
                    sleep(0.1) # Shorter sleep for CPU runs
            
            while not self.process_queue.empty():
                sleep(10)
           
        except KeyboardInterrupt:                        
            KeyboardInterruptmessage = 'KeyboardInterrupt caught, terminating all processes'
            self.handle_death(exception=KeyboardInterrupt, message=KeyboardInterruptmessage)
            
        except SigTermException:
            sigTermMessage = 'SIGTERM (signal 15) recived.'
            self.handle_death(exception=SigTermException, message=sigTermMessage)
        
        except Exception as e:
            unexpected_message = 'Exception encountered in worker manager, terminating all processes'
            self.handle_death(exception=e, message=unexpected_message)
            
        finally: # Ensure cleanup runs even if errors occur
            print("Waiting for any remaining worker processes to finish...")
            for p in self.worker_process_list:
                p.join(timeout=10) # Wait for clean exit with timeout
                if p.is_alive():
                    print(f"Process {p.pid} did not exit cleanly, terminating forcefully.")
                    p.terminate() # Force terminate if join timed out
            self.worker_process_list[:] = [] # Clear the list of processes


        toc = timeit.default_timer()
        print(f'All queued simulations finished in: {toc - tic}')


    def handle_death(self, exception=Exception, message=None):
        """Terminates all tracked worker processes."""
        if message:
            print(message)
        
        # Print traceback if it's not a KeyboardInterrupt (which doesn't usually need a traceback)
        if not isinstance(exception, KeyboardInterrupt):
            print("Traceback (most recent call last):")
            traceback.print_exc() # Print the traceback for the exception that triggered handle_death

        print(f"Terminating {len(self.worker_process_list)} worker process(es)...")
        terminated_count = 0
        # --- CHANGE 3: Iterate through Process objects and use terminate() ---
        for worker_process in self.worker_process_list:
            try:
                if worker_process.is_alive():
                    # print(f"Sending terminate signal to process PID: {worker_process.pid}")
                    worker_process.terminate()
                    terminated_count += 1
            except Exception as e:
                # Catch potential errors during termination (e.g., process already dead)
                print(f"Error trying to terminate process PID {getattr(worker_process, 'pid', 'N/A')}: {e}")
                pass # Continue trying to terminate others
            
        # Check again and potentially join/kill remaining ones (more robust cleanup)
        remaining_processes = []
        for worker_process in self.worker_process_list:
             if worker_process.is_alive():
                 print(f"Process PID {worker_process.pid} still alive after terminate signal. Attempting join...")
                 worker_process.join(timeout=2) # Short timeout join
                 if worker_process.is_alive():
                      print(f"Process PID {worker_process.pid} did not join. Killing...")
                      # process.kill() sends SIGKILL, more forceful than terminate (SIGTERM)
                      try:
                           worker_process.kill() 
                      except Exception as kill_e:
                           print(f"Error killing process PID {worker_process.pid}: {kill_e}")
                 else:
                      print(f"Process PID {worker_process.pid} joined successfully.")
             remaining_processes.append(worker_process) # Keep track even if terminated/killed now
             
        # Update the list to reflect only processes that might still be tracked (even if dead)
        self.worker_process_list = remaining_processes

        # Shutdown manager after attempting process cleanup
        print("Shutting down manager in handle_death...")
        self.manager.shutdown()
    
        print("Exiting due to error condition.")
        sys.exit(1) # Exit the script after handling death

    def worker_job(self, sim: Simulation, gpu_idx: int, gpu_mem_block: bool = False):
        """ Run an allocated oxDNA simulation"""
        if gpu_mem_block is True:
            sim_mem = self.get_sim_mem(sim, gpu_idx)
            self.gpu_memory_queue.put(sim_mem)

        try:
            sim.oxpy_run.run(subprocess=False, custom_observables=self.custom_observables)
        except oxpy.OxDNAError:
            # run_complete() intentionally re-raises OxDNAError for direct/manual use of
            # oxpy_run; here we're the manager's own worker process, so absorb it instead
            # of letting it kill this process - error_message is already set below.
            pass
        if sim.oxpy_run.error_message is not None:
            self.terminate_queue.put(
                f'Simulation exception encountered in {sim.sim_dir}:\n{sim.oxpy_run.error_message}')
        sleep(1)
        self.process_queue.get()

    def run(self, join=False,
            gpu_mem_block=False,
            custom_observables=None,
            run_when_failed=False,
            cpu_run=False):
        """
        Run the worker manager in a subprocess
        todo: ...logging?
        """
        print('spawning')
        if cpu_run is True:
            gpu_mem_block = False

        p = mp.Process(target=self.worker_manager, args=(),
                       kwargs={'gpu_mem_block': gpu_mem_block,
                               'custom_observables': custom_observables,
                               'run_when_failed': run_when_failed,
                               'cpu_run': cpu_run
                               })
        self.manager_process = p
        p.start()
        if join:
            p.join()

    # def terminate_all(self, ):
    #     try:
    #         self.manager_process.terminate()
    #     except:
    #         pass
    #     for process in self.worker_process_list:
    #         try:
    #             os.kill(process, signal.SIGTERM)
    #         except:
    #             pass
    #     self.worker_process_list[:] = []

    def start_nvidia_cuda_mps_control(self, pipe='$SLURM_TASK_PID'):
        """
        Begin nvidia-cuda-mps-server.
        
        Parameters:
            pipe (str): directory to pipe control server information to. Defaults to PID of a slurm allocation
        """
        with open('launch_mps.tmp', 'w') as f:
            f.write(f"""#!/bin/bash
export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps-pipe_{pipe};
export CUDA_MPS_LOG_DIRECTORY=/tmp/mps-log_{pipe};
mkdir -p $CUDA_MPS_PIPE_DIRECTORY;
mkdir -p $CUDA_MPS_LOG_DIRECTORY;
nvidia-cuda-mps-control -d"""
                    )
        os.system('chmod u+rx launch_mps.tmp')
        sp.call('./launch_mps.tmp')
        self.test_cuda_script()
        os.system('./test_script')
        os.system('echo $CUDA_MPS_PIPE_DIRECTORY')

    #         os.system(f"""export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps-pipe_{pipe};
    # export CUDA_MPS_LOG_DIRECTORY=/tmp/mps-log_{pipe};
    # mkdir -p $CUDA_MPS_PIPE_DIRECTORY;
    # mkdir -p $CUDA_MPS_LOG_DIRECTORY;
    # nvidia-cuda-mps-control -d;""")

    def restart_nvidia_cuda_mps_control(self):
        os.system("""echo quit | nvidia-cuda-mps-control""")
        sleep(0.5)
        self.start_nvidia_cuda_mps_control()

    def test_cuda_script(self):
        script = """#include <stdio.h>

#define N 2

__global__
void add(int *a, int *b) {
    int i = blockIdx.x;
    if (i<N) {
        b[i] = 2*a[i];
    }
}

int main() {

    int ha[N], hb[N];

    int *da, *db;
    cudaMalloc((void **)&da, N*sizeof(int));
    cudaMalloc((void **)&db, N*sizeof(int));

    for (int i = 0; i<N; ++i) {
        ha[i] = i;
    }


    cudaMemcpy(da, ha, N*sizeof(int), cudaMemcpyHostToDevice);

    add<<<N, 1>>>(da, db);

    cudaMemcpy(hb, db, N*sizeof(int), cudaMemcpyDeviceToHost);
    
        for (int i = 0; i<N; ++i) {
        printf("%d", hb[i]);
    }

    cudaFree(da);
    cudaFree(db);

    return 0;
}
"""
        with open('test_script.cu', 'w') as f:
            f.write(script)

        os.system('nvcc -o test_script test_script.cu')
        os.system('./test_script')


def load_input_file(plain_text_input: Path) -> dict[str, Any]:
    """
    load a plain text oxDNA input file and convert to a dict
    :param plain_text_input: path to plain text input file
    :return: dict of input parameters
    """
    my_input = dict()
    with plain_text_input.open('r') as f:
        # i hate behavior of python readline
        # a `while line := f.readline():` loop will terminate prematurely if it finds an empty line
        lines = f.readlines()
        my_input = read_input_file_stream(collections.deque(lines))
    return my_input

def read_input_file_stream(lines: collections.dequeue) -> dict[str, Any]:
    my_input = dict()
    while lines:
        line = lines.pop()
        if not line.strip().startswith('#'):
            if "=" in line:
                key, value = line.strip().split('=', 1)
                if value.endswith("{"):
                    value = read_input_file_stream(lines)
                else:
                    value = value.strip()
                my_input[key.strip()] = value
            elif "}" in line:
                break # end of block
    return my_input

class Input(SimulationComponent):
    input_dict: dict[str, str]
    default_input: DefaultInput
    """ Lower level input file methods"""

    def __init__(self, sim: Simulation):
        """ 
        Read input file in simulation dir if it exsists, other wise define default input parameters.
        
        Parameters:
            sim_dir (str): Simulation directory
            parameters: depreciated
        """
        SimulationComponent.__init__(self, sim)
        self.default_input = get_default_input("cuda_MD")

        self.input_dict = {}
        if self.sim.sim_dir.exists():
            self.initalize_input()

    def clear(self):
        """
        deletes existing input file data
        """
        self.input_dict = {}
        self.write_input()

    def initalize_input(self, read_existing_input: Union[bool, None] = None):
        """
        Initializes the input file
        """
        if read_existing_input or read_existing_input is None:
            existing_input = (self.sim.sim_dir / 'input.json').exists() or (self.sim.sim_dir / 'input').exists()
        else:
            existing_input = False

        if existing_input:
            self.read_input()
        elif read_existing_input:
            raise SimBuildMissingFileException(self.sim, "input.json")

    def swap_default_input(self, default_type: str):
        """
        Swap the default input parameters to a different type of simulation.
        Current Options Include:
        cuda_prod, cpu_prod, cpu_relax
        """
        self.default_input = get_default_input(default_type)
        self.default_input.evaluate()
        self.input_dict = self.default_input.get_dict()
        self.write_input()

    def get_last_conf_top(self) -> tuple[str, str]:
        """
        Set attributes containing the name of the inital conf (dat file) and topology
        """
        top, conf = find_top_dat(self.sim.sim_dir, self.sim)
        self.initial_conf = conf.name
        self.top = top.name
        return self.initial_conf, self.top

    def write_input_standard(self):
        """ Write a oxDNA input file to sim_dir"""
        if not self.has_top_conf():
            raise MissingTopConfException(self.sim)
        with oxpy.Context():
            ox_input = oxpy.InputFile()
            for k, v in self.input_dict.items():
                ox_input[k] = v
            with open(os.path.join(self.sim.sim_dir, f'input'), 'w') as f:
                print(ox_input, file=f)

    def write_input(self, production=False):
        """ Write an oxDNA input file as a json file to sim_dir"""
        if production is False:
            if not self.has_top_conf():
                top, conf = find_top_dat(self.sim.sim_dir, self.sim)
                self.set_top_file(top.name)
                self.set_conf_file(conf.name)

        # Write input file
        self.default_input.evaluate(**self.input_dict)
        # local input dict w/ defaults and manually-specified values
        inputdict = {
            **self.default_input.get_dict(),
            **self.input_dict
        }
        # open input file json
        with open(os.path.join(self.sim.sim_dir, f'input.json'), 'w') as f:
            input_json = dumps(inputdict, indent=4) # dump input file in dict form
            f.write(input_json)
        # open actual input file
        write_input_file(self.sim_dir / "input", inputdict)


    def modify_input(self, parameters: dict):
        """ Modify the parameters of the oxDNA input file."""
        if (self.sim.sim_dir / 'input.json').exists():
            self.read_input()
        for k, v in parameters.items():
            # handle non-JSON serializable objects
            if isinstance(v, Path):
                v = str(v)
            if not isinstance(v, (str, int, float, bool)):
                warnings.warn(f"Non-allowed input file value {v} (type {type(v)}) for key {k}. Skipping...")
            else:
                self.input_dict[k] = v
        self.write_input()

    def read_input(self):
        """ Read parameters of exsisting input file in sim_dir"""
        if (self.sim.sim_dir / "input.json").exists() and os.stat(self.sim.sim_dir / "input.json").st_size > 0:
            with (self.sim.sim_dir / "input.json").open("r") as f:
                content = f.read()
                my_input = loads(content)
                self.input_dict = my_input
            # I don't know why you did this
            # it SHOULD throw an error if it finds a mangled JSON file!
            # except json.JSONDecodeError:
            #     self.initalize_input(read_exsisting_input=False)

        else:
            plain_text_input =self.sim.sim_dir / "input"
            my_input = load_input_file(plain_text_input)

            self.input_dict = my_input

    def get_conf_file(self) -> Union[None, str]:
        """
        Returns: the conf file that the simulation will initialize from
        """
        if "conf_file" not in self.input_dict:
            return None
        else:
            return self.input_dict["conf_file"]

    def set_conf_file(self, conf_file_name: str):
        """
        Sets the conf file
        """
        self.input_dict["conf_file"] = conf_file_name

    def get_top_file(self) -> Union[None, str]:
        """
        Returns: the topology file that the simulation will use
        """
        if "topology" not in self.input_dict:
            return None
        else:
            return self.input_dict["topology"]

    def set_top_file(self, top_file_name: str):
        """
        Sets the topology file
        """
        self.input_dict["topology"] = top_file_name

    def has_top_conf(self) -> bool:
        return self.get_conf_file() is not None and self.get_top_file() is not None

    def get_last_conf(self) -> Union[None, str]:
        if "lastconf_file" not in self.input_dict:
            return None
        else:
            return self.input_dict["lastconf_file"]

    def set_last_conf(self, conf_file_name: str):
        self.input_dict["lastconf_file"] = conf_file_name

    initial_conf = property(get_conf_file, set_conf_file)
    top = property(get_top_file, set_conf_file)

    def __getitem__(self, item: str):
        try:
            return self.input_dict[item]
        except KeyError as e:
            raise KeyError(f"Input file {str(self.sim.sim_dir / 'input')} has no key `{item}`")

    def __setitem__(self, key: str, value: Union[str, float, bool]):
        self.input_dict[key] = value
        self.write_input()

    def __contains__(self, item):
        return item in self.input_dict

    def __call__(self, **kwargs):
        self.modify_input(kwargs)

    def __delitem__(self, key:str):
        del self.input_dict[key]
        self.write_input()


class SequenceDependant(SimulationComponent):
    """ Make the targeted sim_dir run a sequence dependant oxDNA simulation"""
    parameters: str
    rna_parameters: str

    def __init__(self, sim: Simulation):
        SimulationComponent.__init__(self, sim)
        # TODO: hardcode sequence-dependant parameters externally
        self.parameters = "\n".join([f"{name} = {value}" for name, value in SEQ_DEP_PARAMS.items()])

        self.na_parameters = "\n".join([f"{name} = {value}" for name, value in NA_PARAMETERS.items()])

        self.rna_parameters = "\n".join([f"{name} = {value}" for name, value in RNA_PARAMETERS.items()])

    def make_sim_sequence_dependant(self):
        self.sequence_dependant_input()
        self.write_sequence_dependant_file()

    def write_sequence_dependant_file(self):
        # TODO: externalize interaction-type stuff?
        int_type = self.sim.input.input_dict['interaction_type']
        if (int_type == 'DNA') or (int_type == 'DNA2') or (int_type == 'NA'):
            with open(os.path.join(self.sim.sim_dir, 'oxDNA2_sequence_dependent_parameters.txt'), 'w') as f:
                f.write(self.parameters)

        if (int_type == 'RNA') or (int_type == 'RNA2') or (int_type == 'NA'):
            with open(os.path.join(self.sim.sim_dir, 'rna_sequence_dependent_parameters.txt'), 'w') as f:
                f.write(self.rna_parameters)

        if int_type == 'NA':
            with open(os.path.join(self.sim.sim_dir, 'NA_sequence_dependent_parameters.txt'), 'w') as f:
                f.write(self.na_parameters)

    def sequence_dependant_input(self):
        int_type = self.sim.input.input_dict['interaction_type']

        if (int_type == 'DNA') or (int_type == 'DNA2'):
            self.sim.input_file({'use_average_seq': False,
                                 'seq_dep_file': 'oxDNA2_sequence_dependent_parameters.txt'
                                 })

        if (int_type == 'RNA') or (int_type == 'RNA2'):
            self.sim.input_file({'use_average_seq': False,
                                 'seq_dep_file': 'rna_sequence_dependent_parameters.txt'
                                 })

        if int_type == 'NA':
            self.sim.input_file({'use_average_seq': False,
                                 'seq_dep_file_DNA': 'oxDNA2_sequence_dependent_parameters.txt',
                                 'seq_dep_file_RNA': 'rna_sequence_dependent_parameters.txt',
                                 'seq_dep_file_NA': 'NA_sequence_dependent_parameters.txt'
                                 })


class Analysis(SimulationComponent):
    """
    Methods used to interface with oxDNA simulation in jupyter notebook (currently in work)
    """

    # this is a Josh original. todo: use
    # todo: make these properties with lazy loading
    observables: dict[str, Observable]
    observables_data: dict[str, pd.DataFrame]

    # pandas dataframe for energy data
    _energy_df: pd.DataFrame

    def __init__(self, simulation):
        """ Set attributes to know all files in sim_dir and the input_parameters"""
        SimulationComponent.__init__(self, simulation)
        self.sim_files = simulation.sim_files
        self.observables = {}
        self.observables_data = {}
        self._energy_df = None

    # ------------------------------------------------------------------ #
    # oat (oxDNA Analysis Tools) methods                                  #
    # ------------------------------------------------------------------ #

    def _describe(self) -> tuple[TopInfo, TrajInfo]:
        """Return (top_info, traj_info) resolving traj → last_conf → dat fallback."""
        if not hasattr(self.sim.sim_files, 'top'):
            raise SimBuildMissingFileException(self.sim, "topology file")
        top = self.sim.sim_files.top.as_posix()
        if hasattr(self.sim.sim_files, 'traj'):
            traj = self.sim.sim_files.traj.as_posix()
        elif hasattr(self.sim.sim_files, 'last_conf'):
            traj = self.sim.sim_files.last_conf.as_posix()
        elif hasattr(self.sim.sim_files, 'dat'):
            traj = self.sim.sim_files.dat.as_posix()
        else:
            raise SimBuildMissingFileException(self.sim, "topology or trajectory file")
        return describe(top, traj)

    def describe(self):
        """Populate self.top_info and self.traj_info from sim files."""
        self.top_info, self.traj_info = self._describe()

    def _load_conf(self, conf: Configuration | str | Path) -> Configuration:
        """Return conf unchanged if already a Configuration; otherwise load from path (relative to sim_dir)."""
        if isinstance(conf, Configuration):
            return conf
        conf_path = str(conf) if os.path.isabs(str(conf)) else str(self.sim.sim_dir / conf)
        top_info, traj_info = describe(self.sim.sim_files.top.as_posix(), conf_path)
        return get_confs(top_info, traj_info, 0, 1)[0]

    def align(self, outfile: str = 'aligned.dat', ncpus: int = 1,
              indexes: list[int] | None = None,
              ref_conf: Configuration | None = None,
              no_center: bool = False) -> None:
        """Align trajectory to a reference structure, writing result to outfile in sim_dir."""
        _oat_align(
            self.sim.sim_files.traj.as_posix(),
            str(self.sim.sim_dir / outfile),
            ncpus, indexes, ref_conf, no_center,
        )

    def centroid(self, ref_conf: Configuration | str | Path = 'mean.dat',
                 outfile: str = 'centroid.dat', ncpus: int = 1,
                 indexes: list[int] | None = None) -> tuple[Configuration, float]:
        """Find the trajectory frame closest to ref_conf. Returns (centroid_conf, rmsd) and writes to outfile."""
        top_info, traj_info = self._describe()
        centroid_conf, rmsd = _oat_centroid(traj_info, top_info, self._load_conf(ref_conf), indexes, ncpus)
        write_conf(str(self.sim.sim_dir / outfile), centroid_conf, append=False, include_vel=False)
        return centroid_conf, rmsd

    def decimate(self, outfile: str = 'strided_trajectory.dat',
                 start: int = 0, stop: int = -1, stride: int = 10) -> None:
        """Reduce the number of configurations in the trajectory by striding."""
        _oat_decimate(self.sim.sim_files.traj.as_posix(), str(self.sim.sim_dir / outfile), start, stop, stride)

    def deviations(self, mean_conf: Configuration | str | Path = 'mean.dat',
                   ncpus: int = 1,
                   indexes: list[int] | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Calculate per-frame RMSD and per-particle RMSF relative to mean_conf. Returns (RMSDs, RMSFs)."""
        top_info, traj_info = self._describe()
        return _oat_deviations(traj_info, top_info, self._load_conf(mean_conf), indexes, ncpus)

    def distance(self, args: str = '') -> None:
        """Run oat distance with the given args string. See `oat distance -h` for options."""
        sp.run(['oat', 'distance'] + args.split(), cwd=self.sim.sim_dir, check=True)

    def generate_force(self, conf_file: Union[None, Path] = None, args: dict = {}) -> None:
        """Generate forces.txt from the current configuration, then load forces into the simulation."""
        if conf_file is None:
            conf_file = self.sim.input.initial_conf
        cmd = ['oat', 'generate_forces', str(self.sim.sim_files.input), str(conf_file)]
        for k, v in args.items():
            cmd += [f'-{k}', str(v)]
        sp.run(cmd, cwd=self.sim.sim_dir, check=True)
        self.sim.forces.load_from(self.sim.sim_dir / 'forces.txt')
        self.sim.add_forces()

    def mean(self, outfile: str = 'mean.dat', ncpus: int = 1,
             indexes: list[int] | None = None,
             ref_conf: Configuration | None = None) -> Configuration:
        """Compute the mean structure of the trajectory. Writes to outfile and returns the Configuration."""
        top_info, traj_info = self._describe()
        mean_conf = _oat_mean(traj_info, top_info, ref_conf, indexes, ncpus)
        write_conf(str(self.sim.sim_dir / outfile), mean_conf, append=False, include_vel=False)
        return mean_conf

    def minify(self, outfile: str = 'mini_trajectory.dat', ncpus: int = 1,
               digits: int | None = None, discard_a_vectors: bool = False) -> None:
        """Reduce trajectory file size by discarding precision."""
        top_info, traj_info = self._describe()
        _oat_minify(traj_info, top_info, str(self.sim.sim_dir / outfile), d=digits, a=discard_a_vectors, ncpus=ncpus)

    def output_bonds(self, visualize: bool = False, ncpus: int = 1,
                     fields: list[str] | None = None) -> tuple[np.ndarray | None, list[str]]:
        """Compute per-nucleotide bond energies. Returns (mean_energies_or_None, active_field_names)."""
        top_info, traj_info = self._describe()
        return _oat_output_bonds(traj_info, top_info, str(self.sim.sim_files.input),
                                 visualize=visualize, ncpus=ncpus, fields=fields)

    def oxDNA_PDB(self, configuration: str = 'mean.dat', direction: str = '35',
                  pdbfiles: str = '', args: str = '') -> None:
        """Convert an oxDNA configuration to PDB format."""
        cmd = ['oat', 'oxDNA_PDB', str(self.sim.sim_files.top), configuration, direction]
        if pdbfiles:
            cmd.append(pdbfiles)
        if args:
            cmd += args.split()
        sp.run(cmd, cwd=self.sim.sim_dir, check=True)

    def pca(self, mean_conf: Configuration | str | Path = 'mean.dat',
            outfile: str = 'pca.json', ncpus: int = 1) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Perform principal component analysis. Writes scree.png and outfile to sim_dir.
        Returns (coordinates, eigenvalues, eigenvectors)."""
        top_info, traj_info = self._describe()
        with PathContext(self.sim.sim_dir):
            coordinates, evalues, evectors = _oat_pca(traj_info, top_info, self._load_conf(mean_conf), ncpus)
        with open(self.sim.sim_dir / outfile, 'w') as f:
            dump({'evalues': evalues.tolist(), 'evectors': evectors.tolist(),
                  'coordinates': coordinates.tolist()}, f)
        return coordinates, evalues, evectors

    def radius_of_gyration(self, ncpus: int = 1,
                           indexes: np.ndarray | list[int] | None = None) -> np.ndarray:
        """Compute radius of gyration for each frame. Returns np.ndarray of shape (nconfs,)."""
        top_info, traj_info = self._describe()
        idx = np.array(indexes) if indexes is not None else np.array([])
        return _oat_rg(top_info, traj_info, idx, ncpus)

    def subset_trajectory(self, args: str = '') -> None:
        """Extract specified nucleotide indexes from the trajectory. See `oat subset_trajectory -h`."""
        sp.run(
            ['oat', 'subset_trajectory', str(self.sim.sim_files.traj), str(self.sim.sim_files.top)] + args.split(),
            cwd=self.sim.sim_dir, check=True,
        )

    def com_distance(self, base_list_file_1: str | None = None, base_list_file_2: str | None = None,
                     base_list_1: str | None = None, base_list_2: str | None = None,
                     args: str = '') -> None:
        """Find COM distance between two groups of particles. See `oat com_distance -h`."""
        def _write(val: str, prefix: str) -> str:
            counter = 0
            while (self.sim.sim_dir / f'{prefix}{counter}.txt').exists():
                counter += 1
            p = self.sim.sim_dir / f'{prefix}{counter}.txt'
            p.write_text(val.replace(',', ' '))
            return str(p)

        if base_list_file_1 is None:
            base_list_file_1 = _write(base_list_1, 'base_list_')
            base_list_file_2 = _write(base_list_2, 'base_list_')
        sp.run(
            ['oat', 'com_distance', '-i', str(self.sim.sim_files.traj),
             base_list_file_1, base_list_file_2] + args.split(),
            cwd=self.sim.sim_dir, check=True,
        )

    def angle(self, base_list_file_1: str | None = None, base_list_file_2: str | None = None,
              base_list_file_3: str | None = None, base_list_1: str | None = None,
              base_list_2: str | None = None, base_list_3: str | None = None,
              args: str = '') -> None:
        """Find angle between COM of three groups. See `oat angle -h`."""
        def _write(val: str, prefix: str) -> str:
            counter = 0
            while (self.sim.sim_dir / f'{prefix}{counter}.txt').exists():
                counter += 1
            p = self.sim.sim_dir / f'{prefix}{counter}.txt'
            p.write_text(val)
            return str(p)

        if base_list_file_1 is None:
            base_list_file_1 = _write(base_list_1, 'angle_index_list_')
            base_list_file_2 = _write(base_list_2, 'angle_index_list_')
            base_list_file_3 = _write(base_list_3, 'angle_index_list_')
        sp.run(
            ['oat', 'angle', '-i', str(self.sim.sim_files.traj),
             base_list_file_1, base_list_file_2, base_list_file_3] + args.split(),
            cwd=self.sim.sim_dir, check=True,
        )

    def determine_cv_sign(self, base_list_file_1: str | None = None, base_list_file_2: str | None = None,
                          base_list_file_3: str | None = None, base_list_1: str | None = None,
                          base_list_2: str | None = None, base_list_3: str | None = None,
                          args: str = '') -> None:
        """Determine CV sign from three groups of particles. See `oat determine_cv_sign -h`."""
        def _write(val: str, prefix: str) -> str:
            counter = 0
            while (self.sim.sim_dir / f'{prefix}{counter}.txt').exists():
                counter += 1
            p = self.sim.sim_dir / f'{prefix}{counter}.txt'
            p.write_text(val)
            return str(p)

        if base_list_file_1 is None:
            base_list_file_1 = _write(base_list_1, 'cv_sign_index_list_')
            base_list_file_2 = _write(base_list_2, 'cv_sign_index_list_')
            base_list_file_3 = _write(base_list_3, 'cv_sign_index_list_')
        sp.run(
            ['oat', 'determine_cv_sign', '-i', str(self.sim.sim_files.traj),
             base_list_file_1, base_list_file_2, base_list_file_3] + args.split(),
            cwd=self.sim.sim_dir, check=True,
        )

    # ------------------------------------------------------------------ #
    # Simulation inspection / notebook methods                            #
    # ------------------------------------------------------------------ #

    def get_init_conf(self) -> tuple[tuple[TopInfo, TrajInfo], Configuration]:
        """ Returns inital topology and dat file paths, as well as x,y,z info of the conf."""
        self.sim_files.parse_current_files()
        ti, di = describe(str(self.sim_files.top),
                          str(self.sim_files.dat))
        return (ti, di), get_confs(ti, di, 0, 1)[0]

    def get_last_conf(self) -> tuple[tuple[TopInfo, TrajInfo], Configuration]:
        """ Returns last topology and dat file paths, as well as x,y,z info of the conf."""
        self.sim_files.parse_current_files()
        ti, di = describe(str(self.sim_files.top),
                          str(self.sim_files.last_conf))
        return (ti, di), get_confs(ti, di, 0, 1)[0]

    def get_num_confs(self) -> int:
        """
        :return: the numbers of confs in the trajectory
        """
        self.sim_files.parse_current_files()
        _, di = describe(str(self.sim_files.top),
                          str(self.sim_files.traj))
        return di.nconfs

    def view_init(self):
        """ Interactivly view inital oxDNA conf in jupyter notebook."""
        (ti, di), conf = self.get_init_conf()
        oxdna_conf(ti, conf)
        sleep(2.5)

    def view_last(self):
        """ Interactivly view last oxDNA conf in jupyter notebook."""
        self.sim_files.parse_current_files()
        try:
            (ti, di), conf = self.get_last_conf()
            oxdna_conf(ti, conf)
        except Exception as e:
            # TODO: custom exception for missing conf, consider adapting one from pypatchy.patchy.stage
            raise Exception('No last conf file avalible')
        sleep(2.5)

    # TODO: code to add observables and run dnaAnalysis

    def observable_data(self, observable_name: str) -> pd.DataFrame:
        """
        loads data from an observable into a pandas dataframe
        :param observable_name: name of observable
        """
        assert observable_name in self.observables, f"No observable named `{observable_name}`!"
        # if data not loaded, load it
        if observable_name not in self.observables_data:
            # load observable data
            # cannot assume that observable name = file name! often not true?
            # todo: nicer read_csv
            self.observables_data[observable_name] = pd.read_csv(self.sim.sim_dir / (self.observables[observable_name].file_name + '.txt'), header=None, engine='pyarrow')
        return self.observables_data[observable_name]

    def get_observable_last_entry_time(self, obs: Observable) -> float:
        """
        :returns: the time of the last entry in the observable data
        """
        return self.runtime() - (self.runtime() % obs.print_every)

    def read_all_observables(self):
        """
        loads all observable data, doesn't return anything
        """
        for observable_name in self.observables:
            self.observable_data(observable_name)

    def get_conf_count(self) -> int:
        """ Returns the number of confs in trajectory file."""
        self.sim_files.parse_current_files()
        ti, di = describe(self.sim_files.top.as_posix(),
                          self.sim_files.traj.as_posix())
        return len(di.idxs)

    def get_conf(self, conf_id: int) -> tuple[tuple[TopInfo, TrajInfo], Configuration]:
        """ Returns x,y,z (and other) info of specified conf."""
        self.sim_files.parse_current_files()
        ti, di = describe(self.sim_files.top,
                          self.sim_files.traj)
        l = len(di.idxs)
        if conf_id < l:
            return (ti, di), get_confs(ti, di, conf_id, 1)[0]
        else:
            # TODO: custom exception
            raise Exception("You requested a conf out of bounds.")

    def get_conf_as_structure(self, conf_id: int) -> DNAStructure:
        """
        :returns: a conf at a given index as a DNAStructure object
        """
        assert 0 <= conf_id < self.get_num_confs(), "You requested a conf out of bounds."
        return load_dna_structure(self.sim_files.top, self.sim_files.traj, conf_id)


    def current_step(self) -> float:
        """ Returns the time-step of the most recently save oxDNA conf."""
        n_confs = float(self.get_conf_count())
        steps_per_conf = float(self.sim.input.input_dict["print_conf_interval"])
        return n_confs * steps_per_conf

    def view_conf(self, conf_id: int):
        """ Interactivly view oxDNA conf in jupyter notebook."""
        (ti, di), conf = self.get_conf(conf_id)
        oxdna_conf(ti, conf)
        sleep(2.5)

    @property
    def energy_df(self) -> pd.DataFrame:
        if self._energy_df is None:
            self.load_energy()
        return self._energy_df

    def load_energy(self):
        """
        Plot energy of oxDNA simulation.
        todo: make this a property, with an accessor that auto-reads the file if needed
        """
        self.sim_files.parse_current_files()
        try:
            sim_type = self.sim.input.input_dict['sim_type']
        except KeyError:
            sim_type = "MD"
        if sim_type == 'MC':
            df = pd.read_csv(self.sim_files.energy, sep='\\s+', names=['time', 'U', "acceptance_translational", "acceptance_rotational", "acceptance_volume"])
        elif sim_type == 'MD':
            df = pd.read_csv(self.sim_files.energy, sep='\\s+', names=['time', 'U', 'K', 'Etot'])
        elif sim_type == "VMMC":
            raise Exception("Use VMMC-specific analysis class for this")
        else:
            raise Exception(f"Unknown simulation type {sim_type}")

        self._energy_df = df


    def plot_energy(self, fig=None, ax=None, label=None, color=None):
        """ Plot energy of oxDNA simulation."""
        try:
            self.sim_files.parse_current_files()
            sim_type = self.sim.input.input_dict['sim_type']
            if (sim_type == 'MC') or (sim_type == 'VMMC'):
                df = pd.read_csv(self.sim_files.energy, sep='\\s+', names=['time', 'U', 'P', 'K', 'empty'])
            else:
                df = pd.read_csv(self.sim_files.energy, sep='\\s+', names=['time', 'U', 'P', 'K'])
            dt = float(self.sim.input.input_dict["dt"])
            steps = float(self.sim.input.input_dict["steps"])
            # df = df[df.U <= 10]
            # df = df[df.U >= -10]
            # make sure our figure is bigger
            if fig is None:
                plt.figure(figsize=(15, 3))
            # plot the energy
            if ax is None:
                if (sim_type == 'MC') or (sim_type == 'VMMC'):
                    plt.plot(df.time, df.U, label=label, color=color)
                else:
                    plt.plot(df.time / dt, df.U, label=label, color=color)
                plt.ylabel("Energy")
                plt.xlabel("Steps")
            else:
                if (sim_type == 'MC') or (sim_type == 'VMMC'):
                    ax.plot(df.time, df.U, label=label, color=color)
                else:
                    ax.plot(df.time / dt, df.U, label=label, color=color)
                ax.set_ylabel("Energy")
                ax.set_xlabel("Steps")

            if np.any(df.U > 10):
                print(self.sim.sim_dir)
                print('Energy is greater than 10, check for errors in the simulation')
            if np.any(df.U < -10):
                print(self.sim.sim_dir)
                print('Energy is less than -10, check for errors in the simulation')

        except Exception as e:
            # TODO: custom exception handling and exception raising
            print(f'{self.sim.sim_dir}: No energy file avalible')
            print(traceback.format_exc())

    def plot_observable(self, observable: Union[Observable, dict],
                        sliding_window: Union[False, Any] = False, fig: bool=True):
        """
        Plots data from an observable in a scatterplot.
        I (Josh) have preserved backwards compatibility for dict form observables
        """
        if isinstance(observable, dict):
            file_name = observable['output']['name']
            conf_interval = float(observable['output']['print_every'])
        else:
            assert isinstance(observable, Observable)
            file_name = observable.file_name
            conf_interval = observable.print_every

        df = pd.read_csv(self.sim.sim_dir / file_name, header=None, engine='pyarrow')
        if sliding_window is not False:
            df = df.rolling(window=sliding_window).sum().dropna().div(sliding_window)
        df = np.concatenate(np.array(df))
        sim_conf_times = np.linspace(0, conf_interval * len(df), num=len(df))
        if fig:
            plt.figure(figsize=(15, 3))
        plt.xlabel('steps')
        plt.ylabel(f'{os.path.splitext(file_name)[0]} (sim units)')
        plt.plot(sim_conf_times, df, label=self.sim.sim_dir.stem, rasterized=True)

    def hist_observable(self, observable: Union[Observable, dict], bins=10, fig=True):
        """
        plots data from an observable as a histogram
        I (Josh) have preserved backwards compatibility for dict form observables
        todo: use observable object
        """
        if isinstance(observable, dict):
            file_name = observable['output']['name']
            conf_interval = float(observable['output']['print_every'])
            df = pd.read_csv(self.sim.sim_dir / file_name, header=None)
            df = np.concatenate(np.array(df))
            sim_conf_times = np.linspace(0, conf_interval * len(df), num=len(df))
            if fig is True:
                plt.figure(figsize=(15, 3))
            plt.xlabel(f'{os.path.splitext(file_name)[0]} (sim units)')
            plt.ylabel(f'Probablity')
            H, bins = np.histogram(df, density=True, bins=bins)
            H = H * (bins[1] - bins[0])
            plt.plot(bins[:-1], H, label=self.sim.sim_dir.stem)
        else:
            assert isinstance(observable, Observable)

    # Unstable
    def view_traj(self, init=0, op=None):
        print('This feature is highly unstable and will crash your kernel if you scroll through confs too fast')
        # get the initial conf and the reference to the trajectory 
        (ti, di), cur_conf = self.get_conf(init)

        slider = widgets.IntSlider(
            min=0,
            max=len(di.idxs),
            step=1,
            description="Select:",
            value=init
        )

        output = widgets.Output()
        if op:
            min_v, max_v = np.min(op), np.max(op)

        def handle(obj=None):
            conf = get_confs(ti, di, slider.value, 1)[0]
            with output:
                output.clear_output()
                if op:
                    # make sure our figure is bigger
                    plt.figure(figsize=(15, 3))
                    plt.plot(op)
                    print(init)
                    plt.plot([slider.value, slider.value], [min_v, max_v], color="r")
                    plt.show()
                oxdna_conf(ti, conf)

        slider.observe(handle)
        display(slider, output)
        handle(None)

    def runtime(self) -> float:
        """ Returns the total runtime of the simulation in time-steps."""
        (top, traj), conf = self.get_last_conf()
        return conf.time


    def get_up_down(self, x_max: float, com_dist_file: str, pos_file: str):
        """
        todo: method description
        why is it structured like this?
        :param x_max: todo
        :param com_dist_file: presumable a file path, todo
        :param pos_file: presumable a file path, todo
        """
        key_names = ['a', 'b', 'c', 'p', 'va', 'vb', 'vc', 'vp']

        def process_pos_file(pos_file: str, key_names: list) -> dict:
            cms_dict = {}
            with open(pos_file, 'r') as f:
                pos = f.readlines()
                pos = [line.strip().split(' ') for line in pos]
                for idx, string in enumerate(key_names):
                    cms = np.transpose(pos)[idx]
                    cms = [np.array(line.split(','), dtype=np.float64) for line in cms]
                    cms_dict[string] = np.array(cms)
            return cms_dict

        def point_in_triangle(a, b, c, p):
            u = b - a
            v = c - a
            n = np.cross(u, v)
            w = p - a
            gamma = (np.dot(np.cross(u, w), n)) / np.dot(n, n)
            beta = (np.dot(np.cross(w, v), n)) / np.dot(n, n)
            alpha = 1 - gamma - beta
            return ((-1 <= alpha) and (alpha <= 1) and (-1 <= beta) and (beta <= 1) and (-1 <= gamma) and (gamma <= 1))

        def point_over_plane(a, b, c, p):
            u = c - a
            v = b - a
            cp = np.cross(u, v)
            va, vb, vc = cp
            d = np.dot(cp, c)
            plane = np.array([va, vb, vc, d])
            point = np.array([p[0], p[1], p[2], 1])
            result = np.dot(plane, point)
            return True if result > 0 else False

        def up_down(x_max: float, com_dist_file: str, pos_file: str) -> list:
            with open(com_dist_file, 'r') as f:
                com_dist = f.readlines()
            com_dist = [line.strip() for line in com_dist]
            com_dist = list(map(float, com_dist))
            cms_list = process_pos_file(pos_file, key_names)
            up_or_down = [point_in_triangle(a, b, c, p) for (a, b, c, p) in
                          zip(cms_list['va'], cms_list['vb'], cms_list['vc'], cms_list['vp'])]
            over_or_under = [point_over_plane(a, b, c, p) for (a, b, c, p) in
                             zip(cms_list['va'], cms_list['vb'], cms_list['vc'], cms_list['vp'])]

            # true_up_down = []
            # # print(up_or_down)
            # # print(over_or_under)
            # new_coms = []
            # for com, u_d, o_u in zip(com_dist, up_or_down, over_or_under):
            #     if u_d != o_u:
            #         if abs(com) > (x_max * 0.75):
            #             if u_d == 0:
            #                 new_coms.append(-com)
            #             else:
            #                 new_coms.append(com)      
            #         else:
            #             if o_u == 0:
            #                 new_coms.append(-com)
            #             else:
            #                 new_coms.append(com)   
            #     else:
            #         if o_u == 0:
            #             new_coms.append(-com)
            #         else:
            #             new_coms.append(com) 
            # com_dist = new_coms       

            com_dist = [-state if direction == 0 else state for state, direction in zip(com_dist, over_or_under)]

            # if np.mean(com_dist) > :
            #     com_dist = [dist for dist in com_dist if (np.sign(dist) == np.sign(np.mean(com_dist)))]
            # if (abs(max(com_dist) + min(com_dist)) < 2) :
            #     print(np.mean(com_dist))
            #     com_dist = [dist for dist in com_dist if (np.sign(dist) == np.sign(np.mean(com_dist)))]

            com_dist = [x_max - state if state > 0 else -x_max - state for state in com_dist]
            # if max(abs(max(com_dist)),  abs(min(com_dist))) > 15:
            #     com_dist = [dist if (np.sign(dist) == np.sign(np.mean(com_dist))) else -dist for dist in com_dist ]

            #             if max(abs(max(com_dist)),  abs(min(com_dist))) > 15:
            #                 com_dist = [abs(dist) if (np.sign(dist) == -1) else dist for dist in com_dist]

            com_dist = [np.round(val, 4) for val in com_dist]
            return com_dist

        return up_down(x_max, com_dist_file, pos_file)

    def view_cms_obs(self, xmax, print_every, sliding_window=False, fig=True):
        self.sim_files.parse_current_files()
        new_com_vals = self.get_up_down(xmax, self.sim_files.com_distance, self.sim_files.cms_positions)
        conf_interval = float(print_every)
        df = pd.DataFrame(new_com_vals)
        if sliding_window is not False:
            df = df.rolling(window=sliding_window).sum().dropna().div(sliding_window)
        df = np.concatenate(np.array(df))
        sim_conf_times = np.linspace(0, conf_interval * len(df), num=len(df))
        if fig is True:
            plt.figure(figsize=(15, 3))
        plt.xlabel('steps')
        plt.ylabel(f'End-to-End Distance (sim units)')
        plt.plot(sim_conf_times, df, label=self.sim.sim_dir.split("/")[-1])

    def hist_cms_obs(self, xmax, print_every, bins=10, fig=True):
        new_com_vals = self.get_up_down(xmax, self.sim_files.com_distance, self.sim_files.cms_positions)
        conf_interval = float(print_every)
        df = pd.DataFrame(new_com_vals)
        df = np.concatenate(np.array(df))
        sim_conf_times = np.linspace(0, conf_interval * len(df), num=len(df))
        if fig is True:
            plt.figure(figsize=(15, 3))
        plt.xlabel(f'End-to-End Distance (sim units)')
        plt.ylabel(f'Probablity')
        H, bins = np.histogram(df, density=True, bins=bins)
        H = H * (bins[1] - bins[0])
        plt.plot(bins[:-1], H, label=self.sim.sim_dir.split("/")[-1])


class SimFiles(SimulationComponent):
    """
    Parse the current files present in simulation directory
    """
    files_list: list[str]

    dat: Path
    top: Path
    traj: Path
    last_conf: Path
    force: Path
    input: Path
    input_js: Path
    observables: Path
    run_file: Path
    energy: Path
    com_distance: Path
    cms_positions: Path
    par: Path
    last_hist: Path
    hb_observable: Path
    potential_energy: Path
    all_observables: Path
    hb_contacts: Path
    run_time_custom_observable: Path

    def __init__(self, sim: Simulation):
        SimulationComponent.__init__(self, sim)
        if os.path.exists(self.sim.sim_dir):
            self.file_list = os.listdir(self.sim.sim_dir)
            self.parse_current_files()

    def parse_current_files(self):
        """
        searches directory for required simulation files and puts them as class member vars
        """
        if self.sim.sim_dir.exists():
            self.file_list: list[str] = os.listdir(self.sim.sim_dir)
        else:
            raise FileNotFoundError(f"Simulation directory {str(self.sim.sim_dir)} does not exist. Current cwd: {str(Path.cwd())}")
        for file in self.file_list:
            if not file.endswith('pyidx'):
                if file == 'trajectory.dat':
                    self.traj = self.sim.sim_dir / file
                elif file == 'last_conf.dat':
                    self.last_conf = self.sim.sim_dir / file
                elif file.endswith(".dat") and not any([
                    file.endswith("energy.dat"),
                    file.endswith("trajectory.dat"),
                    file.endswith("error_conf.dat"),
                    file.endswith("last_hist.dat"),
                    file.endswith("traj_hist.dat"),
                    file.endswith("last_conf.dat")
                ]):
                    self.dat = self.sim.sim_dir / file
                elif file.endswith('.top'):
                    self.top = self.sim.sim_dir / file
                elif file == 'forces.json':
                    self.force = self.sim.sim_dir / file
                elif file == 'input':
                    self.input = self.sim.sim_dir / file
                elif file == 'input.json':
                    self.input_js = self.sim.sim_dir / file
                elif file == 'observables.json':
                    self.observables = self.sim.sim_dir / file
                elif file == 'run.sh':
                    self.run_file = self.sim.sim_dir / file
                elif file.startswith('slurm'):
                    self.run_file = self.sim.sim_dir / file
                elif 'energy.dat' in file:
                    self.energy = self.sim.sim_dir / file
                elif 'com_distance' in file:
                    self.com_distance = self.sim.sim_dir / file
                elif 'cms_positions' in file:
                    self.cms_positions = self.sim.sim_dir / file
                elif 'par' in file:
                    self.par = self.sim.sim_dir / file
                elif 'last_hist.dat' in file:
                    self.last_hist = self.sim.sim_dir / file
                elif 'hb_observable.txt' in file:
                    self.hb_observable = self.sim.sim_dir / file
                elif 'potential_energy.txt' in file:
                    self.potential_energy = self.sim.sim_dir / file
                elif 'all_observables.txt' in file or "observables.txt" in file:
                    self.all_observables = self.sim.sim_dir / file
                elif 'hb_contacts.txt' in file:
                    self.hb_contacts = self.sim.sim_dir / file
                elif 'run_time_custom_observable.json' in file:
                    self.run_time_custom_observable = self.sim.sim_dir / file

    def __contains__(self, item: Union[Path, str]):
        """
        treat path and str quite differently
        """
        if isinstance(item, Path):
            return item in self.file_list
        else:
            return hasattr(self, item)

class SimBuildException(Exception, SimulationComponent, ABC):
    """
    base class for exceptions involving building a simulation.
    I've chosen not to make this an abstract class but generally it's better
    to use a more precise subclass
    """
    def __str__(self) -> str:
        return f"Error building simulation with file directory {str(self.file_dir)} and simulation directory {str(self.sim_dir)}"


class MissingTopConfException(SimBuildException):
    def __str__(self) -> str:
        return f"No specified topology and initial configuration files specified in the input file for simulation at {str(self.sim.sim_dir)}"


class SimBuildMissingFileException(SimBuildException):
    missing_file_descriptor: str

    def __init__(self, sim: Simulation, missing_file: str):
        SimulationComponent.__init__(self, sim)
        self.missing_file_descriptor = missing_file

    def __str__(self) -> str:
        return f"No {self.missing_file_descriptor} in directory {str(self.sim.file_dir)}"


def find_top_dat(directory: Path, sim: Union[Simulation, None] = None) -> tuple[Path, Path]:
    """
    Tries to find a top and dat file in the provided directory. simulation object is provided
    for err-messaging purposes only
    """
    # list files in simulation directory

    # skip inputs where we've already set top

    # skip inputs where we've already set top and
    return find_top_file(directory, sim), find_conf_file(directory, sim)


def find_top_file(directory: Path, sim: Union[Simulation, None] = None) -> Path:
    """
    Tries to find a top file in the provided directory. simulation object is provided
    for err-messaging purposes only
    """
    if not directory.exists():
        raise FileNotFoundError(f"{str(directory)} does not exist")
    try:
        return [file for file in directory.iterdir() if file.name.endswith('.top')][0]
    except IndexError:
        raise FileNotFoundError(f"No valid .top file found in directory {str(directory)}")


def find_conf_file(directory: Path, sim: Union[Simulation, None] = None) -> Path:
    """
    Tries to find a dat file in the provided directory. simulation object is provided
    for err-messaging purposes only
    """
    try:
        last_conf = [file for file in directory.iterdir()
                     if file.name.startswith('last_conf')
                     and not file.name.endswith('pyidx')][0]
    except IndexError:
        try:
            last_conf = [file for file in directory.iterdir() if file.name.endswith(".dat") and not any([
                file.name.endswith("energy.dat"),
                file.name.endswith("trajectory.dat"),
                file.name.endswith("error_conf.dat")])
                         ][0]
        except IndexError:
            if sim is not None:
                raise SimBuildException(sim, "initial conf file")
            else:
                raise FileNotFoundError(errno.ENOENT,
                                        os.strerror(errno.ENOENT),
                                        f"No valid .dat file found in directory {str(directory)}")
    return last_conf

def write_input_file(fp: Path, inputdict: dict):
    with fp.open('w') as f:
        # use oxpy.Context and oxpy.InputFile to write input file
        with oxpy.Context(print_coda=False):
            ox_input = oxpy.InputFile()
            for k, v in inputdict.items():
                if type(v) == list:
                    continue  # todo: throw warning?
                elif type(v) == dict:
                    continue  # todo: there is actually reason to write these
                else:
                    ox_input[k] = str(v)
            print(ox_input, file=f)
