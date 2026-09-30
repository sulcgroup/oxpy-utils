# purpose: create a DNA structure consisting of only selected bases
# authors: Azadeh Zare and Josh Evans
# mostly Josh Evans

import copy
import functools
from pathlib import Path
from typing import Union, Iterable

import numpy as np
from oxDNA_analysis_tools.UTILS.RyeReader import describe
from oxDNA_analysis_tools.deviations import deviations
from oxDNA_analysis_tools.mean import mean

from .dna_structure import DNAStructure, strand_from_info, DNABase, load_dna_structure
from ..utils.force import harmonic_trap, Force, ForceType
from ..oxdna_simulation import Simulation, BuildSimulation


class StructureSlicer(BuildSimulation):

    # uids of bases in the clipped structure which are adjacent to the set_slice points
    __endpoint_base_uids: list[int]

    slice_pts: list[tuple[int, int]]

    # uuids of bases to slice
    __slice_bases: list[int]

    __clipped_structure: Union[DNAStructure, None]

    __endpts_mean: Union[None, np.ndarray]
    __endpts_rmsfs: Union[None,np.ndarray]

    def __init__(self, sim: Simulation):
        super().__init__(sim)
        self.__endpoint_base_uids = list()
        self.__endpts_rmsfs = None
        self.__clipped_structure = None
        self.__endpts_mean = None
        self.__endpts_rmsfs = None
        self.__slice_bases = []
        self.slice_pts = []

    @functools.cache
    def starting_structure(self) -> DNAStructure:
        return load_dna_structure(*self.find_starting_top_dat())

    def clipped_structure(self) -> DNAStructure:
        return self.__clipped_structure

    def load_sampling_from(self, traj_file_name="trajectory.dat"):
        """
        loads rmsfs from trajectory
        """
        assert (self.file_dir / traj_file_name).is_file()
        top_info, traj_info = describe(str(self.file_dir / self.top_file_name), str(self.file_dir / traj_file_name))
        # UIDs are global counters, not 0-based particle indices; convert via base_id_reverse_map
        ep_oxview_idxs = [self.starting_structure().base_id_reverse_map[uid]
                          for uid in self.endpoint_bases()]
        conf = mean(traj_info, top_info, indexes=ep_oxview_idxs)
        self.__endpts_mean = conf.positions
        _, self.__endpts_rmsfs = deviations(traj_info, top_info, conf, indexes=ep_oxview_idxs)

    def endpoint_bases(self) -> list[int]:
        return self.__endpoint_base_uids

    def endpoint_rmsfs(self) -> Union[None, np.ndarray]:
        """Per-particle RMSF for endpoint bases; None until load_sampling_from is called."""
        return self.__endpts_rmsfs

    def set_slice_box(self, corner_1: np.ndarray, corner_2: np.ndarray, reverse: bool=False, behavior="replace") -> np.ndarray:
        """
        sets the slicer to remove nucleotides within or outside of a specified box
        Parameters:
            corner_1: corner of the box
            corner_2: corner of the box
            reverse: exclude nucleotides inside the box, instead of the default of outside
            behavior: how should the selection interact with existing sliced nucleotides. options: "union", "intersection", "replace"
        """
        assert corner_1.shape == corner_2.shape == (3,)
        assert behavior in ["union", "intersection", "replace"], f"Invalid set_slice behavior {behavior}"
        assert not (corner_1 == corner_2).any(), "A box dimension has zero size"
        lo, hi = np.minimum(corner_1, corner_2), np.maximum(corner_1, corner_2)
        base_positions = self.starting_structure().poss()
        inside = ((lo <= base_positions) & (base_positions <= hi)).all(axis=1)
        select = ~inside if reverse else inside
        self.set_slice(self._combine(np.nonzero(select)[0], behavior))

    def _combine(self, idxs: np.ndarray, behavior: str) -> np.ndarray:
        """combines newly selected oxview indexes with the current selection"""
        if behavior == "replace":
            return idxs
        rev = self.starting_structure().base_id_reverse_map
        current = np.array([rev[uid] for uid in self.__slice_bases], dtype=int)
        return np.union1d(idxs, current) if behavior == "union" else np.intersect1d(idxs, current)

    def set_slice_plane(self, points: np.ndarray, norm: np.ndarray, behavior="replace"):
        """
        method written by chatGPT
        Sets the slicer to keep nucleotides on one side of a defined plane.
        Parameters:
            points: 3x3 np array containing 3 points (as xyz coord sets) defining the plane
            norm: point not on the plane; nucleotides on the same side as this point will be kept
            behavior: how should the selection interact with existing sliced nucleotides.
                      Options: "union", "intersection", "replace"
        """
        assert points.shape == (3, 3)
        assert norm.shape == (3,)
        assert behavior in ["union", "intersection", "replace"], f"Invalid set_slice behavior {behavior}"

        # Define the plane
        v1 = points[1] - points[0]
        v2 = points[2] - points[0]
        normal_vec = np.cross(v1, v2)
        assert np.linalg.norm(normal_vec) > 0, "Plane points must not be collinear"

        # Normalize
        normal_vec = normal_vec / np.linalg.norm(normal_vec)

        # Get base positions
        base_positions = self.starting_structure().poss()

        # Determine the side of the plane the norm point lies on
        reference_side = np.dot(norm - points[0], normal_vec)

        # Select bases on the same side as the reference point
        dot_products = (base_positions - points[0]) @ normal_vec
        select = dot_products > 0 if reference_side > 0 else dot_products < 0
        idxs = np.nonzero(select)[0]

        self.set_slice(self._combine(idxs, behavior))

    def set_slice(self, bases: Iterable[int]):
        """

        Parameters:
            bases: oxview indexes of bases to keep when slicing the structure
        """
        bases = sorted(set(int(b) for b in bases))
        bases_set = set(bases)  # oxview-index membership check; __slice_bases holds UIDs, not indices
        struct = self.starting_structure()
        self.__endpoint_base_uids = []
        self.slice_pts = []
        self.__slice_bases = [struct.base_id_map[base].uid for base in bases]
        # nicking strands will mess with the indexes so we need to convert oxview base indexes
        # to uids immediately. A kept base is an endpoint if a neighbour on the same strand is discarded.
        for idx, uid in zip(bases, self.__slice_bases):
            is_endpoint = False
            for nb in (idx + 1, idx - 1):
                if nb in bases_set or not 0 <= nb < struct.nbases:
                    continue
                if struct.base_to_strand(idx) == struct.base_to_strand(nb):
                    self.slice_pts.append((uid, struct.base_id_map[nb].uid))
                    is_endpoint = True
            if is_endpoint:
                self.__endpoint_base_uids.append(uid)

    def slice_bases(self):
        return self.__slice_bases

    def do_slice(self):
        self.__clipped_structure = copy.deepcopy(self.starting_structure())
        # make nicks
        # loop all set_slice points we identified a moment ago
        for keep_base_uid, adj in self.slice_pts:
            # get strand id and local index in strand for the base
            strand_id, strand_slice_idx = self.__clipped_structure.strand_position(uid=keep_base_uid)
            _, adj_idx = self.__clipped_structure.strand_position(uid=adj)
            if strand_slice_idx > adj_idx:
                # if we are slicing in the 3' -> 5' direction
                self.__clipped_structure.nick(strand_id, adj_idx)
            else:
                # nick in the 5' -> 3' direction
                self.__clipped_structure.nick(strand_id, strand_slice_idx)

        # seperate strands to keep from strand to discard
        strands_to_keep: list[int] = list()
        strands_to_discard: list[int] = list()
        uids = self.slice_bases()
        for strand_id, strand in enumerate(self.__clipped_structure.strands):
            if any([base.uid in uids for base in strand]):
                strands_to_keep.append(strand_id)
                assert all(base.uid in uids for base in strand)
            else:
                strands_to_discard.append(strand_id)
                assert all(base.uid not in uids for base in strand)
        strands_to_discard.reverse()
        for strand_id in strands_to_discard:
            self.__clipped_structure.remove_strand(strand_id)
        assert all(uid in self.__clipped_structure.base_id_reverse_map for uid in self.endpoint_bases())
        self.__clipped_structure.export_top_conf(self.sim_dir / self.sim.input.get_top_file(),
                                                 self.sim_dir / self.sim.input.get_conf_file())

    def build_dat_top(self):
        self.do_slice()

    def add_endpoint_forces_simple(self, force_stiffness: float = 1000):
        """
        add simple forces to the sliced structure in order to make it hold together
        Parameters:
            sim_folder_path path to the folder to store the simulation files
        """
        for slice_pt in self.__endpoint_base_uids:
            base: DNABase = self.clipped_structure().get_base_by_uid(slice_pt)
            self.sim.add_force(Force(
                type=ForceType.HARMONIC_TRAP.value.type_name,
                particle=self.clipped_structure().base_index(base),
                pos0=base.pos,
                stiff=force_stiffness,
            ))
        return self.sim

    def add_endpoint_forces(self, force_stiffness: float = 1):
        """
        add forces based on a md simulation of the entire structure: each endpoint is trapped at
        its current position with a stiffness inversely proportional to its RMSF in the full simulation
        """
        if self.__endpts_rmsfs is None:
            raise RuntimeError("No sampling data; call load_sampling_from first (or use add_endpoint_forces_simple)")
        clipped = self.clipped_structure()
        for uid, rmsf in zip(self.endpoint_bases(), np.ravel(self.__endpts_rmsfs)):
            base = clipped.get_base_by_uid(uid)
            self.sim.add_force(Force(force_type="trap",
                                     particle=clipped.base_index(base),
                                     pos0=base.pos,
                                     stiff=force_stiffness / max(float(rmsf), 1e-3)))
        return self.sim
