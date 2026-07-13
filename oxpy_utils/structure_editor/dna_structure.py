from __future__ import annotations

import copy
import itertools
import json
import subprocess
import tempfile
import warnings
from collections import namedtuple
from datetime import datetime
from pathlib import Path
from typing import Union, Generator, Iterable, Optional
from random import choice

import numpy as np
from pygltflib import GLTF2, Scene, Node, Mesh, Buffer, BufferView, Accessor, Asset, Primitive, Material, \
    PbrMetallicRoughness
import struct
from oxDNA_analysis_tools.UTILS.RyeReader import get_traj_info, linear_read
from oxDNA_analysis_tools.UTILS.data_structures import TopInfo, Configuration

from ..defaults.defaults import SEQ_DEP_PARAMS
from ..utils.util import rotation_matrix, process_path, si_units

# Path to tacoxDNA/src directory. Override at runtime if tacoxDNA lives elsewhere.
TACOXDNA_SRC: Optional[Path] = None

# universal indexer for residues
# DOES NOT CORRESPOND TO LINEAR INDEXING IN TOP AND CONF FILES!!!!
# also NO guarentee that all residues will be continuous! or like meaningfully connected at all!
RESIDUECOUNT = 0


def GEN_UIDS(n: int) -> range:
    global RESIDUECOUNT  # hate it
    r = range(RESIDUECOUNT, RESIDUECOUNT + n)
    RESIDUECOUNT += n
    return r


DNABase = namedtuple('DNABase', ['uid', 'base', 'pos', 'a1', 'a3'])

complement_translation = str.maketrans('ACGT', 'TGCA')

def rc(seq: str) -> str:
    return seq.translate(complement_translation)[::-1]

def wcbp(b1: str, b2: str) -> bool:
    """
    checks if two bases or strands are reverse compliments
    """
    return rc(b1) == b2

def get_rotation_matrix(axis: np.ndarray,
                        angle: Union[int, float, np.float64, np.float32],
                        units: Union[None, str] = None):
    """
    Gets. a rotation matrix around an axis? using rodreguis' formula?
    The main thing this provides is flexability with multiple types
    """
    # if additional angle info provided
    if units:
        if units in ["degrees", "deg", "o"]:
            angle = (np.pi / 180.) * angle  # degrees to radians
            # angle = np.deg2rad (anglest[0])
        elif units == "bp":
            # Notice that the choice of 35.9 DOES NOT correspond to the minimum free energy configuration.
            # This is usually not a problem, since the structure will istantly relax during simulation, but it can be
            # if you need a configuration with an equilibrium value for the linking number.
            # The minimum free energy angle depends on a number of factors (salt, temperature, length, and possibly more),
            # so if you need a configuration with 0 twist make sure to carefully choose a value for this angle
            # and force it in some way (e.g. by changing the angle value below to something else in your local copy).
            # Allow partial bp turns
            angle = angle * (np.pi / 180.) * 35.9
            # Older versions of numpy don't implement deg2rad()
            # angle = int(anglest[0]) * np.deg2rad(35.9)
        # otherwise assume radians
    return rotation_matrix(axis, angle)


class DNAStructureStrand:
    """
    Wrapper class for DNA strand top and conf info
    Note that this object does NOT contain base or strand IDs!
    like oxDNA itself, 3'->5'
    """
    bases: np.ndarray  # char representations of bases
    positions: np.ndarray  # xyz coords of bases
    a1s: np.ndarray  # a1s of bases orientations
    a3s: np.ndarray  # a3s of bases orientations

    def get_a2s(self) -> np.ndarray:
        return np.cross(self.a3s, self.a1s)

    a2s: np.ndarray = property(get_a2s)

    def __init__(self,
                 bases: np.ndarray,
                 positions: np.ndarray,
                 a1s: np.ndarray,
                 a3s: np.ndarray,
                 uids: Union[np.ndarray, None] = None):
        assert bases.size == positions.shape[0] == a1s.shape[0] == a3s.shape[0], "Mismatch in strand raws shapes!"
        assert all([b in ("A", "T", "C", "G", "D", "U") for b in bases]), "Invalid base!" # incl. dummy base and uracil
        assert np.all(~np.isnan(positions))
        if uids is None:
            self.very_global_uids = np.array(GEN_UIDS(len(bases)))
        else:
            self.very_global_uids = uids
        self.bases = bases
        self.positions = positions
        self.a1s = a1s
        self.a3s = a3s

    def __getitem__(self, item: Union[int, slice]) -> Union[DNABase, DNAStructureStrand]:
        """
        Returns: a residue object if a position is specified, or a substrand if a slice is specified
        """
        if isinstance(item, int):
            return DNABase(self.very_global_uids[item], self.bases[item], self.positions[item, :], self.a1s[item, :],
                           self.a3s[item, :])
        elif isinstance(item, slice):
            return DNAStructureStrand(self.bases[item],
                                      self.positions[item, :],
                                      self.a1s[item, :],
                                      self.a3s[item, :],
                                      self.very_global_uids[item])
        else:
            raise Exception(f"Invalid indexer {item}")

    def __len__(self):
        return len(self.bases)

    def assign(self, other: DNAStructureStrand, start: int, stop: int, refr_uids="new"):
        """
        Modifies the strand in-place, substituting the data from the other strand at the start and stop positions
        """
        assert start >= 0, f"Invalid start index {start}"
        assert stop <= len(self), f"Invalid stop index {stop}"
        assert stop > start, f"Invalid indexes {start},{stop}"
        assert start - stop == -len(other), f"Length of target region to modify {start - stop} does not match length " \
                                           f" of data source strand object, {len(other)}."
        # assign values
        self.bases[start:stop] = other.bases
        self.positions[start:stop, :] = other.positions
        self.a1s[start:stop, :] = other.a1s
        self.a3s[start:stop, :] = other.a3s
        if refr_uids == "new":
            self.very_global_uids[start:stop] = np.array(GEN_UIDS(stop - start))
        elif refr_uids == "cpy":  # USE WITH CAUTION
            self.very_global_uids[start:stop] = other.very_global_uids

    def append(self, other: Union[DNAStructureStrand, DNABase], cpy_uids=False):
        """
        Depending on the parameter, either concatinates a strand on the 5' of this strand or
        appends a single residue to the end of this strand, in-place
        """
        if isinstance(other, DNABase):
            self.append(DNAStructureStrand(np.array(other.base),
                                           other.pos[np.newaxis, :],
                                           other.a1[np.newaxis, :],
                                           other.a3[np.newaxis, :]))
        elif isinstance(other, DNAStructureStrand):
            self.very_global_uids = np.concatenate([self.very_global_uids, other.very_global_uids if cpy_uids
            else np.array(GEN_UIDS(len(other)))])
            self.bases = np.concatenate([self.bases, other.bases])
            self.positions = np.concatenate([self.positions, other.positions], axis=0)
            self.a1s = np.concatenate([self.a1s, other.a1s], axis=0)
            self.a3s = np.concatenate([self.a3s, other.a3s], axis=0)

    def prepend(self, other: Union[DNAStructureStrand, DNABase], cpy_uids=False):
        """
        Depending on the parameter, either concatinates a strand on the 3' of this strand or
        appends a single residue to the end of this strand, in-place
        """
        if isinstance(other, DNABase):
            self.prepend(DNAStructureStrand(np.array(other.base),
                                            other.pos[np.newaxis, :],
                                            other.a1[np.newaxis, :],
                                            other.a3[np.newaxis, :]))
        elif isinstance(other, DNAStructureStrand):
            self.very_global_uids = np.concatenate([other.very_global_uids if cpy_uids
                                                    else np.array(GEN_UIDS(len(other))), self.very_global_uids])
            self.bases = np.concatenate([other.bases, self.bases])
            self.positions = np.concatenate([other.positions, self.positions], axis=0)
            self.a1s = np.concatenate([other.a1s, self.a1s], axis=0)
            self.a3s = np.concatenate([other.a3s, self.a3s], axis=0)

    def __add__(self, other: Union[DNAStructureStrand, DNABase]):
        """
        Combines a strand with another strand or a DNA base and returns the result, without
        modifying the original strand
        """
        cpy = copy.deepcopy(self)
        cpy.append(other)
        return cpy

    def __iter__(self) -> Generator[DNABase, None, None]:
        # iterate bases in this strand
        for i in range(len(self)):
            yield DNABase(self.very_global_uids[i],
                          self.bases[i],
                          self.positions[i, :],
                          self.a1s[i, :],
                          self.a3s[i, :])

    def transform(self, rot: np.ndarray = np.identity(3), tran: np.ndarray = np.zeros(3)):
        """
        Transforms in place
        """
        assert rot.shape == (3, 3), "Wrong shape for rotation!"
        assert np.linalg.det(rot) - 1 < 1e-6, f"Rotation matrix {rot} has nonzero determinate {np.linalg.det(rot)}"
        assert tran.shape in [(3,), (3, 1)], "Wrong shape for translaton!"
        self.positions[:,:] = np.einsum('ij, ki -> kj', rot, self.positions) + tran
        self.a1s[:,:] = np.einsum('ij, ki -> kj', rot, self.a1s)
        self.a3s[:,:] = np.einsum('ij, ki -> kj', rot, self.a3s)
        return self # so we can chain transformations

    def validate(self) -> bool:
        # a1s should all be unit vectors
        if not np.all((np.sum(self.a1s * self.a1s, axis=1) - 1) < 1e-6):
            return False
        # a3s should all be unit vectors
        if not np.all((np.sum(self.a3s * self.a3s, axis=1) - 1) < 1e-6):
            return False
        # all dot products of a1 and a3 should be zero
        if not np.all((np.sum(self.a1s * self.a3s, axis=1)) < 1e-6):
            return False
        # TODO: optional check for distance between adjacent bases?
        return True

    def seq(self, from_5p: bool = False) -> str:
        if from_5p:
            return "".join([*reversed(self.bases)])
        else:
            return "".join(self.bases)
        
        
    def mutate_sequence(self, new_sequence:str, start: int, stop: Union[int, None] = None, five_prime:bool = True):
        """
        Mutate base sequences for the strand in-place, substituting the using a string in 5` to 3` order
        """
        
        assert start >= 0, f"Invalid start index {start}"
        if stop is not None:
            assert stop <= len(self), f"Invalid stop index {stop}"
            assert stop > start, f"Invalid indexes {start},{stop}"
            assert start - stop == -len(new_sequence), f"Length of target region to modify {start - stop} does not match length " \
                                            f" of data source strand object, {len(new_sequence)}."
        else:
            assert len(new_sequence) == 1
        
        coding_seq_str_a = np.array([char if char != 'U' else 'T' for char in new_sequence], dtype=np.str_)
        if five_prime:
            coding_seq_str_a = coding_seq_str_a[::-1]
        
        if stop is not None:
            self.bases[start:stop] = coding_seq_str_a[:]
        else:
            self.bases[start] = coding_seq_str_a[0]

    def __str__(self):
        return "".join(self.bases)

    def index_of(self, **kwargs) -> int:
        """
        Parameters:
            base (DNABase) a DNA base
        Returns:
            the index of the provided base
        """
        assert len(kwargs) == 1
        if "base" in kwargs:
            assert isinstance(kwargs["base"], DNABase)
            return self.index_of(uid=kwargs["base"].uid)
        else:
            base_uid = kwargs["uid"]
            for (idx, uid) in enumerate(self.very_global_uids):
                if base_uid == uid:
                    return idx
            raise Exception(f"No base in strand {str(self)} with uid {base_uid}")

# this shit was originally in a file called helix.py
# i cannot be held resposible for this
BASE_BASE = 0.3897628551303122  # helical distance between two bases? in oxDNA units?
POS_BASE = 0.4  # I think this is the conversion factor from base pairs to oxDNA units????
# gonna say... distance between the helix center of mass of dsDNA to base position?
CM_CENTER_DS = POS_BASE + 0.2
POS_BACK = -0.4
FENE_EPS = 2.0


def strand_from_info(strand_info: list[tuple[chr, np.ndarray, np.ndarray, np.ndarray]]) -> DNAStructureStrand:
    """
    Parameters:
         strand_info (list): a list representing the strand, where each item is a tuple [base, position, a1, a3]

    """
    bases, positions, a1s, a3s = zip(*strand_info)
    return DNAStructureStrand(np.array(bases),
                              np.array(positions),
                              np.array(a1s),
                              np.array(a3s))


def construct_strands(bases: str,
                      start_pos: np.ndarray,
                      helix_direction: np.ndarray,
                      perp: Union[None, bool, np.ndarray] = None,
                      rot: float = 0.,
                      rbases: Union[str, None] = None
                      ) -> tuple[DNAStructureStrand,
                                 DNAStructureStrand]:
    """
    Constructs a pair of antiparallel complimentary DNA strands
    You can discard one of the strands if it isn't needed
    Parameters:
        bases the bases for the strand, with 5' and 3' ends indicted
        rbases the bases for the compilemnt strant, also in oxDNA 3'->5' notation. if not provided will default to rc(baes)
    """
    # we need numpy array for these
    if bases[1] != "'":
        warnings.warn("Strand does not start with a 5' or 3' indicator! The program will assume 3'->5' direction!!! WARN!!!")

    elif bases[0] == "5":
        assert bases[:3] == "5'-" and bases[-3:] == "-3'"
        bases = bases[-4:2:-1] # rest ofthe method expects 3' -> 5'
    else:
        assert bases[:3] == "3'-" and bases[-3:] == "-5'"
        bases = bases[3:-3]
    # we need to find a vector orthogonal to dir
    # compute magnitude of helix direction vector
    dir_norm = np.sqrt(np.dot(helix_direction, helix_direction))
    # if the vector is zero-length
    if dir_norm < 1e-10:
        # arbitrary default helix direction
        helix_direction = np.array([0, 0, 1])
    else:
        # normalize helix direction vector
        helix_direction /= dir_norm

    if perp is None or perp is False:
        v1 = np.random.random_sample(3)
        v1 -= helix_direction * (np.dot(helix_direction, v1))
        v1 /= np.sqrt(sum(v1 * v1))
    else:
        v1 = perp

    # get bases for sequences
    if rbases is None:
        rbases = rc(bases)
    else:
        assert len(bases) >= len(rbases), "For programmatic reasons, the longer strand should always " \
                                          "be the first arg. I could fix this but don't feel like it"

    # Setup initial parameters
    # and we need to generate a rotational matrix
    # R0 is helix starting rotation
    R0 = get_rotation_matrix(helix_direction, rot)
    # R is the rotation matrix that's applied to each consecutive nucleotide to make the helix a helix
    R = get_rotation_matrix(helix_direction, 1, "bp")
    a1 = v1
    a1 = np.dot(R0, a1)
    # rb = positions along center axis of helix
    rb = np.array(start_pos)
    a3 = helix_direction

    # order here is base, position, a1, a3
    strand_info: list[tuple[chr, np.ndarray, np.ndarray, np.ndarray]] = []
    for i, base in enumerate(bases):
        # compute nucleotide position, a1, a3
        strand_info.append((base, rb - CM_CENTER_DS * a1, a1, a3,))  # , sequence[i]))
        # if this isn't the last base in the sequence
        if i != len(bases) - 1:
            # rotate a1 vector by helix rot
            a1 = np.dot(R, a1)
            # increment position by a3 vector
            rb += a3 * BASE_BASE
    fwd_strand = strand_from_info(strand_info)

    # Fill in complement strand
    strand_info = []  # reuse var
    for ridx, rbase in enumerate(rbases):
        # Note that the complement strand is built in reverse order
        # first compute idx in fwd strand
        i = len(bases) - ridx - 1
        a1 = -fwd_strand[i].a1
        a3 = -fwd_strand[i].a3
        nt2_cm_pos = -(FENE_EPS + 2 * POS_BACK) * a1 + fwd_strand[i].pos
        strand_info.append((rbase, nt2_cm_pos, a1, a3,))
    return fwd_strand, strand_from_info(strand_info)

def gen_helix_coords(n: int,
                     start_pos: np.ndarray,
                      helix_direction: np.ndarray,
                      perp: Union[None, bool, np.ndarray] = None,
                      rot: float = 0.
                      ) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray],
                                 tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    generates sets of helix coordinates
    Parameters:
        n:  number of bases for which to generate coords
        start_pos: the position of the first base
        helix_direction: unit vector specifying direction of helix
        perp: a3 vectpr
        rot: rotation of first nucleotide
    Returns: a pair of tuples of three arrays, in form (pos, a1, a3)
    """
    # todo: make this not redundant with above method
    # we need numpy array for these

    # we need to find a vector orthogonal to dir
    # compute magnitude of helix direction vector
    dir_norm = np.sqrt(np.dot(helix_direction, helix_direction))
    # if the vector is zero-length
    if dir_norm < 1e-10:
        # arbitrary default helix direction
        helix_direction = np.array([0, 0, 1])
    else:
        # normalize helix direction vector
        helix_direction /= dir_norm

    if perp is None or perp is False:
        v1 = np.random.random_sample(3)
        v1 -= helix_direction * (np.dot(helix_direction, v1))
        v1 /= np.sqrt(sum(v1 * v1))
    else:
        v1 = perp

    # Setup initial parameters
    # and we need to generate a rotational matrix
    # R0 is helix starting rotation
    R0 = get_rotation_matrix(helix_direction, rot)
    # R is the rotation matrix that's applied to each consecutive nucleotide to make the helix a helix
    R = get_rotation_matrix(helix_direction, 1, "bp")
    a1 = v1
    a1 = np.dot(R0, a1)
    # rb = positions along center axis of helix
    rb = np.array(start_pos)
    a3 = helix_direction

    # order here is position, a1, a3
    s1_info: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for i in range(n):
        # compute nucleotide position, a1, a3
        s1_info.append((rb - CM_CENTER_DS * a1, a1, a3,))  # , sequence[i]))
        # if this isn't the last base in the sequence
        if i != n - 1:
            # rotate a1 vector by helix rot
            a1 = np.dot(R, a1)
            # increment position by a3 vector
            rb += a3 * BASE_BASE

    # Fill in complement strand
    s2_info: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for ridx in range(n):
        # Note that the complement strand is built in reverse order
        # first compute idx in fwd strand
        i = n - ridx - 1
        a1 = -s1_info[i][1]
        a3 = -s1_info[i][2]
        nt2_cm_pos = -(FENE_EPS + 2 * POS_BACK) * a1 + s1_info[i][0]
        s2_info.append((nt2_cm_pos, a1, a3,))

    s1_info = (np.stack([pos for pos, a1, a3 in s1_info]),
               np.stack([a1 for pos, a1, a3 in s1_info]),
               np.stack([a3 for pos, a1, a3 in s1_info]))
    s2_info = (np.stack([pos for pos, a1, a3 in s2_info]),
               np.stack([a1 for pos, a1, a3 in s2_info]),
               np.stack([a3 for pos, a1, a3 in s2_info]))
    return s1_info, s2_info

def to_oxview_color(color: Union[str, list[float]], basis=1.) -> int:
    """
    converts various normal color formats to oxview's bad one
    :param color: either a hex string like #RRGGBB or a list of three floats between 0 and 1
    :param basis: if color is a list of floats, the basis to use for scaling. defaults to 1.0, but if you have colors in 0-255 range you can set this to 255
    :return: integer color in oxview format
    """
    if isinstance(color, str):
        if not color.startswith("#"):
            raise ValueError("color must be in format #RRGGBB")
        return int(color[1:], 16)
    elif isinstance(color, list):
        assert len(color) == 3, "color list must be length 3"
        assert all([0.0 <= c / basis <= 1.0 for c in color]), f"color list values must be between 0 and {basis}"
        # good lord please tell me this works
        return (int(color[0] / basis * 255) << 16) + (int(color[1] / basis * 255) << 8) + int(color[2] / basis * 255)
    else:
        raise TypeError("color must be a string or list of floats")

class DNAStructure:
    """
    This class is intended to import, edit, and export oxDNA confs, since the oat library to
    edit confs is... very very bad
    TODO: incorporate this into OAT

    The class doesn't explicity store base IDs but rather constructs them on-the-fly

    """
    strands: list[DNAStructureStrand]
    time: int
    box: np.ndarray
    energy: np.ndarray
    clustermap: dict[int, int]  # mapping of base uids to cluster indexes
    clusters: list[set[int]]  # list of cluster info, where each cluster is a set of base uids

    # base uuid -> color
    # colors are stored as
    base_coloration: dict[int, int]

    # mapper for base idxs from file-loaded confs. these should stay constant during edit but will
    # NOT be carried over if the edits are saved!
    # keys are base idxs (like you'd find in OxView), values are DNABase objects
    # TODO: make this a custom object that we can slice like a numpy array
    base_id_map: dict[int, DNABase]
    # mapper for base IDs from the uids used by ipy_oxDNA back onto oxView indexes
    base_id_reverse_map: dict[int, int]

    def __init__(self,
                 strands: list[DNAStructureStrand],
                 t: int = 0,
                 box: Optional[np.ndarray] = None,
                 energy: np.ndarray = np.zeros(3),
                 clusters: list[set[int]] = None):
        assert all([isinstance(strand, DNAStructureStrand) for strand in strands]),\
            f"Invalid type for strands parameter: {type(strands)}"
        self.strands = strands
        assert isinstance(t, int) or isinstance(t, float), f"Invalid type for t {type(t)}"
        self.time = t
        if isinstance(box, list) or isinstance(box, tuple):
            assert len(box) == 3, "Wrong size for box!"
            box = np.array(box)
        elif box is None:
            pass # should be ok to start with an invalidated box
        else:
            assert isinstance(box, np.ndarray), f"Invalid box python object type {type(box)}"
            assert box.shape == (3,), "Wrong size for box!"
        self.box = box
        self.energy = energy
        self.clustermap = dict()
        if clusters is not None:
            self.clusters = clusters
            for i, cluster in enumerate(clusters):
                for uid in cluster:
                    self.clustermap[uid] = i
        else:
            self.clusters = list()
        self.base_id_map = dict()
        self.base_id_reverse_map = dict()

        # write base id mapping
        indexer = 0
        self.base_coloration = dict()
        self.reindex_base_ids()

    def clone(self, copy_uuids: bool = False) -> DNAStructure:
        """
        creates a copy of this structure, with new uids
        """
        cpy: DNAStructure = copy.deepcopy(self)
        if not copy_uuids:
            uid_map: dict[int, int] = dict()
            for strand in cpy.strands:
                new_uids = GEN_UIDS(len(strand))
                uid_map.update(dict(zip(strand.very_global_uids, new_uids)))
                strand.very_global_uids = new_uids

        cpy.clustermap = {
            uid_map[uid]: cluster
            for uid, cluster in cpy.clustermap.items()
        }
        if not copy_uuids:
            cpy.clusters = [[uid_map[base] for base in cluster] for cluster in cpy.clusters]
        else:
            cpy.clusters = [[base for base in cluster] for cluster in cpy.clusters]
        return cpy

    def get_num_strands(self) -> int:
        return len(self.strands)

    nstrands: int = property(get_num_strands)

    def get_num_bases(self) -> int:
        return sum([len(strand) for strand in self.strands])

    nbases: int = property(get_num_bases)

    def molarity(self, interaction_type: str = "DNA", n_per_box=1) -> float:
        """
        compute molarity of the simulation
        :return: molarity in mol/L
        """
        # compute box volume in liters
        box_size = si_units(self.box, interaction_type, "du").prod() / 1000
        return (n_per_box / 6.022e23) / box_size

    def get_base(self, base_idx: int) -> DNABase:
        """
        Returns the residue at the given position in the structure, indexed from 0
        """
        indexer = 0
        for strand_idx, strand in enumerate(self.strands):
            # if the indexer + the length of the strand is greater than the length of the strand
            if indexer + len(strand) > base_idx:
                # return the base in the strand
                return strand[base_idx - indexer]
            # update indexer
            indexer += len(strand)
        raise Exception(f"Base index {base_idx} is greater than total number of bases {indexer}!")

    def get_base_by_uid(self, base_uid: int) -> DNABase:
        for base in self.strands[self.strand_for_uid(base_uid)]:
            if base.uid == base_uid:
                return base
        raise Exception(f"No base with uid {base_uid} in structure!")

    def base_index(self, base: Union[DNABase, int]) -> int:
        """
        given a base (either a DNABase object or a base UID), return its index in the topology
        there should probably be a faster alternative for this
        """
        if isinstance(base, DNABase):
            return self.base_index(base.uid)
        else:
            for i,b in enumerate(self.iter_bases()):
                if b.uid == base:
                    return i
            raise Exception(f"Structure has no base with UID {base}")

    def strand_for_uid(self, base_uid: int) -> int:
        for strand_idx, strand in enumerate(self.strands):
            if base_uid in strand.very_global_uids:
                return strand_idx
        raise Exception(f"No strand in the structure contains a base with uid {base_uid}!")

    def reindex_base_ids(self):
        """
        updates self.base_id_map and self.base_id_reverse_map
        somewhat computationally intensive, avoid overusing
        """
        self.base_id_map = dict()
        self.base_id_reverse_map = dict()
        indexer = 0
        for strand_idx, strand in enumerate(self.strands):
            # if the indexer + the length of the strand is greater than the length of the strand
            for base_idx, base in enumerate(strand):
                self.base_id_map[indexer + base_idx] = base
                self.base_id_reverse_map[base.uid] = indexer + base_idx
            # update indexer
            indexer += len(strand)

    def base_to_strand(self, base_idx: int) -> int:
        """
        Returns the index of the strand containing the base with index i
        """
        indexer = 0
        for strand_idx, strand in enumerate(self.strands):
            # if the indexer + the length of the strand is greater than the length of the strand
            if indexer + len(strand) > base_idx:
                # return the strand index
                return strand_idx
            # update indexer
            indexer += len(strand)
        raise Exception(f"Base index {base_idx} is greater than total number of bases {indexer}!")

    def strand_position(self, **kwargs) -> tuple[int, int]:
        """
        Given a base, finds the position of the base in its strand
        """
        if "idx" in kwargs:
            base_idx = kwargs["idx"]
            indexer = 0
            for strand_idx, strand in enumerate(self.strands):
                # if the indexer + the length of the strand is greater than the length of the strand
                if indexer + len(strand) > base_idx:
                    # return the strand index
                    return strand_idx, base_idx - indexer
                # update indexer
                indexer += len(strand)
            raise Exception(f"Base index {base_idx} is greater than total number of bases {indexer}!")
        else:
            assert "uid" in kwargs, "`strand_position` method envoked without either required arg `idx` or `uid`"
            base_uid = kwargs["uid"]
            strand_idx = self.strand_for_uid(base_uid)
            return strand_idx, self.strands[strand_idx].index_of(uid=base_uid)

    def get_strand(self, strand_id) -> DNAStructureStrand:
        assert -1 < strand_id < self.nstrands, f"Invalid strand {strand_id}"
        return self.strands[strand_id]

    def nick(self, strand_id: int, n: int):
        """
        Adds a nick in a strand, turning it into two strands
        Parameters:
            strand_id: the id of the strand to nick
            n: the position (3' -> 5') to nick after. So n=0 would clip off the first nucleotide.
                If a negative number is provided, the method will index 5'->3'
        Returns: a tuple of the strand IDs of the two strands produced
        """
        assert 0 <= strand_id < self.get_num_strands(), f"Invalid strand ID {strand_id}"
        if n < 0:
            return self.nick(strand_id, n + len(self.get_strand(strand_id)) - 1)
        assert 0 <= n <= len(self.get_strand(strand_id)) - 1, f"Invalid base index {n} on {strand_id}"
        strand = self.strands[strand_id]
        # length-checking
        strand_start_length = len(strand)
        # create a new strand ending at the slice point
        new_strand = strand[n+1:]
        # slice strand in-place
        self.strands[strand_id] = strand[:n+1]
        assert len(new_strand) + len(self.strands[strand_id]) == strand_start_length, \
            "Mismatch between lengths of daughter strands and length of parent strand!"
        self.add_strand(new_strand, False)


    def next_strand_base(self, base: DNABase) -> Union[DNABase, None]:
        """
        Returns:
            the next base on the same strand
        """
        strand_id = self.strand_for_uid(base.uid)
        idx = self.strands[strand_id].index_of(base=base)
        return self.strands[strand_id][idx+1] if idx + 1 < len(self.strands[strand_id]) else None

    def prev_strand_base(self, base: DNABase) -> DNABase:
        """
        Returns:
            the previous base on the same strand
        """
        strand_id = self.strand_for_uid(base.uid)
        idx = self.strands[strand_id].index_of(base=base)
        return self.strands[strand_id][idx-1] if idx > 0 else None

    def is_tip(self, base: DNABase) -> bool:
        """
        Checks if a base is on either end of its strand
        """
        return self.next_strand_base(base) is None or self.prev_strand_base(base) is None

    def export_top_conf(self, top_file_path: Path, conf_file_path: Path):
        if not self.has_valid_box():
            self.inbox().export_top_conf(top_file_path, conf_file_path)
        else:
            # assert self.validate(), "Invaid structure!"
            with top_file_path.open("w") as topfile, conf_file_path.open("w") as conffile:
                # write conf header
                conffile.write(f't = {int(self.time)}\n')
                conffile.write(f"b = {' '.join(self.box.astype(str))}\n")
                conffile.write(f"E = {' '.join(self.energy.astype(str))}\n")

                # write top file header
                topfile.write(f"{self.nbases} {self.nstrands}\n")
                # loop strands
                base_global_idx = 0
                for sid, strand in enumerate(self.strands):
                    # loop base in strands
                    for base_idx, b in enumerate(strand):
                        # if this strand has a previous base, set n3 to base idx - 1, else set to no previous base (-1)
                        n3 = base_global_idx - 1 if base_idx > 0 else -1
                        # if this strand has a following base, set n5 to base idx + 1, else set to no next base (-1)
                        n5 = base_global_idx + 1 if base_idx < len(strand) - 1 else -1
                        # write residue to top
                        topfile.write(f"{sid + 1} {b.base} {n3} {n5}\n")

                        # write residue to conf
                        # (this class doesn't store particle velocities so skip)
                        pos_str = ' '.join(['{:0.6f}'.format(i) for i in b.pos])
                        a1_str = ' '.join(['{:0.6f}'.format(i) for i in b.a1])
                        a3_str = ' '.join(['{:0.6f}'.format(i) for i in b.a3])
                        conffile.write(f"{pos_str} {a1_str} {a3_str} 0.0 0.0 0.0 0.0 0.0 0.0\n")
                        # conffile.write(
                        #     f"{np.array2string(b.pos)[1:-1]} {np.array2string(b.a1)[1:-1]} {np.array2string(b.a3)[1:-1]} 0.0 0.0 0.0 0.0 0.0 0.0\n")

                        base_global_idx += 1  # iter base global idx

                    # conffile.w
                    # conffile.write('{} {} {} 0.0 0.0 0.0 0.0 0.0 0.0\n'.format(' '.join(p.astype(str)), ' '.join(a1.astype(str)),
                    # ' '.join(a3.astype(str))))

    def export_pdb(self, path: Union[str, Path], direction: str = '35',
                   hydrogens: bool = True) -> None:
        """Export structure to PDB format using tacoxDNA's oxDNA_PDB converter."""
        if direction not in ('35', '53'):
            raise ValueError("direction must be '35' or '53'")
        path = Path(path).resolve()
        with tempfile.TemporaryDirectory() as _tmpdir:
            tmpdir = Path(_tmpdir)
            conf_path = tmpdir / 'structure.oxdna'
            top_path = tmpdir / 'structure.top'
            self.export_top_conf(top_path, conf_path)
            tacox = _tacoxdna_src()
            cmd = ['python3', str(tacox / 'oxDNA_PDB.py'),
                   str(top_path), str(conf_path), direction]
            if not hydrogens:
                cmd += ['--hydrogens=false']
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(f"tacoxDNA oxDNA_PDB.py failed:\n{result.stderr}")
            pdb_out = Path(str(conf_path) + '.pdb')
            pdb_out.rename(path)

    def poss(self) -> np.ndarray:
        return np.concatenate([s.positions for s in self.strands], axis=0)

    def cms(self) -> np.ndarray:
        """
        Returns the center of mass of this structure
        """

        return np.concatenate([strand.positions for strand in self.strands], axis=0).mean(axis=0)

    def invalidate_box(self):
        self.box = None

    def has_valid_box(self):
        return self.box is not None

    def inbox(self, relpad: Union[float, np.ndarray] = 0.1,
              extrapad: Union[float, np.ndarray] = 0) -> DNAStructure:
        """
        Copies the structure and translates it so that all position values are positive, and resizes the box so
        that all residues are within the box
        """
        if isinstance(extrapad, (int, float)):
            extrapad = np.full(fill_value=extrapad, shape=(3,))
        else:
            assert extrapad.shape == (3,)
        # get all coordinates
        positions = self.poss()
        # get min and max coords
        mins = np.amin(positions, axis=0)
        maxs = np.amax(positions, axis=0)
        # compute padding
        pad = relpad * (maxs - mins) + extrapad

        # copy self
        cpy = copy.deepcopy(self)
        # translate copy so that mins are at 0
        cpy.transform(tran=-mins + pad)
        # resize box of copy
        cpy.box = maxs - mins + 2 * pad
        new_positions = cpy.poss()
        assert np.all(0 <= new_positions), "Some residues still somehow out of box!"
        assert np.all(new_positions <= cpy.box[np.newaxis, :]), "Some residues still somehow out of box!"
        assert self.validate() == cpy.validate()  # garbage in should imply garbage out and vice versa
        return cpy

    def add_strands(self, strands: Iterable[DNAStructureStrand]):
        for strand in strands:
            self.add_strand(strand)

    def add_strand(self, strand: DNAStructureStrand, update_maps=True):
        """
        if adding multiple strands it is best to update base id maps at end
        """
        n_bases = self.get_num_bases()
        self.strands.append(strand)
        # update bases maps
        if update_maps:
            for base_idx, base in enumerate(strand):
                assert base_idx + n_bases not in self.base_id_map
                assert base.uid not in self.base_id_reverse_map
                self.base_id_map[base_idx + n_bases] = base
                self.base_id_reverse_map[base.uid] = base_idx + n_bases

    # i think these two may be backwards
    def strand_3p(self, strand_idx: int) -> DNABase:
        return self.strands[strand_idx][-1]

    def strand_5p(self, strand_idx: int) -> DNABase:
        return self.strands[strand_idx][0]

    def export_oxview(self, ovfile: Union[Path, str], use_indent=True):
        ovfile = process_path(ovfile)
        assert ovfile.parent.exists()
        oxview_json = self.get_oxview_json()
        with ovfile.open("w") as f:
            json.dump(oxview_json, f, indent=4 if use_indent else None)

    def get_oxview_json(self, cluster_colors: Union[None, dict[int, list[int]]]=None, smart_inbox=True):
        if not self.has_valid_box():
            if smart_inbox:
                return self.inbox().get_oxview_json(cluster_colors=cluster_colors)
            else:
                inboxd = copy.deepcopy(self)
                positions = inboxd.poss()
                inboxd.box = positions.max(axis=0)
                return inboxd.get_oxview_json(cluster_colors=cluster_colors)
        else:
            # assert self.validate(), "Invalid structure!"
            oxview_json = {
                "box": self.box.tolist(),
                "date": datetime.now().isoformat(),
                "forces": [],
                "selections": []
            }

            system = {
                "id": 0,
                "strands": []
            }
            bid = 0  # base ID
            for sid, strand in enumerate(self.strands):
                strand_json = {
                    "class": "NucleicAcidStrand",
                    "id": sid,
                    "end3": bid,
                    "end5": bid + len(strand) - 1,
                    "monomers": []
                }
                for base_local_idx, b in enumerate(strand):
                    # if this strand has a previous base, set n3 to base idx - 1, else set to no previous base (-1)
                    n3 = bid - 1 if base_local_idx > 0 else -1
                    # if this strand has a following base, set n5 to base idx + 1, else set to no next base (-1)
                    n5 = bid + 1 if base_local_idx < len(strand) - 1 else -1
                    nucleotide = {
                        "a1": b.a1.tolist(),
                        "a3": b.a3.tolist(),
                        "class": "DNA",
                        "cluster": bid if b.uid not in self.clustermap else self.clustermap[b.uid],
                        "id": bid,
                        "n3": n3,
                        "n5": n5,
                        "p": b.pos.tolist(),
                        "type": b.base
                    }
                    # todo merge this code into color assignment
                    if cluster_colors is not None and nucleotide["cluster"] in cluster_colors:
                        colornum = cluster_colors[nucleotide["cluster"]][0]
                        colornum = (colornum << 8) + cluster_colors[nucleotide["cluster"]][1]
                        colornum = (colornum << 8) + cluster_colors[nucleotide["cluster"]][2]
                        nucleotide["color"] = colornum
                    if b.uid in self.base_coloration:
                        nucleotide["color"] = self.base_coloration[b.uid]
                    strand_json["monomers"].append(nucleotide)
                    #
                    bid += 1
                system["strands"].append(strand_json)
            oxview_json["systems"] = [system]
            return oxview_json

    def join(self, other: DNAStructure, preserve_uuids: bool = False):
        """
        combines two DNAStructure objects
        """
        # ignore time, box, energy, etc. in other
        # TODO: CLUSTERS
        new_structure = self.clone(preserve_uuids)

        new_structure.add_strands(other.strands)

        cluster_counter = len(new_structure.clusters)  # starting cluster idxing
        for i, cluster in enumerate(other.clusters):
            for uid in cluster:
                assert uid not in new_structure.clustermap
                new_structure.assign_base_to_cluster(uid, i + cluster_counter)
        return new_structure

    def __add__(self, other: Union[DNAStructure, DNAStructureStrand]) -> DNAStructure:
        """
        combines two DNAStucture objects, returns the product of the merge
        """
        # copy self, so we do not modifiy in place
        if isinstance(other, DNAStructureStrand):
            new_structure = self.clone()
            new_structure.strands.append(other)
            return new_structure
        else:
            return self.join(other, preserve_uuids=False)

    @staticmethod
    def merge_many(structures: list[DNAStructure]) -> DNAStructure:
        assert len(structures) > 0
        new_structure = structures[0].clone()
        for other in structures[1:]:
            if isinstance(other, DNAStructureStrand):
                new_structure.strands.append(other)
            else:
                new_structure.strands.extend(other.strands)
                cluster_counter = len(new_structure.clusters)
                for i, cluster in enumerate(other.clusters):
                    for uid in cluster:
                        assert uid not in new_structure.clustermap
                        new_structure.assign_base_to_cluster(uid, i + cluster_counter)
        return new_structure

    from typing import Union, Optional, Tuple

    def assign_base_to_cluster(self,
                               base_identifier: Union[int, Tuple[int, int]],
                               cluster_id: Optional[int]):
        """
        Assigns a base to a cluster.

        Parameters:
            base_identifier: base uid, or a tuple (strand_index, base_index_in_strand)
            cluster_id: cluster identifier (int) or None to create a new cluster
        """
        # resolve uid from tuple or accept uid directly
        if isinstance(base_identifier, tuple):
            strand_idx, base_idx = base_identifier
            uid = self.strands[strand_idx][base_idx].uid
        else:
            uid = base_identifier

        # handle "create new cluster" case
        if cluster_id is None:
            cluster_id = len(self.clusters)
            self.clusters.append(set())
        else:
            if cluster_id < 0:
                raise ValueError(f"cluster_id must be non-negative or None, got {cluster_id}")
            # allow non-sequential cluster ids by growing the list up to cluster_id
            if cluster_id >= len(self.clusters):
                for _ in range(len(self.clusters), cluster_id + 1):
                    self.clusters.append(set())

        # remove uid from previous cluster if assigned
        prev = self.clustermap.get(uid)
        if prev is not None and uid in self.clusters[prev]:
            self.clusters[prev].remove(uid)

        # add to new cluster and update mapping
        self.clusters[cluster_id].add(uid)
        self.clustermap[uid] = cluster_id

    def clear_clusters(self):
        """
        clears cluster information
        """
        self.clusters = []
        self.clustermap = {}

    def transform(self, rot: np.ndarray = np.identity(3), tran: np.ndarray = np.zeros(3)):
        """
        Rotates self in place using the rotation matrix given as an arg
        does NOT update the bounding box!!!
        """
        assert rot.shape == (3, 3), "Wrong shape for rotation!"
        assert abs(np.linalg.det(rot)) - 1 < 1e-6, f"Rotation matrix {rot} has non-one determinate {np.linalg.det(rot)}"
        if isinstance(tran, list):
            tran = np.array(tran)
        assert tran.shape in [(3,), (3, 1)], "Wrong shape for translaton!"
        for strand in self.strands:
            strand.transform(rot, tran)
        self.invalidate_box()

    def validate(self) -> bool:
        return all([s.validate() for s in self.strands])

    def possible_bonds(self,
                       idxs1: list[int],
                       idxs2: list[int],
                       min_hybrid_length: int = 1) -> list[tuple[int, int]]:
        """
        Parameters:
            idxs1: indexes of nucleotides in first sequence
            idxs2: indexes of nucleotides in second sequence
            min_hybrid_length: minimum length of hybridized sequences to check for. TODO: implement
        """

        possible_bonds: list[tuple[int, int]] = []

        for idx1, idx2 in itertools.product(idxs1, idxs2):
            if f"HYDR_{self.get_base(idx1).base}_{self.get_base(idx2).base}" in SEQ_DEP_PARAMS:
                possible_bonds.append((idx1, idx2))
        return possible_bonds

    def remove_strand(self, strand_id: int):
        """
        deletes a strand with the given id from the structure
        """
        assert -1 < strand_id < self.nstrands
        self.strands = self.strands[:strand_id] + self.strands[strand_id+1:]

    def iter_strands(self) -> Generator[DNAStructureStrand]:
        """
        Iterates strands in the structure
        """
        yield from self.strands

    def iter_bases(self) -> Generator[DNABase]:
        for strand in self.iter_strands():
            yield from strand

    def get_mass(self) -> float:
        """
        computes the total mass of the structure in kilodaltons
        todo: at some point we will want to use dna/rna structures
        """
        mass = 0
        # sugar_mass = 134.13
        # phosphate_mass = 94.97
        # A_mass = 135.13 + sugar_mass +phosphate_mass
        # T_mass = 126.11 + sugar_mass+phosphate_mass
        # C_mass = 111.10 + sugar_mass+phosphate_mass
        # G_mass = 151.13 +sugar_mass+phosphate_mass

        # m.ws from https://www.thermofisher.com/us/en/home/references/ambion-tech-support/rna-tools-and-calculators/dna-and-rna-molecular-weights-and-conversions.html
        A_mass = 331.2
        C_mass = 307.2
        G_mass = 347.2
        T_mass = 322.2
        for base in self.iter_bases():
            mass += A_mass if base.base == "A" else T_mass if base.base == "T" else C_mass if base.base == "C" else G_mass
        return mass / 1000

    def check_top_match(self, structure_conf: DNAStructure) -> bool:
        """
        tests that the topology of this structure matches the other structure
        note that this is a **very** inflexable match. we do not account for different base or strand numbering or
        ordering!
        """
        if self.nbases != structure_conf.nbases:
            return False
        if self.nstrands != structure_conf.nstrands:
            return False
        for self_strand, other_strand in zip(self.iter_strands(), structure_conf.iter_strands()):
            if len(self_strand) != len(other_strand):
                return False
        for self_base, other_base in zip(self.iter_bases(), structure_conf.iter_bases()):
            if self_base.base != other_base.base:
                return False
        return True

    def __getitem__(self, key) -> list[DNABase]:
        """
        get list of bases from IDs
        if only one arg is provided, assume ints are idxs
        if a second arg is provided, check what it is
        todo: more arg options
        """
        if not isinstance(key, tuple):
            key = key,
        if len(key) > 1 and key[1] == "uid":
            return [self.get_base_by_uid(i) for i in key[0]]
        else:
            return [self.get_base(i) for i in key[0]]

    def helixify(self, s1: DNAStructureStrand, s2: DNAStructureStrand):
        """
        given two DNA strands in this structure (do not have to be complete strands, can be subsets)
        sets positions, a1s, and a3s so the strands base pair
        """
        assert len(s1) == len(s2), "Cannot helixify strands of different lengths"
        # todo: better calculation of helix start & direction
        helix_start = s1[0].pos
        helix_direction = (s2[0].pos - s1[0].pos) / np.linalg.norm(s2[0].pos - s1[0].pos)
        (s1.positions, s1.a1s, s1.a3s), (s2.positions, s2.a1s, s2.a3s) = gen_helix_coords(len(s1),
                                                                                 helix_start,
                                                                                 helix_direction)

    def export_gltf(self, fp: Union[Path, str], sphere_radius: float = 0.3, default_color: Optional[list[float]] = None):
        """
        Exports the structure as a GLTF file, rendering each base as a colored sphere.
        Generates a single sphere mesh for each color needed and reuses them for all bases of that color.
        :param fp: file path to write to
        :param default_color: RGBA list for default color, e.g. [1.0, 1.0, 1.0, 1.0]. If None, all bases must have assigned color
        """
        fp = process_path(fp)
        assert fp.suffix == ".gltf", "File path must end in .gltf"

        # check that default color is RGBA, add Alpha = 1.0 if not
        if default_color is not None:
            if not isinstance(default_color, list):
                raise TypeError("default_color should be provided as an RGB or (if you want) RGBA list")
            else:
                if len(default_color) == 3:
                    default_color.append(1.0)
                elif len(default_color) != 4:
                    raise ValueError("default_color should be provided as an RGB or (if you want) RGBA list")
            if any([not (0 <= c <= 1.0) for c in default_color]):
                raise ValueError("RGB values in default_color should be between 0 and 1.0")

        def create_sphere_mesh(radius, segments=32, rings=32):
            verts = []
            faces = []
            for i in range(rings + 1):
                phi = np.pi * i / rings
                for j in range(segments):
                    theta = 2 * np.pi * j / segments
                    x = radius * np.sin(phi) * np.cos(theta)
                    y = radius * np.sin(phi) * np.sin(theta)
                    z = radius * np.cos(phi)
                    verts.append((x, y, z))
            for i in range(rings):
                for j in range(segments):
                    a = i * segments + j
                    b = a + segments
                    a_next = i * segments + (j + 1) % segments
                    b_next = a_next + segments
                    faces.append((a, b, a_next))
                    faces.append((b, b_next, a_next))
            return verts, faces

        # Collect all unique colors needed
        color_to_bases = {}
        for base in self.iter_bases():
            color = self.base_coloration.get(base.uid, None)
            if color is not None:
                r = (color >> 16) & 0xFF
                g = (color >> 8) & 0xFF
                b = color & 0xFF
                base_color = (r / 255.0, g / 255.0, b / 255.0, 1.0)
            elif default_color is not None:
                base_color = tuple(default_color)
            else:
                raise Exception(
                    f"Base UID {base.uid} does not have an assigned color and no default_color was provided")
            color_to_bases.setdefault(base_color, []).append(base)

        # Generate a mesh for each color
        meshes = []
        materials = []
        mesh_idx_map = {}
        buffer_datas = []
        buffer_views = []
        accessors = []
        mesh_offset = 0

        for color_idx, (base_color, bases) in enumerate(color_to_bases.items()):
            sphere_verts, sphere_faces = create_sphere_mesh(sphere_radius)
            positions = [v for v in sphere_verts]
            indices = [i for face in sphere_faces for i in face]
            colors = [[base_color[0], base_color[1], base_color[2]] for _ in sphere_verts]

            pos_bytes = b''.join([struct.pack('<3f', *positions[i]) for i in range(len(positions))])
            color_bytes = b''.join([struct.pack('<3f', *colors[i]) for i in range(len(colors))])
            index_bytes = b''.join([struct.pack('<I', idx) for idx in indices])

            buffer_data = pos_bytes + color_bytes + index_bytes
            buffer_datas.append(buffer_data)

            pos_view = BufferView(buffer=color_idx, byteOffset=0, byteLength=len(pos_bytes), target=34962)
            color_view = BufferView(buffer=color_idx, byteOffset=len(pos_bytes), byteLength=len(color_bytes), target=34962)
            index_view = BufferView(buffer=color_idx, byteOffset=len(pos_bytes) + len(color_bytes), byteLength=len(index_bytes), target=34963)

            buffer_views.extend([pos_view, color_view, index_view])

            pos_accessor = Accessor(bufferView=mesh_offset, byteOffset=0, componentType=5126, count=len(positions), type="VEC3",
                                    min=[min([p[0] for p in positions]), min([p[1] for p in positions]), min([p[2] for p in positions])],
                                    max=[max([p[0] for p in positions]), max([p[1] for p in positions]), max([p[2] for p in positions])])
            color_accessor = Accessor(bufferView=mesh_offset + 1, byteOffset=0, componentType=5126, count=len(colors), type="VEC3")
            index_accessor = Accessor(bufferView=mesh_offset + 2, byteOffset=0, componentType=5125, count=len(indices), type="SCALAR")

            accessors.extend([pos_accessor, color_accessor, index_accessor])

            material = Material(pbrMetallicRoughness=PbrMetallicRoughness(baseColorFactor=list(base_color)))
            materials.append(material)

            mesh = Mesh(primitives=[Primitive(attributes={"POSITION": mesh_offset, "COLOR_0": mesh_offset + 1}, indices=mesh_offset + 2, material=color_idx)])
            meshes.append(mesh)

            mesh_idx_map[base_color] = color_idx
            mesh_offset += 3

        # Write all buffers
        buffers = []
        for idx, buffer_data in enumerate(buffer_datas):
            buffer_uri = fp.with_suffix(f'.{idx}.bin').name
            with open(fp.with_suffix(f'.{idx}.bin'), 'wb') as fbin:
                fbin.write(buffer_data)
            buffer = Buffer(byteLength=len(buffer_data), uri=buffer_uri)
            buffers.append(buffer)

        # Create nodes for each base, referencing the correct mesh
        nodes = []
        for base_color, bases in color_to_bases.items():
            mesh_idx = mesh_idx_map[base_color]
            for base in bases:
                px, py, pz = base.pos
                node = Node(mesh=mesh_idx, translation=[float(px), float(py), float(pz)],
                            extras={"base_color": list(base_color)})
                nodes.append(node)

        scene = Scene(nodes=list(range(len(nodes))))

        gltf = GLTF2(
            asset=Asset(),
            buffers=buffers,
            bufferViews=buffer_views,
            accessors=accessors,
            materials=materials,
            meshes=meshes,
            nodes=nodes,
            scenes=[scene],
            scene=0
        )

        gltf.save(fp)

def seq(bases: list[DNABase]) -> str:
    return "".join([b.base for b in bases])

def load_dna_structure(top: Union[str, Path], conf_file: Union[str, Path], conf_idx: int = 0) -> DNAStructure:
    """
    Constructs a topology
    """
    if not isinstance(conf_idx, int):
        raise TypeError("conf_idx must be an integer ")
    if isinstance(top, str):
        top = Path(top)
    if isinstance(conf_file, str):
        conf_file = Path(conf_file)

    assert top.is_file(), f"No file exists {str(top)}"
    assert conf_file.is_file(), f"No file exists {str(conf_file)}"

    with top.open("r") as top_file:
        lines = top_file.readlines()
    # get line info
    nbases, nstrands = map(int, lines[0].split())
    # generate placeholder for bases
    # each base represented as a tuple
    strands_list: list[list[tuple[chr, np.ndarray, np.ndarray, np.ndarray]]] = [[] for _ in range(1, nstrands + 1)]
    # generate the return object
    topology = TopInfo(str(top), nbases)
    traj_info = get_traj_info(str(conf_file))
    idx = 0
    conf = None
    for chunk in linear_read(traj_info, topology):
        if conf_idx < idx + len(chunk):
            conf = chunk[conf_idx - idx]
            break
        idx += len(chunk)
    if conf is None:
        raise IndexError(f"Configuration index {conf_idx} out of range! Number of confs is {traj_info.nconfs}")
    # conf: Configuration = next(linear_read(traj_info, topology))[conf_idx]
    for base_idx, line in enumerate(lines[1:]):
        # find base info
        # strand id, base, next base id, prev base id
        sid, t, p3, p5 = line.split()
        sid, p3, p5 = map(int, [sid, p3, p5])
        # b = Base(t, p3, p5)
        strands_list[sid - 1].append((t,
                                      conf.positions[base_idx, :],
                                      conf.a1s[base_idx, :],
                                      conf.a3s[base_idx, :],))
    strands = [strand_from_info(strand) for strand in strands_list]
    return DNAStructure(strands,
                        conf.time,
                        conf.box,
                        conf.energy)


def _tacoxdna_src() -> Path:
    """Locate the tacoxDNA/src directory."""
    if TACOXDNA_SRC is not None:
        p = Path(TACOXDNA_SRC)
    else:
        # Convention: sibling git checkout of ipy_oxDNA
        p = Path(__file__).parents[3] / 'tacoxDNA' / 'src'
    if not p.is_dir():
        raise FileNotFoundError(
            f"tacoxDNA not found at {p}. Set dna_structure.TACOXDNA_SRC to the tacoxDNA/src directory.")
    return p


def _fasta_to_sqs(fasta_path: Path, out_dir: Path) -> Path:
    """Strip FASTA header lines and join the sequence into one line for tacoxDNA."""
    text = fasta_path.read_text()
    if not any(line.startswith('>') for line in text.splitlines()):
        return fasta_path.resolve()
    plain = ''.join(line.strip() for line in text.splitlines() if not line.startswith('>'))
    out = out_dir / 'sequence.sqs'
    out.write_text(plain)
    return out


def _tacox_run(script: str, args: list[str], work_dir: Path) -> tuple[Path, Path]:
    """
    Run a tacoxDNA *-to-oxDNA conversion script and return (conf_path, top_path).

    The script is run with cwd=work_dir so output lands there.
    All tacoxDNA input-format scripts use the same output naming convention:
    basename(input_arg) + ".oxdna" and + ".top".
    """
    tacox = _tacoxdna_src()
    cmd = ['python3', str(tacox / script)] + args
    result = subprocess.run(cmd, cwd=work_dir, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"tacoxDNA script {script} failed:\n{result.stderr}")
    input_basename = Path(args[0]).name
    conf_path = work_dir / (input_basename + '.oxdna')
    top_path = work_dir / (input_basename + '.top')
    return conf_path, top_path


def load_dna_structure_from_pdb(pdb_path: Union[str, Path],
                                 direction: str = '35') -> DNAStructure:
    """Load a DNAStructure from a PDB file using tacoxDNA's PDB_oxDNA converter.

    direction: '35' if nucleotides are listed 3'→5' in the PDB (default),
               '53' for 5'→3'.
    """
    if direction not in ('35', '53'):
        raise ValueError("direction must be '35' or '53'")
    pdb_path = Path(pdb_path).resolve()
    with tempfile.TemporaryDirectory() as _tmpdir:
        tmpdir = Path(_tmpdir)
        conf, top = _tacox_run('PDB_oxDNA.py', [str(pdb_path), direction], tmpdir)
        return load_dna_structure(top, conf)


def load_dna_structure_from_cadnano(json_path: Union[str, Path],
                                     lattice: str,
                                     sequence_file: Optional[Union[str, Path]] = None) -> DNAStructure:
    """Load a DNAStructure from a cadnano JSON file using tacoxDNA's cadnano_oxDNA converter.

    lattice: 'sq' for square lattice, 'he' for hexagonal lattice.
    sequence_file: optional FASTA file; when provided, bases are assigned from it
                   instead of randomly, making the output deterministic.
    """
    if lattice not in ('sq', 'he'):
        raise ValueError("lattice must be 'sq' (square) or 'he' (hexagonal)")
    json_path = Path(json_path).resolve()
    with tempfile.TemporaryDirectory() as _tmpdir:
        tmpdir = Path(_tmpdir)
        args = [str(json_path), lattice]
        if sequence_file is not None:
            sqs = _fasta_to_sqs(Path(sequence_file), tmpdir)
            args += ['--sequence', str(sqs)]
        conf, top = _tacox_run('cadnano_oxDNA.py', args, tmpdir)
        return load_dna_structure(top, conf)


def load_dna_structure_from_rcsb(pdb_id: str, direction: str = '35') -> DNAStructure:
    """Download a structure from the RCSB PDB by ID and load it as a DNAStructure."""
    from Bio.PDB import PDBList
    with tempfile.TemporaryDirectory() as _tmpdir:
        tmpdir = Path(_tmpdir)
        PDBList().retrieve_pdb_file(pdb_id, file_format='pdb', pdir=str(tmpdir), overwrite=True)
        pdb_files = list(tmpdir.glob('*.ent'))
        if not pdb_files:
            raise FileNotFoundError(f"RCSB download for '{pdb_id}' returned no file.")
        return load_dna_structure_from_pdb(pdb_files[0], direction=direction)


def load_oxview(oxview: Union[str, Path]) -> DNAStructure:
    if isinstance(oxview, str):
        oxview = Path(oxview)
    # if not oxview.is_absolute():
    #     oxview = get_input_dir() / oxview
    with oxview.open("r") as f:
        ovdata = json.load(f)
        box = ovdata["box"]
        s = DNAStructure([], 0, box)
        # frankly i have no idea how to handle multiple-system files
        if len(ovdata["systems"]) > 1:
            print("Warning: multiple systems will be merged")
        for ox_sys in ovdata["systems"]:
            for strand_data in ox_sys["strands"]:
                if strand_data["class"] == "NucleicAcidStrand":
                    assert strand_data["end3"] < strand_data["end5"], "unexpected strand ordering"
                    strand = []  # 3' -> 5' list of nucleotides
                    strand_data["monomers"] = sorted(strand_data["monomers"], key=lambda x: x["id"])
                    for i, nuc in enumerate(strand_data["monomers"]):
                        a1 = np.array(nuc["a1"])
                        a3 = np.array(nuc["a3"])
                        pos = np.array(nuc["p"])
                        base = nuc["type"]
                        # goddammit why is it 3'->5'
                        if "n3" in nuc and strand_data['end3'] != nuc["id"]:
                            assert strand_data["monomers"][i - 1]["id"] == nuc["n3"], "Topology problem!"

                        if "n5" in nuc and strand_data['end5'] != nuc["id"]:
                            assert strand_data["monomers"][i + 1]["id"] == nuc[
                                "n5"], "Topology problem!"
                        strand.append((base, pos, a1, a3))
                    s.add_strand(strand_from_info(strand))
                    # load extra stuff (clusters, etc.)
                    for i, (nuc, nuc_data) in enumerate(zip(s.strands[-1], strand_data["monomers"])):
                        if "cluster" in nuc_data:
                            s.assign_base_to_cluster(nuc.uid, nuc_data["cluster"])
                        if "color" in nuc_data:
                            s.base_coloration[nuc.uid] = nuc_data["color"]
                        s.base_id_map[nuc_data['id']] = nuc
                        s.base_id_reverse_map[nuc.uid] = nuc_data["id"]

                else:
                    print(f"Unrecognized system data type {strand_data['class']}")
    return s

def rand_seq(length: int) -> str:
    return "".join(
        choice(["A", "T", "C", "G"]) for _ in range(length))

def recolor_top_dat(color_oxview_source: Union[str, Path],
                    structure_top_source: Union[str, Path],
                    structure_conf_source: Union[str, Path],
                    recolor_write_to: Union[str, Path]):
    """
    applies the nucleotide coloring in recolor_oxview_source to the conf ibn recolor_conf_source
    Parameters:
        recolor_oxview_source the oxview file to use for color inforation
        structure_top_source the topology file to use for the actual structure (should match recolor_oxview_source)
        structure_conf_source the conf file to use for the actual structure
        recolor_write_to path to write oxview file
    """

    if isinstance(color_oxview_source, str):
        recolor_top_dat(Path(color_oxview_source), structure_top_source, structure_conf_source, recolor_write_to)
    elif isinstance(structure_top_source, str):
        recolor_top_dat(color_oxview_source, Path(structure_top_source), structure_conf_source, recolor_write_to)
    elif isinstance(structure_conf_source, str):
        recolor_top_dat(color_oxview_source, structure_top_source, Path(structure_conf_source), recolor_write_to)
    elif isinstance(recolor_write_to, str):
        recolor_top_dat(color_oxview_source, structure_top_source, structure_conf_source, Path(recolor_write_to))
    else:
        assert color_oxview_source.exists(), f"Coloration oxview file {color_oxview_source} does not exist!"
        assert color_oxview_source.suffix == ".oxview", "Coloration file does not have `.oxview` extension"
        assert structure_top_source.exists(), f"Topology file {structure_top_source} does not exist!"
        assert structure_top_source.suffix == ".top", "Structure file does not have `.top` extension"
        assert structure_conf_source.exists(), f"Conf file {structure_conf_source} does not exist!"
        assert structure_conf_source.suffix == ".conf" or structure_conf_source.suffix == ".dat",\
            "Conf file does not have `.conf` or `.dat` extension"
        structure_conf = load_dna_structure(structure_top_source, structure_conf_source)

        recolor_structure(color_oxview_source, structure_conf)
        structure_conf.export_oxview(recolor_write_to)


def recolor_structure(color_oxview_source: Union[str,Path, DNAStructure], structure_conf: DNAStructure):
    """
    recolors a structure in place using info from another structure
    """
    if not isinstance(color_oxview_source, DNAStructure):
        color_source_structure = load_oxview(color_oxview_source)
    else:
        color_source_structure = color_oxview_source
    assert color_source_structure.check_top_match(structure_conf), (f"Structure topology in file "
                                                                    f"`{str(color_source_structure)}` does not match"
                                                                    f" color source structure topology`!")
    for i, (base1, base2) in enumerate(zip(color_source_structure.iter_bases(), structure_conf.iter_bases())):
        if base1.uid in color_source_structure.base_coloration:
            structure_conf.base_coloration[base2.uid] = color_source_structure.base_coloration[base1.uid]
