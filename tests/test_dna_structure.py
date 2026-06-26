import random
from pathlib import Path

import pytest

import numpy as np
from oxpy_utils.structure_editor.dna_structure import DNAStructure, construct_strands, load_dna_structure, \
    to_oxview_color


def test_generate_dna_structure():
    fwd, rev = construct_strands("".join([random.choice(["A", "C", "T", "G"]) for _ in range(25)]),
                                 np.array([0., 0., 0.]),
                                 np.array([1., 0., 0.]))
    structure = DNAStructure([fwd, rev],
                             0,
                             fwd.positions.max(axis=0) * 1.1)
    # idk what to do now

def test_load_dna_structure() -> DNAStructure:
    """
    load an example structure
    """
    examples_dir = Path(
        __file__).parent.parent / 'examples' / 'tutorials' / '8_nt_duplex_melting_cpu' / 'oxdna_files'
    return load_dna_structure(examples_dir / "duplex_box_30.top",
                       examples_dir / "duplex_box_30.dat")

def test_dna_structure_to_gltf(tmp_path):
    """
    load an example structure and export to gltf
    """
    structure = test_load_dna_structure()
    for base in structure.iter_bases():
        if base.base == "A":
            structure.base_coloration[base.uid] = to_oxview_color([252, 186, 3], basis=255.)
        elif base.base == "T":
            structure.base_coloration[base.uid] = to_oxview_color([0, 153, 51], basis=255.)
        elif base.base == "C":
            structure.base_coloration[base.uid] = to_oxview_color([204, 0, 102], basis=255.)
        elif base.base == "G":
            structure.base_coloration[base.uid] = to_oxview_color([0, 102, 204], basis=255.)
    structure.export_gltf(tmp_path / "duplex_box_30.gltf")
    # todo validate the gltf file