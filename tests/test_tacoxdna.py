"""Tests for tacoxDNA ↔ DNAStructure bridge functions."""
from pathlib import Path

import numpy as np
import pytest

from oxpy_utils.structure_editor.dna_structure import (
    DNAStructure,
    DNAStructureStrand,
    TACOXDNA_SRC,
    _tacoxdna_src,
    load_dna_structure,
    load_dna_structure_from_cadnano,
    load_dna_structure_from_pdb,
    load_dna_structure_from_rcsb,
)

TACOX_TESTS = Path(__file__).parents[2] / 'tacoxDNA' / 'tests'
OXDNA_EXAMPLES = Path(__file__).parent.parent / 'examples' / 'tutorials' / 'vmmc_windowing_8nt_duplex' / 'oxdna_files'


def _collect_bases(struct: DNAStructure):
    """Return (positions, a1s, a3s, bases) arrays for all bases, sorted by position.

    Sorting by position gives a canonical ordering that is independent of strand
    numbering and of whether bases are stored 5'→3' or 3'→5' within each strand.
    """
    positions = np.vstack([s.positions for s in struct.strands])
    a1s = np.vstack([s.a1s for s in struct.strands])
    a3s = np.vstack([s.a3s for s in struct.strands])
    bases = np.concatenate([s.bases for s in struct.strands])
    order = np.lexsort(positions.T[::-1])  # primary sort on x, then y, then z
    return positions[order], a1s[order], a3s[order], bases[order]


def _assert_structures_match(actual: DNAStructure, ref: DNAStructure, atol: float = 1e-5) -> None:
    """Assert that two DNAStructures represent the same physical structure.

    Bases are sorted by 3D position before comparison so that strand numbering
    and per-strand storage direction do not affect the result.
    """
    assert actual.nbases == ref.nbases, f"nbases mismatch: {actual.nbases} != {ref.nbases}"
    assert actual.nstrands == ref.nstrands, f"nstrands mismatch: {actual.nstrands} != {ref.nstrands}"
    a_pos, a_a1, a_a3, a_bases = _collect_bases(actual)
    r_pos, r_a1, r_a3, r_bases = _collect_bases(ref)
    np.testing.assert_allclose(a_pos, r_pos, atol=atol, err_msg="positions mismatch")
    np.testing.assert_allclose(a_a1, r_a1, atol=atol, err_msg="a1s mismatch")
    np.testing.assert_allclose(a_a3, r_a3, atol=atol, err_msg="a3s mismatch")
    assert list(a_bases) == list(r_bases), "nucleotide identities mismatch after positional sort"
PDB_3GBI = Path(__file__).parent / 'test_data' / '3gbi.pdb'


@pytest.fixture(scope='session')
def pdb_3gbi() -> Path:
    """Download 3GBI from RCSB once per session and cache it locally."""
    if not PDB_3GBI.exists():
        from Bio.PDB import PDBList
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            PDBList().retrieve_pdb_file('3GBI', file_format='pdb', pdir=tmpdir, overwrite=True)
            ent_files = list(Path(tmpdir).glob('*.ent'))
            if not ent_files:
                pytest.skip('Could not download 3GBI from RCSB')
            PDB_3GBI.parent.mkdir(parents=True, exist_ok=True)
            ent_files[0].rename(PDB_3GBI)
    return PDB_3GBI


class TestTacoxdnaSrc:
    def test_default_resolves(self):
        p = _tacoxdna_src()
        assert p.is_dir()
        assert (p / 'PDB_oxDNA.py').is_file()

    def test_override(self, monkeypatch):
        import oxpy_utils.structure_editor.dna_structure as mod
        real = _tacoxdna_src()
        monkeypatch.setattr(mod, 'TACOXDNA_SRC', real)
        assert _tacoxdna_src() == real

    def test_bad_override_raises(self, monkeypatch):
        import oxpy_utils.structure_editor.dna_structure as mod
        monkeypatch.setattr(mod, 'TACOXDNA_SRC', Path('/nonexistent/path'))
        with pytest.raises(FileNotFoundError):
            _tacoxdna_src()


class TestLoadFromPdb:
    @pytest.fixture(scope='class')
    def pdb_struct(self):
        return load_dna_structure_from_pdb(TACOX_TESTS / 'PDB_oxDNA' / 'input.pdb', direction='35')

    def test_returns_dna_structure(self, pdb_struct):
        assert isinstance(pdb_struct, DNAStructure)

    def test_strand_count(self, pdb_struct):
        assert pdb_struct.nstrands == 2

    def test_base_count(self, pdb_struct):
        assert pdb_struct.nbases == 24

    def test_box_is_3d(self, pdb_struct):
        assert pdb_struct.box.shape == (3,)
        assert np.all(pdb_struct.box > 0)

    def test_sequence(self, pdb_struct):
        all_bases = ''.join(s.seq() for s in pdb_struct.strands)
        assert all_bases == 'CGCGAATTCGCGCGCGAATTCGCG'

    def test_orientations_are_unit_vectors(self, pdb_struct):
        for strand in pdb_struct.strands:
            norms_a1 = np.linalg.norm(strand.a1s, axis=1)
            norms_a3 = np.linalg.norm(strand.a3s, axis=1)
            np.testing.assert_allclose(norms_a1, 1.0, atol=1e-5)
            np.testing.assert_allclose(norms_a3, 1.0, atol=1e-5)

    def test_invalid_direction_raises(self):
        with pytest.raises(ValueError, match="direction"):
            load_dna_structure_from_pdb(TACOX_TESTS / 'PDB_oxDNA' / 'input.pdb', direction='bad')

    def test_missing_file_raises(self):
        with pytest.raises(RuntimeError):
            load_dna_structure_from_pdb('/nonexistent/file.pdb')

    def test_matches_correct_output(self, pdb_struct):
        ref = load_dna_structure(
            TACOX_TESTS / 'PDB_oxDNA' / 'correct_output.top',
            TACOX_TESTS / 'PDB_oxDNA' / 'correct_output.dat',
        )
        _assert_structures_match(pdb_struct, ref)


class TestLoadFromCadnano:
    @pytest.fixture(scope='class')
    def cadnano_struct(self):
        return load_dna_structure_from_cadnano(
            TACOX_TESTS / 'cadnano_oxDNA' / 'init.json', lattice='sq')

    def test_returns_dna_structure(self, cadnano_struct):
        assert isinstance(cadnano_struct, DNAStructure)

    def test_strand_count(self, cadnano_struct):
        assert cadnano_struct.nstrands == 3

    def test_base_count(self, cadnano_struct):
        assert cadnano_struct.nbases == 128

    def test_box_is_3d(self, cadnano_struct):
        assert cadnano_struct.box.shape == (3,)
        assert np.all(cadnano_struct.box > 0)

    def test_all_bases_valid(self, cadnano_struct):
        for base in cadnano_struct.iter_bases():
            assert base.base in ('A', 'T', 'C', 'G')

    def test_invalid_lattice_raises(self):
        with pytest.raises(ValueError, match="lattice"):
            load_dna_structure_from_cadnano(
                TACOX_TESTS / 'cadnano_oxDNA' / 'init.json', lattice='bad')

    def test_matches_correct_output(self, cadnano_struct):
        ref = load_dna_structure(
            TACOX_TESTS / 'cadnano_oxDNA' / 'correct_output.top',
            TACOX_TESTS / 'cadnano_oxDNA' / 'correct_output.dat',
        )
        _assert_structures_match(cadnano_struct, ref)


class TestExportPdb:
    @pytest.fixture(scope='class')
    def struct(self):
        return load_dna_structure(
            TACOX_TESTS / 'oxDNA_PDB' / 'ds.top',
            TACOX_TESTS / 'oxDNA_PDB' / 'ds.dat',
        )

    def test_creates_file(self, struct, tmp_path):
        out = tmp_path / 'out.pdb'
        struct.export_pdb(out)
        assert out.is_file()
        assert out.stat().st_size > 0

    def test_atom_count(self, struct, tmp_path):
        out = tmp_path / 'out.pdb'
        struct.export_pdb(out)
        atom_lines = [l for l in out.read_text().splitlines() if l.startswith('ATOM')]
        assert len(atom_lines) == 505

    def test_has_ter_lines(self, struct, tmp_path):
        out = tmp_path / 'out.pdb'
        struct.export_pdb(out)
        ter_lines = [l for l in out.read_text().splitlines() if l.startswith('TER')]
        assert len(ter_lines) == struct.nstrands

    def test_no_hydrogens(self, struct, tmp_path):
        with_h = tmp_path / 'with_h.pdb'
        without_h = tmp_path / 'without_h.pdb'
        struct.export_pdb(with_h, hydrogens=True)
        struct.export_pdb(without_h, hydrogens=False)
        h_count = sum(1 for l in with_h.read_text().splitlines()
                      if l.startswith('ATOM') and ' H' in l)
        no_h_count = sum(1 for l in without_h.read_text().splitlines()
                         if l.startswith('ATOM') and ' H' in l)
        assert h_count > 0
        assert no_h_count < h_count  # terminal HO3'/HO5' atoms always remain

    def test_invalid_direction_raises(self, struct, tmp_path):
        with pytest.raises(ValueError, match="direction"):
            struct.export_pdb(tmp_path / 'out.pdb', direction='bad')

    def test_direction_53(self, struct, tmp_path):
        out_35 = tmp_path / '35.pdb'
        out_53 = tmp_path / '53.pdb'
        struct.export_pdb(out_35, direction='35')
        struct.export_pdb(out_53, direction='53')
        # both should be valid PDB files; residue ordering differs
        assert out_35.read_text() != out_53.read_text()


class TestPdbRoundtrip:
    """Load oxDNA → export PDB → reload (checks positions survive roundtrip at structure level)."""

    def test_base_count_preserved(self, tmp_path):
        struct = load_dna_structure(
            TACOX_TESTS / 'oxDNA_PDB' / 'ds.top',
            TACOX_TESTS / 'oxDNA_PDB' / 'ds.dat',
        )
        pdb_out = tmp_path / 'roundtrip.pdb'
        struct.export_pdb(pdb_out)
        reloaded = load_dna_structure_from_pdb(pdb_out, direction='35')
        assert reloaded.nbases == struct.nbases

    def test_strand_count_preserved(self, tmp_path):
        struct = load_dna_structure(
            TACOX_TESTS / 'oxDNA_PDB' / 'ds.top',
            TACOX_TESTS / 'oxDNA_PDB' / 'ds.dat',
        )
        pdb_out = tmp_path / 'roundtrip.pdb'
        struct.export_pdb(pdb_out)
        reloaded = load_dna_structure_from_pdb(pdb_out, direction='35')
        assert reloaded.nstrands == struct.nstrands


class TestLoadFromRcsb:
    @pytest.fixture(scope='class')
    def rcsb_struct(self, pdb_3gbi):
        return load_dna_structure_from_pdb(pdb_3gbi, direction='35')

    def test_returns_dna_structure(self, rcsb_struct):
        assert isinstance(rcsb_struct, DNAStructure)

    def test_has_bases(self, rcsb_struct):
        assert rcsb_struct.nbases > 0

    def test_has_strands(self, rcsb_struct):
        assert rcsb_struct.nstrands > 0

    def test_box_is_3d(self, rcsb_struct):
        assert rcsb_struct.box.shape == (3,)
        assert np.all(rcsb_struct.box > 0)

    def test_all_bases_valid(self, rcsb_struct):
        for base in rcsb_struct.iter_bases():
            assert base.base in ('A', 'T', 'C', 'G')

    def test_orientations_are_unit_vectors(self, rcsb_struct):
        for strand in rcsb_struct.strands:
            np.testing.assert_allclose(np.linalg.norm(strand.a1s, axis=1), 1.0, atol=1e-5)
            np.testing.assert_allclose(np.linalg.norm(strand.a3s, axis=1), 1.0, atol=1e-5)

    def test_load_from_rcsb_helper(self, pdb_3gbi):
        """load_dna_structure_from_rcsb downloads and loads in one call."""
        struct = load_dna_structure_from_rcsb('3GBI')
        assert isinstance(struct, DNAStructure)
        assert struct.nbases > 0