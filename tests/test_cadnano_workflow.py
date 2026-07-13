"""Integration test: cadnano → DNAStructure → MC relax → MD relax → production → analysis."""
import urllib.request
from pathlib import Path

import numpy as np
import pytest

from oxpy_utils.oxdna_simulation import Simulation
from oxpy_utils.structure_editor.dna_structure import load_dna_structure_from_cadnano

CADNANO_URL = 'https://bionano.physics.illinois.edu/sites/all/modules/origami/simple.json'
CADNANO_CACHE = Path(__file__).parent / 'test_data' / 'simple.json'


@pytest.fixture(scope='session')
def simple_cadnano() -> Path:
    """Download simple.json once per session and cache it locally."""
    if not CADNANO_CACHE.exists():
        try:
            urllib.request.urlretrieve(CADNANO_URL, CADNANO_CACHE)
        except Exception as e:
            pytest.skip(f'Could not download cadnano file: {e}')
    return CADNANO_CACHE


@pytest.fixture(scope='module')
def workflow(tmp_path_factory, simple_cadnano):
    """Run the full MC relax → MD relax → production pipeline once per module."""
    work_dir = tmp_path_factory.mktemp('cadnano_workflow')

    structure = load_dna_structure_from_cadnano(simple_cadnano, lattice='sq')

    mc_dir = work_dir / 'mc_relax'
    mc_sim = Simulation(structure, mc_dir)
    mc_sim.build(clean_build='force')
    mc_sim.input.swap_default_input('cpu_MC_relax')
    mc_sim.input_file({
        'steps': '1e4',
        'print_energy_every': '1e3',
        'print_conf_interval': '1e4',
        'T': '30C',
    })
    mc_sim.oxpy_run.run(subprocess=False, log=False, verbose=False)

    md_relax_dir = work_dir / 'md_relax'
    md_relax_sim = Simulation(mc_sim, md_relax_dir)
    md_relax_sim.build(clean_build='force')
    md_relax_sim.input.swap_default_input('cpu_MD_relax')
    md_relax_sim.input_file({
        'steps': '1e3',
        'print_energy_every': '5e2',
        'print_conf_interval': '1e3',
        'T': '30C',
    })
    md_relax_sim.oxpy_run.run(subprocess=False, log=False, verbose=False)

    prod_dir = work_dir / 'production'
    prod_sim = Simulation(md_relax_sim, prod_dir)
    prod_sim.build(clean_build='force')
    prod_sim.input_file({
        'steps': '2e3',
        'print_energy_every': '1e3',
        'print_conf_interval': '1e3',
        'T': '30C',
    })
    prod_sim.oxpy_run.run(subprocess=False, log=False, verbose=False)

    return mc_sim, md_relax_sim, prod_sim


class TestStructureLoad:
    def test_returns_dna_structure_from_cadnano(self, simple_cadnano):
        from oxpy_utils.structure_editor.dna_structure import DNAStructure
        s = load_dna_structure_from_cadnano(simple_cadnano, lattice='sq')
        assert isinstance(s, DNAStructure)

    def test_has_bases(self, simple_cadnano):
        s = load_dna_structure_from_cadnano(simple_cadnano, lattice='sq')
        assert s.nbases > 0

    def test_all_bases_valid(self, simple_cadnano):
        s = load_dna_structure_from_cadnano(simple_cadnano, lattice='sq')
        for base in s.iter_bases():
            assert base.base in ('A', 'T', 'C', 'G')


class TestMcRelax:
    def test_last_conf_exists(self, workflow):
        mc_sim, _, _ = workflow
        assert mc_sim.sim_files.last_conf.exists()

    def test_energy_file_exists(self, workflow):
        mc_sim, _, _ = workflow
        assert mc_sim.sim_files.energy.exists()

    def test_energy_values_finite(self, workflow):
        mc_sim, _, _ = workflow
        energy = mc_sim.analysis.energy_df
        # cpu_MC_relax uses a modified backbone potential; energy need not decrease
        assert np.isfinite(energy['U'].values).all()


class TestMdRelax:
    def test_last_conf_exists(self, workflow):
        _, md_relax_sim, _ = workflow
        assert md_relax_sim.sim_files.last_conf.exists()

    def test_energy_file_exists(self, workflow):
        _, md_relax_sim, _ = workflow
        assert md_relax_sim.sim_files.energy.exists()


class TestProduction:
    def test_trajectory_exists(self, workflow):
        _, _, prod_sim = workflow
        assert prod_sim.sim_files.traj.exists()

    def test_energy_file_exists(self, workflow):
        _, _, prod_sim = workflow
        assert prod_sim.sim_files.energy.exists()

    def test_energy_df_columns(self, workflow):
        _, _, prod_sim = workflow
        df = prod_sim.analysis.energy_df
        assert set(df.columns) >= {'time', 'U'}


class TestAnalysis:
    def test_mean_structure(self, workflow):
        from oxDNA_analysis_tools.UTILS.data_structures import Configuration
        _, _, prod_sim = workflow
        mean = prod_sim.analysis.mean()
        assert isinstance(mean, Configuration)

    def test_deviations(self, workflow):
        _, _, prod_sim = workflow
        mean = prod_sim.analysis.mean()
        rmsds, rmsfs = prod_sim.analysis.deviations(mean_conf=mean)
        assert len(rmsds) > 0
        assert np.all(rmsds >= 0)
        assert len(rmsfs) == prod_sim.analysis._describe()[0].nbases

    def test_radius_of_gyration(self, workflow):
        _, _, prod_sim = workflow
        rgs = prod_sim.analysis.radius_of_gyration()
        assert len(rgs) > 0
        assert np.all(rgs > 0)