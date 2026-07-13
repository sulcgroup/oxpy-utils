"""Tests for Analysis (merged with oat methods) and the sim.oat alias."""
import shutil
from pathlib import Path

import numpy as np
import pytest

from oxpy_utils.oxdna_simulation import Analysis, Simulation
from oxDNA_analysis_tools.UTILS.get_confs import Configuration

EXAMPLES = Path(__file__).parent.parent / 'examples' / '8nt_duplex_files'


@pytest.fixture(scope='module')
def ran_sim(tmp_path_factory):
    """Build and run a short MC simulation once for the whole module."""
    d = tmp_path_factory.mktemp('analysis')
    for f in ('duplex_box_30.top', 'duplex_box_30.dat'):
        shutil.copy(EXAMPLES / f, d / f)
    sim = Simulation(d, d / 'sim')
    sim.build()
    sim.input.swap_default_input('cpu_MC_relax')
    sim.input_file({'print_conf_interval': '1e3', 'print_energy_every': '1e3', 'steps': '5e3'})
    sim.oxpy_run.run(subprocess=False)
    return sim


class TestAnalysisOatAlias:
    def test_oat_is_analysis(self, ran_sim):
        assert ran_sim.oat is ran_sim.analysis

    def test_analysis_is_analysis_instance(self, ran_sim):
        assert isinstance(ran_sim.analysis, Analysis)


class TestDescribe:
    def test_describe_returns_top_and_traj_info(self, ran_sim):
        top_info, traj_info = ran_sim.analysis._describe()
        assert top_info.nbases > 0
        assert traj_info.nconfs > 0

    def test_describe_populates_attributes(self, ran_sim):
        ran_sim.analysis.describe()
        assert hasattr(ran_sim.analysis, 'top_info')
        assert hasattr(ran_sim.analysis, 'traj_info')
        assert ran_sim.analysis.top_info.nbases > 0

    def test_describe_nconfs_matches_steps(self, ran_sim):
        _, traj_info = ran_sim.analysis._describe()
        # 5e3 steps / 1e3 print_conf_interval = 5 confs
        assert traj_info.nconfs == 5


class TestLoadConf:
    def test_load_conf_from_path_string(self, ran_sim):
        conf = ran_sim.analysis._load_conf('last_conf.dat')
        assert isinstance(conf, Configuration)

    def test_load_conf_passthrough(self, ran_sim):
        conf_in = ran_sim.analysis._load_conf('last_conf.dat')
        conf_out = ran_sim.analysis._load_conf(conf_in)
        assert conf_out is conf_in


class TestMean:
    def test_mean_returns_configuration(self, ran_sim):
        conf = ran_sim.analysis.mean(outfile='mean.dat')
        assert isinstance(conf, Configuration)

    def test_mean_writes_file(self, ran_sim):
        ran_sim.analysis.mean(outfile='mean.dat')
        assert (ran_sim.sim_dir / 'mean.dat').exists()

    def test_mean_shape(self, ran_sim):
        top_info, _ = ran_sim.analysis._describe()
        conf = ran_sim.analysis.mean()
        assert conf.positions.shape == (top_info.nbases, 3)


class TestDeviations:
    def test_deviations_returns_arrays(self, ran_sim):
        ran_sim.analysis.mean(outfile='mean.dat')
        rmsds, rmsfs = ran_sim.analysis.deviations(mean_conf='mean.dat')
        assert isinstance(rmsds, np.ndarray)
        assert isinstance(rmsfs, np.ndarray)

    def test_deviations_shapes(self, ran_sim):
        ran_sim.analysis.mean(outfile='mean.dat')
        top_info, traj_info = ran_sim.analysis._describe()
        rmsds, rmsfs = ran_sim.analysis.deviations(mean_conf='mean.dat')
        assert rmsds.shape == (traj_info.nconfs,)
        assert rmsfs.shape == (top_info.nbases,)

    def test_rmsds_nonnegative(self, ran_sim):
        ran_sim.analysis.mean(outfile='mean.dat')
        rmsds, _ = ran_sim.analysis.deviations(mean_conf='mean.dat')
        assert np.all(rmsds >= 0)


class TestCentroid:
    def test_centroid_returns_conf_and_rmsd(self, ran_sim):
        ran_sim.analysis.mean(outfile='mean.dat')
        conf, rmsd = ran_sim.analysis.centroid(ref_conf='mean.dat', outfile='centroid.dat')
        assert isinstance(conf, Configuration)
        assert isinstance(rmsd, float)
        assert rmsd >= 0

    def test_centroid_writes_file(self, ran_sim):
        ran_sim.analysis.mean(outfile='mean.dat')
        ran_sim.analysis.centroid(ref_conf='mean.dat', outfile='centroid.dat')
        assert (ran_sim.sim_dir / 'centroid.dat').exists()


class TestDecimate:
    def test_decimate_writes_file(self, ran_sim):
        ran_sim.analysis.decimate(outfile='strided.dat', stride=2)
        assert (ran_sim.sim_dir / 'strided.dat').exists()

    def test_decimate_reduces_nconfs(self, ran_sim):
        from oxDNA_analysis_tools.UTILS.RyeReader import describe as oat_describe
        ran_sim.analysis.decimate(outfile='strided2.dat', stride=2)
        top = ran_sim.sim_files.top.as_posix()
        _, traj_orig = ran_sim.analysis._describe()
        _, traj_strided = oat_describe(top, str(ran_sim.sim_dir / 'strided2.dat'))
        assert traj_strided.nconfs < traj_orig.nconfs


class TestRadiusOfGyration:
    def test_rg_returns_array(self, ran_sim):
        rg = ran_sim.analysis.radius_of_gyration()
        assert isinstance(rg, np.ndarray)

    def test_rg_length_matches_nconfs(self, ran_sim):
        _, traj_info = ran_sim.analysis._describe()
        rg = ran_sim.analysis.radius_of_gyration()
        assert len(rg) == traj_info.nconfs

    def test_rg_positive(self, ran_sim):
        rg = ran_sim.analysis.radius_of_gyration()
        assert np.all(rg > 0)