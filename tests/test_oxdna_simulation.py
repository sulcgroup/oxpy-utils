import pytest
import shutil
from pathlib import Path
from oxpy_utils.oxdna_simulation import Analysis, Simulation, SimulationManager
from oxpy_utils.utils.observable import Observable, ObservableColumn
from unittest.mock import Mock, patch


class TestSimulation:
    @pytest.fixture
    def sim(self, tmp_path):
        # Define the path for the file_dir within the temporary directory
        file_dir = tmp_path / "files"

        # Create the file_dir
        file_dir.mkdir()

        # Path to the examples directory in your package
        # Adjust the path as necessary depending on your package structure
        examples_dir = Path(
            __file__).parent.parent / 'examples' / '8nt_duplex_files'

        # Files to be copied
        files_to_copy = ['duplex_box_30.top', 'duplex_box_30.dat']

        # Copy each file to the file_dir
        for file_name in files_to_copy:
            shutil.copy(examples_dir / file_name, file_dir / file_name)

        # Define sim_dir, which will not be created here but should be managed by Simulation
        sim_dir = file_dir / "simulation"

        # Return a new Simulation instance initialized with these directories
        return Simulation(file_dir, sim_dir)

    @pytest.fixture
    def sim_with_last_conf(self, tmp_path):
        # Define the path for the file_dir within the temporary directory
        file_dir = tmp_path / "files"

        # Create the file_dir
        file_dir.mkdir()

        # Path to the examples directory in your package
        # Adjust the path as necessary depending on your package structure
        examples_dir = Path(
            __file__).parent.parent / 'examples' / '8nt_duplex_files'

        # Files to be copied
        files_to_copy = ['duplex_box_30.top', 'duplex_box_30.dat']

        # Copy each file to the file_dir
        for file_name in files_to_copy:
            if file_name == 'duplex_box_30.dat':
                shutil.copy(examples_dir / file_name, file_dir / 'last_conf.dat')
            else:
                shutil.copy(examples_dir / file_name, file_dir / file_name)

        # Define sim_dir, which will not be created here but should be managed by Simulation
        sim_dir = file_dir / "simulation"

        # Return a new Simulation instance initialized with these directories
        return Simulation(file_dir, sim_dir)

    def test_sim_init(self, sim: Simulation):
        # Test if sim_dir and file_dir are set correctly
        assert Path(sim.file_dir).exists(), "file_dir does not exist"

        # Test if the files were copied correctly
        assert (Path(sim.file_dir) / 'duplex_box_30.top').exists(), "Topology file does not exist in file_dir"
        assert (Path(sim.file_dir) / 'duplex_box_30.dat').exists(), "Conformation file does not exist in file_dir"

    def test_sim_last_conf_init(self, sim_with_last_conf: Simulation):
        sim = sim_with_last_conf
        # Test if sim_dir and file_dir are set correctly
        assert Path(sim.file_dir).exists(), "file_dir does not exist"

        # Test if the files were copied correctly
        assert (Path(sim.file_dir) / 'duplex_box_30.top').exists(), "Topology file does not exist in file_dir"
        assert (Path(sim.file_dir) / 'last_conf.dat').exists(), "Conformation file does not exist in file_dir"

    def test_build_no_existing_dir(self, sim: Simulation):
        sim.build()
        assert sim.sim_dir.exists(), "Simulation directory should be created"

        # Check if both top and dat files are copied
        assert (sim.sim_dir / 'duplex_box_30.top').exists(), "Top file was not copied correctly"
        assert (sim.sim_dir / 'init.dat').exists(), "Dat file was not copied correctly"

        # Check for the creation of input files
        assert (sim.sim_dir / 'input').exists(), "Input file was not created"
        assert (sim.sim_dir / 'input.json').exists(), "Input.json file was not created"

    def test_build_no_existing_dir_last_conf(self, sim_with_last_conf):
        sim = sim_with_last_conf
        sim.build()
        assert sim.sim_dir.exists(), "Simulation directory should be created"

        # Check if both top and dat files are copied
        assert (sim.sim_dir / 'duplex_box_30.top').exists(), "Top file was not copied correctly"
        assert (sim.sim_dir / 'init.dat').exists(), "Dat file was not copied correctly"

        # Check for the creation of input files
        assert (sim.sim_dir / 'input').exists(), "Input file was not created"
        assert (sim.sim_dir / 'input.json').exists(), "Input.json file was not created"

    def test_build_existing_dir_no_overwrite(self, sim):
        # Pre-create the sim_dir to simulate existing conditions
        sim_dir = Path(sim.sim_dir)
        sim_dir.mkdir()
        with patch('builtins.print') as mock_print:
            sim.build(clean_build=False)
            mock_print.assert_called_with(
                'The simulation directory already exists, if you wish to write over the directory set clean_build=force')

    @patch('builtins.input', return_value='y')
    def test_build_clean_true_user_confirms(self, mock_input, sim):
        sim_dir = Path(sim.sim_dir)
        sim_dir.mkdir()
        sim.build(clean_build=True)
        assert sim_dir.exists(), "Simulation directory should be rebuilt"

    @patch('builtins.input', return_value='n')
    def test_build_clean_true_user_aborts(self, mock_input, sim):
        sim_dir = Path(sim.sim_dir)
        sim_dir.mkdir()
        sim.build(clean_build=True)
        assert sim_dir.exists(), "Simulation directory should not be deleted"

    def test_build_clean_force(self, sim):
        sim_dir = Path(sim.sim_dir)
        sim_dir.mkdir()
        sim.build(clean_build='force')
        assert sim_dir.exists(), "Simulation directory should be forcibly rebuilt"

    def test_oxpy_run_run_subprocess_false(self, sim):
        sim.build()
        sim.input.swap_default_input("cpu_MC_relax")
        sim.oxpy_run.run(subprocess=False)

        assert Path(sim.sim_files.traj).exists(), "Trajectory file was not created"
        assert Path(sim.sim_files.last_conf).exists(), "Log file was not created"
        assert Path(sim.sim_files.energy).exists(), "Log file was not created"
        # assert lasf_conf and traj are not empty
        assert Path(sim.sim_files.traj).stat().st_size > 0, "Trajectory file is empty"
        assert Path(sim.sim_files.last_conf).stat().st_size > 0, "Log file is empty"
        assert Path(sim.sim_files.energy).stat().st_size > 0, "Log file is empty"

    def test_last_conf_oxpy_run_run_subprocess_false(self, sim_with_last_conf):
        sim = sim_with_last_conf
        sim.build()
        sim.input.swap_default_input("cpu_MC_relax")
        sim.oxpy_run.run(subprocess=False)

        assert Path(sim.sim_files.traj).exists(), "Trajectory file was not created"
        assert Path(sim.sim_files.last_conf).exists(), "Log file was not created"
        assert Path(sim.sim_files.energy).exists(), "Log file was not created"
        # assert lasf_conf and traj are not empty
        assert Path(sim.sim_files.traj).stat().st_size > 0, "Trajectory file is empty"
        assert Path(sim.sim_files.last_conf).stat().st_size > 0, "Log file is empty"
        assert Path(sim.sim_files.energy).stat().st_size > 0, "Log file is empty"

    def test_oxpy_run_run_subprocess_true(self, sim):
        sim.build()
        sim.input.swap_default_input("cpu_MC_relax")
        sim.oxpy_run.run(join=True)

        assert Path(sim.sim_files.traj).exists(), "Trajectory file was not created"
        assert Path(sim.sim_files.last_conf).exists(), "Log file was not created"
        assert Path(sim.sim_files.energy).exists(), "Log file was not created"
        # assert lasf_conf and traj are not empty
        assert Path(sim.sim_files.traj).stat().st_size > 0, "Trajectory file is empty"
        assert Path(sim.sim_files.last_conf).stat().st_size > 0, "Log file is empty"
        assert Path(sim.sim_files.energy).stat().st_size > 0, "Log file is empty"

    def test_oxpy_run_run_subprocess_true_join_false(self, sim):
        sim.build()
        sim.input.swap_default_input("cpu_MC_relax")
        sim.oxpy_run.run(join=False)
        # wait until subprocess finishes
        sim.oxpy_run.process.join()
        sim.sim_files.parse_current_files()
        assert Path(sim.sim_files.traj).exists(), "Trajectory file was not created"
        assert Path(sim.sim_files.last_conf).exists(), "Log file was not created"
        assert Path(sim.sim_files.energy).exists(), "Log file was not created"
        # assert lasf_conf and traj are not empty
        assert Path(sim.sim_files.traj).stat().st_size > 0, "Trajectory file is empty"
        assert Path(sim.sim_files.last_conf).stat().st_size > 0, "Log file is empty"
        assert Path(sim.sim_files.energy).stat().st_size > 0, "Log file is empty"

    def test_continue_run_oxpy_run_run_subprocess_false(self, sim):
        sim.build()
        sim.input.swap_default_input("cpu_MC_relax")
        sim.input_file({'print_energy_every': '1e2', "steps": "1e3"})
        sim.oxpy_run.run(subprocess=False)

        assert Path(sim.sim_files.traj).exists(), "Trajectory file was not created"
        assert Path(sim.sim_files.last_conf).exists(), "Log file was not created"
        assert Path(sim.sim_files.energy).exists(), "Log file was not created"
        # assert lasf_conf and traj are not empty
        assert Path(sim.sim_files.traj).stat().st_size > 0, "Trajectory file is empty"
        assert Path(sim.sim_files.last_conf).stat().st_size > 0, "Log file is empty"
        assert Path(sim.sim_files.energy).stat().st_size > 0, "Log file is empty"
        # check that energy file has 11 lines
        assert len(open(sim.sim_files.energy).readlines()) == 11, "Energy file has incorrect number of lines"
        sim.oxpy_run.run(subprocess=False, continue_run=1e3)
        assert len(open(sim.sim_files.energy).readlines()) == 22, "Energy file has incorrect number of lines"


class TestSimulationManager:
    @pytest.fixture
    def sim_list(self, tmp_path):
        # Define the path for the file_dir within the temporary directory
        n_sims = 3

        file_dirs = [tmp_path / f"files_{i}" for i in range(n_sims)]

        # Create the file_dir
        for file_dir in file_dirs:
            file_dir.mkdir()

        # Path to the examples directory in your package
        # Adjust the path as necessary depending on your package structure
        examples_dir = Path(
            __file__).parent.parent / 'examples' / '8nt_duplex_files'

        # Files to be copied
        files_to_copy = ['duplex_box_30.top', 'duplex_box_30.dat']

        # Copy each file to the file_dir
        for file_dir in file_dirs:
            for file_name in files_to_copy:
                shutil.copy(examples_dir / file_name, file_dir / file_name)

        # Define sim_dir, which will not be created here but should be managed by Simulation
        sim_dirs = [file_dir / f"simulation_{i}" for i, file_dir in enumerate(file_dirs)]

        # Return a new Simulation instance initialized with these directories
        sim_list = [Simulation(file_dir, sim_dir) for file_dir, sim_dir in zip(file_dirs, sim_dirs)]
        return sim_list

    def test_sim_manager_worker_manager(self, sim_list):
        sim_manager = SimulationManager()
        for sim in sim_list:
            sim.build()
            sim.input.swap_default_input("cpu_MC_relax")
            sim_manager.queue_sim(sim)
        sim_manager.worker_manager(cpu_run=True)

        for sim in sim_list:
            sim.sim_files.parse_current_files()
            assert Path(sim.sim_files.traj).exists(), "Trajectory file was not created"
            assert Path(sim.sim_files.last_conf).exists(), "Log file was not created"
            assert Path(sim.sim_files.energy).exists(), "Log file was not created"
            # assert lasf_conf and traj are not empty
            assert Path(sim.sim_files.traj).stat().st_size > 0, "Trajectory file is empty"
            assert Path(sim.sim_files.last_conf).stat().st_size > 0, "Log file is empty"
            assert Path(sim.sim_files.energy).stat().st_size > 0, "Log file is empty"
            
    #TODO: Write this test
    # def test_sim_manager_worker_manager_handle_death(self, sim_list):
    #     sim_manager = SimulationManager()
    #     for sim in sim_list:
    #         sim.build()
    #         sim.input.swap_default_input("cpu_MC_relax")
    #         sim_manager.queue_sim(sim)
    #     # Simulate a process death


class TestAnalysisObservableData:
    """
    Analysis.observable_data must read the observable's output file at exactly its
    `name` (oxDNA appends no extension), parsing oxDNA's whitespace-delimited,
    possibly multi-column output.
    """

    def _analysis(self, tmp_path) -> Analysis:
        sim = Mock()
        sim.sim_dir = tmp_path
        sim.sim_files = Mock()
        return Analysis(sim)

    def test_reads_multicolumn_whitespace_file_at_bare_name(self, tmp_path):
        # what build_op_trajectory_observable produces: step + one int per order parameter,
        # whitespace-delimited, trailing space, file named "op_trajectory" (no .txt)
        (tmp_path / "op_trajectory").write_text(
            "           0 6 0 0 0.383386 \n"
            "       10000 6 0 0 0.357645 \n"
            "       20000 5 1 0 0.408355 \n"
        )
        analysis = self._analysis(tmp_path)
        obs = Observable("op_trajectory", 10000,
                         ObservableColumn("step"), ObservableColumn("order_parameters"))
        analysis.observables["op_trajectory"] = obs

        df = analysis.observable_data("op_trajectory")

        assert df.shape == (3, 5)
        # every column is numeric (not one collapsed string column)
        assert list(df[1].astype(int)) == [6, 6, 5]
        assert list(df[2].astype(int)) == [0, 0, 1]

    def test_reads_single_column_file(self, tmp_path):
        (tmp_path / "hb_count").write_text("12\n11\n13\n")
        analysis = self._analysis(tmp_path)
        analysis.observables["hb_count"] = Observable(
            "hb_count", 1000, ObservableColumn("hb_list", only_count=True))

        df = analysis.observable_data("hb_count")

        assert df.shape == (3, 1)
        assert list(df[0]) == [12, 11, 13]

    def test_result_is_cached(self, tmp_path):
        (tmp_path / "obs").write_text("1 2\n3 4\n")
        analysis = self._analysis(tmp_path)
        analysis.observables["obs"] = Observable("obs", 1, ObservableColumn("step"))

        first = analysis.observable_data("obs")
        (tmp_path / "obs").write_text("9 9\n9 9\n")   # change on disk
        assert analysis.observable_data("obs").equals(first)   # served from cache
