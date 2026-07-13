import shutil

import itertools
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import pytest

from oxpy_utils.replicas.generate_replicas import ReplicaGroup
from oxpy_utils.vmmc_umbrella.vmmc_replicas import VmmcReplicas, VmmcReplicasGroup
from oxpy_utils.vmmc_umbrella.vmmc_data import VMMCData, average_vmmc_data

from oxpy_utils.oxdna_simulation import SimulationManager
from oxpy_utils.vmmc_umbrella.vmmc import VirtualMoveMonteCarlo


# copied this from Matt's test_oxdna_simulation.py file
@pytest.fixture
def vmmc_sim(tmp_path) -> VirtualMoveMonteCarlo:
    assert tmp_path.exists()
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
    return VirtualMoveMonteCarlo(file_dir, sim_dir)

def test_vmmc(vmmc_sim: VirtualMoveMonteCarlo):
    """
    test vmmc?
    """

    # will set nucleotides to test melt
    # order params will be set automatically
    vmmc_sim.set_nucleotides(list(range(0, 7)),
                             list(range(8, 15)))

    vmmc_sim.build()
    # set order parameter file name
    vmmc_sim.input["op_file"] = "op.txt"
    # set weights file name
    vmmc_sim.input["weights_file"] = "weights.txt"
    # build vmmc-specific stuff - needs to wait for file names to be assigned
    vmmc_sim.build_vmmc()
    # todo: more tests?

@pytest.fixture
def vmmc_replicas(tmp_path) -> VmmcReplicasGroup:
    reps = VmmcReplicasGroup()
    # construct VmmcReplicas object
    # set vmmc temperatures
    reps.temperatures = [19, 21, 23, 25, 27, 29, 31]

    # todo: actually two distinct systems instead of the same one twice
    source_dir_path = Path(__file__).parent.parent / 'examples' / '8nt_duplex_files'
    a_src_path = source_dir_path
    b_src_path = source_dir_path
    # set up 3 replicas each of system "a" and system "b", located at tmp_path
    reps.multisystem_replica(systems=[
            ("a", a_src_path, tmp_path / "a"),
            ("b", b_src_path, tmp_path / "b")
        ],
        n_replicas_per_system=3)
    reps.build()
    return reps

def test_process_vmmc_data(vmmc_replicas: VmmcReplicasGroup):
    """
    """
    mgr = SimulationManager()
    # run our vmmc simulations
    for sim in vmmc_replicas.sim_list:
        sim.set_nucleotides(list(range(8)),
                                 list(range(15, 7, -1)))
        sim.build()
        # set simulation time to something short enough that we can actually run it
        sim.input["steps"] = int(1e5)
        sim.input["T"] = "25C"
        sim.weights[...] = sim.generate_weights(11.)

        mgr.queue_sim(sim)
    mgr.run(join=True)

    for system in vmmc_replicas.sim_list:
        system.analysis.read_vmmc_op_data()
        system.analysis.plot_energy()
        system.analysis.plot_sampling_pie_chart()
        # system.analysis.calculate_and_estimate_melting_profiles()

    # todo: figure out what i was trying to do here, then fix
    # finding free energy
    #
    # # split into 3 groups of 5 vmmc simulations
    # n_groups = len(vmmc_replicas.systems)
    # n_total_reps = len(vmmc_replicas.sim_list)
    # n_sims_per_group = int(n_total_reps / n_groups)
    #
    # cases_data = []
    #
    # cases = list(vmmc_replicas.systems.keys())
    # for group_idx, vmmc_group in enumerate(vmmc_replicas.systems.values()):
    #     # iter our 3 groups of 5
    #     last_hists = []
    #     for sim in vmmc_group.simulations:
    #         # for rep_group in range(n_groups):
    #         # iter simulations in group of 5
    #         # for i in range(n_sims_per_group):
    #         #     idx = i * n_groups + rep_group
    #         #     assert 0 <= idx < n_total_reps
    #         # get simulation data
    #         sim.read_bond_op()
    #         # last_hist (in wrapper class) for simulation
    #         data: VMMCData = sim.get_vmmc_data()
    #         last_hist_data = data.df
    #         # print(last_hist_data)
    #         last_hists.append(data)
    #     avgs = average_vmmc_data(last_hists)
    #     # compute means of slices
    #     # todo: more possible combinations of groups?
    #     partition_avgs =     average_vmmc_data(last_hists)
    #
    #     # compute difference between slice means
    #     partition_avg_diffs = [
    #         (a.free_energy - b.free_energy).abs()
    #         for a, b in itertools.combinations(partition_avgs, 2)
    #     ]
    #     errors = np.average(partition_avg_diffs, axis=0)
    #     fe = avgs.free_energy
    #     fe = fe - fe[0] + 1e-12  # shift curve so 0 h bonds = 0 energy, w/ small offset to prevent divide-by-0 problems
    #     cases_data.append((fe[:-1],
    #                        errors[avgs.h_bonds < 6],
    #                        avgs.h_bonds[avgs.h_bonds < 6]))
    #     # propegate error for 0-bond to others
    #     errors[0] = 0
    #     if group_idx == 0:
    #         case_0_data = (fe,
    #                        errors,
    #                        avgs.h_bonds)
    #
    # # Create the plot
    # plt.figure(figsize=(8, 5))
    #
    # # X-axis: num hbonds, Y-axis: Free energy
    #
    # plt.errorbar(case_0_data[2], case_0_data[0], yerr=case_0_data[1], fmt='-o', capsize=5, label="No Mismatch")
    # # Plotting free energy with error bars
    # for i, (free_energies, errors, h_bonds) in list(enumerate(cases_data))[1:]:
    #     plt.errorbar(h_bonds, free_energies, yerr=errors, fmt='-o', capsize=5, label=f"Case {cases[i]}")
    #
    # # Add labels and title
    # plt.xlabel('Number of H-Bonds')
    # plt.ylabel('Free Energy [kB * T')
    # plt.title('Free Energies of Bonding States')
    #
    # # Add grid for better visualization
    # plt.grid(True)
    #
    # # Show the plot
    # plt.legend(title="Mismatch Position")
    # plt.tight_layout()
    # plt.savefig("free_energy.svg")
    # plt.show()