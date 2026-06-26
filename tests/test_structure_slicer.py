from pathlib import Path

import numpy as np
import pytest

from oxpy_utils.oxdna_simulation import Simulation
from oxpy_utils.structure_editor.dna_structure import load_dna_structure
from oxpy_utils.structure_editor.structure_slicer import StructureSlicer

CUBE_DIR = Path(__file__).parent / "test_data" / "cube"

# A contiguous sub-region of the cube origami used as the slice target.
BASES = [
    530, 531, 532, 533, 534, 535, 536, 537, 538, 539, 540, 541, 542, 543, 544, 545, 546,
    547, 548, 549, 550, 551, 552, 553, 554, 555, 556, 557, 558, 559, 560, 561, 562, 563,
    564, 565, 566, 567, 568, 569, 570, 571, 572, 573, 574, 575, 576, 577, 578, 579, 580,
    581, 582, 583, 584, 585, 586, 587, 588, 589, 590, 591, 592, 593, 594, 595, 596, 597,
    598, 599, 600, 601, 602, 603, 604, 605, 606, 607, 608, 609, 610, 611, 612, 613, 1521,
    1522, 1523, 1524, 1525, 1526, 1527, 1528, 1529, 1530, 1531, 1532, 1533, 1534, 1535,
    1536, 1537, 1538, 1539, 1540, 1541, 1542, 1543, 1544, 1545, 1546, 1547, 1548, 1549,
    1550, 1551, 1552, 1553, 1554, 1555, 1556, 1557, 1558, 1559, 1560, 1561, 1562, 1563,
    1564, 1565, 1566, 1567, 1568, 1569, 1570, 1571, 1572, 1573, 1574, 1575, 1576, 1577,
    1578, 1579, 1580, 1581, 1582, 1583, 1584, 1585, 1586, 1587, 1588, 1589, 1590, 1591,
    1592, 1593, 1594, 1595, 1596, 1597, 1598, 1599, 1600, 1601, 1602, 1603, 1604, 1617,
    1618, 1619, 1620, 3437, 3438, 3439, 3440, 3441, 3442, 3443, 3444, 3445, 3446, 3447,
    3448, 3449, 3450, 3451, 3452, 3453, 3454, 3455, 3456, 3492, 3493, 3494, 3495, 3496,
    3497, 3498, 3499, 3500, 3501, 3502, 3503, 3504, 3505, 3506, 3507, 3508, 3509, 3510,
    3511, 3512, 3513, 3514, 3515, 3516, 3517, 3518, 3519, 3520, 3521, 3522, 3523, 3524,
    3525, 3526, 3527, 3528, 3529, 3530, 3531, 3532, 3533, 4164, 4165, 4166, 4167, 4168,
    4169, 4170, 4171, 4172, 4173, 4174, 4175, 4176, 4177, 4178, 4179, 4180, 4181, 4182,
    4183, 4184, 4185, 4186, 4187, 4188, 4189, 4190, 4191, 4192, 4193, 4194, 4195, 4196,
    4197, 4198, 4199, 4200, 4201, 4202, 4203, 4204, 4205, 4206, 4207, 4208, 4209, 4210,
    4211, 4212, 4213, 4214, 4215, 4216, 4217, 4218, 4219, 4220, 4221, 4222, 4223, 4224,
    4225, 4226, 4227, 4228, 4229, 4230, 4231, 4232, 4233, 4234, 4235, 4236, 4237, 4238,
    4239, 4240, 4241, 4242, 4243, 4244, 4245, 4250, 4251, 4252, 4253, 4254, 4255, 4256,
    4257, 4258, 4259, 4260, 4261, 4262, 5176, 5177, 5178, 5179, 5180, 5181, 5182, 5183,
    5184, 5185, 5186, 5187, 5188, 5189, 5190, 5191, 5192, 5193, 5194, 5195, 5196, 5197,
    5198, 5199, 5200, 5201, 5202, 5203, 5204, 5205, 5206, 5207, 5208, 5209, 5210, 5211,
    5212, 5213, 5214, 5215, 5242, 5243, 5244, 5245, 5246, 5247, 5248, 5249, 5250, 5251,
    5252, 5253, 5254, 5258, 5259, 5260, 5261, 5262, 5263, 5743, 5744, 5745, 5746, 5747,
    5748, 5749, 5750, 5751, 5752, 5753, 5754, 5755, 5756, 5757, 5758, 5759, 5760, 5761,
    5762, 5763, 5764, 5765, 5766, 5767, 5768, 5769, 5770, 5771, 5772, 5773, 5774, 5775,
    5776, 5777, 5778, 5779, 5780, 5781, 5782, 5783, 5784, 6407, 6408, 6409, 6410, 6411,
    6412, 6413, 6414, 6415, 6416, 6417, 6418, 6419, 6420, 6421, 6422, 6423, 6424, 6425,
    6426, 6427, 6428, 6429, 6430, 6431, 6432, 6433, 6434, 9785, 9786, 9787, 9788, 9789,
    9790, 9791, 9792, 9793, 9794, 9795, 9796, 9797, 9798, 9799, 9800, 9801, 9802, 9803,
    9804, 9805, 9806, 9807, 9808, 9809, 9810, 9811, 9812, 9813, 9814, 9815, 9816, 9817,
    9818, 9819, 9820, 9821, 9822, 9823, 9824, 9825, 9826, 9827, 9828, 9829, 9830, 9831,
    9832, 9833, 9834, 9835, 9836, 9837, 9838, 9839, 9840, 9841, 9842, 9843, 9844, 9845,
    9846, 9847, 9848, 9849, 9850, 9851, 9852, 9853, 9854, 9855, 9856, 10948, 10949, 10950,
    10951, 10952, 10953, 10954, 10955, 10956, 10957, 10958, 10959, 10960, 10961, 10962,
    10963, 10964, 10965, 10966, 10967, 10968, 10969, 10970, 10971, 10972, 10973, 10974,
    10975, 10976, 10977, 10978, 10979, 10980, 10981, 10982, 10983, 10984, 10985, 10986,
    10987, 10988, 10989, 10990, 10991, 10992, 10993, 10994, 10995, 10996, 10997, 10998,
    10999, 11000, 11001, 11002, 11003, 11004, 11005, 11006, 11007, 11008, 11009, 11010,
    11011, 11012, 11013, 11014, 11015, 11016, 11017, 11018, 11019, 11020, 11021, 11022,
    11427, 11428, 11429, 11430, 11431, 11432, 11433, 11434, 11435, 11436, 11437, 11438,
    11439, 11440, 11441, 11442, 11443, 11444, 11445, 11446, 11447, 11448, 11449, 11450,
    11451, 11452, 11453, 11454, 11455, 11456, 11457, 11458, 11459, 11460, 11461, 11462,
    11463, 11464, 11465, 11466, 11467, 11468, 11469, 11470, 11471, 11472, 11473, 11474,
    11475, 11476, 11477, 11478, 11479, 11480, 11481, 11482, 11483, 11484, 11485, 11486,
    11487, 11488, 11489, 11490, 11491, 11492, 11493, 11494, 11495, 11496, 11497, 11498,
    11499, 11500, 11501, 11502, 11706, 11707, 11708, 11709, 11710, 11711, 11712, 11713,
    11714, 11715, 11716, 11717, 11718, 11719, 11720, 11721, 11722, 11723, 11724, 11725,
    11726, 11727, 11728, 11729, 11730, 11731, 11732, 11733, 11734, 11735, 11736, 11737,
    11738, 11739, 11740, 11741, 11742, 11743, 11744, 11745, 11746, 11747, 11748, 11749,
    11750, 11751, 11752, 11753, 11754, 11755, 11756, 11757, 11758, 11759, 11760, 11761,
    11762, 11763, 11764, 11765, 11766, 11767, 11768, 11769, 11770, 11771, 11772, 11773,
    11774, 11775, 11776, 11777, 11778, 11779, 11780, 11781, 11782, 11783, 11784, 11785,
    11858, 11859, 11860, 11861, 11862, 11863, 11864, 11865, 11866, 11867, 11868, 11869,
    11870, 11871, 11872, 11873, 11874, 11875, 11876, 11877, 11878, 11879, 11880, 11881,
    11882, 11883, 11884, 11885, 11886, 11887, 11888, 11889, 11890, 11891, 11892, 11893,
    11894, 11895, 11896, 11897, 11898, 11899, 11900, 11901, 11902, 11903, 11904, 11905,
    11906, 11907, 11908, 11909, 11910, 11911, 11912, 11913, 11914, 11915, 11916, 11917,
    11918, 11919, 11920, 11921, 11922, 11923, 11924, 11925, 11926, 11927, 11928, 11929,
    11930, 11931, 11932, 11933, 11934, 11935, 11936, 11937, 12421, 12422, 12423, 12424,
    12425, 12426, 12427, 12428, 12429, 12430, 12431, 12432, 12433, 12434, 12435, 12436,
    12437, 12438, 12439, 12440, 12441, 12442, 12443, 12444, 12445, 12446, 12447, 12448,
    12449, 12450, 12451, 12452, 12453, 12454, 12455, 12456, 12457, 12458, 12459, 12460,
    12461, 12462, 12463, 12464, 12465, 12466, 12467, 12468, 12469, 12470, 12471, 12472,
    12473, 12474, 12475, 12476, 12477, 12478, 12479, 12480, 12481, 12482, 12483, 12484,
    12485, 12486, 12487, 12488, 12489, 12490, 12491, 12492, 12493, 12494, 12495, 12496,
    12497,
]


def _run_sim(sim: Simulation, default_type: str, steps: int, print_every: int = None,
             extra_params: dict = None):
    """Build, configure, and run a simulation in-process."""
    sim.build()
    sim.input.swap_default_input(default_type)
    params = {"steps": str(steps)}
    if print_every is not None:
        params["print_conf_interval"] = str(print_every)
        params["print_energy_every"] = str(print_every)
    if extra_params:
        params.update(extra_params)
    sim.input_file(params)
    sim.oxpy_run.run(subprocess=False)
    assert sim.oxpy_run.error_message is None, \
        f"Simulation in {sim.sim_dir} failed:\n{sim.oxpy_run.error_message}"


def test_structure_slicer_workflow(tmp_path: Path):
    # ------------------------------------------------------------------ #
    # Step 1: MC relax — bring the cube out of any strained initial state
    # ------------------------------------------------------------------ #
    mc_dir = tmp_path / "1_mc_relax"
    mc_sim = Simulation(CUBE_DIR, mc_dir)
    mc_sim.build()
    mc_sim.input.swap_default_input("cpu_MC_relax")
    mc_sim.oxpy_run.run(subprocess=False)

    assert mc_sim.sim_files.last_conf.exists()
    assert mc_sim.sim_files.last_conf.stat().st_size > 0
    assert mc_sim.sim_files.traj.stat().st_size > 0

    # ------------------------------------------------------------------ #
    # Step 2: MD relax — equilibrate velocities and softer constraints
    # ------------------------------------------------------------------ #
    md_relax_dir = tmp_path / "2_md_relax"
    md_relax_sim = Simulation(mc_dir, md_relax_dir)
    _run_sim(md_relax_sim, "cuda_MD_relax", steps=2500, print_every=100)

    assert md_relax_sim.sim_files.last_conf.exists()
    assert md_relax_sim.sim_files.last_conf.stat().st_size > 0

    # ------------------------------------------------------------------ #
    # Step 3: Short production run — record a trajectory for RMSF-based
    # endpoint forces and sample the configuration space
    # ------------------------------------------------------------------ #
    prod_dir = tmp_path / "3_production"
    prod_sim = Simulation(md_relax_dir, prod_dir)
    _run_sim(prod_sim, "cuda_MD", steps=1000, print_every=100)

    assert prod_sim.sim_files.last_conf.exists()
    assert prod_sim.sim_files.traj.stat().st_size > 0

    # ------------------------------------------------------------------ #
    # Step 4: Build sliced simulation
    # The slicer reads the production last_conf as its starting structure
    # and the production trajectory to estimate per-particle RMSF for
    # the endpoint harmonic traps.
    # ------------------------------------------------------------------ #
    slice_dir = tmp_path / "4_sliced"
    slice_sim = Simulation(prod_dir, slice_dir)
    slicer = StructureSlicer(slice_sim)
    slice_sim.set_builder(slicer)

    slicer.set_slice(BASES)
    slicer.load_sampling_from("trajectory.dat")

    # build() writes the sliced top/conf and a default input file
    slice_sim.build()

    clipped = slicer.clipped_structure()
    assert clipped is not None
    assert clipped.nbases == len(BASES)

    # Both output files must exist in the slice directory
    assert (slice_dir / slicer.top_file_name).exists(), "sliced topology not written"
    assert (slice_dir / "init.dat").exists(), "sliced conf not written"

    # swap_default_input resets input_dict entirely, so it must come before
    # add_endpoint_forces; the latter sets external_forces=1 via Input.__setitem__
    # and that write persists to disk for the subsequent run.
    slice_sim.input.swap_default_input("cpu_MD")
    slice_sim.input_file({"steps": "5000", "print_conf_interval": "100", "print_energy_every": "100"})

    # Add RMSF-scaled harmonic traps at each strand endpoint, write forces
    slicer.add_endpoint_forces()
    slicer.build_force()

    assert (slice_dir / "forces.json").exists(), "forces.json not written"
    assert len(slice_sim.forces) == len(slicer.endpoint_bases()), \
        "number of forces does not match number of endpoint bases"
    assert len(slicer.endpoint_bases()) > 0, "no endpoint bases detected"

    # ------------------------------------------------------------------ #
    # Step 5: Run the sliced simulation
    # ------------------------------------------------------------------ #
    slice_sim.oxpy_run.run(subprocess=False)

    assert slice_sim.sim_files.last_conf.exists()
    assert slice_sim.sim_files.traj.stat().st_size > 0

    # ------------------------------------------------------------------ #
    # Step 6: Compare sliced simulation to the production run
    #
    # The sliced sim starts from the production last_conf, with harmonic
    # traps anchoring the strand endpoints to their production-run
    # positions.  We verify:
    #   (a) The endpoint bases (held by forces) stayed near their
    #       starting positions — confirming the forces are active.
    #   (b) The overall selected region did not drift catastrophically —
    #       confirming the sliced simulation reproduces the local behavior
    #       of the same region in the full production run.
    #
    # "Starting positions" == the production last_conf restricted to the
    # selected bases, which is exactly clipped.poss() before any
    # simulation.  Both the sliced sim and clipped share the same
    # simulation box and coordinate frame, so direct subtraction is valid.
    # ------------------------------------------------------------------ #
    initial_poss = clipped.poss()  # (N_selected, 3) — production last_conf

    final_struct = load_dna_structure(slice_sim.sim_files.top, slice_sim.sim_files.last_conf)
    final_poss = final_struct.poss()  # (N_selected, 3)

    assert initial_poss.shape == final_poss.shape, \
        "topology mismatch between initial and final sliced structures"

    # (a) Endpoint RMSD: forces constrain these bases near their starting
    #     positions, so RMSD should be small even with a short run.
    endpoint_idxs = [clipped.base_index(uid) for uid in slicer.endpoint_bases()]
    ep_initial = initial_poss[endpoint_idxs]
    ep_final = final_poss[endpoint_idxs]
    ep_rmsd = float(np.sqrt(np.mean(np.sum((ep_final - ep_initial) ** 2, axis=1))))
    assert ep_rmsd < 5.0, (
        f"Endpoint RMSD {ep_rmsd:.2f} too large; harmonic traps may not be active"
    )

    # (b) Global RMSD: interior bases can thermally fluctuate, but the
    #     sliced region as a whole should not drift far from the
    #     production-run configuration.
    global_rmsd = float(np.sqrt(np.mean(np.sum((final_poss - initial_poss) ** 2, axis=1))))
    assert global_rmsd < 10.0, (
        f"Global RMSD {global_rmsd:.2f} too large; sliced simulation diverged from "
        "the production-run configuration"
    )