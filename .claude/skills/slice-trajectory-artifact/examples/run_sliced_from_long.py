"""Sliced run whose endpoint traps are derived from the long full-cube production run (GPU).
usage: python run_sliced_from_long.py TAG SEED [scale] [mode] [steps] [print_every]
  scale : multiplier on the equipartition stiffness (1.0 = k = 3kT / rmsf^2 straight from the control)
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/root/Github/oxpy-utils/tests")
from test_structure_slicer import BASES
from oxpy_utils.oxdna_simulation import Simulation
from oxpy_utils.structure_editor.structure_slicer import StructureSlicer

RUNS = Path("/root/scripts/slice_demo/runs_gpu")
tag, seed = sys.argv[1], sys.argv[2]
scale = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
mode = sys.argv[4] if len(sys.argv) > 4 else "equipartition"
steps = int(float(sys.argv[5])) if len(sys.argv) > 5 else 1_000_000
every = int(float(sys.argv[6])) if len(sys.argv) > 6 else 10000

sim = Simulation(RUNS / "6_prod_long", RUNS / f"7_sliced_{tag}")
sl = StructureSlicer(sim)
sim.set_builder(sl)
sl.set_slice(BASES)
sl.load_sampling_from("trajectory.dat")
sim.build()
sim.input.swap_default_input("cuda_MD")
sim.input_file({"steps": str(steps), "print_conf_interval": str(every), "print_energy_every": str(every), "seed": seed})
sl.add_endpoint_forces(scale, mode=mode)
sl.build_force()
k = sl.endpoint_stiffnesses(scale, mode)
print(f"{tag}: {len(k)} traps, stiffness min/median/max = {k.min():.3f}/{np.median(k):.3f}/{k.max():.3f}, kT = {sl.sim_kT():.4f}", flush=True)
sim.oxpy_run.run(subprocess=False)
assert sim.oxpy_run.error_message is None, sim.oxpy_run.error_message
print("DONE", tag, flush=True)
