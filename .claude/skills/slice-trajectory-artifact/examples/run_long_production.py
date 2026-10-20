"""Long production run of the full cube (GPU): the sampling source for the slicer's endpoint traps and the control.
usage: python run_long_production.py [steps] [print_every] [out_name] [seed] [source_dir]
"""
import sys
from pathlib import Path

from oxpy_utils.oxdna_simulation import Simulation

RUNS = Path("/root/scripts/slice_demo/runs_gpu")
steps = int(float(sys.argv[1])) if len(sys.argv) > 1 else 1_000_000
every = int(float(sys.argv[2])) if len(sys.argv) > 2 else 10000
name = sys.argv[3] if len(sys.argv) > 3 else "6_prod_long"
seed = sys.argv[4] if len(sys.argv) > 4 else "300"
source = sys.argv[5] if len(sys.argv) > 5 else "2_md_relax"

sim = Simulation(RUNS / source, RUNS / name)
sim.build()
sim.input.swap_default_input("cuda_MD")
sim.input_file({"steps": str(steps), "print_conf_interval": str(every), "print_energy_every": str(every), "seed": seed, "max_io": "10"})
sim.oxpy_run.run(subprocess=False)
assert sim.oxpy_run.error_message is None, sim.oxpy_run.error_message
print("DONE", flush=True)
