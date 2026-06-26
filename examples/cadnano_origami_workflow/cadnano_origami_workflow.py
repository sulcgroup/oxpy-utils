"""
cadnano → oxDNA workflow example
=================================
Downloads a cadnano design, converts it to oxDNA format, runs an MC relax
followed by an MD relax and a short production run, then computes basic
statistics with oxDNA_analysis_tools.

Steps
-----
1. Download cadnano JSON from Illinois bionano server
2. Load as DNAStructure via tacoxDNA bridge
3. MC relax   — removes clashes from the idealized cadnano geometry
4. MD relax   — equilibrates with a gentle langevin thermostat
5. Production — short MD run collecting a trajectory
6. Analysis   — mean structure, per-frame RMSD, and radius of gyration
"""
from pathlib import Path
import urllib.request
import tempfile

from oxpy_utils.oxdna_simulation import Simulation
from oxpy_utils.structure_editor.dna_structure import load_dna_structure_from_cadnano

CADNANO_URL = 'https://bionano.physics.illinois.edu/sites/all/modules/origami/simple.json'

# ── 1. download cadnano design ────────────────────────────────────────────────

def download_cadnano(dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        urllib.request.urlretrieve(CADNANO_URL, dest)
    return dest


# ── 2. convert to DNAStructure ────────────────────────────────────────────────

def load_structure(json_path: Path):
    return load_dna_structure_from_cadnano(json_path, lattice='sq')


# ── 3-5. build and run simulations ───────────────────────────────────────────

MC_RELAX_PARAMS = {
    'steps': '1e4',
    'print_energy_every': '1e3',
    'print_conf_interval': '1e4',
    'T': '30C',
}

MD_RELAX_PARAMS = {
    'steps': '1e5',
    'print_energy_every': '1e4',
    'print_conf_interval': '1e5',
    'T': '30C',
}

PRODUCTION_PARAMS = {
    'steps': '1e6',
    'print_energy_every': '1e5',
    'print_conf_interval': '1e5',
    'T': '30C',
}


def run_workflow(work_dir: Path):
    work_dir.mkdir(parents=True, exist_ok=True)

    json_path = download_cadnano(work_dir / 'simple.json')
    structure = load_structure(json_path)

    # MC relax — starts from DNAStructure so file_dir == sim_dir
    mc_dir = work_dir / 'mc_relax'
    mc_sim = Simulation(structure, mc_dir)
    mc_sim.build(clean_build='force')
    mc_sim.input.swap_default_input("cpu_mc_relax")
    # mc_sim.input_file(MC_RELAX_PARAMS)
    mc_sim.oxpy_run.run(subprocess=False, log=False, verbose=False)

    # MD relax — seed from MC last conf
    md_relax_dir = work_dir / 'md_relax'
    md_relax_sim = Simulation(mc_sim, md_relax_dir)
    md_relax_sim.build(clean_build='force')
    md_relax_sim.input.swap_default_input("cpu_md_relax")
    # md_relax_sim.input_file(MD_RELAX_PARAMS)
    md_relax_sim.oxpy_run.run(subprocess=False, log=False, verbose=False)

    # Production — seed from MD relax last conf
    prod_dir = work_dir / 'production'
    prod_sim = Simulation(md_relax_sim, prod_dir)
    prod_sim.build(clean_build='force')
    prod_sim.input_file(PRODUCTION_PARAMS)
    prod_sim.oxpy_run.run(subprocess=False, log=False, verbose=False)

    return mc_sim, md_relax_sim, prod_sim


# ── 6. analysis ───────────────────────────────────────────────────────────────

def analyse(prod_sim: Simulation):
    mean_conf = prod_sim.analysis.mean()
    rmsds, rmsfs = prod_sim.analysis.deviations(mean_conf=mean_conf)
    rgs = prod_sim.analysis.radius_of_gyration()
    energy = prod_sim.analysis.energy_df

    print(f"Frames in trajectory : {len(rmsds)}")
    print(f"Mean RMSD            : {rmsds.mean():.4f}")
    print(f"Mean Rg              : {rgs.mean():.4f}")
    print(f"Final potential energy: {energy['U'].iloc[-1]:.4f}")

    return {
        'mean_conf': mean_conf,
        'rmsds': rmsds,
        'rmsfs': rmsfs,
        'rgs': rgs,
        'energy': energy,
    }


if __name__ == '__main__':
    work_dir = Path(__file__).parent / 'run'
    mc_sim, md_relax_sim, prod_sim = run_workflow(work_dir)
    results = analyse(prod_sim)