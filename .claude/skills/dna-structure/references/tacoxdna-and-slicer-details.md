# tacoxDNA loaders & StructureSlicer — deeper reference

Companion to `../SKILL.md`.

## tacoxDNA-backed loaders

`_tacoxdna_src()` resolves `Path(__file__).parents[3] / 'tacoxDNA' / 'src'`
unless `dna_structure.TACOXDNA_SRC` is set. All three tacoxDNA-backed
loaders round-trip through a temp directory and a subprocess call
(`_tacox_run`), then call `load_dna_structure` on the resulting `.top`/
`.oxdna` files:

- `load_dna_structure_from_pdb(pdb_path, direction='35'|'53')` — shells out
  to tacoxDNA's `PDB_oxDNA.py`. `direction` describes the nucleotide listing
  order in the source PDB, not the output structure's storage order (which
  is always 3'→5' once loaded, per the main skill file).
- `load_dna_structure_from_cadnano(json_path, lattice, sequence_file=None)`
  — `lattice` must be `'sq'` (square) or `'he'` (honeycomb), matching
  whatever lattice the cadnano design actually used. `sequence_file` (a
  FASTA path) is converted to tacoxDNA's `.sqs` format via `_fasta_to_sqs`
  (strips `>` headers, joins to one line) — supplying this gives
  deterministic base assignment; omitting it lets tacoxDNA randomize bases,
  which is what the shipped `cadnano_origami_workflow.py` example does.
- `load_dna_structure_from_rcsb(pdb_id, direction='35')` — downloads via
  `Bio.PDB.PDBList` to a `.ent` file, then defers to the PDB loader above.

`export_pdb(path, direction='35'|'53', hydrogens=True)` round-trips the
other direction: writes a temp `.top`/`.oxdna`, then shells out to
tacoxDNA's `oxDNA_PDB.py`.

**None of this is a pip package dependency** — it's a sibling git checkout.
If you're running in an environment where the Dockerfile/setup didn't clone
`tacoxDNA` next to `oxpy-utils`, `oxDNA`, and `polycubes` (see the
container-level layout), every PDB/cadnano/RCSB loader call raises
`FileNotFoundError` with a message telling you to set `TACOXDNA_SRC`
yourself.

## `load_oxview` — the one loader that doesn't use tacoxDNA

Parses OxView-format JSON directly. Builds an empty `DNAStructure([], 0, box)`
and adds strands per `NucleicAcidStrand` entry in the file, validating
topology consistency (asserting `n3`/`n5` neighbor references match) and
reconstructing cluster/color metadata (`assign_base_to_cluster`,
`base_coloration`) plus rehydrating `base_id_map`/`base_id_reverse_map`
directly from the file's own ids. Only cleanly handles a single system in
the OxView file — multiple systems print a warning and get merged.

## Export format details

`export_top_conf(top_file_path, conf_file_path)` writes:

- Conf header: `t = <time>`, `b = <box x y z>`, `E = <energy x y z>`.
- Top header: `<nbases> <nstrands>`.
- Per base — top row: `<strand_id+1> <base> <n3> <n5>` (global-linear
  neighbor indices, `-1` at strand ends); conf row:
  `pos a1 a3 0.0 0.0 0.0 0.0 0.0 0.0` — **velocities and angular
  velocities are always zeroed**, since `DNAStructure` doesn't store
  particle velocities at all. If you're exporting mid-simulation state for
  something velocity-sensitive, this loses that information.

`export_oxview`/`get_oxview_json` encode `base_coloration` and cluster
membership per nucleotide, auto-inboxing the same way as `export_top_conf`
if the box isn't already valid.

`export_gltf(path, sphere_radius=0.3, default_color=None)` renders one
sphere per base, instanced/grouped by color — raises if any base has no
assigned color and you didn't supply `default_color`.

## `StructureSlicer` internals

`StructureSlicer(sim)` subclasses `BuildSimulation` and overrides
`build_dat_top()`, so it plugs directly into the ordinary `Simulation`
lifecycle via `sim.set_builder(slicer)`.

Selection methods (composable via a `behavior` of `"union"`,
`"intersection"`, or `"replace"`):

- `set_slice(bases: Iterable[int])` — direct OxView base indices. Converts
  to persistent uids immediately (`__slice_bases`), then walks each
  selected base's strand-neighbor to find **cut points**: selected bases
  whose 3'/5' neighbor is *not* selected. These get recorded as
  `(keep_uid, adjacent_uid)` pairs in `slice_pts` — this is what
  `do_slice()` later nicks at.
- `set_slice_box(corner_1, corner_2, reverse=False, behavior=...)` —
  axis-aligned box keep/exclude.
- `set_slice_plane(points, norm, behavior=...)` — keep bases on one side of
  a plane (the source docstring literally attributes this one to being
  "written by chatGPT" — treat it as less battle-tested than `set_slice`).

`do_slice()`:
1. Deep-copies the parent structure (`starting_structure()`, cached).
2. For each cut point, calls `nick(strand_id, adj_idx)` at the index
   determined by direction (respecting the 3'→5' convention — see the main
   skill file's Gotchas).
3. Discards any strand not fully contained in the selection, asserting
   every remaining strand is all-in or all-out after nicking.
4. Calls `export_top_conf` straight into the sim's directory — this is the
   `build_dat_top()` override itself.

`load_sampling_from(traj_file_name="trajectory.dat")` computes per-endpoint
mean position and RMSF from a parent trajectory (via oat's `mean`/
`deviations`), so `add_endpoint_forces()` can scale each cut point's
`HARMONIC_TRAP` stiffness by `1/rmsf` — a base that moved around a lot in
the full-structure trajectory gets a looser restraint in the sliced
sub-simulation than one that barely moved. `add_endpoint_forces_simple()`
is the fallback that just applies one fixed stiffness everywhere, no
trajectory required.

Concrete validated use case (`tests/test_structure_slicer.py`): relax a
full cube origami (MC then MD), run production MD to get a trajectory,
slice out a contiguous ~500-base region, seed a new simulation with just
that region plus RMSF-scaled endpoint restraints, and verify the sliced
sub-simulation's local dynamics track the full-structure production run via
RMSD checks.
