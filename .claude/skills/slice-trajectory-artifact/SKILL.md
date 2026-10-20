---
name: slice-trajectory-artifact
description: Use when asked to visualize, share, or compare a finished StructureSlicer workflow (a sliced oxDNA run with endpoint traps) as an interactive Artifact page - 3D playback of the slice, aligned RMSD from the starting structure against an unsliced baseline, energy, and the raw-file paths. Also use when asked how to run a slicer workflow so it can be validated (production run, equipartition traps, matched control). Not for general oxDNA trajectory plotting unrelated to StructureSlicer.
---

# Slice trajectory artifact

`build_slice_page.py` (next to this file) turns a finished `StructureSlicer` workflow into one
self-contained HTML page: a 3D player of the sliced region (colored by displacement from the
start, trapped cut ends highlighted, the rest of the structure as faint context), a linked RMSD
chart and energy chart, and a frame table. It needs no list of slice bases, so it works however
the slice was chosen (`set_slice`, `set_slice_box`, `set_slice_plane`).

## Inputs (default oxpy_utils file names)

| flag | directory holds | notes |
|---|---|---|
| `--sliced` | `*.top`, `init.dat`, `trajectory.dat`, `energy.dat`, `forces.json`, `input.json` | the sliced run |
| `--source` | `*.top`, `last_conf.dat` | the full-structure run whose **final frame was sliced** |
| `--control` | `*.top`, `trajectory.dat` | optional but recommended: the unsliced structure **continued from the same final frame**, with the same `steps` and `print_conf_interval` as the sliced run |

```bash
python .claude/skills/slice-trajectory-artifact/build_slice_page.py \
  --sliced runs/7_sliced --source runs/6_prod --control runs/8_control \
  --out page.html --title "Cube Slice Rerun" --structure "the relaxed DNA-origami cube" \
  --stiffness-note "with a stiffness of 3kT/RMSF² set by a 1,000,000-step run of the full structure" \
  --build-note "One paragraph: relaxation steps, production length, how the traps were set."
```

Then publish `page.html` with the Artifact tool (`icon: "chart"`). The page is a fixed template, so
this does not need the artifact-design pass. It prints the mean and final RMSD for each series so
you can sanity-check before publishing. Page size is about 1.8 MB for 918 nt x 101 frames; use
`--max-frames` / `--max-context` for larger slices (the page limit is 16 MB).

## Running a workflow that validates properly

A sliced run's RMSD from its starting snapshot means little on its own: the unsliced structure
also drifts from that snapshot. Always run the control, and judge the slice against it. The recipe
that gave matching results (see `examples/` for the two scripts; paths there are specific to the
original cube test):

1. MC relax, MD relax (`cpu_MC_relax` has no CUDA variant; `cuda_MD_relax` and `cuda_MD` do).
2. **Production run of the full structure: aim for 1e6+ steps** (a 2,000-step run is only a smoke
   test; its RMSFs are too small and the traps come out far too stiff). Print every `1e4` steps.
3. `slicer.set_slice(...)`, then `slicer.load_sampling_from("trajectory.dat")`, `sim.build()`,
   `sim.input.swap_default_input("cuda_MD")` (this must precede force setup because it resets the
   input), then `slicer.add_endpoint_forces(mode="equipartition")` and `slicer.build_force()`.
   Equipartition stiffness is `3 kT / RMSF^2` per endpoint; `force_stiffness` is a multiplier
   (1.0 = straight from the production run).
4. Run the sliced simulation, and a control: the full structure continued from the production
   run's `last_conf.dat` for the same steps and print interval (new seed).
5. Build the page. For a stricter check, also run an independent second production run and compare
   per-nucleotide RMSF (sliced vs control should correlate about as well as control vs control).

## Gotchas learned the hard way

- **Sliced particle order is not sorted-base order.** Match sliced particle `i` to its source
  particle by exact position in the starting frame (as the builder does). Never index by
  `sorted(bases)`.
- **Align on the whole structure** when computing RMSF for trap stiffness (`load_sampling_from`
  does this by default now). Aligning on the endpoints alone strips out the motion the absolute
  traps must hold and gives traps that are far too stiff.
- OAT `mean` / `deviations` return arrays for **every** particle, not just `indexes`; take the
  endpoint rows yourself. RMSF from `deviations` is in nm (x0.8518 of simulation units).
- **oxDNA aborts on "too much data (x MB/s)"** when `max_io` (default 5 MB/s) is exceeded. On a GPU
  with a ~14k nt structure, printing every 2e3 steps trips it. Print less often (1e4 is plenty:
  consecutive frames are strongly correlated), and raise `max_io` only modestly.
- Give replicates explicit `seed`s; without one they can share a seed.
- The builder refuses a control whose frame steps do not line up with the sliced run's.
- Trap centers are the starting snapshot, not the mean position.

## Related skills

`dna-structure` covers the `StructureSlicer` API itself (selection methods, `do_slice`, force modes);
`oxdna-simulation-base` and `sim-chaining` cover the `Simulation` and relax->production plumbing the recipe above uses.

## Requires

The `equipartition` mode and whole-structure alignment in `StructureSlicer`
(`add_endpoint_forces(mode="equipartition")`, `load_sampling_from`). Python deps:
`oxDNA_analysis_tools`, `scipy`, `numpy`.
