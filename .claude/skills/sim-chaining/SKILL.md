---
name: sim-chaining
description: Use when chaining multiple oxpy-utils Simulations into a pipeline where one stage's final configuration becomes the next stage's starting configuration - the import -> MC relax -> MD relax -> production pattern. Covers the Simulation(prev_sim, next_dir) constructor overload, why MC relax then MD relax is needed before production on a freshly-imported structure (cadnano/PDB import has severe steric clashes MD integration can't safely start from), the parameter differences between relax and production default inputs, and real bugs in the one shipped example of this pattern (examples/cadnano_origami_workflow/). Only this one example in the repo demonstrates the full chain - VMMC/FFS examples all start from an already-relaxed structure and skip straight to sampling; say so plainly rather than implying otherwise. Not for the Simulation class fundamentals themselves (see oxdna-simulation-base) or building/loading the initial DNAStructure (see dna-structure).
---

# Chaining simulations: import → MC relax → MD relax → production

The only place in this repo that demonstrates a full multi-stage pipeline is
`examples/cadnano_origami_workflow/cadnano_origami_workflow.py` (mirrored by
`tests/test_cadnano_workflow.py`). VMMC and FFS examples all start from an
already-relaxed `.top`/`.dat` and go straight to sampling — if you want to
relax before running VMMC or FFS, you build this chain yourself using the
same mechanism described here; nothing does it for you automatically in
those skills.

## The chaining mechanism

The first stage's `file_dir` doesn't have to be a `DNAStructure` from
cadnano — it can equally be a plain directory already containing a
`.top`/`.dat` pair (the same "path" shape `oxdna-simulation-base` describes
for a standalone `Simulation`). Use whichever starting point matches your
actual structure; only the *first* stage's construction differs, stages 2+
always chain off the previous `Simulation` object either way:

```python
structure = load_dna_structure_from_cadnano(json_path, lattice="sq")  # DNAStructure
# or: skip this and pass an existing directory of .top/.dat files directly
# as the first stage's file_dir instead — both are valid starting points.

mc_sim = Simulation(structure, mc_dir)              # file_dir = a DNAStructure
mc_sim.build(clean_build="force")
mc_sim.input.swap_default_input("cpu_MC_relax")     # exact case matters - see Gotchas
mc_sim.oxpy_run.run(subprocess=False)

md_relax_sim = Simulation(mc_sim, md_relax_dir)     # file_dir = a Simulation object
md_relax_sim.build(clean_build="force")
md_relax_sim.input.swap_default_input("cpu_MD_relax")
md_relax_sim.oxpy_run.run(subprocess=False)

prod_sim = Simulation(md_relax_sim, prod_dir)       # same pattern again
prod_sim.build(clean_build="force")
prod_sim.input.swap_default_input("cpu_MD")         # do NOT skip this - see Gotchas
prod_sim.oxpy_run.run(subprocess=False)
```

**`Simulation(prev_sim, next_dir)` is a first-class, supported constructor
overload** (`isinstance(file_dir, Simulation)` in `oxdna_simulation.py`),
not something you build yourself with manual file copying. It sets
`self.file_dir = prev_sim.sim_dir`, and the ordinary `build()` machinery's
`find_conf_file` preferentially picks up a conf file literally named
`last_conf*` (which every shipped default input JSON writes via
`"lastconf_file": "last_conf.dat"`). Get the previous stage's input to
actually produce that file and the hand-off "just works" — see
`oxdna-simulation-base` for `build()`/`clean_build` mechanics in general.

The only genuinely manual requirements: (a) make sure the upstream stage's
input really does write a `last_conf*`-named file (it does by default), and
(b) call `.build()` again for each new stage so the copy/rename actually
happens.

## Why three stages instead of one

There's essentially one real explanation in the repo — the example script's
own docstring: **MC relax "removes clashes from the idealized cadnano
geometry"**, **MD relax "equilibrates with a gentle langevin thermostat."**
The rest follows from the parameter differences:

- **MC relax has no time integration.** It proposes discrete
  translation/rotation moves (`delta_translation`/`delta_rotation`, both
  `0.22` by default) and accepts/rejects via Metropolis. A freshly-imported
  cadnano lattice geometry can have severely strained/overlapping bases —
  MC relax can't "blow up" the way an MD integrator would, making it the
  safe first step for resolving gross steric clashes.
- **MD relax then takes over** once clashes are gone, but still guards
  against residual instability with `max_backbone_force`/
  `max_backbone_force_far` (caps on the backbone-spring force used during
  integration) and a comparatively generous `verlet_skin`/
  `max_density_multiplier`, at a mild `T=10C`.
- **Production (`cpu_MD`/`cuda_MD`) drops these relax-only safety caps
  entirely** and runs standard `john`-thermostat integration for actual
  sampling, at whatever temperature and step count you actually want.

| param | MC relax | MD relax | Production |
|---|---|---|---|
| `sim_type` | `MC` | `MD` | `MD` |
| `steps` (defaults) | `5e3` | `1e6` | `1e8` (CPU) / `1e9` (CUDA) |
| `T` (defaults) | `30C` | `10C` | `20C` |
| `thermostat` | n/a | `langevin` | `john` |
| `max_backbone_force`/`_far` | present (MC's is typo'd, see Gotchas) | present | **absent** |
| `delta_translation`/`_rotation` | `0.22`/`0.22` | n/a | n/a |

**There is no automated energy-convergence check anywhere in this pipeline.**
Each stage runs to completion unconditionally regardless of whether the
physics actually settled; the analysis step (`mean()`, `deviations()`,
`radius_of_gyration()`, `energy_df['U']`) only runs *after* production, as a
post-hoc diagnostic, not a gate between stages. If you want to verify a
relax stage actually converged before proceeding, you have to check that
yourself (e.g. inspect `energy_df['U']` for a plateau) — nothing in the
codebase does it automatically.

## Gotchas — real bugs in the one shipped example

Verified directly against the current source, not just inferred:

- **`swap_default_input` is called with the wrong case in the example
  script.** `cadnano_origami_workflow.py` calls
  `swap_default_input("cpu_mc_relax")` and `swap_default_input("cpu_md_relax")`
  (all-lowercase) — but the actual files are `cpu_MC_relax.json` /
  `cpu_MD_relax.json` (mixed case). This raises `FileNotFoundError` on a
  case-sensitive filesystem. **Use the correctly-cased names** shown in the
  code block above (which match `tests/test_cadnano_workflow.py`, not the
  example script) — don't copy the example script's calls verbatim.
- **The production stage silently defaults to CUDA if you don't explicitly
  swap its default input.** Every fresh `Input` starts as `cuda_MD`
  regardless of intent. The example script's `prod_sim` only calls
  `input_file(PRODUCTION_PARAMS)` (steps/T/print-interval overrides) and
  never calls `swap_default_input` for the production stage — verified
  directly that this leaves `"backend": "CUDA"` in `input.json`. **On a
  CPU-only box, call `prod_sim.input.swap_default_input("cpu_MD")`
  explicitly before running production**, exactly as shown above; the
  reference example omits this and would fail to launch on CPU-only
  hardware as written.
- **The example's `MC_RELAX_PARAMS`/`MD_RELAX_PARAMS` dicts are dead
  code** — defined but never applied (the calls that would use them are
  commented out). Only `PRODUCTION_PARAMS` is actually used. Don't assume
  the example's relax stages run with the values suggested by those unused
  dicts; they run with whatever `swap_default_input` alone set.
- **`cpu_MC_relax.json` has a typo'd key**: `max_backhone_force_far`
  (missing the `b`) instead of `max_backbone_force_far`. oxDNA's input
  parser won't recognize the misspelled key, so it silently falls back to
  oxDNA's own internal default rather than the intended value — a subtle
  no-op in the shipped default, not something that raises or warns.
- **Skipping MC relax and going straight from a freshly-imported
  `DNAStructure` to MD relax is mechanically possible** (nothing stops
  `Simulation(structure, md_dir)`), but per the docstring's own framing this
  risks feeding MD integration a structure with severe, unresolved steric
  clashes that MD's force caps mitigate but don't eliminate. No test in the
  repo exercises this skip, so treat any specific claim about the failure
  mode (crash vs. a degenerate frozen structure) as inference, not a
  verified fact — check current behavior yourself if you try it.
- **`find_conf_file`'s fallback ordering is filesystem-dependent** if a
  stage's directory ever contains multiple ambiguous `.dat` files with none
  literally prefixed `last_conf` — always let the upstream stage's
  `lastconf_file` setting do its job rather than relying on the fallback.

## Where to look next

- `oxdna-simulation-base` — `build()`/`clean_build` semantics and the
  `Simulation(file_dir, sim_dir)` constructor overloads in full.
- `dna-structure` — how the initial `DNAStructure` gets built/loaded before
  stage 1.
- `vmmc-simulations` / `ffs-simulations` — what to chain *after* this
  pipeline if your production stage is VMMC or FFS rather than plain MD;
  neither skill documents a relax step, so bring one over from here if your
  starting structure needs it.
