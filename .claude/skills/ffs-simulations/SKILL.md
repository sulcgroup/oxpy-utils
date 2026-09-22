---
name: ffs-simulations
description: Use when setting up, running, resuming, or analyzing a Forward Flux Sampling (FFS) run via oxpy-utils - estimating a rare-event rate constant (e.g. duplex dissociation/melting) by ratcheting trial trajectories across a sequence of order-parameter interfaces. Covers oxpy_utils.ffs.*: FFSProgram (the entry point), FFSFluxGenerator, FFSShooter, FFSInterface/Condition, and the YAML-driven ffs_cli.py/ffs_analyze_cli.py scripts. Not for VMMC/umbrella-sampling free-energy profiles (see vmmc-simulations) or TLM/libtlm polycube simulations (see pypatchy's tlm-simulations skill) - FFS here estimates a rate, not a free-energy landscape. Several pieces of this code have known bugs/dead paths (see Gotchas) - verify against current source before trusting an old explanation, including this one.
---

# FFS simulations via oxpy-utils

Forward Flux Sampling estimates a rare-event rate constant (classically:
duplex melting/dissociation) by partitioning phase space with a sequence of
**interfaces** — thresholds on an order parameter — and firing trial
trajectories forward across them, tracking the conditional probability of
reaching each next interface before falling back to the stable basin.

The entry point is **`FFSProgram`** (`oxpy_utils/ffs/ffs_program.py`). It
owns one `FFSFluxGenerator` (computes the initial crossing rate) plus a
chain of `FFSShooter`s (one per remaining interface), both built on the
shared abstract base `BaseFluxSampler`.

## Core vocabulary

- **`OrderParameter`** (`oxpy_utils/utils/order_parameter.py`) — same class
  used by VMMC: `"bond"` (count of specific WC pairs) or `"mindistance"`
  (binned min inter-group distance). These are the only two order parameter
  types oxDNA itself supports.
- **`FFSInterface(op, val, compare)`** — "trajectory crosses this interface
  when `op.compute_value(...) compare val` becomes true." `~iface` flips the
  comparison operator (for building fail conditions); `.flip()` reverses
  phase-space direction — a distinct operation from invert, don't confuse
  the two.
- **`Condition`** — an OR/AND combination of interfaces, written out as an
  oxDNA FFS stopping-condition file.
- **Flux** — the initial-crossing rate over the first interface region
  (λ₋₁ → λ₀), computed by `FFSFluxGenerator`.
- **Shooting** — from configs that reached one interface, fire new trials to
  see if they reach the next interface (success) or fall back past λ₋₁
  (failure) — **always** λ₋₁, not the immediately preceding interface (see
  Gotchas).

## Minimal end-to-end example

```python
from oxpy_utils.ffs.ffs_program import FFSProgram
from oxpy_utils.ffs.ffs_interface import FFSInterface, Comparison
from oxpy_utils.utils.order_parameter import OrderParameter

bonds = list(zip([0,1,2,3,4,5,6,7], reversed([8,9,10,11,12,13,14,15])))
native  = OrderParameter("native", "bond", bonds)
mindist = OrderParameter("distance", "mindistance", bonds)

interfaces = (
    FFSInterface(native, 7, Comparison.LEQ),    # lambda_-1
    FFSInterface(native, 6, Comparison.LEQ),    # lambda_0
    FFSInterface(native, 4, Comparison.LEQ),    # lambda_1
    FFSInterface(native, 2, Comparison.LEQ),    # lambda_2
    FFSInterface(mindist, 5.0, Comparison.GT),  # lambda_success (full melt)
)

program = FFSProgram(
    "run_name", tmp_path, num_cpus, desired_n_successes=12,
    source_directory=examples_dir,   # dir with starting .top/.dat
)
program.input_file_params["T"] = "70C"
# The shipped "ffs"/"ffs_cuda" default inputs have NO backbone-force cap (unlike
# cpu_MD_relax.json). Without one, the reset-to-lambda sub-simulation can blow up
# ("distance between bonded neighbors exceeded acceptable values") within the first
# ~25 steps, especially at higher T. Set these explicitly — they aren't optional in
# practice even though nothing requires them syntactically:
program.input_file_params["max_backbone_force"] = 5
program.input_file_params["max_backbone_force_far"] = 10
program.ffs_default_input_name = "ffs"      # or "ffs_cuda"
program.set_interfaces(*interfaces)          # >= 3 interfaces required
program.run()
program.save_graph()
program.plot_graph()
```

`set_interfaces` requires `len(interfaces) >= 3` and builds **one shooter
per remaining interface after the first two** (`interfaces[2:]`) — N
interfaces total gives you N-2 shooters, `shoot1..shoot(N-2)`, each chained
so stage *i*'s `destination_directory` becomes stage *i+1*'s source. The
final interface passed doubles as both the fluxer's overall-success
condition and the last shooter's target — don't think of "fully committed"
as a separate thing from the final shooting stage's target.

Analysis-only re-run (no simulation re-executed):
```python
program = FFSProgram(...)                    # same construction
program.set_interfaces(*interfaces)          # same interfaces, needed to rebuild shooter names
program.load()                               # reload success logs from disk
program.load_graph()                         # reload process_graph.json
program.export_results()                     # -> results.csv
program.plot_graph()                         # -> process_graph.svg
```

`export_results()` computes the FFS rate estimator directly:
`k ≈ Φ · Π_i P(λ_{i+1} | λ_i)` — flux's `success_ratio` times the product of
every shooter's `success_ratio`. This is a point estimate only — there's no
committor calculation and no built-in uncertainty/error-bar estimate.

## YAML / CLI path

`ffs_cli.py --input run.yml` runs the full pipeline;
`ffs_analyze_cli.py --input run.yml` re-does only the analysis against an
existing output dir, reusing the same YAML schema. Both are thin wrappers —
for anything beyond straightforward batch/HPC submission, drive `FFSProgram`
directly from Python (see the class docs above); the CLIs don't add
capability, just YAML config parsing.

```yaml
order_parameters:
  native:
    order_parameter: "bond"
    nucleotide_indexes_0: [0,1,2,3,4,5,6,7]
    nucleotide_indexes_1: [15,14,13,12,11,10,9,8]
  distance:
    order_parameter: "mindistance"
    nucleotide_indexes_0: [0,1,2,3,4,5,6,7]
    nucleotide_indexes_1: [15,14,13,12,11,10,9,8]
interfaces:
  - {op: "native", value: 7, compare: "<="}
  - {op: "native", value: 6, compare: "<="}
  - {op: "native", value: 4, compare: "<="}
  - {op: "native", value: 2, compare: "<="}
  - {op: "distance", value: 5.0, compare: ">"}
```

The example under `examples/ffs_example/` demonstrates this shape end to
end — but see Gotchas: **its shipped YAML's `file_dir` path is stale**. Use
`examples/8nt_duplex_files/` as the real starting-conf directory (that's
what `tests/test_forward_flux_sample_OO.py` actually points at).

## Gotchas

- **`examples/ffs_example/ffs_cli_input.yml`'s `file_dir` is stale** — it
  points at `../8_nt_duplex_melting_cpu/oxdna_files`, which doesn't exist.
  The real starting files are at `examples/8nt_duplex_files/`.
- **`ffs_seperation.py` (`SeperationFluxer`) is dead code** — it duplicates
  `FFSFluxGenerator`'s algorithm almost exactly but nothing in `FFSProgram`,
  the CLIs, or the shooter chain references it. Don't build on it; use
  `FFSFluxGenerator` via `FFSProgram.set_interfaces`.
- **Resuming a completed flux stage doesn't actually skip work.**
  `FFSFluxGenerator.run()` logs "Found enough existing successes...
  Exiting..." but has no `return` — it falls through and still spawns
  worker processes (they exit near-instantly since the success target is
  already met, so cost is process-spawn overhead, not wasted simulation —
  but the log message is misleading). `FFSShooter.run()` does correctly
  return early; only the flux stage has this bug.
- **The shooting fail condition is always `~lambda_neg1`** (the *original*
  first interface), not the previous stage's interface — this is standard,
  correct FFS behavior, but easy to misread from the code.
- **`FFSProgram.keep_sim_dirs` defaults to `True`**, propagated to all
  shooters — for a large `desired_n_successes` this can leave enormous
  numbers of per-trial scratch directories on disk. `FFSFluxGenerator` has
  no cleanup path for its own per-trial dirs regardless of this flag —
  watch disk usage on long flux runs.
- **`FFSProgram.graph_update_queue` is a class attribute, not per-instance.**
  Multiple `FFSProgram` objects created in the same process (e.g. in a
  notebook, or a parameter-sweep loop) share the same underlying
  multiprocessing queue — a real cross-talk risk for the process graph.
  Prefer one `FFSProgram` per process/run when scripting a sweep.
- **Checkpointing is not fully idempotent.** `init()` recomputes existing
  success count by globbing `success_*.dat` files; `FFSShooter.init()`
  additionally requires its `<name>_success_log.csv` row count to match, or
  you get an assert mismatch. The `auto_save` background thread only saves
  every 300s, so a hard crash can lose up to 5 minutes of shooting
  statistics (though the success configs themselves are written immediately
  on success).
- **Look at per-stage logs, not the top-level one.** The top-level
  `<name>.log` file tends to end up empty; real diagnostics live in
  `initial_flux.log`, `shoot1.log`, etc.
- **`OrderParameter` only supports `"bond"`/`"mindistance"`** — a hardcoded
  oxDNA limitation, not something oxpy-utils can extend without upstream
  oxDNA support.

## Where to look next

- `examples/ffs_example/` — real end-to-end script/YAML (mind the stale
  path above).
- `tests/test_ffs_interface.py`, `test_forward_flux_sample_OO.py` — smallest
  correct usage patterns, including the Python-direct and YAML/CLI forms
  side by side.
- `references/graph-and-analysis.md` — how the live process graph is built,
  what `plot_graph()` actually renders, and the exact `results.csv` schema.
