---
name: oxdna-simulation-base
description: Use when working with the foundational oxpy_utils.oxdna_simulation.Simulation class - the base every oxpy-utils simulation (plain MD/MC, VMMC, FFS) is built on. Covers construction (from a directory, a DNAStructure, or another Simulation), build()/clean_build semantics, the input property and swap_default_input/f(args)=expr dynamic defaults, attaching forces (SimForces/Force) and observables (Observable), oxpy_run's subprocess/join modes, the analysis/oat alias, and pickle_sim. Several real bugs live here (clear_forces KeyError, com_force/rotating_harmonic_trap calling the wrong helper, dead input_file_params arg - see Gotchas). Not for VMMC-specific analysis/windowing (see vmmc-simulations), FFS (see ffs-simulations), building/editing structures (see dna-structure), or chaining multiple Simulations into a relax->production pipeline (see sim-chaining) - though this skill explains the mechanism sim-chaining relies on.
---

# The base `Simulation` class (`oxpy_utils/oxdna_simulation.py`)

`Simulation` is what `VirtualMoveMonteCarlo` (VMMC) and the FFS simulation
classes both subclass. If you're doing plain MD/MC, or you're trying to
understand what VMMC/FFS are quietly relying on underneath, this is the
layer. The extension pattern subclasses use: they call
`super().__init__(...)` then directly assign `self._analysis = ...` /
`self._oxpy_run = ...` in their own `__init__`, bypassing the base
properties' lazy-init guard (`if self._X is None: ...`) — that's how
`VirtualMoveMonteCarlo` swaps in `VmmcAnalysis`/`VmmcOxpyRun` without
overriding the properties themselves.

## Minimal lifecycle

```python
from oxpy_utils.oxdna_simulation import Simulation

sim = Simulation(file_dir, sim_dir)   # file_dir has a .top + .dat (or last_conf*.dat)
sim.build()                            # copies/renames conf+top, writes input + input.json
sim.input.swap_default_input("cpu_MC_relax")   # or cpu_MD, cpu_MD_relax, cuda_MD, cuda_MD_relax, vmmc, ffs, ffs_cuda

sim.add_force({...})        # optional, before running
sim.add_observable({...})   # optional, before running

sim.oxpy_run.run(subprocess=False)   # blocking, in-process
# sim.sim_files.traj / .last_conf / .energy now reflect the run's output
```

`file_dir` accepts three shapes, each triggering different construction
logic — see `dna-structure` for the first, `sim-chaining` for the third:
- a path with an existing `.top`/`.dat` (or `last_conf*.dat`) — plain
  `BuildSimulation`.
- a `DNAStructure` object — installs `BuildSimulationFromStructure`, which
  calls `structure.export_top_conf(...)` into `sim_dir` on `build()`.
- another `Simulation` object — sets `file_dir = other.sim_dir`, so
  `build()` picks up *that* simulation's `last_conf*` file as this one's
  starting configuration. This is exactly how multi-stage pipelines chain
  (`Simulation(prev_sim, next_dir)`).

**A brand-new `Simulation`'s `input` defaults to `cuda_MD`** (`Input.__init__`
hardcodes `get_default_input("cuda_MD")`) regardless of what you eventually
call `swap_default_input` with — if you only ever call `input_file({...})`
and never `swap_default_input`, you'll silently get a CUDA backend even on
a CPU-only box.

## `build()` and `clean_build`

| `clean_build` | `sim_dir` doesn't exist yet | `sim_dir` exists, `file_dir == sim_dir` (build in place) | `sim_dir` exists, `file_dir != sim_dir` |
|---|---|---|---|
| (any) | Just builds: dir, dat/top, input. | — | — |
| `False` (default) | — | Prints a message, **does nothing** — no exception, the sim is silently not rebuilt. | Same — no-op. |
| `True` | — | Blocking `input()` prompt ("Type y/yes...") before wiping `input_dict` + `rmtree`. **Hangs non-interactive runs.** | Same prompt. |
| `"force"` | — | Deletes everything in `sim_dir` **except** the current `.dat`/`.top` — preserves the conformation, wipes input/forces/observables/trajectory. | `shutil.rmtree`s the **entire** `sim_dir` — full wipe, not just non-conf files. |

The `"force"` row is the one to internalize: it is *not* equally destructive
in both cases. Build-in-place `"force"` keeps your conformation; fresh-build
`"force"` doesn't need to (there's a separate source directory), so it burns
the whole thing down.

`build()` requires `file_dir` to already contain a real `.top` and a conf
file before it's called — `find_starting_top_dat()` raises `FileNotFoundError`
otherwise. It prefers a conf file literally named `last_conf*` over any other
`.dat` (important for chaining — see `sim-chaining`).

## `input` — `swap_default_input` and dynamic values

`sim.input` is dict-like (`__getitem__`/`__setitem__`/`__contains__`) backed
by `input_dict`. **Every single key write immediately rewrites the whole
input file** — no batching, so a hot loop of individual sets does real I/O
each time.

`swap_default_input(name)` loads a named JSON from
`oxpy_utils/defaults/inputs/` (`cpu_MD`, `cpu_MD_relax`, `cpu_MC_relax`,
`cuda_MD`, `cuda_MD_relax`, `vmmc`, `ffs`, `ffs_cuda`) and **unconditionally
replaces `input_dict`** — call it *before* setting your own custom
parameters, not after, or your custom values get silently discarded.

Some default JSONs use a `f(args) = expr` string value (e.g.
`cpu_MC_relax.json`'s `"print_conf_interval": "f(steps) = steps / 10"`).
`write_input()` always calls `default_input.evaluate(**input_dict)` before
merging (`{**default, **input_dict}` — your manual overrides win), so a
dynamic expression can react to a value you set yourself. The substitution
is **naive string replacement, not real templating** — an argument name that's
a substring of another argument name could corrupt the expression; nothing
guards against this.

## Forces and observables

```python
sim.add_force({"type": "mutual_trap", "particle": 0, ...})   # dict or Force object
sim.add_forces(load_from="forces.json")                       # reload an existing force file
sim.clear_forces()                                             # see Gotchas: currently broken

sim.add_observable(distance(...))    # from oxpy_utils.utils.observable convenience functions
```

`Force` (`utils/force.py`) validates required/optional params per
`ForceType` (`MORSE`, `SKEW_TRAP`, `CENTER_OF_MASS`, `MUTUAL_TRAP`, `STRING`,
`HARMONIC_TRAP`, `ROT_HARMONIC_TRAP`, `REPULSION_PLANE`, `REPULSION_SPHERE`)
— missing a required param or passing an unrecognized one both raise
`ValueError`. `SimForces`'s static convenience constructors
(`.morse`, `.mutual_trap`, etc.) return **raw dicts, not `Force` objects** —
mind the inconsistency if you're checking `isinstance`.

`Observable` (`utils/observable.py`) convenience factories
(`simulation_time`, `distance`, `hb_list`, `particle_position`,
`potential_energy`, `kinetic_energy`, `pair_energy`, `force_energy`) each
return the `{"output": {...}}` dict shape `add_observable` expects directly
— you don't need to construct `Observable` yourself for the common cases.
**`potential_energy()`'s default `split` param is `None`, which stringifies
to `"None"` and oxDNA's input parser rejects as an invalid boolean**
(`OxDNAError: boolean key 'split' is invalid ('none')`) — pass
`split=True` (or `False`) explicitly; don't rely on the default.

**Observables written to disk are never automatically visible to
`sim.analysis` — this is unconditional, not just a cross-process/reload
concern.** `add_observable`/`build_observable` only ever write
`observables.json`; nothing keeps `Analysis.observables` (the in-memory dict
`observable_data()` reads) in sync, even within the same process right
after a successful run. Always call
`sim.analysis.load_observables_from_json()` before
`sim.analysis.observable_data(...)`, every time — skipping it raises
`AssertionError: No observable named ...` even immediately after `add_observable`
+ a completed run in the same script.

## `oxpy_run` — execution modes

`sim.oxpy_run.run(subprocess=True|False, join=True|False, continue_run=False, log=True)`
(also callable directly: `sim.oxpy_run(...)`)

| `subprocess` | `join` | Behavior |
|---|---|---|
| `False` | (ignored) | Runs `run_complete()` directly in the **parent process** — fully blocking, no isolation. A crash/OOM inside oxpy can kill the caller. |
| `True` | `True` | Spawns a `multiprocessing.Process`, then blocks (`.join()`) until it finishes, then refreshes `sim.sim_files`. Looks synchronous but isolates crashes/CUDA OOM from the parent — this is usually what you want for scripting. |
| `True` | `False` | Spawns the process and returns immediately. You must manually `sim.oxpy_run.process.join()` later and re-`parse_current_files()` yourself before reading outputs. |

`continue_run=<n>` resumes from `sim_files.last_conf` for `n` more steps
instead of overwriting (patches `conf_file`/`refresh_vel`/
`restart_step_counter`/`steps` on the input before running).

Only `oxpy.OxDNAError` is re-raised after being recorded in
`self.error_message`; other exception types are caught, logged, and
swallowed — a run that "fails silently" (no output but no exception either)
is a real possibility worth checking `error_message` for.

## `analysis` / `oat`

**`sim.oat` is a plain alias for `sim.analysis`** — same object, not a
separate wrapper. `Analysis` mixes three different things under one class:
in-process calls into the `oxDNA_analysis_tools` (`oat`) Python library
(`.mean()`, `.deviations()`, `.centroid()`, `.pca()`, `.rg()`/
`.radius_of_gyration()`), calls that shell out to the **`oat` CLI**
(`.distance()`, `.generate_force()`, `.oxDNA_PDB()`, `.angle()` — these need
`oat` on `PATH` separately from the library import), and oxpy-utils' own
non-oat conveniences (`get_conf_as_structure`, `plot_energy`,
`hist_observable`, Jupyter-only `view_*` widgets).

`Analysis.load_energy()` explicitly refuses `sim_type == "VMMC"` — this
confirms VMMC's overridden analysis class is required, not optional cosmetic
sugar.

## `pickle_sim` / `from_pickle`

Plain whole-object `pickle.dump`/`pickle.load` to `sim_dir/sim.pkl` — no
custom `__getstate__`/`__setstate__`, no test coverage found. Known risk: if
you ran with `subprocess=True`, `sim.oxpy_run.process` holds a live/finished
`multiprocessing.Process`, which is not reliably picklable. Also:
unpickling on a different machine/filesystem layout produces a `Simulation`
with `file_dir`/`sim_dir` `Path`s that may not exist there — no validation
happens on load.

## Gotchas

- **`clear_forces()` has a real key-casing bug, and it partially corrupts
  state before crashing rather than failing cleanly.** It does
  `del self.input["external_forces_as_json"]` (lowercase `json`) while every
  setter uses `"external_forces_as_JSON"` (uppercase), so it raises
  `KeyError` on that third delete. But `Input.__delitem__` writes the input
  file on every individual delete, so the first two deletes
  (`external_forces`, `external_forces_file`) already succeeded and were
  already flushed to disk *before* the crash — a caller who catches/ignores
  the `KeyError` ends up with `external_forces` silently reverted to `0`
  (oxDNA's `cpu_MD` default) while `forces.json` and
  `external_forces_as_JSON` are left dangling. Confirmed by direct testing;
  don't rely on `clear_forces()` at all until this is fixed upstream.
- **`SimForces.com_force` and `.rotating_harmonic_trap` call the wrong
  helper, and calling them with their own name's natural arguments outright
  fails rather than just building the wrong force type.** `com_force`
  actually calls `skew_force(**kwargs)` — pass it `com_force`'s real
  arguments (`com_list`, `ref_list`, ...) and you get
  `TypeError: skew_force() got an unexpected keyword argument 'com_list'`,
  confirmed directly. It only "succeeds" if you happen to pass
  `skew_force`'s arguments instead, in which case it silently returns a
  `skew_trap` dict, not a `com` force. `rotating_harmonic_trap` calls
  `harmonic_trap(**kwargs)` — this one *does* silently succeed if called
  with `harmonic_trap`-shaped arguments (`particle`, `pos0`, `stiff`),
  returning `{"type": "trap"}` instead of the intended `{"type": "twist"}`,
  confirmed directly. Likely copy-paste bugs either way; don't trust either
  method name over what it actually builds.
- **`Simulation.__init__`'s `input_file_params` argument is dead code** —
  accepted in the signature, never read or used anywhere.
- **`clean_build='force'`'s destructiveness depends entirely on
  `file_dir == sim_dir`** — see the table above. This is the single most
  likely "why did my forces.json disappear" surprise.
- **`clean_build=True` (not `"force"`) blocks on a real `input()` prompt** —
  will hang a non-interactive script/CI run.
- A `Simulation`'s `input` starts as `cuda_MD` regardless of your intent —
  always call `swap_default_input` explicitly rather than relying on the
  constructor default.

## Where to look next

- `dna-structure` — the `DNAStructure` object you can pass as `file_dir`,
  and how a running/finished `Simulation` hands a conformation back as one
  (`get_conf`, `last_conf_structure`).
- `sim-chaining` — the `Simulation(prev_sim, next_dir)` pattern for a full
  import → relax → relax → production pipeline, plus real bugs found in the
  one shipped example that uses it.
- `vmmc-simulations` / `ffs-simulations` — the two concrete subclasses built
  on everything above.
- `references/forces-and-observables.md` — full `Force`/`Observable` field
  reference if the summary above isn't enough.
