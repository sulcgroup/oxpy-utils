# FFS process graph & results schema

Deep-dive companion to `../SKILL.md`.

## How the process graph is built

`FFSProgram` maintains a `networkx.DiGraph` recording every simulation event
across all stages: flux sub-steps, shoot nodes, success/failure reports. It's
populated **live** via a `multiprocessing.Queue`
(`FFSProgram.graph_update_queue`) plus a listener thread
(`_graph_updater`) that consumes queue messages from the worker processes
each stage spawns.

Remember this queue is a **class attribute**, shared across every
`FFSProgram` instance in the same Python process — see the Gotchas section
in the main skill file.

Persistence: the listener appends every message to
`process_graph_events.jsonl` in the root dir (one line `{"t": unix time,
"cmd": [message]}`, line-buffered, so it survives a crash) before applying
it. That event log is the complete record:

- `run()` replays an existing log first (`replay_graph_events()`), so a
  relaunch in the same root dir (resume after a crash, or more interfaces
  added) extends the graph instead of starting over, and logs a `LAUNCH`
  event (`process_graph.graph["launches"]`).
- Sim numbering: all steps share one counter per worker process
  (`FFSProgram.sim_counters`, a multiprocessing Array; steps read it with
  `first_sim_index()` and advance it with `record_sim_index()`), so worker
  `i`'s sims are numbered consecutively through the flux stage and every
  shooter, and each `p<i>/sim<k>` dir and node id `(step, process, sim)`
  belongs to one step. `run()` seeds it with `next_sim_indices()`: one past
  every index in the replayed graph or found as a `p<i>/sim<k>` dir in any
  step's working directory.
- Every node has a `path` (the sim dir), reported by the worker.
- `load()` replays the log (falls back to `process_graph.json`).
- `save_graph()` / `load_graph()` still write/read the `process_graph.json`
  snapshot (now with a `"graph"` key holding `launches`); it only contains
  what was in memory, so prefer the log. `ffs_cli.py` registers `save_graph`
  via `safe_exit.register(...)`.

Before 2026-11 (branch ffs-process-graph) there was no event log: each
launch's `save_graph()` replaced the previous graph, `load()` did not load
anything, and every step numbered its sims from 0 in the same
`root_dir/p<i>/sim<k>` dirs. Each shooter's sim was built with
`clean_build='force'` over the flux (or previous shooter's) sim of the same
index, wiping it (and deleting it with `keep_sim_dirs=False`); scripts worked
around it by giving shooters their own `set_working_directory()`.

## Lineage attributes

- A node that wrote a success conf (`CPY_CONF`) has `success_index` (n of
  `success_<n>.dat`) and `success_conf` (its path).
- A shoot node has `source_conf`, `source_step` (the stage that wrote it:
  `initial_flux` for shoot1, else the previous shooter) and
  `source_success_index`, plus `seed`, and an edge from the node that wrote
  its starting conf — also when that node is from an earlier launch.
- Flux nodes started from a pool conf (`equilibrate` in
  `FFSFluxGenerator`, `flux_fwd` in `RestartOnFailFluxGenerator`) carry
  `source_conf` (and `seed` for restart-on-fail).

So a success's full history is the path from `origin` to its node, e.g.
`nx.shortest_path(G, "origin", node)`. The shooter logs also name the file
each success wrote (`SUCCESS: ...; wrote success_<n>.dat`).

## `plot_graph()`

A large, bespoke NetworkX/matplotlib visualization (~360 lines) — colors
flux sub-steps distinctly from shooter columns, lays out each shooter stage
as its own column, labels success fractions, and draws dividers at each
interface's order-parameter value. It includes an internal "DIAGNOSTIC"
assertion block that raises if it finds orphaned flux nodes in the graph —
if you hit that assertion, it usually means the graph was corrupted by the
class-attribute queue cross-talk gotcha (two `FFSProgram`s writing to the
same queue), not a bug in the plotting code itself.

Treat this whole subsystem (graph construction + rendering) as the least
stable part of the FFS code — the core flux/shoot simulation loop is
comparatively settled; the live graph tracking has known race-condition
acknowledgments in comments around worker process exit / queue flushing.

## `results.csv` schema (`export_results()`)

Columns: `flux, shoot1, ..., shootK, total`. Rows: `num_attempts`,
`num_successes`, `success_ratio`, `rate_per_oxdna_time_unit`, `rate_per_<unit>`.

- **`flux` column**: `num_successes` = count of flux-stage success configs;
  `num_attempts` = total simulation steps summed over those success
  configs (`num_successes / fluxer.compute_flux()`) — a step count, not a
  literal attempt count. `success_ratio` = `fluxer.compute_flux()` itself,
  in successes per step (not successes/attempts like the shooter columns).
- **Each `shootK` column**: `num_attempts = sum(shooter.attempt_from)` across
  all starting configs used by that shooter; `num_successes =
  shooter.success_count.value`.
- **`success_ratio`** per shooter column = successes / attempts; for the
  `flux` column it is the flux itself (successes per step).
- **`total`** = `numpy.prod` of every column's `success_ratio` — this is
  exactly the FFS rate-constant estimator
  `k ≈ Φ_A · Π_i P(λ_{i+1} | λ_i)`, in successes per simulation step
  (multiply by 1/dt for per-oxDNA-time-unit).
- **`rate_per_oxdna_time_unit` / `rate_per_<unit>`** rows: only the `flux` and
  `total` columns are filled (shooter columns are dimensionless). The first is
  `success_ratio / dt`. The second's name and unit adapt: `<unit>` is the
  smallest SI time unit (ps, ns, us, ms, s, ks, ... from `SI_TIME_UNITS`) in
  which both rates are > 1 (falls back to `s` if neither rate is positive and
  finite, and to the largest unit if none qualifies). `dt` and
  `interaction_type` come from `set_input_params` if set, otherwise the default
  ffs input file; interaction types containing "RNA" use RNA units, else DNA.

There is no uncertainty/error-bar column and no committor calculation in
this output — if you need error bars, you'll need to run repeated
independent `FFSProgram`s and compute variance across them yourself; nothing
in this module does it for you.

## Per-starting-conf bookkeeping (`FFSShooter`)

Each shooter tracks, per starting configuration (indexed by the
`success_N.dat` file it came from), how many times it was picked as a
shooting origin (`attempt_from[i]`) and how many of those attempts succeeded
(`success_from[i]`). This is persisted to `<name>_success_log.csv` and
reloaded via `load_success_info` on `init()`. If this CSV goes missing or
falls out of sync with the actual `success_*.dat` files on disk (e.g. from a
partial/interrupted run), `init()`'s reconciliation will assert-fail rather
than silently guess — treat that assertion as a sign the run directory is in
an inconsistent state, not a bug to route around.
