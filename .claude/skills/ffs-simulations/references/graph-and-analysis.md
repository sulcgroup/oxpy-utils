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

Persistence: `save_graph()` / `load_graph()` serialize/deserialize the graph
to/from `process_graph.json` at the run's top-level output directory. The
`ffs_cli.py` script registers `save_graph` via `safe_exit.register(...)` so
the graph is flushed even on interruption (Ctrl-C, unhandled exception).

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
`num_successes`, `success_ratio`.

- **`flux` column**: `num_successes` = count of flux-stage success configs;
  `num_attempts` = `fluxer.total_success_time()` — i.e. this column's
  "attempts" is actually total simulated time used as the flux-rate
  denominator, not a literal attempt count. Don't read it as "N trials were
  attempted."
- **Each `shootK` column**: `num_attempts = sum(shooter.attempt_from)` across
  all starting configs used by that shooter; `num_successes =
  shooter.success_count.value`.
- **`success_ratio`** per column = successes / attempts.
- **`total`** = `numpy.prod` of every column's `success_ratio` — this is
  exactly the FFS rate-constant estimator
  `k ≈ Φ_A · Π_i P(λ_{i+1} | λ_i)`.

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
