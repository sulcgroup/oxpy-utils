# Windowing & reweighting mechanics (`oxpy_utils/vmmc_umbrella/`)

Deep-dive companion to `../SKILL.md`. Read that first for the workflow shape;
this file is for when something in `setup()`/`wham()`/reweighting goes wrong
and you need to know exactly what the code does.

## Class hierarchy

- `VMMCMetaSimulation` (`metasimulation.py`, ABC) — shared base for "groups of
  related VMMC sims beyond plain replicas." Owns order-parameter bookkeeping
  (`add_order_parameter(s)`, at most one `mindistance` op, bond ops always
  first), `n_reps`, and the legal/desired/possible state machinery:
  - *possible* = physically allowed (`possible_states()` filters out e.g.
    bond counts exceeding half the participating nucleotides)
  - *legal* = possible AND reachable in this setup
    (`filter_legal_states` callback over possible states; windowing further
    intersects with a window's own `state_space_area`)
  - *desired* = legal states you actually want flat-sampled
  Abstract methods subclasses implement: `check_ready()`, `run(join=True)`,
  `visualize(*args)`.
- `VMMCAutoReweight(VMMCMetaSimulation)` (`auto_reweight.py`) — iterative
  weight refinement across successive VMMC runs (not windowed).
  `VMMCGraphReweight` is a more sophisticated variant using per-state visit
  counts (TMMC-style) rather than raw transition-count ratios — see "Why
  raw transition counts don't work" below.
- `VmmcWindowing(VMMCMetaSimulation)` (`windowing.py`) — orchestrates
  multiple `VmmcWindow`s (each a `VmmcReplicas` subclass) sharing one
  top-level directory.
- `VmmcWindowedAutoReweight` (`windowed_auto_reweight.py`) — chains
  `VmmcWindowing` with `VMMCAutoReweight`/`VMMCGraphReweight` **per window**,
  so each window iterates to convergence instead of using one static weight
  mask.

## `VmmcWindowing.setup()` — what actually happens

1. `check_ready()` — every state in `possible_states(*order_parameters())`
   must be covered by at least one window's `state_space_area`, else
   `ValueError`.
2. For each window: build its `n_reps` replicas via `build_replica`.
3. **Probe the starting configuration.** `_probe_starting_state()` spawns a
   real, blocking 1-step oxpy simulation (`oxpy.Context()` +
   `oxpy.OxpyManager`) to read off the starting config's initial
   order-parameter state, and raises `ValueError` if that state isn't inside
   the window's `state_space_area`. This is slow but exists because handing
   e.g. a fully-bonded starting structure to a "melted" window would deadlock
   VMMC via a division-by-zero in the acceptance ratio.
4. **Construct the per-window weight mask.** States outside the window get
   weight `0` (hard exclusion — a VMMC move into a weight-0 state is
   unconditionally rejected). In-window "legal" states get
   `sim.weights / min(sim.weights[legal])`, i.e. renormalized so the minimum
   in-window weight is 1.
5. Write `setup.json` via `cache_settings()` so the windowing object can be
   reloaded later with `VmmcWindowing(tld).load()`.

**The starting_conf path is not persisted in `setup.json`** — after
`load()`, every window's `starting_conf` is `None`. A windowing object
reloaded this way can be analyzed (`wham()`, plotting) but not rebuilt or
rerun without re-supplying the original starting-conf directories.

## WHAM (`VmmcWindowing.wham()`)

Ported from an older WHAM implementation ("Petr's old code," per an inline
comment, plus the code's own self-deprecating `# todo: make not explode`).
Iteratively solves for `rho[state]` (a joint probability over the full state
space, or the marginal over one order parameter if `op=<index>` is passed),
using each window's combined-replica histogram and its weight mask, until
`max|Δexp(-logf)| < 1e-4` or 10000 iterations. Treat it as a working-but-not-
hardened numerical routine — if it fails to converge, suspect either
insufficient sampling in one window or an under-normalized weight mask,
before suspecting the WHAM code itself.

`get_merged_weights()` / `save_merged_weights()` is a simpler, non-WHAM
alternative: it averages the (already per-window normalized) weight arrays
across windows, but first aligns the overlapping states' scale via a
**geometric mean of the ratio at the overlap** — the code explicitly warns
that naive averaging without this step "misaligns" the windows' scales. This
produces one global weight matrix (not a probability distribution) — useful
for e.g. constructing a single follow-up unwindowed run that covers the full
range with one weight file.

## `VmmcWindowedAutoReweight` — iterating each window to convergence

Instead of `setup()`'s one static weight mask per window, this class builds
one `VMMCAutoReweight` (or `VMMCGraphReweight`) *per window* and iterates it:

```python
windowing = VmmcWindowing(tld)
windowing.add_order_parameters([...])
windowing.add_window(state_space_area, starting_conf)
windowing.extrapolate_hist_Ts = [...]

chained = VmmcWindowedAutoReweight(windowing)  # writes to tld/"reweighted"
chained.build_replica = my_build_replica
chained.steps_per_iter = 5e7
chained.max_iterations = 10
chained.run()

windowing.wham()  # now reflects the reweighted, converged per-window data
```

Internally, `build_window_reweighter` scopes `filter_legal_states` to
intersect the window's own states, sets `filter_desired_states` to "every
legal state in this window" (flat sampling within the window is the whole
point), and — critically — sets `illegal_state_weight = 0.0` (vs. the base
`VMMCAutoReweight` default of `1.0`) because here "illegal" means "belongs
to a different window," a perfectly reachable, sometimes thermodynamically
favored state that must be actively excluded, not just soft-discouraged.
After each window converges, `_splice_into_windowing` replaces that window's
data in the parent windowing object's `_subgroups` in place, so
`windowing.wham()` / `plot_free_energy_profile()` transparently reflect the
reweighted result afterward. `VmmcWindowing.load(splice_reweighted=True)`
reconstructs this spliced state from disk in a later session.

**Multi-order-parameter windowing + reweighting isn't fully turnkey**: the
default `build_start_weights` fallback logic explicitly says to "provide
your own for multi-op windowing." If your window uses both a bond op and a
mindistance op, expect to write a custom `build_start_weights`.

**Reweighting a completed window that lost its starting_conf raises a clear
error** — `build_window_reweighter` checks for this and fails loudly rather
than silently reusing a stale starting-conf-less window.

## Why raw transition counts don't work (`VMMCGraphReweight`)

An extensive inline comment in `auto_reweight.py` explains: because VMMC
satisfies detailed balance, comparing raw transition counts `c[i→j]` vs.
`c[j→i]` directly is *wrong* — over a long run those converge to equal
**regardless** of whether sampling is flat. You must normalize by per-state
visit counts (TMMC-style per-visit transition rates) to recover the true
equilibrium population ratio. A naive first implementation of this
reweighter would silently collapse to trivial (flat) weights no matter what
the real free-energy landscape looks like. Treat `VMMCGraphReweight`'s
damping/trust-region constants (`max_log_weight_step`,
`rare_event_damping`, `undersampled_boost_factor`,
`undersampled_occ_threshold`) as provisional, actively-tuned values, not
settled defaults.

## Other quirks worth knowing

- `filter_legal_states` **must return a `list`**, not a `set` — a `set`
  silently breaks later `list`-only indexing inside `setup()`.
- `read_op_hist_file`'s parsing infers 1-op vs. multi-op histograms purely
  from column count matching an expected formula; a malformed or
  unexpected-shape `last_hist.dat` raises a generic `Exception`, not
  something diagnostic — if you hit this, check the histogram file's column
  count against how many order parameters you actually configured.
- `possible_states()`'s "impossible state" filter only checks a bond op
  against its own nucleotide count — it does not model cross-op physical
  constraints between a bond op and a mindistance op (an explicit `# TODO`
  in the source acknowledges this).
- `compute_heat_map` has a known architectural TODO to make
  `sim.analysis.statistics` a proper `VMMCData` object — the analysis layer
  and the standalone `VMMCData` dataclass (`vmmc_data.py`) aren't unified
  yet.
