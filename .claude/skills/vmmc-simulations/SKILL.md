---
name: vmmc-simulations
description: Use when setting up, running, or analyzing a VMMC (Virtual Move Monte Carlo) simulation via oxpy-utils - including umbrella sampling / windowing (VmmcWindowing), weight reweighting (VMMCAutoReweight / VMMCGraphReweight / VmmcWindowedAutoReweight), multi-replica statistics (VmmcReplicas), and Tm/melting-curve analysis. Covers oxpy_utils.vmmc_umbrella.*: VirtualMoveMonteCarlo, order parameters (bond/mindistance), weight files, WHAM-based window merging. oxpy-utils has no separate "umbrella sampling" module - windowed umbrella sampling here IS VMMC with zero-weighted excluded regions, so this one skill covers both plain VMMC and umbrella sampling. Not for TLM/libtlm polycube VMMC (a completely different C++ lattice engine for polycube assembly, not oxDNA - see pypatchy's tlm-simulations skill for that) and not for FFS rare-event sampling (see ffs-simulations). This code is under active iteration - WHAM has a literal "todo: make not explode" comment and reweighting constants read like live tuning notes - verify surprising behavior against current source before trusting an old explanation, including this one.
---

# VMMC simulations via oxpy-utils

VMMC (Virtual Move Monte Carlo) here is oxDNA's own cluster-move Monte Carlo
engine, driven through `oxpy` and wrapped by
`oxpy_utils.vmmc_umbrella.vmmc.VirtualMoveMonteCarlo` — a subclass of the base
`Simulation` class in `oxpy_utils/oxdna_simulation.py`. This is a completely
different simulation system from TLM/`libtlm` (the polycube lattice engine
pypatchy's `tlm-simulations` skill covers) — same acronym ("VMMC"), unrelated
code, unrelated physics (continuous DNA nucleotides vs. a discrete polycube
lattice).

**Umbrella sampling is not a separate feature here** — it's VMMC run with a
*weight file* that hard-excludes ("zero-weights") states outside a target
window of an order parameter. `VmmcWindowing` orchestrates multiple such
windows and stitches them back together with WHAM. If you just want plain
VMMC with a bias (e.g. favoring low-bond states to see a full melting
transition), skip windowing entirely — see "Plain VMMC" below.

## Plain VMMC — minimal working example

```python
from oxpy_utils.vmmc_umbrella.vmmc import VirtualMoveMonteCarlo

sim = VirtualMoveMonteCarlo(file_dir, sim_dir)   # file_dir has a .top/.dat (duplex_box_30 etc.)

# Defines the "native" bond order parameter: number of these base pairs formed (0..N).
# Pair antiparallel, 5'->3' vs 3'->5' — for two 8-nt complementary strands sharing one
# duplex (nt 0-7 and 8-15), that's 0<->15, 1<->14, ..., 7<->8, NOT 0<->8, 1<->9, ...
sim.set_nucleotides(list(range(0, 8)), list(range(15, 7, -1)))
# sim.veryify_bond_ops() should return True — if it doesn't, the pairing above is wrong
# for your structure and the bond op will silently track base pairs that never form.

sim.build()                     # base Simulation.build() + swap_default_input("vmmc")
                                 # + writes op_file/weights_file — REQUIRES an order
                                 # parameter to already be set (assert on self.bond_op)

sim.input["steps"] = int(1e5)
sim.input["T"] = "25C"
sim.weights[...] = sim.generate_weights(11.)   # exponential bias toward low-bond states
sim.build_vmmc_weight_file()

sim.oxpy_run(subprocess=True, join=True)
```

Analysis:
```python
sim.analysis.read_vmmc_op_data()                    # populates sim.analysis.vmmc_df from last_hist.dat
sim.analysis.calculate_sampling_and_probabilities()  # sampling_percent / wt_prob / wt_free
sim.analysis.plot_melting_profiles()                 # Tm curve from extrapolate_hist temperatures
```

Order parameter types (`oxpy_utils/utils/order_parameter.py`) are hardcoded to
what oxDNA itself supports: `"bond"` (count of specific WC pairs formed) and
`"mindistance"` (binned minimum distance between two nucleotide groups). A
`VirtualMoveMonteCarlo` can carry one bond op (`set_nucleotides` →
`sim.bond_op`, named `"native"`) plus optionally one mindistance op
(`sim.add_dist_op(interfaces=..., p1=..., p2=...)`). The `weights` array is
N-dimensional, shaped by every order parameter's number of possible values —
bond op(s) first, then the dist op.

Parallel tempering (`sim.parallel_tampering = True`, `sim.PT_Ts = [...]`) is
also implemented on this class but must run via CLI/MPI (`sim.cli_run(...)`),
not `oxpy_run` — treat it as a separate, less-traveled path with no example
coverage.

## Umbrella sampling / windowing

A **window** (`VmmcWindow`) is a subset of order-parameter state space —
concretely a `set` of state tuples, e.g. `{(s,) for s in range(5, 9)}` for a
"folded" window on a single bond op. `VmmcWindowing` orchestrates a set of
windows that deliberately overlap at their boundary states so the pieces can
be stitched back together with WHAM. Full end-to-end reference:
`examples/vmmc_windowing_8nt_duplex/vmmc_windowing_8nt_duplex.py`.

```python
from oxpy_utils.vmmc_umbrella.windowing import VmmcWindowing

w = VmmcWindowing(output_dir)
w.add_order_parameter(op)                 # bond and/or one mindistance op
w.n_reps = N
w.extrapolate_hist_Ts = ["30C", ...]       # required

w.filter_legal_states = lambda states: sorted(states)  # MUST return a list, not a set
w.build_replica = build_replica_fn         # (windowing, sim) -> None : build(), swap_default_input,
                                            #   set input["steps"]/["T"]/etc.
w.build_start_weights = build_start_weights_fn  # (sim, window_idx) -> None for VmmcWindowing
                                                 # -- NOT the 1-arg (sim) signature used elsewhere

w.add_window(state_space_area, starting_conf_dir_or_structure)  # once per window
w.setup()     # validates full coverage, probes each starting conf's OP state, builds
              # the zero/normalized weight mask per window, writes setup.json
w.run(join=True)

w.wham()                          # dict[state -> probability] across all windows
w.plot_free_energy_profile()
```

Reload later without rerunning: `VmmcWindowing(tld).load()` — but the
starting-conf paths are **not** persisted, so a reloaded windowing object can
analyze but cannot rebuild/rerun.

See `references/windowing-mechanics.md` for: exactly what `setup()` does to
the weight arrays, the WHAM algorithm, `get_merged_weights()` as a
non-probabilistic alternative, `VmmcWindowedAutoReweight` (iterating each
window to convergence instead of a static weight mask), and every quirk
called out below in more depth.

## Replicas

`VmmcReplicas` runs N independent copies of one VMMC config in parallel
(mainly for SEM/confidence-interval statistics on Tm and free energy, not for
windows) — `VmmcWindow` itself is a `VmmcReplicas` subclass, so a window
*is* a replica group confined to a region. `VmmcReplicasGroup` is the
analogous container for comparing several distinct systems (e.g. mismatch
variants) side by side, each with its own replica set. This package's
statistics/plotting machinery (`get_analysis_stat_mean/sem/ci`, melting-curve
fits) is clearly built around duplex melting-temperature (Tm) as the primary
science use case.

## Gotchas

- **Order parameter must exist before `build()`.** `build_vmmc_op_file()`
  asserts `self.bond_op is not None` — call `set_nucleotides`/`add_dist_op`
  first.
- **`build_start_weights` signature is context-dependent**: `(sim)` for the
  base `VMMCMetaSimulation`/`VMMCAutoReweight`, but `(sim, window_idx)` when
  used with `VmmcWindowing`. Mixing these up is a real, easy-to-hit trap.
- **`illegal_state_weight` default (1.0) is wrong for windowing.** The base
  auto-reweight default treats "illegal" as *physically impossible* so a
  nonzero placeholder is harmless; `VmmcWindowedAutoReweight` overrides it to
  `0.0` per window because there "illegal" means "belongs to a different,
  perfectly reachable window" and must be hard-excluded.
- **`extrapolate_hist` is hardcoded to Celsius**, and the setter always
  writes bare floats (drops a trailing `"C"` on round-trip) even though the
  reader accepts either form — a code comment literally flags this as a
  known wart.
- **Weight-file units/scale are per-window normalized to min=1** by
  `setup()`; never average raw window weight files directly — use
  `get_merged_weights()`, which geometric-mean-aligns overlapping states
  first (naive averaging silently misaligns window scales).
- **`VmmcWindowing.visualize()` is an unimplemented `pass` stub** — don't
  expect it to do anything.
- **oxDNA-unit → Celsius conversion is a hardcoded formula**
  (`T_celsius = T_simunits * 3000 - 273.15`) inside `read_vmmc_data` — worth
  knowing if you're reading `vmmc_df` temperatures directly instead of via
  the provided analysis helpers.
- **`VmmcReplicas.load()` hard-fails if replicas' weight files disagree**
  beyond float tolerance — don't hand-edit one replica's `wfile.txt`.

## Where to look next

- `references/windowing-mechanics.md` — WHAM internals, weight-mask
  construction, `VmmcWindowedAutoReweight`, the full quirk list with line
  references.
- `examples/vmmc_windowing_8nt_duplex/` and
  `examples/vmmc_graph_reweight_8nt_duplex/` in this repo — real, runnable
  end-to-end scripts; read these before writing a new windowing/reweighting
  workflow from scratch.
- `tests/test_vmmc.py`, `test_vmmc_windowing*.py`, `test_auto_reweight.py` —
  smallest correct usage patterns if the examples are more than you need.
