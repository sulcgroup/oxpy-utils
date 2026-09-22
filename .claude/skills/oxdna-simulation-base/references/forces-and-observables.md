# Forces and observables — full reference

Companion to `../SKILL.md`. Covers `utils/force.py`, `utils/observable.py`,
and the `SimForces` component in more depth.

## `Force` (`utils/force.py`)

`Force` is a `collections.abc.Mapping`, constructed as
`Force(force_type=ForceType.X, particle=..., **force_params)` (or
`type=` as a string alias for `force_type`). Each `ForceType` enum member
carries a `ForceTypeInfo(type_name, type_req_params, type_param_defaults)`:

- `MORSE`, `SKEW_TRAP`, `CENTER_OF_MASS` (`com`), `MUTUAL_TRAP`, `STRING`,
  `HARMONIC_TRAP` (`trap`), `ROT_HARMONIC_TRAP` (`twist`),
  `REPULSION_PLANE`, `REPULSION_SPHERE` (`sphere`).
- Missing a required param (with no matching default) raises
  `ValueError("Missing...")`.
- An unrecognized extra kwarg raises `ValueError("Excess...")`.
- Values are stringified on `__getitem__` except `np.ndarray` params, which
  must have `shape == (3,)` and are serialized comma-joined.

`zip_strands(strand_a_indices, strand_b_indices)` builds paired
`MUTUAL_TRAP` forces (one per index pair, `PBC=False`) — the mechanism
behind `SimForces.fold_forces` (see below) and useful directly if you want
to hold two complementary strands together (e.g. to pre-form/fold a duplex
before an equilibration run).

`load_forces_from_txt` / `load_forces_from_json` are the module-level parser
functions `SimForces.load_from` delegates to based on file extension.

## `SimForces` (the `sim.forces` component)

Backed by `_forces: dict[str, Force]`.

- `add_force(force: Force, name: str = None)` — auto-names `force_{n}` if
  `name` is omitted.
- `fold_forces(fold_str)` — parses dot-bracket secondary-structure notation
  into base-pair index tuples, then `zip_strands(*zip(*pairs))` to generate
  one `MUTUAL_TRAP` per pair and adds each.
- `load_from(file: Path)` — `.json` → `{name: {...force dict...}}` mapping,
  each value wrapped into a `Force(**value)` via `SimForces.__setitem__`;
  anything else assumed `.txt`, parsed via `load_forces_from_txt`.
- `save(file_name="forces.json")` — dumps `{name: {**force}}` for every
  force (relies on `Force.__iter__`/`Mapping` protocol).
- `dump_old_fmt(file="forces.txt")` — legacy oxDNA plain-text force-block
  format, for compatibility with tools that don't read the JSON form.
- Convenience static methods, **all returning raw dicts, not `Force`
  objects**: `morse`, `skew_force`, `mutual_trap`, `string`,
  `harmonic_force`, `repulsion_plane`, `repulsion_sphere`, plus the two
  mismatched ones flagged in the main skill file (`com_force` actually
  calls `skew_force`; `rotating_harmonic_trap` actually calls
  `harmonic_trap`) — don't rely on the name matching the force type it
  produces for those two specifically.

`Simulation.add_force` always calls `build_sim.build_force()` afterward,
which just does `self.sim.forces.save()` — so every single `add_force` call
rewrites the entire `forces.json`, not just appends. For many forces added
in a loop, this means O(n²) total file writes; harmless for a handful of
forces, worth knowing if you're adding hundreds.

## `Observable` (`utils/observable.py`)

`Observable(name, print_every, *ObservableColumn_or_dict, **kwargs)`.
`.export()` returns `{"output": to_dict()}` where
`to_dict() = {"print_every": str, "name": ..., "cols": [col.export() for col in cols], **extra_kwargs}`.

`ObservableColumn(type_name, **col_attrs)` stringifies all attrs on
`.export()`.

Convenience factory functions — each builds an `Observable` with one
`ObservableColumn` and returns `.export()` directly, so you rarely need to
construct `Observable`/`ObservableColumn` by hand:

- `simulation_time(...)`
- `distance(particle_1, particle_2, ...)`
- `hb_list(...)` — hydrogen-bond list
- `particle_position(particle, ...)`
- `potential_energy(...)`, `kinetic_energy(...)`, `force_energy(...)` —
  `potential_energy`'s `split` kwarg defaults to `None`, which stringifies
  to the literal text `"None"`; oxDNA's input parser rejects that as an
  invalid boolean at run time (`OxDNAError: boolean key 'split' is invalid
  ('none')`), confirmed directly. Always pass `split=True` or `split=False`
  explicitly — never rely on the default.
- `pair_energy(particle_1, particle_2, ...)`

`Simulation.add_observable` ensures
`input_file({'observables_file': 'observables.json'})` is set on first
call, then `build_sim.build_observable(observable_js)` either creates
`observables.json` fresh or merges into an existing one. The merge logic
includes a "multi-column" path that appends a new `cols` entry to an
*existing* observable's block if the second value of the two dicts happens
to match — this is a fairly fragile heuristic; if you're getting an
unexpected merge (or an unexpected lack of one), check whether your new
observable's dict shape happens to collide with this check.

**Reminder from the main skill**: `add_observable` only ever writes to
disk. `Analysis.observable_data()` reads from an in-memory
`self.observables` dict that is *not* automatically populated from
`observables.json` — call `sim.analysis.load_observables_from_json()`
explicitly whenever you're working with a simulation object that wasn't
the one that originally called `add_observable` in the same process (e.g.
after reloading via a fresh `Simulation(existing_sim_dir)` or via
`from_pickle`).
