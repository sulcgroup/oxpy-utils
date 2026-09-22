---
name: dna-structure
description: Use when building, loading, editing, or exporting a DNAStructure/DNAStructureStrand via oxpy-utils (oxpy_utils.structure_editor.dna_structure) - constructing a duplex from scratch (construct_strands), loading from oxDNA .top/.dat, PDB, cadnano JSON, RCSB, or oxView, mutating a strand's sequence or bases (mutate_sequence, assign, nick), exporting back to .top/.dat/PDB/oxView/glTF, or extracting a sub-region with StructureSlicer. This is the shared foundation seqdesign, oxdna-simulation-base, and sim-chaining all build on - most oxpy-utils workflows touch it even when it isn't the headline task. Storage is 3'->5' throughout (opposite of how most sequences are written/read), a real source of bugs - see Gotchas. Not for the Simulation class itself (see oxdna-simulation-base) or a multi-stage relax pipeline (see sim-chaining), though both hand off to/from a DNAStructure constantly.
---

# `DNAStructure` (`oxpy_utils/structure_editor/dna_structure.py`)

An in-memory, editable mirror of an oxDNA topology (`.top`) + configuration
(`.dat`) pair — built because "the oat library to edit confs is very very
bad" (the module's own docstring). `DNAStructure` holds
`strands: list[DNAStructureStrand]`, plus `time`/`box`/`energy` and OxView
cluster/color bookkeeping. `DNAStructureStrand` holds the actual per-base
data as parallel numpy arrays: `bases` (chars), `positions`, `a1s`/`a3s`
(orientation vectors; `a2s` is derived as `cross(a3s, a1s)`).

**Storage convention: everything is 3'→5', matching oxDNA itself.** This is
the single most important thing to internalize before touching this module
— indices, `nick()`, and `mutate_sequence`'s default all follow it, and
getting it backwards silently edits/reads the wrong end of a strand. See
Gotchas.

## Constructing / loading

```python
from oxpy_utils.structure_editor.dna_structure import (
    construct_strands, load_dna_structure, load_dna_structure_from_cadnano,
)

# From scratch — a straight antiparallel duplex, geometry generated for you.
# Direction MUST be spelled out in the string or you get a loud warning and
# an assumed-3'->5' fallback:
fwd, rev = construct_strands("5'-ACGTACGT-3'", start_pos, helix_direction)
structure = DNAStructure([fwd, rev], t=0, box=box)

# From existing oxDNA files (the most common loader in practice):
structure = load_dna_structure(top_path, conf_path, conf_idx=0)

# From a cadnano design (the real origami-design entry point):
structure = load_dna_structure_from_cadnano(json_path, lattice="sq")
# lattice must be "sq" or "he". Pass sequence_file=<fasta> for deterministic
# bases instead of tacoxDNA's randomized default assignment.
```

Other loaders: `load_dna_structure_from_pdb(pdb_path, direction='35'|'53')`,
`load_dna_structure_from_rcsb(pdb_id, direction=...)` (downloads via
`Bio.PDB.PDBList`, then defers to the PDB loader), `load_oxview(path)`
(parses OxView JSON directly, no tacoxDNA involved).

**PDB/cadnano/RCSB loaders all shell out to tacoxDNA**, a sibling checkout
(`/root/Github/tacoxDNA`), not a pip package — resolved via
`_tacoxdna_src()` (`Path(__file__).parents[3] / 'tacoxDNA' / 'src'` by
default). If your checkout layout differs, set the module-level
`dna_structure.TACOXDNA_SRC` yourself before calling these loaders, or you
get a clear `FileNotFoundError`.

## Editing

```python
# In-place, 5'->3' input by default (reversed internally to match 3'->5' storage):
strand.mutate_sequence("ACGT", start=0, stop=4)          # exact-length replace
strand.mutate_sequence("G", start=5)                     # single-base mutation
strand.mutate_sequence("acgt", start=0, stop=4, five_prime=False)  # already 3'->5'

structure.nick(strand_id, n)   # split a strand after position n (3'->5' indexing;
                                # negative n counts from the 5' end)
structure.transform(rot, tran) # rigid-body move — does NOT update the bounding box
cpy = structure.clone()        # deep copy, mints fresh global uids by default
```

`DNAStructureStrand.assign(other, start, stop, refr_uids="new"|"cpy")` is
the more general block-replace (bases + positions + orientations, not just
sequence) — same strict length-match rule as `mutate_sequence`.
`refr_uids="cpy"` reuses uids from `other` rather than minting new ones —
the docstring itself says "USE WITH CAUTION."

`helixify(s1, s2)` repositions two equal-length (sub)strands into a proper
double helix using the same geometry as `construct_strands` — useful after
you've assembled/edited strands that aren't yet in helical coordinates.

## Exporting

```python
structure.export_top_conf(top_path, conf_path)   # the canonical oxDNA round-trip
structure.export_pdb(path, direction='35', hydrogens=True)
structure.export_oxview(ovfile)
structure.export_gltf(path, default_color=...)   # visualization; raises if any
                                                    # base has no color and no default given
```

`export_top_conf` calls `.open()` directly on `top_file_path`/
`conf_file_path` — it does **not** coerce a plain `str` to `Path` for you.
Pass `Path` objects (e.g. `Path("duplex.top")`), not strings, or you get
`AttributeError: 'str' object has no attribute 'open'`.

If `box` isn't set, every export path **transparently calls
`self.inbox()` first** — which returns a *new*, re-centered structure
rather than mutating `self`. Don't expect `structure.box` to be populated
after an export call just because the export succeeded.

## `StructureSlicer` — extracting a sub-region

`StructureSlicer(sim: Simulation)` (`structure_editor/structure_slicer.py`)
plugs into a `Simulation`'s builder (`sim.set_builder(slicer)`) to carve out
a spatial/selection-based sub-structure as its own standalone, runnable
simulation — e.g. pulling a ~500-base region out of a full origami to study
its local dynamics in isolation.

```python
slicer = StructureSlicer(sim)
slicer.set_slice(base_indices)               # OxView indices, or:
slicer.set_slice_box(corner_1, corner_2)     # axis-aligned box selection
slicer.set_slice_plane(points, norm)         # keep one side of a plane

slicer.load_sampling_from("trajectory.dat")  # per-endpoint mean position + RMSF
slicer.add_endpoint_forces()                 # HARMONIC_TRAP at each cut point,
                                              # scaled by 1/rmsf so the excised
                                              # region doesn't unravel at the cuts
```

Internally, `set_slice` converts selected indices to persistent uids and
detects **cut points** (selected bases whose strand-neighbor isn't
selected). `do_slice()` deep-copies the parent structure, `nick()`s at each
cut point, discards any strand not fully inside the selection, and exports
straight into the new sim's directory — this *is* the builder's
`build_dat_top()` override, so it composes with the ordinary `Simulation`
lifecycle from `oxdna-simulation-base`.

## Connecting to `Simulation`

- **`DNAStructure` → `Simulation`**: pass the structure directly as
  `file_dir`: `Simulation(structure, sim_dir)`. `build()` then calls
  `structure.export_top_conf(...)` into `sim_dir` for you.
- **`Simulation` → `DNAStructure`** (reading a conformation back):
  `sim.get_conf(conf_idx)` (frame N — index 0 reads the initial conf, any
  other index reads `trajectory_file` at `conf_idx - 1`),
  `sim.last_conf_structure()` (the sim's final/checkpoint conf), or
  `sim.analysis.get_conf_as_structure(conf_id)`. All three ultimately call
  `load_dna_structure` on whatever files the simulation has written so
  far — there's no separate "read from a running sim" path.

See `oxdna-simulation-base` for the `Simulation` side of this, and
`sim-chaining` for how this plays into `Simulation(prev_sim, next_dir)`.

## Gotchas

- **3'→5' storage is pervasive and easy to get backwards.** `nick()`'s
  negative-index convention, `mutate_sequence`'s `five_prime=True` reversal,
  and `StructureSlicer`'s own code comment ("oxDNA does things 3'→5'
  because Mistakes Were Made") all point at the same trap — verify direction
  explicitly (e.g. via `seq()`) rather than assuming.
- **`clone(copy_uuids=True)` looks like a real bug**: the `uid_map` used to
  remap `clustermap` entries is only ever defined in the `copy_uuids=False`
  branch, but referenced unconditionally afterward. Cloning with
  `copy_uuids=True` on a structure with any clustermap entries is likely to
  raise `UnboundLocalError`/`NameError` — untested in the repo's own test
  suite, so verify against current source if you hit this rather than
  trusting this note indefinitely.
- **`append`/`prepend`'s docstrings don't obviously match their
  implementation** — both just describe 5'/3' concatenation in prose, but
  given the 3'→5' storage convention, verify which physical end you're
  actually extending (e.g. via `seq()` before/after) rather than trusting
  the docstring language at face value.
- **`transform()` invalidates the box and does not recompute it** — the
  docstring says so explicitly ("does NOT update the bounding box!!!").
  Any whole-structure rotation/translation means the next export auto-boxes
  via `inbox()`, returning a new re-centered structure.
- **Topology/recoloring comparisons are positional-only** — `check_top_match`
  and `recolor_structure` explicitly do not account for different base or
  strand ordering/numbering. Two structures that are physically identical
  but built/loaded in a different strand order will not compare or
  recolor correctly.
- **`base_id_map`/`base_id_reverse_map` are not preserved across edits** —
  documented directly in the source. After adding/removing/nicking strands,
  call `reindex_base_ids()` again before relying on these maps.
- **Global mutable uid counter** (`RESIDUECOUNT`) — uids are process-global
  and monotonically increasing, not reproducible across runs/processes.
  Fine for uniqueness within one run, not useful as a stable identifier
  across sessions.
- **`construct_strands` needs an explicit `5'-...-3'`/`3'-...-5'` decorated
  string** — a bare undecorated sequence silently gets treated as
  already-3'→5', which is a likely source of accidentally-reversed
  sequences if you forget the decoration.

## Where to look next

- `references/tacoxdna-and-slicer-details.md` — fuller tacoxDNA loader
  mechanics, `StructureSlicer` internals, and export format specifics.
- `oxdna-simulation-base` — the `Simulation` side of the hand-off.
- `sequence-design` — `mutate_sequence`/`rc()` as the (currently manual)
  hand-off point for assigning designed sequences onto a structure.
- `sim-chaining` — the real multi-stage pipeline example that starts from
  a cadnano-loaded `DNAStructure`.
