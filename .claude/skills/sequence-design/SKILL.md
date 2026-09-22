---
name: sequence-design
description: Use when generating, screening, or optimizing DNA sequences for a nanostructure via oxpy-utils - unique-sequence generation (seqgen), matching melting temperature (Tm) across many strands (create_same_tm_seq_library), or screening a candidate pool for orthogonal (non-cross-hybridizing) sequences (orthogonal_seq_screen_parallel / sequence_optimization_functions). Covers oxpy_utils.seqdesign.*. Every module except seqgen.py does `from nupack import *` at module scope, so NUPACK (a separately-licensed, non-pip-installable dependency not present by default) is required just to import them - even the "biopy" Tm code path - unless you stub the import (see Dependency note). generate_same_tm_seqs_to_count has a real infinite-loop bug on infeasible Tm/length/GC targets and an intermittent reverse-complement-collision assertion crash - both covered in Gotchas. The higher-level domain/motif designer (OrthogonalSequenceSetDesigner.compute()) is an unfinished stub - don't present it as a working entry point. Not for assigning designed sequences onto a DNAStructure/cadnano model (that hand-off is manual today, via DNAStructureStrand.mutate_sequence or a FASTA file - see structure_editor code directly) or for the VMMC/FFS simulation steps that would follow (see vmmc-simulations / ffs-simulations).
---

# Sequence design via oxpy-utils

`oxpy_utils.seqdesign` is a standalone sequence-generation/screening toolkit,
not (yet) an automated pipeline. Nothing in the repo currently wires its
output into `DNAStructure`/cadnano strand assignment or into a VMMC/FFS run
automatically — that hand-off is manual (see "Connecting to a structure"
below). Treat this skill as "how to produce a set of good sequences," and
the structure/simulation skills as separate downstream steps.

## Dependency note — read this first

Only `seqgen.py` has zero external dependencies and is safe to import in any
environment. Every other module here (`sequence_set_designer.py`,
`sequence_optimization_functions.py`, `create_same_tm_seq_library.py`,
`orthogonal_seq_screen_parallel.py`) does `from nupack import *` **at module
scope** — so NUPACK (not on PyPI/conda, requires separate
license/registration: `pip install nupack -f
https://nupack.org/download/latest`) is required just to **import** any of
them, regardless of which function or `mt_algo` you actually intend to use.
Confirmed not installed by default in this container
(`python -c "import nupack"` → `ModuleNotFoundError`).

**This means `create_same_tm_seq_library.py`'s `mt_algo="biopy"` path is not
actually usable NUPACK-free as-is** — the import fails before you ever reach
the `mt_algo` branch, even though nothing in the `"biopy"` code path calls
NUPACK at runtime. Confirmed by actually running it in a NUPACK-less
environment. To use the biopy-only functions without installing NUPACK, drop
a throwaway stub module named `nupack.py` earlier on `sys.path`, defining
no-op placeholders for whatever names `from nupack import *` needs
(`Tube`, `Strand`, `Complex`, `SetSpec`, `Model`, `tube_analysis` as of this
writing — check the current top of the target file for the exact names) —
none of them get called when `mt_algo="biopy"`. This is a real gap in the
module, not a documented "feature flag"; if it gets fixed upstream (e.g. a
lazy/deferred NUPACK import), this workaround becomes unnecessary.
Biopython (`Bio.SeqUtils.MeltingTemp.Tm_NN`) is installed and is what
actually computes Tm on the biopy path.

## Unique sequence generation — `seqgen.py`

No external deps. Lazily yields unique, randomly-ordered DNA sequences of an
exact length and exact GC count, using a combinatorial-number-system
encoding so it never materializes the full sequence space:

```python
from oxpy_utils.seqdesign.seqgen import generate_unique_sequence

gen = generate_unique_sequence(8, 4)   # 8-nt sequences, exactly 4 GC bases
seq = next(gen)
```

Raises `ValueError` if `gc_count > length` or either arg is negative; raises
a generic `Exception` once the combinatorial space is exhausted for that
(length, gc_count) pair — check `count_sequences(length, gc_count)` first if
you need many sequences and want to know the ceiling. Also usable as a CLI:
`python seqgen.py <handle_length> <gc_count> <size> [<filename>]`.

## Matching melting temperature across many strands — `create_same_tm_seq_library.py`

Problem: in a multi-strand nanostructure you typically want many *different*
sequences (so strands don't cross-react) that all melt at ~the same
temperature, so the structure assembles/disassembles cooperatively instead
of some domains melting far below/above others.

```python
# Requires stubbing nupack first if it's not installed — see Dependency note above.
from oxpy_utils.seqdesign.create_same_tm_seq_library import generate_same_tm_seqs_to_count

seqs = generate_same_tm_seqs_to_count(
    celsius=45, gc_count=7, sequence_length=10,   # NOTE: sequence_length, not hlen
    mt_algo="biopy",          # or "nupack" / "both" — see caveat below
    num_seqs_to_generate=12,
    tolerance=2.0,
)   # -> {sequence: Tm}, e.g. {"GCGTTGGCCA": 43.06, "ACGTGCTCGC": 43.16, ...}
```

(`hlen` is the sibling function `generate_same_tm_seqs`'s length parameter —
`generate_same_tm_seqs_to_count` uses `sequence_length` instead. Easy to
mix up since they otherwise take near-identical arguments.)

`generate_same_tm_seqs_to_count` keeps generating in chunks until it
accumulates the requested count — prefer it over `generate_same_tm_seqs`,
which instead over/under-shoots against a fixed candidate pool size. **But
verify your target is physically reachable before calling it** — see
Gotchas below; an infeasible (Tm, length, gc_count) combination doesn't
raise, it spins forever.

**Use `mt_algo="biopy"` unless you specifically need NUPACK** (and even then
you're stubbing `nupack` at import time regardless — see Dependency note).
The `"nupack"` and `"both"` code paths use NUPACK tube analysis (sweeping
temperature to find where unbound fraction crosses 0.5) — functional, but
the module's own TODO admits the `"nupack"` and `"biopy"` branches look like
they were never actually reconciled after being copy-pasted, and this is one
of two independent seq-validity filters in the package (see Gotchas).

## Screening for orthogonal (non-cross-hybridizing) sequences

Given a candidate pool, find the subset of `n` sequences (+ complements)
that maximizes the **minimum** mismatch-binding free energy (ddG) across all
non-complementary pairs — sequences that bind their intended partner
strongly but don't cross-talk with anything else in the set. Requires
NUPACK (all-pairs ΔG comes from `nupack.complex_analysis`).

```
python orthogonal_seq_screen_parallel.py \
    --candidate_sequences_file candidates.txt \
    -n 20 -s 1000 \
    -o orthogonal_output.txt \
    -t TTTTT \
    -c 20
```

The optimization itself (`sequence_optimization_functions.py`) computes an
all-pairs ΔG matrix (`compute_dGs`, pickle-cacheable via `fill_dGs`), derives
a ddG matrix (`set_column_complement_to_inf`: for each sequence, how much
weaker an off-target interaction is vs. its intended duplex), then runs a
minimax hill-climbing search (`ddG_optimization_algorithm` /
`optimization_iteration`, the inner loop numba-JIT'd) that **is** genuinely
parallelized across CPU cores — despite the *filename*
`orthogonal_seq_screen_parallel.py` containing none of the parallelism
itself; it lives in `sequence_optimization_functions.py`. Note this
parallelism depends on the `fork` multiprocessing start method (the ΔG
matrix is shared via a process-global, not passed explicitly) — it will
break under `spawn`.

GC content is a **generation constraint** (via `seqgen`'s `gc_count`), not an
optimization objective here — the thing actually being maximized is the
worst-case orthogonality gap, not GC content or average ΔG.

## The higher-level domain/motif designer is not usable yet

`sequence_set_designer.OrthogonalSequenceSetDesigner` models multi-domain
strands (a **domain** = one named sequence; a **motif** = an ordered list of
domain names forming one physical strand) and exposes
`request_domains(...)`, `request_motifs(...)`, and `compute(...)` as what
looks like the intended top-level API. **Don't use `compute()`** — its core
loop calls `__iter_domain_assigments()`, which is an empty stub (`pass`), so
it cannot run to completion. Lower-level pieces on the same class do work
today: `generate_sequence`, `screen_tms`, `get_tm_seq`, `check_seq_valid`,
`is_rc`/`name_rc`. `screen_tms_nupack` is separately self-flagged in the
source as "90% sure this method is broken" — avoid it even for the pieces
that do run.

## Connecting to a structure (manual today)

Nothing in the repo automatically assigns seqdesign output onto a
`DNAStructure`. The mechanism that *would* do it is
`DNAStructureStrand.mutate_sequence(new_sequence, start, stop, five_prime=True)`
(in `structure_editor/dna_structure.py`) — an in-place base substitution on
an existing strand — or supplying a FASTA `sequence_file` to
`load_dna_structure_from_cadnano(...)`, which passes it through to
tacoxDNA's `--sequence` flag. The `examples/cadnano_origami_workflow/`
example does the latter (cadnano scaffold/staple sequences, not
seqdesign-generated ones) and never imports `oxpy_utils.seqdesign` — treat
seqdesign as producing a sequence set you hand off yourself, not an
automated step in that example's pipeline.

## Gotchas

- **`generate_same_tm_seqs_to_count` hangs forever on an infeasible target
  instead of raising.** Its outer loop only detects "can't generate more"
  by checking whether a freshly-built chunk came back shorter than
  `chunk_size` — but rejected-on-Tm sequences aren't excluded from future
  generator calls (only accepted ones are), so if the *raw* candidate space
  for your `(sequence_length, gc_count)` is large, chunks never come back
  short even when zero candidates in the entire space hit your target Tm
  window. Confirmed by direct testing: at `sequence_length=8, gc_count=4`,
  the single best-case 8-mer (`GCGCGCGC`) tops out at 44.9°C under this
  module's Tm model (Biopython nearest-neighbor, dnac1=dnac2=10nM, Na=50mM,
  Mg=12.5mM) — so a `celsius=45` target at that length is unreachable, and
  the call spins at 100% CPU indefinitely rather than erroring.
  **Before calling it, sanity-check feasibility yourself** — e.g. brute-force
  or spot-check a handful of high/low-GC sequences at your target length
  through `Bio.SeqUtils.MeltingTemp.Tm_NN` to confirm your target Tm falls
  inside the achievable range, especially for short sequences (≤8-9nt) where
  the achievable Tm range is narrow. If you hit this, increasing
  `sequence_length` is usually the fix (10nt comfortably reaches 45°C
  targets with plenty of GC-count headroom to find 12+ distinct sequences).
- **Intermittent `AssertionError` from its own internal duplicate check.**
  The function's closing sanity check
  (`assert all(rc(seq) not in good_sequences for seq in good_sequences.keys())`)
  can and does fail — confirmed reproducible in ~2 of 3 unseeded trials even
  on an otherwise-feasible target — because a sequence and its reverse
  complement can both get accepted across different generation chunks
  before the check runs. This is a real bug in the function, not user
  error; currently the only workaround is to retry the call (it succeeds
  when no such collision happens to occur in that trial).
- **Two different, inconsistent "is this sequence OK" filters exist** —
  `sequence_set_designer.check_seq_valid` (configurable min/max polybase
  run length, default reject runs ≥3) vs.
  `create_same_tm_seq_library.seq_check` (hardcoded, narrower rule set, with
  a docstring that literally says `???`). **Prefer `check_seq_valid`** as
  the canonical filter if you're writing new code that needs one.
- **Direction conventions are a real footgun.** oxDNA/oxpy stores sequences
  3'→5' internally; the seqdesign code is written/documented 5'→3', and
  `seqs_oxDNA()` handles the conversion by **reversing** (not
  reverse-complementing) the domain sequence on export. Confusing "reverse"
  with "reverse complement" here silently produces a nonsense structure.
- **Reverse-complement domain naming is asymmetric.** A domain name
  prefixed with `-` or suffixed with `'` both mean "this is the RC of some
  other domain," but `name_rc()` always *produces* the `'`-suffix form —
  round-tripping a `-`-prefixed name through `name_rc(name_rc(x))` does not
  return the original string.
- **`generate_unique_sequence`'s `generated_sequences` param is a mutable
  default argument (`= {}`)** — currently harmless since the function never
  reads or mutates it, but if that TODO (wiring up de-duplication) is ever
  implemented naively, it will leak state across independent calls sharing
  the default.
- **Alphabet validation is inconsistent about RNA.** Most of the pipeline
  assumes DNA (`rc()` uses `str.maketrans('ACGT','TGCA')`), but
  `orthogonal_seq_screen_parallel.py`'s own input-file validation accepts
  `U` — a sequence with `U` would pass validation but then silently
  mis-translate under `rc()` (U isn't in the translation table).
- **This subpackage was recently migrated** from an older `ipy_oxDNA`
  codebase (thin git history) and has already had at least two silent
  correctness bugs fixed in `generate_unique_sequence`'s core algorithm
  (duplicate-yield bugs in `nth_combination` and the shuffle helper) — the
  area has a real history of subtle exhaustiveness/uniqueness bugs, so treat
  "should be exhaustive and unique" claims about this generator with mild
  suspicion and check `count_sequences()` against expectations if something
  looks off.

## Where to look next

- `tests/test_seqdesign.py`, `test_generate_sequences.py` — smallest correct
  usage patterns, and the module docstring in `test_seqdesign.py` documents
  the two historical `generate_unique_sequence` bugs mentioned above in more
  detail.
- `oxpy_utils/structure_editor/dna_structure.py` — `mutate_sequence` and
  `rc()`, the two points of actual code coupling to seqdesign.
