---
name: oxview-visualization
description: Use when asked to look at, screenshot, or sanity-check an oxDNA/DNA/RNA structure or trajectory in oxView (the browser-based oxDNA viewer/editor) - a DNA origami, a duplex, a sliced region, an exported .oxview, a relaxation or production run. Covers which files oxView takes from oxpy-utils (top+dat, .oxview from DNAStructure.export_oxview/get_oxview_json, overlays), opening it interactively, render.js in this skill (headless PNG, camera fit, trajectory frame, and running oxView's scripting API to read things back), and how oxView's strand/element ids and sequence direction map onto DNAStructure. Not for patchy-particle simulations (use ppview in the pypatchy repo - it is far better for those), for building/editing structures in Python (see dna-structure), or for the interactive slice-trajectory Artifact page (see slice-trajectory-artifact).
---

# Looking at oxDNA structures in oxView

oxView (github.com/sulcgroup/oxdna-viewer, Three.js, TypeScript compiled into a committed `dist/`) is the
viewer/editor for oxDNA configurations. This skill is about DNA/RNA structures, such as origami,
duplexes, and the outputs of [[dna-structure]], [[oxdna-simulation-base]] and [[sim-chaining]] runs. It
is not for patchy-particle simulations: ppview, documented in the pypatchy repo, is the viewer for those.

Checkouts: `~/git/oxdna-viewer-navsikand` (clone of `navsikand/oxdna-viewer`, the more up to date fork:
latest commit 2026-02-18, adds a dashboard, library and cloud features) is the default for `render.js`.
`~/git/oxdna-viewer` is an older checkout (JeDeveloper fork of the sulcgroup original, last commit 2023).
The hosted original is https://sulcgroup.github.io/oxdna-viewer/ (it may lag the fork).

## Which files to give it

oxView decides what each file is by extension, in a first-match order (`handleFiles` in
`ts/file_handling/file_handling.ts`):

| File | What oxView does |
|---|---|
| `.top` + `.dat`/`.conf`/`.oxdna` | the system and its configuration; a multi-frame `.dat` is a trajectory |
| `.oxview` | JSON with one or more systems (and cluster colours); no `.top` needed |
| `.pdb`, `.unf`, `.xyz`, `.mgl` | other structure formats |
| `.json` | overlay data (colour maps) or other auxiliary data |
| `.txt` with `trap`/`force` in the name; `.idx`, `select` files; `.hb`, `.db`, `.par`, `.csv`, `.cam`, `.js` | forces, selections, hydrogen bonds, dot-bracket, camera, scripts |

Only **one** file may define a system per load (a second `.top`/`.oxview` in the same load is an error).
Names matter in one place: a file whose name contains `particles`, `patches`, `loro` or `matrix` is taken as
a patchy-particle helper file whatever its extension, so don't give a DNA file such a name.

### Producing them from oxpy-utils

```python
from oxpy_utils.structure_editor.dna_structure import load_dna_structure
s = load_dna_structure("x.top", "x.dat")
s.export_top_conf("out.top", "out.dat")   # canonical top/dat pair
s.export_oxview("out.oxview")             # or get_oxview_json(cluster_colors=...) for the dict
```

(See [[dna-structure]] for the exports, `load_oxview`, and recolouring a top/dat from an `.oxview`.)
A simulation directory already has what oxView needs: its topology plus `last_conf.dat`,
`trajectory.dat` or `init.dat`.

## Interactive

Hosted (above), or serve a checkout statically: `cd ~/git/oxdna-viewer-navsikand && python -m http.server`
and open `index.html`; there is no build step for using it (`dist/` is committed). Drag files in, or
File > Open. Useful: `S` select, `D` transform, the `Download Output Files` button for the edited structure.
The scripting console is the browser's: `api.*`, `edit.*` and the `systems`/`elements` globals.

## Render a PNG headlessly (so Claude can look at it)

`render.js` (next to this file) serves the checkout, starts a throwaway headless Chrome, puts the files into
oxView's own file input (then calls `handleFiles`, as the Open dialog's button does), optionally steps to a
frame, frames the camera, runs console snippets, screenshots and cleans up. No build; the checkout is not
modified. Then `Read` the PNG.

One-time setup, outside any repo: `npm install --prefix ~/.cache/oxpy-utils-render-deps ws`. Needs Node and
Chrome/Chromium (`--chrome`/`$CHROME_PATH`); `--repo`/`$OXVIEW_DIR` overrides the checkout.

```bash
R=~/git/oxpy-utils/.claude/skills/oxview-visualization/render.js

# an origami, camera framed on it, with a summary of what loaded
node $R --fit --stats --out /tmp/o.png tests/test_data/cube/cube.top tests/test_data/cube/cube.dat

# an exported .oxview
node $R --fit --out /tmp/o.png structure.oxview

# trajectory frame, and run the scripting API (printed as JSON)
node $R --fit --frame last --out /tmp/o.png --eval "api.highlight5ps()" run/x.top run/trajectory.dat
```

Options: `--fit`, `--frame first|last|N` (0-based), `--eval JS` (repeatable; a returned value is printed),
`--stats`, `--size WxH`, `--settle MS`. If no system loads, it still saves a screenshot and prints the page's
errors. oxView always throws two harmless errors on load (a tacoxdna worker loaded as a plain script, and a
Metro dialog handler); the script hides those.

Verified here with `oxpy-utils/tests/test_data/cube` (a 170-strand, 14,348-nucleotide origami; renders as the
whole cube with helices visible), the same structure exported to `.oxview` (identical counts: 1 system, 170
strands, 14,348 elements, matching `load_dna_structure`), the 8-nt duplex, and `--frame` on a synthetic
3-frame trajectory (oxView's per-frame particle distances matched the file's, frame by frame).

## Reading the structure back: oxView API vs oxpy-utils

For counts, sequences and anything computed, prefer oxpy-utils. Use oxView's scripting API when the question
is about what oxView itself shows or does.

| Question | oxView (`--eval`) | oxpy-utils |
|---|---|---|
| number of strands / nucleotides | `systems[0].strands.length`, `elements.size` | `len(s.strands)`, `sum(len(x) for x in s.strands)` |
| strand lengths (find the scaffold) | `Object.entries(api.countStrandLength()).map(([n, ss]) => [Number(n), ss.map(x => x.id)])` | `sorted((len(x), i) for i, x in enumerate(s.strands))` |
| a strand's sequence | `edit.getSequence(new Set(systems[0].strands[i].getMonomers()))` | `s.strands[i].seq(from_5p=True)` |
| nucleotide position | `elements.get(id).getPos()` | `s.strands[i].positions` |

Checked on the cube origami: **oxView's strand ids are the indices of `DNAStructure.strands`** (the 7072-nt
scaffold is strand 169 in both; the 59-nt strands are 12, 110 and 123 in both), and **oxView lists a strand
5'→3'**: `edit.getSequence` equals `strand.seq(from_5p=True)`, whereas oxpy-utils' plain `seq()` is the
3'→5' storage order (see [[dna-structure]] on direction).

Things the fork's README gets wrong or hides:
- `getSequence` is `edit.getSequence(elems: Set)`, not `api.getSequence`, and it needs a `Set` of elements.
- `api.countStrandLength()` returns an object keyed by length whose values are arrays of **whole strand
  objects**; printing it as is gives megabytes of JSON. Map to ids as above.
- The real API is wider than the README: `Object.keys(api)` and `Object.keys(edit)` list it (for example
  `api.highlight3ps`, `api.selectElementIDs`, `api.findElement`, `edit.setSequence`, `edit.createBP`).

### Gotchas

- **The default camera frames small structures tiny**, and oxView has no fit command, so `--fit` computes one
  from the element positions (it keeps your current viewing direction).
- oxView **re-centres and re-boxes every frame** (its centring/inboxing settings), so a rigid shift of a
  structure between frames does not show; only shape changes do. Don't read a frame comparison as drift.
- The toolbar and ribbon are in every screenshot; there is no option to hide them here.
- Software WebGL (SwiftShader): the 14k-nucleotide origami is fine; much bigger systems will be slow.
- A render can't confirm counts or sequences; use `--stats`, `--eval`, or oxpy-utils.
- Not tested: overlay `.json` colour maps, forces/selection files, editing through `edit.*`, multi-system
  loads, and UNF/PDB inputs. Behaviour for those is from the README and source only.
