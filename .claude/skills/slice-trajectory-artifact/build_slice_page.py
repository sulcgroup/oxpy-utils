#!/usr/bin/env python
"""Build the "slice trajectory" HTML page from a finished StructureSlicer workflow.

Needs, per run directory (the names oxpy_utils writes by default):
  --sliced  DIR   the sliced simulation: a *.top, init.dat, trajectory.dat, energy.dat, forces.json, input.json
  --source  DIR   the full-structure run the slice was cut from: a *.top and last_conf.dat
  --control DIR   (optional) the unsliced structure continued from the SAME starting frame (source's last_conf.dat),
                  same frame spacing. Gives the dashed baseline on the RMSD chart.
No list of slice bases is needed: sliced particle i is matched to its source particle by exact position
in the starting frame, so this works however the slice was selected (set_slice, box, plane, ...).

usage: python build_slice_page.py --sliced S --source P --out page.html [--control C] [--title T] ...
Then publish --out with the Artifact tool.
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

SU_NM = 0.8518
HERE = Path(__file__).parent


def read_confs(top: Path, traj: Path, max_frames: int = None):
    """positions (T, N, 3) and the step each frame was written at, via oxDNA_analysis_tools"""
    from oxDNA_analysis_tools.UTILS.RyeReader import describe, get_confs
    ti, tj = describe(str(top), str(traj))
    conf = get_confs(ti, tj, 0, tj.nconfs)
    return np.stack([c.positions for c in conf]), [int(round(c.time)) for c in conf]


def one(d: Path, pattern: str) -> Path:
    hits = sorted(d.glob(pattern))
    if not hits:
        raise SystemExit(f"{d} has no {pattern}")
    return hits[0]


def kabsch_rmsd(P, Q, sel=None):
    """rmsd of each frame in P (T,N,3) from Q (N,3) after optimal rigid alignment on all N; optionally score only `sel`"""
    qc = Q - Q.mean(axis=0)
    out = []
    for p in P:
        pc = p - p.mean(axis=0)
        U, _, Vt = np.linalg.svd(pc.T @ qc)
        R = U @ np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))]) @ Vt
        diff = pc @ R - qc
        if sel is not None:
            diff = diff[sel]
        out.append(np.sqrt((diff ** 2).sum(axis=1).mean()))
    return np.array(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sliced", required=True, type=Path)
    ap.add_argument("--source", required=True, type=Path)
    ap.add_argument("--control", type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--title", default="Slice Trajectory", help="page name: 2-4 words")
    ap.add_argument("--structure", default="the full structure", help='what was sliced, e.g. "the relaxed DNA-origami cube"')
    ap.add_argument("--stiffness-note", default="", help='how the traps were set, e.g. "with a stiffness of 3kT/RMSF^2 from a 1,000,000-step run of the full structure"')
    ap.add_argument("--build-note", default="", help="one paragraph on how the run was made (relaxation, production length, trap rule)")
    ap.add_argument("--max-frames", type=int, default=101, help="thin the trajectory to at most this many frames (page size)")
    ap.add_argument("--max-context", type=int, default=40000, help="max rest-of-structure points drawn")
    a = ap.parse_args()

    sl_top, src_top = one(a.sliced, "*.top"), one(a.source, "*.top")
    cfg = json.load(open(a.sliced / "input.json"))
    dt = float(cfg.get("dt", 0.003))

    # ---- sliced run ----
    init, _ = read_confs(sl_top, a.sliced / "init.dat")
    traj, tsteps = read_confs(sl_top, a.sliced / "trajectory.dat")
    frames = np.concatenate([init[:1], traj])
    steps = raw_steps = [0] + tsteps
    n = frames.shape[1]
    keep = np.unique(np.linspace(0, len(frames) - 1, min(a.max_frames, len(frames))).round().astype(int))
    frames, steps = frames[keep], [steps[i] for i in keep]
    start = frames[0]
    center = start.mean(axis=0)

    # ---- map sliced particle i -> source particle m[i] by position in the starting frame ----
    src_all = read_confs(src_top, a.source / "last_conf.dat")[0][0]
    dist, m = cKDTree(src_all).query(start)
    if dist.max() > 1e-3 or len(set(m)) != n:
        raise SystemExit(f"sliced starting frame does not match source last_conf.dat (max dist {dist.max():.3g}); "
                         "--source must be the run whose final frame was sliced")

    # ---- topology: strands + bonds ----
    lines = sl_top.read_text().split("\n")
    nn, ns = map(int, lines[0].split()[:2])
    strand = [int(l.split()[0]) for l in lines[1:1 + nn]]
    strand = [s - min(strand) for s in strand]
    bonds = [[i, i + 1] for i in range(nn - 1) if strand[i] == strand[i + 1]]
    assert nn == n

    # ---- traps ----
    traps = []
    for f in json.load(open(a.sliced / "forces.json")).values():
        pos0 = [float(x) for x in f["pos0"].split(",")] if isinstance(f["pos0"], str) else f["pos0"]
        traps.append({"particle": int(f["particle"]), "pos0": [round(float(p - c), 3) for p, c in zip(pos0, center)],
                      "stiff": round(float(f["stiff"]), 3)})
    ep = [t["particle"] for t in traps]

    F = frames - center
    disp = np.linalg.norm(F - F[0], axis=2)
    data = {"n_nt": n, "n_strands": ns, "n_cube": len(src_all), "dt": dt, "strand": strand, "bonds": bonds, "traps": traps,
            "steps": steps, "frames": [np.round(f.ravel(), 2).tolist() for f in F],
            "disp_max": round(float(np.percentile(disp, 95)), 3),
            "rmsd_all": [round(float(x), 3) for x in kabsch_rmsd(frames, start)],
            "rmsd_ep": [round(float(x), 3) for x in kabsch_rmsd(frames, start, ep)]}

    # ---- energy (oxDNA energy.dat columns: time, potential/nt, kinetic/nt, total/nt) ----
    E = np.loadtxt(a.sliced / "energy.dat", ndmin=2)
    data["energy"] = {"time": [round(float(x), 2) for x in E[:, 0]], "U": [round(float(x), 4) for x in E[:, 1]],
                      "K": [round(float(x), 4) for x in E[:, 2]]}

    # ---- rest of the structure, for context ----
    rest = np.delete(src_all, m, axis=0) - center
    if len(rest) > a.max_context:
        rest = rest[np.linspace(0, len(rest) - 1, a.max_context).astype(int)]
    data["cube"] = np.round(rest.ravel(), 1).tolist()

    # ---- unsliced baseline ----
    if a.control:
        ct_top = one(a.control, "*.top")
        ctrl, csteps = read_confs(ct_top, a.control / "trajectory.dat")
        ctrl = np.concatenate([src_all[m][None], ctrl[:, m]])
        csteps = [0] + csteps
        if csteps != raw_steps:
            raise SystemExit(f"control frames (steps {csteps[:3]}..{csteps[-1]}, {len(csteps)} frames) do not line up with the sliced run "
                             f"(steps {raw_steps[:3]}..{raw_steps[-1]}, {len(raw_steps)} frames): use the same steps and print interval")
        ctrl = ctrl[keep]
        data["ctrl_rmsd_all"] = [round(float(x), 3) for x in kabsch_rmsd(ctrl, start)]
        data["ctrl_rmsd_ep"] = [round(float(x), 3) for x in kabsch_rmsd(ctrl, start, ep)]

    # ---- text ----
    backend = cfg.get("backend", "CPU")
    eyebrow = f"{cfg.get('interaction_type', 'oxDNA2')} · StructureSlicer · {backend} backend · {cfg.get('T', '')}".strip(" ·")
    lede = (f"A {n:,}-nucleotide region cut out of {a.structure} and simulated on its own. Each of the {len(traps)} cut ends "
            f"is held by a harmonic trap at its position in the full structure"
            + (f", {a.stiffness_note}." if a.stiffness_note else "."))
    if a.control:
        rmsd_sub = ("Each frame is aligned to the starting structure on all nucleotides. Dashed lines are the same nucleotides in "
                    "the unsliced structure, run from the same start. Click a point to jump to that frame. "
                    f"1 SU = {SU_NM} nm.")
    else:
        rmsd_sub = f"Each frame is aligned to the starting structure on all nucleotides. 1 SU = {SU_NM} nm. Click a point to jump to that frame."
    build_note = a.build_note or ("StructureSlicer cut the selected region out of the final frame of the source run, nicked every strand "
                                  "that leaves it, and kept the strands inside. Each cut end is held by a harmonic trap at its position in that frame.")

    html = (HERE / "template.html").read_text()
    for k, v in {"TITLE": a.title, "EYEBROW": eyebrow, "LEDE": lede, "RMSD_SUB": rmsd_sub, "BUILD_NOTE": build_note,
                 "RAW_TOP": str((a.sliced / sl_top.name).resolve()), "RAW_TRAJ": str((a.sliced / "trajectory.dat").resolve()),
                 "N_NT": f"{n:,}", "N_EP": str(len(traps))}.items():
        html = html.replace("{{" + k + "}}", v)
    assert not re.search(r"\{\{[A-Z_]+\}\}", html)
    html = html.replace("__SIM_DATA__", json.dumps(data, separators=(",", ":")))
    a.out.write_text(html)

    print(f"wrote {a.out} ({a.out.stat().st_size / 1e6:.2f} MB): {n} nt, {ns} strands, {len(traps)} traps, {len(frames)} frames, "
          f"steps 0-{steps[-1]}, {'with' if a.control else 'no'} unsliced baseline")
    for name, key in [("sliced, all", "rmsd_all"), ("sliced, cut ends", "rmsd_ep"), ("unsliced, all", "ctrl_rmsd_all"), ("unsliced, cut ends", "ctrl_rmsd_ep")]:
        if key in data:
            v = np.array(data[key]); print(f"  RMSD from start, {name:18s} mean over frames {v[1:].mean():.3f}  final {v[-1]:.3f} SU")


if __name__ == "__main__":
    main()
