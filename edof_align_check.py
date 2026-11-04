#!/usr/bin/env python3
"""Alignment quality of finished MagicScan EDOF images (see msprog/check.py for the metrics).

Image spec: '<dataset_dir>:<image_id>' for fm-builder datasets (grayscale/<id>_470.tif, _528.tif, z variants
<id>__z<k>_*, tissue_mask/<id>_patches.h5, z-map in <root>/_qc/<dataset>/flatfield/<id>_edof_zmap.png), or
'<nuc.tif>,<stro.tif>[,<zmap.png>]' for any other pair of grayscale TIFFs.

  run       edof_align_check.py run SPEC [SPEC ...] --out DIR
  compare   edof_align_check.py compare A_SPEC B_SPEC --out DIR      (B is measured on A's windows)
  selftest  edof_align_check.py selftest SPEC                         (inject known shifts, check the metrics)
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from msprog import check as C  # noqa: E402


def cmd_run(a):
    rc = 0
    for spec in a.spec:
        try:
            im = C.resolve(spec)
            wins = C.read_windows(a.windows) if a.windows else None
            rows, shape = C.measure(im, wins, a.workers, a.max_windows)
            s = C.summarize(im, rows, shape, a.mpp)
            C.write_outputs(a.out, s, rows, im)
            print(json.dumps({k: s[k] for k in ("image", "n_windows", "channel", "planes", "seam", "sharpness")}))
        except Exception as e:                       # keep going over a list of images
            print(f"ERROR {spec}: {type(e).__name__}: {e}", file=sys.stderr)
            rc = 1
    return rc


def cmd_compare(a):
    ia, ib = C.resolve(a.a), C.resolve(a.b)
    ra, shape_a = C.measure(ia, None, a.workers, a.max_windows)
    rb, shape_b = C.measure(ib, [(r["y"], r["x"]) for r in ra], a.workers)
    if shape_a != shape_b:
        print(f"note: image sizes differ ({shape_a} vs {shape_b}); windows are matched by position", file=sys.stderr)
    sa, sb = C.summarize(ia, ra, shape_a, a.mpp), C.summarize(ib, rb, shape_b, a.mpp)
    oa, ob = os.path.join(a.out, "A"), os.path.join(a.out, "B")
    C.write_outputs(oa, sa, ra, ia)
    C.write_outputs(ob, sb, rb, ib)
    rows = C.compare(sa, sb)
    w = max(len(r[0]) for r in rows)
    lines = [f"A = {a.a}", f"B = {a.b}", "", f"{'metric':{w}s}  {'A':>10s}  {'B':>10s}  {'B-A':>8s}  verdict"]
    lines += [f"{r[0]:{w}s}  {str(r[1]):>10s}  {str(r[2]):>10s}  {str(r[3]):>8s}  {r[4]}" for r in rows]
    txt = "\n".join(lines)
    print(txt)
    with open(os.path.join(a.out, "compare.txt"), "w") as f:
        f.write(txt + "\n")
    with open(os.path.join(a.out, "compare.json"), "w") as f:
        json.dump(dict(A=sa, B=sb, rows=rows), f, indent=1)
    return 0


def cmd_selftest(a):
    from msprog import selftest
    verdict, _ = selftest.run(a.spec, a.n_windows)
    return 0 if all(verdict.values()) else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="measure one or more images")
    r.add_argument("spec", nargs="+")
    r.add_argument("--windows", help="window list (an earlier *_align_windows.tsv) instead of the tissue grid")
    c = sub.add_parser("compare", help="measure A, then B on A's windows, and print the deltas")
    c.add_argument("a")
    c.add_argument("b")
    t = sub.add_parser("selftest", help="inject known shifts into an image's own windows")
    t.add_argument("spec")
    t.add_argument("--n_windows", type=int, default=6)
    for p in (r, c):
        p.add_argument("--out", required=True)
        p.add_argument("--max_windows", type=int, default=600)
    for p in (r, c, t):
        p.add_argument("--workers", type=int, default=16)
        p.add_argument("--mpp", type=float, default=0.7)
    a = ap.parse_args()
    return {"run": cmd_run, "compare": cmd_compare, "selftest": cmd_selftest}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
