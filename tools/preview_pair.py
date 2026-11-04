#!/usr/bin/env python3
"""Side-by-side crops of two builds of a slide (A | B), 470 in magenta and 528 in green (aligned = white/grey
where both stain), for the windows where A was worst. Reads a finished `edof_align_check.py compare` directory.

  crops   the --n windows with the largest A channel offset, plus the --n with the lowest A seam continuity
          (470), plus strip boundaries with the largest A offset; each crop 1,024 px at full resolution,
          shown at --scale
  overview whole slide of A and B at ~1/32 (470 only), for orientation

usage: preview_pair.py COMPARE_DIR --out DIR [--n 4] [--scale 0.5]
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from msprog.tiffio import TiledReader, thumbnail  # noqa: E402

WIN = 1024


def read_tsv(p):
    if not os.path.exists(p):
        return []
    with open(p) as f:
        hdr = f.readline().rstrip("\n").split("\t")
        rows = []
        for line in f:
            v = line.rstrip("\n").split("\t")
            rows.append({k: (float(x) if x not in ("", "True", "False") else (x == "True" if x else np.nan))
                         for k, x in zip(hdr, v)})
    return rows


def n8(a, lo, hi):
    return np.clip((a - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)


def overlay(rn, rs, y, x, h, w, lims):
    a, b = rn.region(y, x, h, w), rs.region(y, x, h, w)
    m, g = n8(a, *lims[0]), n8(b, *lims[1])
    return np.dstack([m, g, m])                      # BGR: magenta = 470, green = 528


def label(img, text):
    img = img.copy()
    cv2.putText(img, text, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
    cv2.putText(img, text, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return img


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("compare_dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--scale", type=float, default=0.5)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    j = json.load(open(os.path.join(a.compare_dir, "compare.json")))
    A, B = j["A"], j["B"]
    iid = A["image"]
    R = {k: (TiledReader(s["files"]["nuc"]), TiledReader(s["files"]["stro"])) for k, s in (("A", A), ("B", B))}
    wa = read_tsv(os.path.join(a.compare_dir, "A", f"{iid}_align_windows.tsv"))
    wb = {(r["y"], r["x"]): r for r in read_tsv(os.path.join(a.compare_dir, "B", f"{iid}_align_windows.tsv"))}
    ba = read_tsv(os.path.join(a.compare_dir, "A", f"{iid}_align_strip_boundaries.tsv"))
    # display limits from A's tissue windows (same for both builds)
    smp = [np.concatenate([R["A"][c].region(int(r["y"]), int(r["x"]), 256, 256).ravel() for r in wa[::max(1, len(wa) // 40)]])
           for c in (0, 1)]
    lims = [(float(np.percentile(s, 1)), float(np.percentile(s, 99.5))) for s in smp]

    def chan_off(r):
        return np.hypot(r.get("chan_dy", np.nan), r.get("chan_dx", np.nan)) if r.get("chan_ok") is True else -1

    picks = [("channel", r) for r in sorted(wa, key=chan_off, reverse=True)[:a.n]]
    seam = [r for r in wa if np.isfinite(r.get("seam", np.nan))]
    picks += [("seam", r) for r in sorted(seam, key=lambda r: r["seam"])[:a.n]]
    tiles = []
    for kind, r in picks:
        y, x = int(r["y"]), int(r["x"])
        rb = wb.get((r["y"], r["x"]), {})
        ta = label(overlay(*R["A"], y, x, WIN, WIN, lims), f"A  {kind}: chan {chan_off(r):.1f}px seam {r.get('seam', np.nan):.2f}")
        tb = label(overlay(*R["B"], y, x, WIN, WIN, lims), f"B  chan {chan_off(rb):.1f}px seam {rb.get('seam', np.nan):.2f}")
        t = np.hstack([ta, np.full((WIN, 8, 3), 255, np.uint8), tb])
        tiles.append(t)
    bnd = [r for r in ba if r.get("nuc_ok") is True]
    for r in sorted(bnd, key=lambda r: abs(r["nuc_dy"]), reverse=True)[:a.n]:
        xb, y = int(r["xb"]), int(r["y"])
        ta = label(overlay(*R["A"], y, xb - WIN // 2, WIN, WIN, lims), f"A  strip boundary x={xb}: dy {r['nuc_dy']:.1f}px")
        tb = label(overlay(*R["B"], y, xb - WIN // 2, WIN, WIN, lims), "B")
        tiles.append(np.hstack([ta, np.full((WIN, 8, 3), 255, np.uint8), tb]))
    for i, t in enumerate(tiles):
        t = cv2.resize(t, None, fx=a.scale, fy=a.scale, interpolation=cv2.INTER_AREA)
        cv2.imwrite(os.path.join(a.out, f"{iid}_crop{i:02d}.jpg"), t, [cv2.IMWRITE_JPEG_QUALITY, 90])
    ov = []
    for k, s in (("A", A), ("B", B)):
        th, _ = thumbnail(s["files"]["nuc"], 2048)
        ov.append(label(cv2.cvtColor(n8(th, *np.percentile(th, (1, 99.5))), cv2.COLOR_GRAY2BGR), k))
    h = min(o.shape[0] for o in ov)
    cv2.imwrite(os.path.join(a.out, f"{iid}_overview.jpg"), np.hstack([o[:h] for o in ov]), [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(f"{iid}: {len(tiles)} crops -> {a.out}")


if __name__ == "__main__":
    main()
