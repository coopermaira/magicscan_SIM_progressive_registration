#!/usr/bin/env python3
"""Register one real strip (a row range of it) straight from the raw 3-phase frames and measure the result.

Loads SIM = RMS of the 3 phases per plane and channel (as build_ms.py does, without its flat-field), runs
msprog.register.register_strip, then measures with the check's window matcher, before and after registration:
  channel  470 <-> 528 offset per plane
  planes   nuclear plane k <-> reference plane, for every k
on a grid of 1,024 px tissue windows. Writes before/after numbers (JSON) and overlay crops (reference plane in
magenta, outermost plane in green) for the windows with the largest initial plane offset.

usage: try_strip.py RAW_SLICE_DIR --strip 3 --rows 20480 32768 [--planes 2 3 4 5 6] --out DIR
"""
import argparse
import json
import os
import re
import sys
import time

import cv2
import numpy as np
import tifffile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from msprog import check as C, pearson as P, register as R  # noqa: E402

# Va026_S4_location1_470_20mm_s_3_0_0.tiff  |  VA039_cartridge1_strip3_z80.100000_470nm_0.tiff
LOC = re.compile(r"_loc(?:ation)?(\d+)_(470|528)_.*_(\d+)_0_(\d)\.tiff?$", re.I)
STRZ = re.compile(r"_strip(\d+)_z([\d.]+)_(470|528)nm_(\d)\.tiff?$", re.I)


def discover(d, strip):
    found = {}
    for f in os.listdir(d):
        m = LOC.search(f)
        if m and int(m.group(3)) == strip:
            found.setdefault((float(m.group(1)), m.group(2)), {})[int(m.group(4))] = os.path.join(d, f)
            continue
        m = STRZ.search(f)
        if m and int(m.group(1)) == strip:
            found.setdefault((float(m.group(2)), m.group(3)), {})[int(m.group(4))] = os.path.join(d, f)
    zs = sorted({z for z, c in found if (z, "470") in found and (z, "528") in found})
    return found, zs


def sim_rows(phases, y0, y1):
    acc = None
    for k in sorted(phases):
        with tifffile.TiffFile(phases[k]) as t:
            pg = t.pages[0]
            a = np.memmap(phases[k], np.dtype(t.byteorder + "u2"), "r", int(pg.dataoffsets[0]), pg.shape)
            x = np.asarray(a[y0:y1], np.float32)
        acc = x * x if acc is None else acc + x * x
    s = np.sqrt(acc / len(phases))
    return s - np.median(s[::16], axis=0)[None]                       # column destripe


def tissue_windows(img, step=1024):
    m = cv2.GaussianBlur(img, (0, 0), 8)
    thr = np.percentile(m, 60)
    wins = []
    for y in range(C.MARGIN, img.shape[0] - step - C.MARGIN + 1, step):
        for x in range(C.MARGIN, img.shape[1] - step - C.MARGIN + 1, step):
            if (m[y:y + step, x:x + step] > thr).mean() > 0.5:
                wins.append((y, x))
    return wins


def offsets(A, B, wins):
    out = []
    M = C.MARGIN
    for y, x in wins:
        m = P.match_window(A[y:y + C.WIN, x:x + C.WIN], B[y - M:y + C.WIN + M, x - M:x + C.WIN + M], M)
        out.append(np.hypot(m["dy"], m["dx"]) if C.accepted(m) else np.nan)
    return np.array(out)


def st(v):
    v = v[np.isfinite(v)]
    return dict(n=int(v.size), median_px=round(float(np.median(v)), 2) if v.size else None,
                p90_px=round(float(np.percentile(v, 90)), 2) if v.size else None,
                frac_gt_2px=round(float((v > 2).mean()), 3) if v.size else None)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("raw_dir")
    ap.add_argument("--strip", type=int, required=True)
    ap.add_argument("--rows", type=int, nargs=2, required=True)
    ap.add_argument("--planes", type=int, nargs="*", default=None, help="indices into the sorted z list")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    cv2.setNumThreads(1)
    found, zs = discover(a.raw_dir, a.strip)
    if a.planes:
        zs = [zs[i] for i in a.planes]
    y0, y1 = a.rows
    t0 = time.time()
    ph = [sim_rows(found[(z, "470")], y0, y1) for z in zs]
    pe = [sim_rows(found[(z, "528")], y0, y1) for z in zs]
    print(f"loaded {len(zs)} planes x 2 ch, rows {y0}-{y1}, {ph[0].shape} in {time.time() - t0:.0f}s", flush=True)
    ph_r, pe_r, rep = R.register_strip(ph, pe, workers=a.workers)
    r = len(zs) // 2
    wins = tissue_windows(ph[r])
    print(f"{len(wins)} tissue windows", flush=True)
    res = dict(raw_dir=a.raw_dir, strip=a.strip, rows=[y0, y1], z=zs, report=rep, n_windows=len(wins),
               before={}, after={})
    for tag, H_, E_ in (("before", ph, pe), ("after", ph_r, pe_r)):
        for k in range(len(zs)):
            res[tag][f"channel_z{k}"] = st(offsets(H_[k], E_[k], wins))
            if k != r:
                res[tag][f"plane_z{k}_vs_ref"] = st(offsets(H_[r], H_[k], wins))
    json.dump(res, open(os.path.join(a.out, "try_strip.json"), "w"), indent=1, default=str)
    for key in res["before"]:
        b, f = res["before"][key], res["after"][key]
        print(f"{key:18s} before median {b['median_px']} p90 {b['p90_px']} (n {b['n']})   "
              f"after median {f['median_px']} p90 {f['p90_px']} (n {f['n']})")
    # overlays for the 4 windows with the largest initial outer-plane offset
    outer = 0 if r > 0 else len(zs) - 1
    d0 = offsets(ph[r], ph[outer], wins)
    order = np.argsort(-np.nan_to_num(d0, nan=-1))[:4]

    def n8(u):
        return np.clip(u / max(np.percentile(u, 99.5), 1e-6) * 255, 0, 255).astype(np.uint8)
    tiles = []
    for i in order:
        y, x = wins[i]
        row = []
        for H_ in (ph, ph_r):
            ref, mov = n8(H_[r][y:y + 512, x:x + 512]), n8(H_[outer][y:y + 512, x:x + 512])
            row.append(np.dstack([ref, mov, ref]))
        tiles.append(np.hstack(row))
    if tiles:
        cv2.imwrite(os.path.join(a.out, "overlay_before_after.jpg"), np.vstack(tiles))
    print(f"done in {time.time() - t0:.0f}s -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
