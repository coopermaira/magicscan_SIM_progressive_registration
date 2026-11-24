#!/usr/bin/env python3
"""CPU checks of a --snap e1 build (tools/run_snap_e1.sh output). Arms: A = <id> (the builder's EDOF), B = <id>__snapE1.

  exposure  how many E1 inference tiles hold more than one z-plane in the production EDOF (A), overall, in labelled
            cells and per label; tiles cut by a strip boundary are counted separately (snapping cannot make them
            single-source). From _qc/<ds>/snap/<id>_snap_tiles.npz and the E1 coord h5.
  identity  arm A vs the existing dataset image (Prostate_VA_MS_SYBR_EO_reg): equal-pixel fraction and differences
            (A is the builder's own output; differences come from code changes since that build), and A vs B: the
            pixels that changed must be exactly the snapped ones.
  sharpness Tenengrad of arm B / arm A on the E1 tiles whose content changed (the price of a tile-constant plane).

usage: snap_eval.py exposure|identity|sharpness ROOT ID [ID ...] [--out DIR]
"""
import argparse
import collections
import csv
import json
import os
import sys

import h5py
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from msprog import snap as SN  # noqa: E402
from msprog.tiffio import TiledReader  # noqa: E402

DS = "Prostate_VA_MS_SYBR_EO_snapE1"
PROD = "/sci-it/hosts/olympus/magicscan/SIM/0_foundation_model_datasets/Prostate_VA_MS_SYBR_EO_reg"
COORD = "/sci-it/hosts/olympus/magicscan/SIM/0_Datasets/0.5_mpp/0_coord_h5"
LABELS = "/sci-it/hosts/olympus/magicscan/SIM/2_MS_mil/labels/loco/loco_labels_e1_heunion_g44"


def coords_of(sid):
    with h5py.File(f"{COORD}/{sid}_fullstack_he_patches.h5", "r") as f:
        return f["coords"][:]


def labels_of(sid):
    case = sid.split("_")[0]
    out = {}
    for r in csv.DictReader(open(f"{LABELS}/fold_{case}/labels/coarse_labels_2mm.tsv"), delimiter="\t"):
        if r["slide_id"] == sid:
            out[(int(r["grid_col"]), int(r["grid_row"]))] = r["class_name"]
    return out


def tile_table(root, sid):
    st = np.load(f"{root}/_qc/{DS}/snap/{sid}_snap_tiles.npz")
    T = collections.defaultdict(lambda: [0, 0, False, 1.0])        # n_px, max n_planes, straddle, min maj_frac
    for ty, tx, npx, nz, st_, mf in zip(st["ty"], st["tx"], st["n_px"], st["n_planes"], st["straddle"], st["maj_frac"]):
        t = T[(int(ty), int(tx))]
        t[0] += int(npx); t[1] = max(t[1], int(nz)); t[2] = t[2] or bool(st_); t[3] = min(t[3], float(mf))
    return T


def exposure(a):
    rows = []
    for sid in a.ids:
        T, C, L = tile_table(a.root, sid), coords_of(sid), labels_of(sid)
        per = collections.defaultdict(lambda: dict(n=0, multi=0, straddle=0, minority=[]))
        cells = collections.defaultdict(lambda: [0, 0])
        for x, y in C:
            k = (int(y) // 256, int(x) // 256)
            t = T.get(k)
            if t is None:
                continue
            lab = L.get((int(x) // 4000, int(y) // 4000), "unlabelled")
            for g in ("all", lab) + (("labelled",) if lab != "unlabelled" else ()):
                p = per[g]
                p["n"] += 1
                if t[2]:
                    p["straddle"] += 1
                elif t[1] > 1:
                    p["multi"] += 1
                    p["minority"].append(1 - t[3])
            if lab != "unlabelled":
                c = cells[(int(x) // 4000, int(y) // 4000, lab)]
                c[0] += 1; c[1] += int(t[1] > 1 and not t[2])
        for g, p in per.items():
            cl = [v for k, v in cells.items() if g in ("labelled", k[2])]
            rows.append(dict(slide=sid, group=g, tiles=p["n"],
                             multi_plane_pct=round(100 * p["multi"] / max(p["n"], 1), 2),
                             straddle_pct=round(100 * p["straddle"] / max(p["n"], 1), 2),
                             minority_px_median_pct=round(100 * float(np.median(p["minority"])), 1) if p["minority"] else None,
                             cells=len(cl) if cl else None,
                             cells_with_boundary_tile_pct=round(100 * np.mean([v[1] > 0 for v in cl]), 1) if cl else None,
                             boundary_tiles_per_cell_median_pct=round(100 * float(np.median([v[1] / v[0] for v in cl])), 1) if cl else None))
    _write(a, "exposure", rows)


def _bands(r, step=2048):
    for y in range(0, r.h, step):
        yield y, min(step, r.h - y)


def identity(a):
    rows = []
    for sid in a.ids:
        for c in ("470", "528"):
            A = TiledReader(f"{a.root}/{DS}/grayscale/{sid}_{c}.tif")
            B = TiledReader(f"{a.root}/{DS}/grayscale/{sid}{'__snapE1'}_{c}.tif")
            P = TiledReader(f"{PROD}/grayscale/{sid}_{c}.tif")
            r = dict(slide=sid, channel=c, shape_A=[A.h, A.w], shape_prod=[P.h, P.w])
            if (A.h, A.w) != (P.h, P.w):
                r["note"] = "shape differs from production"
            n = eqP = eqB = tis = 0
            dP, dB = [], []
            for y, h in _bands(A):
                a_ = A.region(y, 0, h, A.w)
                b_ = B.region(y, 0, h, A.w)
                t = a_ > 0
                n += a_.size; tis += int(t.sum())
                eqB += int((a_ == b_).sum())
                d = np.abs(a_ - b_)[t]
                dB.append(d[d > 0][::16])
                if (A.h, A.w) == (P.h, P.w):
                    p_ = P.region(y, 0, h, A.w)
                    eqP += int((a_ == p_).sum())
                    e = np.abs(a_ - p_)[t]
                    dP.append(e[e > 0][::16])
            dP = np.concatenate(dP) if dP else np.zeros(0)
            dB = np.concatenate(dB) if dB else np.zeros(0)
            r.update(px=n, A_eq_prod_pct=round(100 * eqP / n, 4) if dP.size or eqP else None,
                     A_vs_prod_absdiff_p50_p99=[float(np.percentile(dP, 50)), float(np.percentile(dP, 99))] if dP.size else [0, 0],
                     A_eq_B_pct=round(100 * eqB / n, 4),
                     A_vs_B_absdiff_p50_p99=[float(np.percentile(dB, 50)), float(np.percentile(dB, 99))] if dB.size else [0, 0])
            rows.append(r)
            print(json.dumps(r), flush=True)
    _write(a, "identity", rows)


def tenengrad(x):
    import cv2
    gx = cv2.Sobel(x, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(x, cv2.CV_32F, 0, 1, ksize=3)
    return float(np.mean(gx * gx + gy * gy))


def sharpness(a):
    rng = np.random.default_rng(0)
    rows = []
    for sid in a.ids:
        T, C = tile_table(a.root, sid), coords_of(sid)
        multi = [(int(x), int(y)) for x, y in C if (t := T.get((int(y) // 256, int(x) // 256))) and t[1] > 1 and not t[2]]
        pick = [multi[i] for i in rng.choice(len(multi), min(a.n, len(multi)), replace=False)] if multi else []
        for c in ("470", "528"):
            A = TiledReader(f"{a.root}/{DS}/grayscale/{sid}_{c}.tif")
            B = TiledReader(f"{a.root}/{DS}/grayscale/{sid}__snapE1_{c}.tif")
            rat = []
            for x, y in pick:
                (fx, lx), (fy, ly) = SN.tile_canvas_span(x // 256), SN.tile_canvas_span(y // 256)
                ta = tenengrad(A.region(int(fy), int(fx), int(ly - fy), int(lx - fx)))
                tb = tenengrad(B.region(int(fy), int(fx), int(ly - fy), int(lx - fx)))
                if ta > 0:
                    rat.append(tb / ta)
            rat = np.asarray(rat)
            rows.append(dict(slide=sid, channel=c, multi_plane_tiles=len(multi), measured=len(rat),
                             ratio_median=round(float(np.median(rat)), 4) if rat.size else None,
                             ratio_p10=round(float(np.percentile(rat, 10)), 4) if rat.size else None,
                             ratio_p90=round(float(np.percentile(rat, 90)), 4) if rat.size else None,
                             tiles_sharper_pct=round(100 * float((rat > 1).mean()), 1) if rat.size else None,
                             tiles_10pct_blurrier_pct=round(100 * float((rat < 0.9).mean()), 1) if rat.size else None))
            print(json.dumps(rows[-1]), flush=True)
    _write(a, "sharpness", rows)


def _write(a, name, rows):
    out = a.out or os.path.join(a.root, "eval")
    os.makedirs(out, exist_ok=True)
    json.dump(rows, open(os.path.join(out, f"{name}.json"), "w"), indent=1, default=str)
    if rows:
        cols = list(dict.fromkeys(k for r in rows for k in r))
        with open(os.path.join(out, f"{name}.tsv"), "w") as f:
            f.write("\t".join(cols) + "\n")
            f.writelines("\t".join("" if r.get(k) is None else str(r.get(k)) for k in cols) + "\n" for r in rows)
        print("\n".join("\t".join(str(r.get(k, "")) for k in cols) for r in rows))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["exposure", "identity", "sharpness"])
    ap.add_argument("root")
    ap.add_argument("ids", nargs="+")
    ap.add_argument("--out", default=None)
    ap.add_argument("--n", type=int, default=3000, help="sharpness: boundary tiles sampled per slide")
    a = ap.parse_args()
    {"exposure": exposure, "identity": identity, "sharpness": sharpness}[a.cmd](a)


if __name__ == "__main__":
    main()
