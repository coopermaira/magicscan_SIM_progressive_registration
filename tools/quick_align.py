#!/usr/bin/env python3
"""Fast alignment survey from raw strips: for each slide, a row band from the middle of its central strips is
loaded, every adjacent pass pair is measured as acquired (470 z<->z+1, 528 z<->z+1, 470<->528 same z), the band
is registered, and the pairs are measured again. No EDOF is built.

Writes OUT/quick_align.jsonl (one registration report per slide band) and OUT/quick_align.tsv (one row per slide).

usage: quick_align.py SLIDES_TSV --out DIR [--strips 3] [--rows 6144] [--workers 28]
SLIDES_TSV: image_id <TAB> slice dir [<TAB> name filter or -]
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import tifffile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
from msprog import edof as E, register as R  # noqa: E402
import rebuild_edof as RB  # noqa: E402


def rows_of(entry, src, y0, y1):
    if src == "recon":
        a = RB._arr(entry["recon"])
        return np.clip(np.asarray(a[y0:y1], np.float32), 0, None)
    x = None
    for k in sorted(entry["phases"]):
        a = np.asarray(RB._arr(entry["phases"][k])[y0:y1], np.float32)
        x = a * a if x is None else x[:a.shape[0]] + a[:x.shape[0]] ** 2
    return np.sqrt(x / len(entry["phases"]))


def med(v):
    v = [x for x in v if x is not None and np.isfinite(x)]
    return round(float(np.median(v)), 2) if v else None


def mx(v):
    v = [x for x in v if x is not None and np.isfinite(x)]
    return round(float(np.max(v)), 2) if v else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("slides")
    ap.add_argument("--out", required=True)
    ap.add_argument("--strips", type=int, default=3)
    ap.add_argument("--rows", type=int, default=6144)
    ap.add_argument("--workers", type=int, default=28)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    jl = open(os.path.join(a.out, "quick_align.jsonl"), "a")
    rows = []
    for line in open(a.slides):
        f = line.rstrip("\n").split("\t")
        if not f[0] or f[0].startswith("#"):
            continue
        sid, d = f[0], f[1]
        filt = f[2] if len(f) > 2 and f[2] not in ("", "-") else None
        t0 = time.time()
        try:
            raw = RB.discover(d, filt)
            P = RB.plan(raw, "auto")
        except Exception as e:
            print(f"{sid}: discovery failed: {e}", flush=True)
            continue
        strips = sorted(P)
        mid = len(strips) // 2
        pick = strips[max(0, mid - a.strips // 2):max(0, mid - a.strips // 2) + a.strips]
        acc = {k: [] for k in ("cb", "cb90", "zb", "zb90", "ob", "ob90", "sy", "sx", "ca", "ca90", "za", "za90")}
        for s in pick:
            pairs, src = P[s]
            first = raw[s]["470"][pairs[0][0]]
            H = tifffile.TiffFile(first["recon"] if src == "recon" else first["phases"][0]).pages[0].shape[0]
            y0 = max(0, H // 2 - a.rows // 2)
            y1 = min(H, y0 + a.rows)
            ph = [E.destripe(rows_of(raw[s]["470"][z4], src, y0, y1)) for z4, _ in pairs]
            pe = [E.destripe(rows_of(raw[s]["528"][z5], src, y0, y1)) for _, z5 in pairs]
            h = min(x.shape[0] for x in ph + pe)
            w = min(x.shape[1] for x in ph + pe)
            ph, pe = [x[:h, :w] for x in ph], [x[:h, :w] for x in pe]
            _, _, rep = R.register_strip(ph, pe, workers=a.workers, log=lambda *x: None)
            rep.update(slide=sid, strip=s, rows=[y0, y1], width=w, source=src, z_470=[p[0] for p in pairs])
            jl.write(json.dumps(rep, default=float) + "\n")
            jl.flush()
            for e, st in (rep.get("edges") or {}).items():
                b, af = st.get("before_um") or {}, st.get("after_um") or {}
                k = "c" if st["kind"] == "c" else "z"
                acc[k + "b"].append(b.get("median"))
                acc[k + "b90"].append(b.get("p90"))
                acc[k + "a"].append(af.get("median"))
                acc[k + "a90"].append(af.get("p90"))
                if k == "z":
                    acc["sy"].append(st.get("stretch_y_median_pct"))
                    acc["sx"].append(st.get("stretch_x_median_pct"))
            apl = rep.get("applied_plane_shift_um") or {}
            ks = sorted(apl, key=int)
            for kk in ({ks[0], ks[-1]} if ks else set()):
                if apl[kk] and int(kk) != rep.get("ref_plane"):
                    acc["ob"].append(apl[kk]["median"])
                    acc["ob90"].append(apl[kk]["p90"])
        row = dict(slide=sid, n_strips=len(strips), strips_measured=",".join(map(str, pick)),
                   planes=len(P[pick[0]][0]) if pick else 0, source=P[pick[0]][1] if pick else "",
                   chan_before_med_um=med(acc["cb"]), chan_before_p90_um=mx(acc["cb90"]),
                   adj_plane_before_med_um=med(acc["zb"]), adj_plane_before_p90_um=mx(acc["zb90"]),
                   outer_plane_vs_ref_med_um=med(acc["ob"]), outer_plane_vs_ref_p90_um=mx(acc["ob90"]),
                   stretch_along_scan_pct=med(acc["sy"]), stretch_across_pct=med(acc["sx"]),
                   chan_after_med_um=med(acc["ca"]), chan_after_p90_um=mx(acc["ca90"]),
                   plane_after_med_um=med(acc["za"]), plane_after_p90_um=mx(acc["za90"]),
                   seconds=round(time.time() - t0))
        rows.append(row)
        print("\t".join(f"{k}={v}" for k, v in row.items()), flush=True)
        with open(os.path.join(a.out, f"quick_align_{os.path.basename(a.slides)}.tsv"), "w") as fo:
            fo.write("\t".join(rows[0]) + "\n")
            fo.writelines("\t".join("" if r[k] is None else str(r[k]) for k in rows[0]) + "\n" for r in rows)


if __name__ == "__main__":
    main()
