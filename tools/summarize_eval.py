#!/usr/bin/env python3
"""Collect a rebuild run (tools/rebuild_edof.py + edof_align_check.py compare per slide) into tables.

  edges.tsv    one row per slide, strip and adjacent pass pair: kind (z = same channel z<->z+1 nuclear/stromal,
               c = 470<->528 same plane), shift before registration (median, p90, max um), median shift
               direction, stretch along / across the scan, residual after registration
  slides.tsv   one row per slide: as-acquired misalignment (channel pairs, adjacent planes, outermost plane vs the
               reference plane, stretch), residual after registration, and the EDOF check of the unregistered
               ('off') vs registered build: 470-528 offset, seam continuity per channel, strip-boundary offset,
               sharpness ratio
  summary.md   slides.tsv as a readable table

usage: summarize_eval.py RUN_DIR [RUN_DIR ...] --out DIR
"""
import argparse
import glob
import json
import os

import numpy as np

MPP = 0.7


def med(v):
    v = [x for x in v if x is not None and np.isfinite(x)]
    return round(float(np.median(v)), 2) if v else None


def mx(v):
    v = [x for x in v if x is not None and np.isfinite(x)]
    return round(float(np.max(v)), 2) if v else None


def g(d, *ks):
    for k in ks:
        if not isinstance(d, dict) or k not in d or d[k] is None:
            return None
        d = d[k]
    return d


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    erows, srows = [], []
    for run in a.runs:
        for rj in sorted(glob.glob(os.path.join(run, "*", "pearson", "*_register.json"))):
            sid = os.path.basename(rj)[:-len("_register.json")]
            R = json.load(open(rj))
            cmpf = os.path.join(os.path.dirname(os.path.dirname(rj)), "compare", "compare.json")
            C = json.load(open(cmpf)) if os.path.exists(cmpf) else {}
            acc = {"c": [], "z": [], "c90": [], "z90": [], "zmax": [], "sx": [], "sy": [], "rz": [], "rc": [],
                   "rz90": [], "rc90": [], "outer": [], "outer90": [], "chanappl": []}
            planes = []
            for s, rep in sorted(R["strips"].items(), key=lambda t: int(t[0])):
                n = rep.get("n_planes", 0)
                planes.append(n)
                for e, st in (rep.get("edges") or {}).items():
                    b, af = st.get("before_um") or {}, st.get("after_um") or {}
                    a_, b_ = (int(x) for x in e.split("-"))
                    kind = st["kind"] if st["kind"] == "c" else ("z470" if a_ % 2 == 0 else "z528")
                    erows.append(dict(slide=sid, strip=int(s), edge=e, kind=kind, plane=a_ // 2,
                                      before_median_um=b.get("median"), before_p90_um=b.get("p90"),
                                      before_max_um=b.get("max"), dy_median_px=st.get("before_dy_median_px"),
                                      dx_median_px=st.get("before_dx_median_px"),
                                      stretch_y_pct=st.get("stretch_y_median_pct"),
                                      stretch_x_pct=st.get("stretch_x_median_pct"),
                                      after_median_um=af.get("median"), after_p90_um=af.get("p90"),
                                      n_nodes=b.get("n")))
                    k = "c" if st["kind"] == "c" else "z"
                    acc[k].append(b.get("median"))
                    acc[k + "90"].append(b.get("p90"))
                    if k == "z":
                        acc["zmax"].append(b.get("p90"))
                        acc["sx"].append(st.get("stretch_x_median_pct"))
                        acc["sy"].append(st.get("stretch_y_median_pct"))
                    acc["r" + k].append(af.get("median"))
                    acc["r" + k + "90"].append(af.get("p90"))
                ap_ = rep.get("applied_plane_shift_um") or {}
                if ap_:
                    ks = sorted(ap_, key=lambda t: int(t))
                    for kk in (ks[0], ks[-1]):
                        if ap_[kk] and int(kk) != rep.get("ref_plane"):
                            acc["outer"].append(ap_[kk]["median"])
                            acc["outer90"].append(ap_[kk]["p90"])
                for v in (rep.get("applied_channel_shift_um") or {}).values():
                    if v:
                        acc["chanappl"].append(v["median"])
            A, B = C.get("A", {}), C.get("B", {})
            row = dict(slide=sid, strips=len(R["strips"]), planes=f"{min(planes)}-{max(planes)}" if planes else "",
                       skipped_strips=len(R.get("skipped_strips", [])),
                       chan_before_median_um=med(acc["c"]), chan_before_p90_um=mx(acc["c90"]),
                       plane_adjacent_before_median_um=med(acc["z"]), plane_adjacent_before_p90_um=mx(acc["z90"]),
                       outer_plane_vs_ref_median_um=med(acc["outer"]), outer_plane_vs_ref_p90_um=mx(acc["outer90"]),
                       stretch_y_median_pct=med(acc["sy"]), stretch_x_median_pct=med(acc["sx"]),
                       chan_after_median_um=med(acc["rc"]), chan_after_p90_um=mx(acc["rc90"]),
                       plane_after_median_um=med(acc["rz"]), plane_after_p90_um=mx(acc["rz90"]),
                       edof_chan_off_median_px=g(A, "channel", "median_px"),
                       edof_chan_off_p90_px=g(A, "channel", "p90_px"),
                       edof_chan_reg_median_px=g(B, "channel", "median_px"),
                       edof_chan_reg_p90_px=g(B, "channel", "p90_px"),
                       seam470_off=g(A, "seam", "median"), seam470_reg=g(B, "seam", "median"),
                       seam528_off=g(A, "seam_stro", "median"), seam528_reg=g(B, "seam_stro", "median"),
                       strip_dy470_off_median_px=g(A, "strips", "nuc", "median_dy_px"),
                       strip_dy470_reg_median_px=g(B, "strips", "nuc", "median_dy_px"),
                       strip_dy470_off_p90_px=g(A, "strips", "nuc", "p90_dy_px"),
                       strip_dy470_reg_p90_px=g(B, "strips", "nuc", "p90_dy_px"),
                       strip_cont470_off=g(A, "strips", "nuc", "continuity_median"),
                       strip_cont470_reg=g(B, "strips", "nuc", "continuity_median"),
                       sharpness_ratio=(round(B["sharpness"]["median"] / A["sharpness"]["median"], 3)
                                        if g(A, "sharpness", "median") and g(B, "sharpness", "median") else None))
            srows.append(row)
    for name, rows in (("edges.tsv", erows), ("slides.tsv", srows)):
        if not rows:
            continue
        cols = list(rows[0])
        with open(os.path.join(a.out, name), "w") as f:
            f.write("\t".join(cols) + "\n")
            for r in rows:
                f.write("\t".join("" if r[c] is None else str(r[c]) for c in cols) + "\n")
    if srows:
        cols = [("slide", "slide"), ("planes", "planes"),
                ("chan_before_median_um", "470-528 before med um"), ("chan_before_p90_um", "p90"),
                ("plane_adjacent_before_median_um", "adj. planes before med um"),
                ("outer_plane_vs_ref_median_um", "outer plane vs ref med um"), ("outer_plane_vs_ref_p90_um", "p90"),
                ("stretch_y_median_pct", "stretch along scan %"),
                ("chan_after_median_um", "470-528 after med um"), ("plane_after_median_um", "planes after med um"),
                ("edof_chan_off_median_px", "EDOF 470-528 off med px"), ("edof_chan_reg_median_px", "reg med px"),
                ("seam470_off", "seam 470 off"), ("seam470_reg", "reg"), ("seam528_off", "seam 528 off"),
                ("seam528_reg", "reg"), ("strip_dy470_off_median_px", "strip dy off px"),
                ("strip_dy470_reg_median_px", "reg"), ("sharpness_ratio", "sharpness reg/off")]
        lines = ["| " + " | ".join(h for _, h in cols) + " |", "|" + "---|" * len(cols)]
        lines += ["| " + " | ".join("" if r[k] is None else str(r[k]) for k, _ in cols) + " |" for r in srows]
        open(os.path.join(a.out, "summary.md"), "w").write("\n".join(lines) + "\n")
        print("\n".join(lines))


if __name__ == "__main__":
    main()
