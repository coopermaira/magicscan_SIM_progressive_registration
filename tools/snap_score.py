#!/usr/bin/env python3
"""Whole-slide check 4 for E1 tile snapping: per-cell P(cancer) of arm A (<id>) vs arm B (<id>__snapE1), using the
existing LOCO heads of the fold that held each case out (8 seeds, softmax averaged), on every labelled E1 cell of the
slides. Features come from distill/extract_fm_bench.py run on each arm (h5 named <id>.h5 in FEAT_A / FEAT_B).

The heads were trained on unsnapped images, so this check is biased towards arm A: a gain for B is strong evidence, a
small loss is not evidence against snapping.

Reports per encoder: AUC and log-loss on these slides for A and B, the paired per-cell difference (B - A) by label,
and the share of cells whose P(cancer) moved by more than 0.05.

usage: snap_score.py --heads mw_e1_twin2i_080k_e1reg --dim 768 --feat_a DIR --feat_b DIR --out TSV ID [ID ...]
"""
import argparse
import csv
import json

import h5py
import numpy as np
import torch
import sys

sys.path.insert(0, "/sci-it/hosts/olympus/magicscan/SIM/2_MS_mil/code/mil")
LABELS = "/sci-it/hosts/olympus/magicscan/SIM/2_MS_mil/labels/loco/loco_labels_e1_heunion_g44"
RUNS = "/sci-it/hosts/olympus/magicscan/SIM/3_SIM_foundation_model/bench_prostate/runs_loco"
SEEDS = (42, 1, 7, 2, 3, 4, 5, 6)


def auc(p, y):
    p, y = np.asarray(p), np.asarray(y)
    o = np.argsort(p, kind="mergesort"); r = np.empty(len(p)); r[o] = np.arange(1, len(p) + 1)
    n1 = y.sum(); n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else float("nan")


def logloss(p, y):
    p = np.clip(np.asarray(p), 1e-6, 1 - 1e-6)
    return float(-np.mean(np.asarray(y) * np.log(p) + (1 - np.asarray(y)) * np.log(1 - p)))


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ids", nargs="+")
    ap.add_argument("--heads", required=True)
    ap.add_argument("--dim", type=int, required=True)
    ap.add_argument("--feat_a", required=True)
    ap.add_argument("--feat_b", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import joint_stage2 as J
    J.EMBED_DIM = a.dim
    rows = []
    for sid in a.ids:
        case = sid.split("_")[0]
        hs = []
        for s in SEEDS:
            c = torch.load(f"{RUNS}/{a.heads}_s{s}_{case}/best_model.pt", map_location="cpu", weights_only=False)
            h = J.build_mil_head("abmil", 2).cuda().eval(); h.load_state_dict(c.get("model", c)); hs.append(h)
        lab = [r for r in csv.DictReader(open(f"{LABELS}/fold_{case}/labels/coarse_labels_2mm.tsv"), delimiter="\t")
               if r["slide_id"] == sid]
        with h5py.File(f"{LABELS}/fold_{case}/bag_mapping/2mm/{sid}.h5", "r") as f:
            idx = f["indices"][:]
        F = {}
        for arm, d in (("A", a.feat_a), ("B", a.feat_b)):
            with h5py.File(f"{d}/{sid}.h5", "r") as f:
                F[arm] = f["features"][:]
        for r in lab:
            ids = idx[int(r["h5_idx"])]; ids = ids[ids >= 0]
            if not len(ids):
                continue
            p = {}
            for arm in ("A", "B"):
                x = torch.from_numpy(F[arm][ids]).float().cuda().unsqueeze(0)
                p[arm] = float(np.mean([torch.softmax(h(x)[0]["logits"].float(), 1)[0, 1].item() for h in hs]))
            rows.append(dict(slide=sid, col=r["grid_col"], row=r["grid_row"], label=int(r["label"]),
                             class_name=r["class_name"], p_A=p["A"], p_B=p["B"]))
    with open(a.out, "w") as f:
        f.write("\t".join(rows[0]) + "\n")
        f.writelines("\t".join(str(r[k]) for k in rows[0]) + "\n" for r in rows)
    Y = np.array([r["label"] for r in rows]); PA = np.array([r["p_A"] for r in rows]); PB = np.array([r["p_B"] for r in rows])
    d = PB - PA
    summ = dict(heads=a.heads, cells=len(rows), cancer=int(Y.sum()), AUC_A=auc(PA, Y), AUC_B=auc(PB, Y),
                logloss_A=logloss(PA, Y), logloss_B=logloss(PB, Y),
                dP_cancer_mean=float(d[Y == 1].mean()) if Y.sum() else None, dP_benign_mean=float(d[Y == 0].mean()),
                cells_moved_gt_0p05_pct=float(100 * (np.abs(d) > 0.05).mean()),
                per_slide={s: dict(AUC_A=auc(PA[m], Y[m]), AUC_B=auc(PB[m], Y[m]), mean_abs_dP=float(np.abs(d[m]).mean()))
                           for s in a.ids for m in [np.array([r["slide"] == s for r in rows])]})
    json.dump(summ, open(a.out.rsplit(".", 1)[0] + "_summary.json", "w"), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
