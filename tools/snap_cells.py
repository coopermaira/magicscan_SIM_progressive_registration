#!/usr/bin/env python3
"""Pick the 2 mm E1 cells whose registered planes a --snap e1 build saves for the boundary-sensitivity probe.

Per slide: N cancer and N benign cells, drawn at random (seed 0) from the cells with the most patches (the bag
width caps num_patches at 256, so fully tiled cells tie; the draw spreads them over the slide), from the slide's own
LOCO test fold
(SIM/2_MS_mil/labels/loco/loco_labels_uniform2mm_20x_g40/fold_<case>/labels/coarse_labels_2mm.tsv).
Chosen before any snapped image exists, so the probe set does not depend on the result.

usage: snap_cells.py VA030_S1 VA034_S3 VA026_S5 --n 6 --out cells.tsv
"""
import argparse
import csv
import random

LABELS = "/sci-it/hosts/olympus/magicscan/SIM/2_MS_mil/labels/loco/loco_labels_uniform2mm_20x_g40"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("slides", nargs="+")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--labels", default=LABELS)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = []
    for sid in a.slides:
        case = sid.split("_")[0]
        cells = [r for r in csv.DictReader(open(f"{a.labels}/fold_{case}/labels/coarse_labels_2mm.tsv"), delimiter="\t")
                 if r["slide_id"] == sid and r["fold_0"] == "Testing"]
        for lab in ("cancer", "benign"):
            pool = sorted((r for r in cells if r["class_name"] == lab), key=lambda r: (-int(r["num_patches"]),
                          int(r["grid_row"]), int(r["grid_col"])))
            top = [r for r in pool if int(r["num_patches"]) == int(pool[0]["num_patches"])] if pool else []
            c = random.Random(f"{sid}:{lab}").sample(top, a.n) if len(top) >= a.n else pool[:a.n]
            rows += [(sid, r["grid_col"], r["grid_row"], lab, r["num_patches"]) for r in c]
            print(f"{sid} {lab}: {len(c)} cells, patches {[int(r[4]) for r in rows if r[0] == sid and r[3] == lab]}")
    with open(a.out, "w") as f:
        f.write("image_id\tcol\trow\tlabel\tnum_patches\n")
        f.writelines("\t".join(r) + "\n" for r in rows)


if __name__ == "__main__":
    main()
