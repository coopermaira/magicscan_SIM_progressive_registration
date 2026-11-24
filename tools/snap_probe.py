#!/usr/bin/env python3
"""Boundary-sensitivity probe for E1 tile snapping (GPU; olympus-compute, Mei's prostate_env).

Inputs: a --snap e1 build (tools/run_snap_e1.sh): arm A <id>, arm B <id>__snapE1, and the probe-cell crops
_qc/<ds>/snap/probe/<id>/cell_c<col>_r<row>_s<strip>.npz (every registered plane before strip gain/normalisation, the
production and snapped plane maps). Tiles are rendered and embedded with the benchmark's own code, imported read-only:
distill/extract_fm_bench.py fm_tile (cubic resample of canvas [x/1.4, (x+256)/1.4)) and build(), distill/common.py
TileReader / pack8 / GRAY_SCALE.

1. Planes to final image space. Arm A's final pixels are an affine function of the pre-final production EDOF per strip
   and row band (strip gain x slide normalisation x 16-bit scale); it is fitted per 64-row band on the crop and applied
   to every plane, and its residual is reported.
2. Clean tiles: E1 tiles of the cell whose whole sampling footprint (fm_tile reads ceil(256/1.4)+6 px from floor(x/1.4)-2)
   has ONE production plane p and lies inside one strip crop. Each gets a neighbouring plane q = p+1 (else p-1) and
   these renders, embedded and compared with the unaltered tile (cosine distance):
     pure_q     the whole tile from q                  (what a different plane choice does)
     mix50_v    p | q, vertical boundary at the centre (what a mid-tile boundary does)
     mix25_v    q on the right quarter
     mix50_h    horizontal boundary at the centre
     edge       q only beyond the tile's right edge    (what a snapped boundary leaves: kernel bleed)
     shift      p shifted by 0.5 canvas px             (resampling noise floor)
   Boundary penalty = d(mix50_v) relative to the halfway point of p and q: d(mix50_v) - d(pure_q)/2 per tile.
3. Real boundaries: tiles whose production footprint holds >1 plane, arm A vs arm B (d(A,B)).
4. Cell P(cancer) with the LOCO heads that held the case out (8 seeds, softmax averaged), the cell's own bag
   (bag_mapping), every scenario substituting only its probed tiles' features: clean tiles -> each render; boundary
   tiles -> arm B. Delta vs arm A is reported per cell.

usage: snap_probe.py ROOT ID [ID ...] --out DIR [--encoders student,teacher] [--max_tiles_per_cell 120]
"""
import argparse
import collections
import csv
import glob
import json
import os
import sys
import time
import types

import h5py
import numpy as np
import torch

FM = "/sci-it/hosts/olympus/magicscan/SIM/3_SIM_foundation_model"
sys.path.insert(0, f"{FM}/distill")
sys.path.insert(0, "/sci-it/hosts/olympus/magicscan/SIM/2_MS_mil/code/mil")
DS = "Prostate_VA_MS_SYBR_EO_snapE1"
COORD = "/sci-it/hosts/olympus/magicscan/SIM/0_Datasets/0.5_mpp/0_coord_h5"
LABELS = "/sci-it/hosts/olympus/magicscan/SIM/2_MS_mil/labels/loco/loco_labels_e1_heunion_g44"
RUNS = f"{FM}/bench_prostate/runs_loco"
SEEDS = (42, 1, 7, 2, 3, 4, 5, 6)
ENC = {"student": dict(heads="mw_e1_twin2i_080k_e1reg", dim=768,
                       args=dict(encoder="student", ckpt="/scratch/scratch_nvme/mwang/fm_phase4/runs/twin_2i/ckpt_080000.pt",
                                 view="gray")),
       "teacher": dict(heads="mw_e1_7b_packed", dim=4096, args=dict(encoder="dinov3_7b", ckpt="none", view="packed"))}
VARIANTS = ("pure_q", "mix50_v", "mix25_v", "mix50_h", "edge", "shift")
FOOT = int(np.ceil(256 / 1.4)) + 6     # fm_tile source window


class Arr:
    """fm_tile reader over an in-memory canvas crop (canvas origin ox, oy)."""

    def __init__(self, a, ox, oy):
        self.a, self.ox, self.oy = a, ox, oy

    def region(self, x, y, size):
        return self.a[y - self.oy:y - self.oy + size, x - self.ox:x - self.ox + size]


def final_space(prod, final, planes, band=64):
    """Fit final = g * prod + c per row band (unclipped pixels), apply to the planes -> uint16; returns planes, residual."""
    out = np.empty(planes.shape, np.uint16)
    res = []
    for y in range(0, prod.shape[0], band):
        p, f = prod[y:y + band].ravel(), final[y:y + band].ravel()
        m = (f > 0) & (f < 65535)
        if m.sum() < 100:
            g, c = 0.0, 0.0
        else:
            g, c = np.polyfit(p[m], f[m], 1)
            res.append(np.abs(g * p[m] + c - f[m]))
        out[:, y:y + band] = np.clip(np.rint(g * planes[:, y:y + band] + c), 0, 65535).astype(np.uint16)
    r = np.concatenate(res) if res else np.zeros(1)
    return out, float(np.median(r)), float(np.percentile(r, 99))


def load_parts(root, sid):
    cells = collections.defaultdict(list)
    for f in sorted(glob.glob(f"{root}/_qc/{DS}/snap/probe/{sid}/cell_c*_r*_s*.npz")):
        z = np.load(f)
        cells[(int(z["col"]), int(z["row"]))].append(f)
    return cells


def heads_for(enc, case):
    import joint_stage2 as J
    J.EMBED_DIM = ENC[enc]["dim"]
    hs = []
    for s in SEEDS:
        ck = f"{RUNS}/{ENC[enc]['heads']}_s{s}_{case}/best_model.pt"
        c = torch.load(ck, map_location="cpu", weights_only=False)
        h = J.build_mil_head("abmil", 2).cuda().eval()
        h.load_state_dict(c.get("model", c))
        hs.append(h)
    return hs


@torch.no_grad()
def p_cancer(hs, feats):
    f = torch.from_numpy(np.asarray(feats, np.float32)).cuda().unsqueeze(0)
    return float(np.mean([torch.softmax(h(f)[0]["logits"].float(), 1)[0, 1].item() for h in hs]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root")
    ap.add_argument("ids", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--encoders", default="student,teacher")
    ap.add_argument("--max_tiles_per_cell", type=int, default=120, help="clean tiles probed per cell (random, seeded)")
    ap.add_argument("--batch", type=int, default=64)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    import extract_fm_bench as X
    from common import TileReader, pack8, GRAY_SCALE
    rng = np.random.default_rng(0)
    tile_rows, real_rows, cell_rows, fit_rows = [], [], [], []
    for enc in a.encoders.split(","):
        ns = types.SimpleNamespace(arch="vitb", nuc_only=False, stro_only=False, uni_no_norm=False,
                                   input_stats="/scratch/scratch_nvme/mwang/fm_scale500k/shards/student_input_stats.json",
                                   **ENC[enc]["args"])
        _, fwd, dim = X.build(ns)

        def embed(tiles):            # tiles: list of (2, 256, 256) uint16, preprocessed exactly as extract_fm_bench.slide
            out = []
            for i in range(0, len(tiles), a.batch):
                b = tiles[i:i + a.batch]
                if ns.encoder == "student":
                    x = torch.from_numpy(np.stack(b).astype(np.float32)).cuda() / GRAY_SCALE
                else:
                    x = torch.from_numpy(np.stack([pack8(t[0], t[1]) for t in b])).cuda().permute(0, 3, 1, 2).float()
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                    out.append(fwd(x).float().cpu().numpy())
            return np.concatenate(out) if out else np.zeros((0, dim), np.float32)

        def cos_d(u, v):
            return 1 - np.sum(u * v, -1) / (np.linalg.norm(u, axis=-1) * np.linalg.norm(v, axis=-1) + 1e-12)

        for sid in a.ids:
            t0 = time.time()
            case = sid.split("_")[0]
            hs = heads_for(enc, case)
            with h5py.File(f"{COORD}/{sid}_fullstack_he_patches.h5", "r") as f:
                C = f["coords"][:]
            lab = {(int(r["grid_col"]), int(r["grid_row"])): r for r in
                   csv.DictReader(open(f"{LABELS}/fold_{case}/labels/coarse_labels_2mm.tsv"), delimiter="\t")
                   if r["slide_id"] == sid}
            with h5py.File(f"{LABELS}/fold_{case}/bag_mapping/2mm/{sid}.h5", "r") as f:
                bagidx = f["indices"][:]
            RA = [TileReader(f"{a.root}/{DS}/grayscale/{sid}_{c}.tif") for c in ("470", "528")]
            RB = [TileReader(f"{a.root}/{DS}/grayscale/{sid}__snapE1_{c}.tif") for c in ("470", "528")]
            for (col, row), files in load_parts(a.root, sid).items():
                r = lab.get((col, row))
                if r is None:
                    continue
                ids = bagidx[int(r["h5_idx"])]
                ids = ids[ids >= 0]
                feats_A = embed([np.stack([X.fm_tile(rd, int(C[i, 0]), int(C[i, 1])) for rd in RA]) for i in ids])
                pos = {int(i): k for k, i in enumerate(ids)}
                renders = {v: {} for v in VARIANTS}
                real_B = {}
                for fpath in files:
                    z = np.load(fpath)
                    px0, py0 = int(z["x0"]), int(z["y0"])
                    h, w = z["prod_nuc"].shape
                    finA = [RA[k].region(px0, py0, max(h, w))[:h, :w].astype(np.float32) for k in range(2)]
                    pl, fits = [], []
                    for k, ch in enumerate(("nuc", "stro")):
                        q, med, p99 = final_space(z[f"prod_{ch}"], finA[k], z[f"planes_{ch}"])
                        pl.append(q)
                        fit_rows.append(dict(encoder=enc, slide=sid, cell=f"{col},{row}", strip=int(z["strip"]), channel=ch,
                                             resid_median_dn=round(med, 3), resid_p99_dn=round(p99, 3)))
                    P = np.stack(pl, 1)                                   # (n_planes, 2, h, w) uint16
                    zp, zs = z["z_prod"], z["z_snap"]
                    nz = P.shape[0]
                    clean, real = [], []
                    for i in ids:
                        x, y = int(C[i, 0]), int(C[i, 1])
                        fx, fy = int(np.floor(x / 1.4)) - 2, int(np.floor(y / 1.4)) - 2
                        lx, ly = fx - px0, fy - py0
                        if lx < 0 or ly < 0 or lx + FOOT + 1 > w or ly + FOOT > h:      # +1: the 0.5 px shift render
                            continue
                        zz = zp[ly:ly + FOOT, lx:lx + FOOT]
                        if zz.min() == zz.max():
                            clean.append((int(i), x, y, int(zz[0, 0])))
                        else:
                            real.append((int(i), x, y))
                    sel = [clean[k] for k in rng.choice(len(clean), min(a.max_tiles_per_cell, len(clean)), replace=False)] if clean else []
                    for i, x, y, p in sel:
                        q = p + 1 if p + 1 < nz else p - 1
                        if q < 0:
                            continue
                        cx, cy = (x + 128) / 1.4 - px0, (y + 128) / 1.4 - py0
                        def mix(boundary, axis):
                            m = np.zeros((h, w), bool)
                            if axis == "v":
                                m[:, int(np.ceil(boundary - 0.5)):] = True     # pixels whose centre is past the line
                            else:
                                m[int(np.ceil(boundary - 0.5)):, :] = True
                            return np.where(m[None], P[q], P[p])
                        canv = {"pure_q": P[q], "mix50_v": mix(cx, "v"), "mix25_v": mix((x + 192) / 1.4 - px0, "v"),
                                "mix50_h": mix(cy, "h"), "edge": mix((x + 256) / 1.4 - px0, "v")}
                        for v, cv in canv.items():
                            renders[v][i] = np.stack([X.fm_tile(Arr(cv[k], px0, py0), x, y) for k in range(2)])
                        renders["shift"][i] = np.stack([X.fm_tile(Arr(P[p][k], px0, py0), x + 0.7, y) for k in range(2)])
                        # unaltered render from the planes must equal arm A's tile (checks step 1 and the geometry)
                        ref = np.stack([X.fm_tile(Arr(P[p][k], px0, py0), x, y) for k in range(2)])
                        renders.setdefault("_ref", {})[i] = ref
                    for i, x, y in real:
                        real_B[i] = np.stack([X.fm_tile(rd, x, y) for rd in RB])
                # embeddings
                E = {v: dict(zip(d.keys(), embed(list(d.values())))) for v, d in renders.items() if d}
                EB = dict(zip(real_B.keys(), embed(list(real_B.values())))) if real_B else {}
                for i in E.get("pure_q", {}):
                    fa = feats_A[pos[i]]
                    rowd = dict(encoder=enc, slide=sid, cell=f"{col},{row}", label=r["class_name"], tile=i,
                                ref_vs_A=float(cos_d(E["_ref"][i], fa)))
                    for v in VARIANTS:
                        rowd[v] = float(cos_d(E[v][i], fa))
                    rowd["boundary_penalty"] = rowd["mix50_v"] - rowd["pure_q"] / 2
                    tile_rows.append(rowd)
                for i, eb in EB.items():
                    real_rows.append(dict(encoder=enc, slide=sid, cell=f"{col},{row}", label=r["class_name"], tile=i,
                                          d_A_B=float(cos_d(eb, feats_A[pos[i]]))))
                pA = p_cancer(hs, feats_A)
                crow = dict(encoder=enc, slide=sid, cell=f"{col},{row}", label=r["class_name"], bag=len(ids),
                            probed_clean=len(E.get("pure_q", {})), boundary_tiles=len(EB), p_A=round(pA, 4))
                for v in VARIANTS:
                    f2 = feats_A.copy()
                    for i, e in E.get(v, {}).items():
                        f2[pos[i]] = e
                    crow[f"dp_{v}"] = round(p_cancer(hs, f2) - pA, 4)
                f2 = feats_A.copy()
                for i, e in EB.items():
                    f2[pos[i]] = e
                crow["dp_snapB"] = round(p_cancer(hs, f2) - pA, 4)
                cell_rows.append(crow)
                print(json.dumps(crow), flush=True)
            print(f"[{enc}] {sid} done in {time.time() - t0:.0f}s", flush=True)
        del fwd
        torch.cuda.empty_cache()
    for name, rows in (("tiles", tile_rows), ("real_boundaries", real_rows), ("cells", cell_rows), ("final_fit", fit_rows)):
        if rows:
            cols = list(rows[0])
            with open(os.path.join(a.out, f"probe_{name}.tsv"), "w") as f:
                f.write("\t".join(cols) + "\n")
                f.writelines("\t".join(str(r[k]) for k in cols) + "\n" for r in rows)
    summ = {}
    for enc in a.encoders.split(","):
        T = [r for r in tile_rows if r["encoder"] == enc]
        R = [r for r in real_rows if r["encoder"] == enc]
        Cc = [r for r in cell_rows if r["encoder"] == enc]
        if not T:
            continue
        s = {k: dict(median=float(np.median([r[k] for r in T])), p90=float(np.percentile([r[k] for r in T], 90)))
             for k in ("ref_vs_A",) + VARIANTS + ("boundary_penalty",)}
        s["n_tiles"] = len(T)
        s["mix50_over_pure_q_median"] = float(np.median([r["mix50_v"] / max(r["pure_q"], 1e-9) for r in T]))
        s["real_A_vs_B"] = dict(n=len(R), median=float(np.median([r["d_A_B"] for r in R])) if R else None)
        s["cell_dp_abs_median"] = {k: float(np.median([abs(r[k]) for r in Cc])) for k in [f"dp_{v}" for v in VARIANTS] + ["dp_snapB"]}
        summ[enc] = s
    json.dump(summ, open(os.path.join(a.out, "probe_summary.json"), "w"), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
