#!/usr/bin/env python3
"""Rebuild a MagicScan EDOF slide from raw 3-phase strips, without and with Pearson registration, for evaluation
on machines where the group's builders are not installed (e.g. voodoo).

Per strip (as fm_build/build_ms.py, minus its flat-field): SIM = RMS of the 3 phases per plane and channel ->
column destripe -> EDOF (msprog/edof.py: SML argmax, 129 median, hard select). Variant 'off' fuses the planes as
they are; 'pearson' registers them first (msprog/register.py) and fuses again. Both variants come from the same
loaded planes, are scaled with the same factor per channel, and are written as

  OUT/<variant>/<id>_470.tif, <id>_528.tif   tiled pyramidal uint16 BigTIFF (tissue P99.9 -> 40000)
  OUT/<variant>/<id>_edof_zmap.png            source plane per 1/4-res pixel (plane * 255 / (n - 1))
  OUT/pearson/<id>_register.json              per-strip registration reports
  OUT/specs.tsv                               variant -> edof_align_check.py image spec

The flat-field is left out on purpose: it changes intensities, not geometry, and needs the instrument files.

usage: rebuild_edof.py SLICE_DIR --image_id VA026_S4 --out DIR [--name_filter RE] [--workers 32]
"""
import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import tifffile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from msprog import edof as E, register as R  # noqa: E402
from msprog.tiffio import write_pyramid  # noqa: E402

CH_TOK = re.compile(r"(?:^|_)(470|528)(?:nm)?(?=_|\.)", re.I)
Z_TOK = re.compile(r"(?:^|_)(?:location|loc)(\d+)(?=_|\.)|_z(\d+\.\d+)(?=_|\.)", re.I)
TAIL3 = re.compile(r"_(\d+)_(\d+)_(\d)\.tiff?$")
STRIP_TOK = re.compile(r"_strip(\d+)(?=_|\.)", re.I)
TAIL1 = re.compile(r"nm_(\d)\.tiff?$", re.I)


def discover(d, name_filter=None):
    """{ch: {z: {strip: {phase: path}}}} for the raw phase frames in d."""
    raw = {}
    for f in sorted(os.listdir(d)):
        if not f.lower().endswith((".tif", ".tiff")) or "pseudo" in f.lower() or f.lower().startswith("correction"):
            continue
        if name_filter and not re.search(name_filter, f):
            continue
        c, z = CH_TOK.search(f), Z_TOK.search(f)
        if not c or not z:
            continue
        zk = float(z.group(1) or z.group(2))
        st, t1, t3 = STRIP_TOK.search(f), TAIL1.search(f), TAIL3.search(f)
        if st and t1:
            s, ph = int(st.group(1)), int(t1.group(1))
        elif t3:
            s, ph = int(t3.group(1)), int(t3.group(3))
        else:
            continue
        raw.setdefault(c.group(1), {}).setdefault(zk, {}).setdefault(s, {})[ph] = os.path.join(d, f)
    return raw


def shape_of(p):
    with tifffile.TiffFile(p) as t:
        return t.pages[0].shape


def sim(phases, H, SW):
    acc = None
    for k in sorted(phases):
        with tifffile.TiffFile(phases[k]) as t:
            pg = t.pages[0]
            if pg.is_contiguous and pg.compression == 1:
                a = np.memmap(phases[k], np.dtype(t.byteorder + "u2"), "r", int(pg.dataoffsets[0]), pg.shape)
            else:
                a = pg.asarray()
            x = np.asarray(a[:H, :SW], np.float32)
        if acc is None:
            acc = x * x
        else:
            acc += x * x
    np.sqrt(acc / len(phases), out=acc)
    return acc


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("slice_dir")
    ap.add_argument("--image_id", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--name_filter", default=None)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--io_workers", type=int, default=12)
    ap.add_argument("--strips", type=int, nargs="*", default=None, help="only these strips (testing)")
    ap.add_argument("--tmp", default=None, help="scratch dir for the float canvases (default OUT/.tmp)")
    a = ap.parse_args()
    t0 = time.time()
    cv2.setNumThreads(4)
    raw = discover(a.slice_dir, a.name_filter)
    if not {"470", "528"} <= set(raw):
        raise SystemExit(f"{a.image_id}: need 470 and 528 raw frames, found {sorted(raw)}")
    zs = sorted(set(raw["470"]) & set(raw["528"]))
    strips = sorted(set.intersection(*[set(raw[c][z]) for c in ("470", "528") for z in zs]))
    if a.strips:
        strips = [s for s in strips if s in a.strips]
    shapes = [shape_of(p) for c in ("470", "528") for z in zs for s in strips for p in raw[c][z][s].values()]
    H, SW = min(s[0] for s in shapes), min(s[1] for s in shapes)
    W = SW * (max(strips) + 1)
    print(f"{a.image_id}: {len(zs)} planes {zs}, strips {strips}, canvas {W}x{H}", flush=True)
    tmp = a.tmp or os.path.join(a.out, ".tmp")
    for d in (tmp, os.path.join(a.out, "off"), os.path.join(a.out, "pearson")):
        os.makedirs(d, exist_ok=True)
    can = {(v, c): np.lib.format.open_memmap(os.path.join(tmp, f"{a.image_id}_{v}_{c}.npy"), "w+", np.float32, (H, W))
           for v in ("off", "pearson") for c in ("470", "528")}
    zmap = {v: np.zeros((H // 4 + 1, W // 4 + 1), np.uint8) for v in ("off", "pearson")}
    reports = {}
    for s in strips:
        ts = time.time()
        jobs = [(c, z) for c in ("470", "528") for z in zs]
        with ThreadPoolExecutor(a.io_workers) as ex:
            loaded = dict(zip(jobs, ex.map(lambda j: E.destripe(sim(raw[j[0]][j[1]][s], H, SW)), jobs)))
        ph, pe = [loaded[("470", z)] for z in zs], [loaded[("528", z)] for z in zs]
        del loaded
        tl = time.time() - ts
        x0 = s * SW
        for v in ("off", "pearson"):
            if v == "pearson":
                ph, pe, rep = R.register_strip(ph, pe, workers=a.workers)
                reports[str(s)] = rep
            zq = E.z_surface(ph, pe)
            fh, fe = E.fuse(ph, pe, zq)
            can[(v, "470")][:, x0:x0 + SW] = fh
            can[(v, "528")][:, x0:x0 + SW] = fe
            zz = np.clip(np.rint(zq) * (255.0 / max(1, len(zs) - 1)), 0, 255).astype(np.uint8)
            zmap[v][:zz.shape[0], x0 // 4:x0 // 4 + zz.shape[1]] = zz
            del fh, fe
        del ph, pe
        print(f"  strip {s}: load {tl:.0f}s, total {time.time() - ts:.0f}s", flush=True)

    # one scale per channel for both variants: P99.9 of the tissue of the unregistered build
    specs = []
    for c in ("470", "528"):
        smp = np.asarray(can[("off", c)][::32, ::32])
        v = smp[smp > np.percentile(smp, 50)]
        scale = 40000.0 / max(float(np.percentile(v, 99.9)) if v.size else 1.0, 1e-6)
        for var in ("off", "pearson"):
            u = np.lib.format.open_memmap(os.path.join(tmp, f"{a.image_id}_{var}_{c}_u16.npy"), "w+", np.uint16, (H, W))
            for y in range(0, H, 4096):
                u[y:y + 4096] = np.clip(can[(var, c)][y:y + 4096] * scale, 0, 65535).astype(np.uint16)
            write_pyramid(os.path.join(a.out, var, f"{a.image_id}_{c}.tif"), u, workers=a.workers)
            del u
            os.remove(os.path.join(tmp, f"{a.image_id}_{var}_{c}_u16.npy"))
    for var in ("off", "pearson"):
        cv2.imwrite(os.path.join(a.out, var, f"{a.image_id}_edof_zmap.png"), zmap[var])
        d = os.path.join(a.out, var)
        specs.append((var, f"{d}/{a.image_id}_470.tif,{d}/{a.image_id}_528.tif,{d}/{a.image_id}_edof_zmap.png"))
    for k in list(can):
        p = can[k].filename
        del can[k]
        os.remove(p)
    with open(os.path.join(a.out, "pearson", f"{a.image_id}_register.json"), "w") as f:
        json.dump(dict(version=R.VERSION, slice_dir=a.slice_dir, planes=zs, strips=reports), f, indent=1)
    with open(os.path.join(a.out, "specs.tsv"), "w") as f:
        f.writelines(f"{v}\t{p}\n" for v, p in specs)
    print(f"{a.image_id}: done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
