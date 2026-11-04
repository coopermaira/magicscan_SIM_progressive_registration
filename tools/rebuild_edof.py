#!/usr/bin/env python3
"""Rebuild a MagicScan EDOF slide from its raw strips, without and with Pearson registration, for evaluation
on machines where the group's builders are not installed (e.g. voodoo).

Input per strip and plane, both channels:
  phases  the 3 SIM phase frames (..._loc<z>_<ch>_..._<strip>_0_<phase>.tiff, or ..._strip<s>_z<mm>_<ch>nm_<phase>.tiff)
          -> SIM = RMS of the phases (as fm_build/build_ms.py)
  recon   the scanner-reconstructed SIM (..._strip<s>_z<mm>_<ch>nm.tiff), which equals the phase SIM up to gain
          and offset; a third of the reading
--source auto uses recon where every plane of a strip has it, else phases.
Planes are matched per strip by their order in z (newer acquisitions set z per strip). Then: column destripe ->
EDOF (msprog/edof.py: SML argmax, 129 median, hard select). Variant 'off' fuses the planes as they are; 'pearson'
registers them first (msprog/register.py). The next strip is read while the current one is processed.

Outputs (both variants from the same loaded planes, same intensity scale per channel):
  OUT/<variant>/<id>_470.tif, <id>_528.tif   tiled pyramidal uint16 BigTIFF (tissue P99.9 -> 40000)
  OUT/<variant>/<id>_edof_zmap.png            source plane per 1/4-res pixel (plane rank * 255 / (n - 1))
  OUT/pearson/<id>_register.json              per-strip registration reports (per-edge shifts before/after)
  OUT/specs.tsv                               variant -> edof_align_check.py image spec
The builder's flat-field is left out on purpose: it changes intensities, not geometry.

usage: rebuild_edof.py SLICE_DIR --image_id ID --out DIR [--name_filter RE] [--source auto|recon|phases]
"""
import argparse
import faulthandler
import json
import os
import re
import signal
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
LOC = re.compile(r"(?:^|_)(?:location|loc)(\d+)(?=_|\.)", re.I)
ZMM = re.compile(r"_z(\d+\.\d+)(?=_|\.)", re.I)
STRIP_TOK = re.compile(r"_strip(\d+)(?=_|\.)", re.I)
TAIL3 = re.compile(r"_(\d+)_(\d+)_(\d)\.tiff?$")
PHASE_NM = re.compile(r"nm_(\d)\.tiff?$", re.I)
RECON_NM = re.compile(r"nm\.tiff?$", re.I)


def discover(d, name_filter=None):
    """{strip: {ch: {zkey: {'phases': {k: path}, 'recon': path}}}} for the raw files in d (top level only)."""
    raw = {}
    for f in sorted(os.listdir(d)):
        low = f.lower()
        if not low.endswith((".tif", ".tiff")) or "pseudo" in low or low.startswith("correction"):
            continue
        if name_filter and not re.search(name_filter, f):
            continue
        p = os.path.join(d, f)
        if os.path.getsize(p) == 0:
            continue
        c = CH_TOK.search(f)
        if not c:
            continue
        st = STRIP_TOK.search(f)
        if st:                                          # cartridge / strip / z_mm naming (z per strip)
            z = ZMM.search(f)
            if not z:
                continue
            s, zk = int(st.group(1)), float(z.group(1))
            m = PHASE_NM.search(f)
            phase, recon = (int(m.group(1)) if m else None), bool(RECON_NM.search(f))
        else:                                           # location naming: <strip>_<m>_<phase> tail
            z, t3 = LOC.search(f) or ZMM.search(f), TAIL3.search(f)
            if not z or not t3:
                continue
            s, zk = int(t3.group(1)), float(z.group(1))
            phase, recon = int(t3.group(3)), False
        e = raw.setdefault(s, {}).setdefault(c.group(1), {}).setdefault(zk, {"phases": {}, "recon": None})
        if phase is not None:
            e["phases"][phase] = p
        elif recon:
            e["recon"] = p
    return raw


def plan(raw, source):
    """Per strip: list of (zkey_470, zkey_528) pairs by z order, and the source to read."""
    out = {}
    for s, chans in sorted(raw.items()):
        if "470" not in chans or "528" not in chans:
            continue
        ok = {c: sorted(z for z, e in chans[c].items() if e["recon"] or len(e["phases"]) == 3) for c in ("470", "528")}
        n = min(len(ok["470"]), len(ok["528"]))
        if n == 0:
            continue
        if len(ok["470"]) != len(ok["528"]):
            # keep the planes whose z agree best across channels
            a, b = (ok["470"], ok["528"]) if len(ok["470"]) > len(ok["528"]) else (ok["528"], ok["470"])
            a = [min(a, key=lambda z: abs(z - zb)) for zb in b]
            ok = {"470": a, "528": b} if len(ok["470"]) > len(ok["528"]) else {"470": b, "528": a}
        pairs = list(zip(ok["470"][:n], ok["528"][:n]))
        has_recon = all(chans["470"][a]["recon"] and chans["528"][b]["recon"] for a, b in pairs)
        has_ph = all(len(chans["470"][a]["phases"]) == 3 and len(chans["528"][b]["phases"]) == 3 for a, b in pairs)
        src = {"auto": "recon" if has_recon else "phases", "recon": "recon", "phases": "phases"}[source]
        if (src == "recon" and not has_recon) or (src == "phases" and not has_ph):
            continue
        out[s] = (pairs, src)
    return out


def _arr(path):
    with tifffile.TiffFile(path) as t:
        pg = t.pages[0]
        if pg.is_contiguous and pg.compression == 1:
            return np.memmap(path, np.dtype(t.byteorder + pg.dtype.char), "r", int(pg.dataoffsets[0]), pg.shape)
        return pg.asarray()


def _drop_cache(path):
    try:
        fd = os.open(path, os.O_RDONLY)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        os.close(fd)
    except (OSError, AttributeError):
        pass


def load_plane(entry, src):
    if src == "recon":
        x = np.asarray(_arr(entry["recon"]), np.float32)
        _drop_cache(entry["recon"])
    else:
        arrs = [_arr(entry["phases"][k]) for k in sorted(entry["phases"])]
        h = min(a.shape[0] for a in arrs)                # phase frames can differ by a row
        w = min(a.shape[1] for a in arrs)
        x = None
        for k, a in zip(sorted(entry["phases"]), arrs):
            a = np.asarray(a[:h, :w], np.float32)
            _drop_cache(entry["phases"][k])
            x = a * a if x is None else x + a * a
        np.sqrt(x / len(arrs), out=x)
    return E.destripe(np.clip(x, 0, None))


def load_strip(raw, s, pairs, src, io_workers):
    jobs = [("470", a) for a, _ in pairs] + [("528", b) for _, b in pairs]
    with ThreadPoolExecutor(io_workers) as ex:
        planes = list(ex.map(lambda j: load_plane(raw[s][j[0]][j[1]], src), jobs))
    n = len(pairs)
    h = min(p.shape[0] for p in planes)
    w = min(p.shape[1] for p in planes)
    return [p[:h, :w] for p in planes[:n]], [p[:h, :w] for p in planes[n:]]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("slice_dir")
    ap.add_argument("--image_id", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--name_filter", default=None)
    ap.add_argument("--source", choices=["auto", "recon", "phases"], default="auto")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--io_workers", type=int, default=8)
    ap.add_argument("--strips", type=int, nargs="*", default=None, help="only these strips (testing)")
    ap.add_argument("--no_prefetch", action="store_true", help="read strips one at a time (less memory)")
    ap.add_argument("--tmp", default=None, help="scratch dir for the float canvases (default OUT/.tmp)")
    a = ap.parse_args()
    faulthandler.register(signal.SIGUSR1, all_threads=True)       # `kill -USR1 <pid>` prints every stack to stderr
    t0 = time.time()
    cv2.setNumThreads(4)
    raw = discover(a.slice_dir, a.name_filter)
    P = plan(raw, a.source)
    if a.strips:
        P = {s: v for s, v in P.items() if s in a.strips}
    if not P:
        raise SystemExit(f"{a.image_id}: no usable strips in {a.slice_dir}")
    strips = sorted(P)
    dims = {}
    for s in strips:
        pairs, src = P[s]
        e = raw[s]["470"][pairs[0][0]]
        dims[s] = tifffile.TiffFile(e["recon"] if src == "recon" else e["phases"][0]).pages[0].shape
    SW = min(d[1] for d in dims.values())
    H = max(d[0] for d in dims.values())
    W = SW * (max(strips) + 1)
    print(f"{a.image_id}: strips {strips}, planes per strip {[len(P[s][0]) for s in strips]}, "
          f"source {sorted({P[s][1] for s in strips})}, canvas {W}x{H}", flush=True)
    skipped = sorted(set(raw) - set(P))
    if skipped:
        print(f"  strips without a complete 470+528 plane set, left empty: {skipped}", flush=True)
    tmp = a.tmp or os.path.join(a.out, ".tmp")
    for d in (tmp, os.path.join(a.out, "off"), os.path.join(a.out, "pearson")):
        os.makedirs(d, exist_ok=True)
    can = {(v, c): np.lib.format.open_memmap(os.path.join(tmp, f"{a.image_id}_{v}_{c}.npy"), "w+", np.float32, (H, W))
           for v in ("off", "pearson") for c in ("470", "528")}
    zmap = {v: np.zeros((H // 4 + 1, W // 4 + 1), np.uint8) for v in ("off", "pearson")}
    nmax = max(len(P[s][0]) for s in strips)
    reports = {}
    pre = ThreadPoolExecutor(1)
    fut = pre.submit(load_strip, raw, strips[0], *P[strips[0]], a.io_workers)
    for i, s in enumerate(strips):
        ts = time.time()
        ph, pe = fut.result()
        tl = time.time() - ts
        if i + 1 < len(strips):
            nxt = strips[i + 1]
            fut = pre.submit(load_strip, raw, nxt, *P[nxt], a.io_workers) if not a.no_prefetch else None
        x0 = s * SW
        h, w = ph[0].shape
        w = min(w, SW)
        for v in ("off", "pearson"):
            if v == "pearson":
                if len(ph) >= 1:
                    ph, pe, rep = R.register_strip(ph, pe, workers=a.workers)
                    rep["z_470"], rep["z_528"] = [z for z, _ in P[s][0]], [z for _, z in P[s][0]]
                    rep["source"] = P[s][1]
                    reports[str(s)] = rep
            zq = E.z_surface(ph, pe)
            fh, fe = E.fuse(ph, pe, zq)
            can[(v, "470")][:h, x0:x0 + w] = fh[:, :w]
            can[(v, "528")][:h, x0:x0 + w] = fe[:, :w]
            zz = np.clip(np.rint(zq) * (255.0 / max(1, nmax - 1)), 0, 255).astype(np.uint8)
            zmap[v][:zz.shape[0], x0 // 4:x0 // 4 + min(zz.shape[1], w // 4)] = zz[:, :w // 4]
            del fh, fe
        del ph, pe
        if a.no_prefetch and i + 1 < len(strips):
            nxt = strips[i + 1]
            fut = pre.submit(load_strip, raw, nxt, *P[nxt], a.io_workers)
        print(f"  strip {s}: {len(P[s][0])} planes ({P[s][1]}), waited {tl:.0f}s for data, "
              f"total {time.time() - ts:.0f}s", flush=True)
    pre.shutdown()

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
        json.dump(dict(version=R.VERSION, slice_dir=a.slice_dir, strip_width=SW, strips=reports,
                       skipped_strips=skipped), f, indent=1)
    with open(os.path.join(a.out, "specs.tsv"), "w") as f:
        f.writelines(f"{v}\t{p}\n" for v, p in specs)
    print(f"{a.image_id}: done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
