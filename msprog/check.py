"""Alignment quality of a finished EDOF image (nuclear + stromal grayscale), measured on a fixed grid of tissue
windows so two builds of the same slide can be compared window for window.

Metrics (1,024 px windows):
  channel   470 <-> 528 offset of the EDOF (Pearson, pearson.match_window); what a model sees as colour fringing.
  planes    top <-> bottom z-plane offset of the nuclear channel, when the build wrote the z variants (<id>__z<k>);
            measures how well the passes were registered before fusion.
  seam      cross-seam continuity where the EDOF switches source plane (needs the build's z-map): correlation of
            the image across the seam / the same across matched in-plane lines. 1 = seamless; fusing misaligned
            planes cuts structures at the seam and lowers it. Flat below ~2 px of seam offset, clear from ~4 px.
  sharpness Tenengrad focus measure of the nuclear EDOF (guards against registration that blurs).
  strips    at every boundary where two scan strips abut (x = k * strip width): vertical offset of the image
            across the boundary (Pearson of the columns either side, +-48 px) and its continuity (correlation
            across the boundary at zero offset / the same between columns inside the strip), per channel.
The seam and strip measures are made on both channels, so a discontinuity shared by both channels (channels
still aligned to each other) shows up even when the channel offset is small.
"""
import glob
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

from . import pearson as P
from .tiffio import TiledReader, thumbnail

WIN = 1024
MARGIN = 96               # search margin of the 1/4-res level (+-24 at 1/4 = +-96 px)
R_ACCEPT = 0.2            # full-res Pearson peak to accept a window
RATIO_MAX = 0.9           # second / first peak at 1/4 res (ambiguity)
SEAM_D = (1.5, 2.5)       # px either side of a seam
SEAM_CTRL = 10            # control lines this far into each plane
SEAM_MIN_PTS = 60
STRIP_W = 11680           # MagicScan strip width (px)
BOUND_S = 48              # vertical search across a strip boundary


# ---------------------------------------------------------------- inputs
def resolve(spec, chans=("470", "528")):
    """'<dataset_dir>:<image_id>' (fm builder layout) or '<nuc.tif>,<stro.tif>[,<zmap.png>]'.
    Returns dict(id, nuc, stro, zmap, z_top, z_bot (nuclear files), patches_h5)."""
    if ":" in spec and os.path.isdir(spec.rsplit(":", 1)[0]):
        d, iid = spec.rsplit(":", 1)
        g = os.path.join(d, "grayscale")
        im = dict(id=iid, nuc=f"{g}/{iid}_{chans[0]}.tif", stro=f"{g}/{iid}_{chans[1]}.tif")
        zs = []
        for f in glob.glob(f"{g}/{iid}__z*_{chans[0]}.tif"):
            m = re.search(r"__z([\d.]+)_", os.path.basename(f))
            if m:
                zs.append((float(m.group(1)), f))
        zs.sort()
        im["z_top"], im["z_bot"] = (zs[0][1], zs[-1][1]) if len(zs) >= 2 else (None, None)
        h5 = os.path.join(d, "tissue_mask", f"{iid}_patches.h5")
        im["patches_h5"] = h5 if os.path.exists(h5) else None
        root, name = os.path.split(os.path.normpath(d))
        zm = os.path.join(root, "_qc", name, "flatfield", f"{iid}_edof_zmap.png")
        im["zmap"] = zm if os.path.exists(zm) else None
    else:
        parts = spec.split(",")
        im = dict(id=os.path.basename(parts[0]).rsplit(".", 1)[0], nuc=parts[0], stro=parts[1],
                  zmap=parts[2] if len(parts) > 2 and parts[2] else None, z_top=None, z_bot=None, patches_h5=None)
    for k in ("nuc", "stro"):
        if not os.path.exists(im[k]):
            raise FileNotFoundError(im[k])
    return im


def tissue_windows(im, H, W, min_patches=8, max_windows=600):
    """Window origins (y, x) on the fixed 1,024 grid that hold tissue, away from the image border."""
    cand = {}
    if im.get("patches_h5"):
        import h5py
        with h5py.File(im["patches_h5"]) as f:
            c = f["coords"][:].astype(np.int64)
            ps = int(f["coords"].attrs.get("patch_size_level0", 183))
        cx, cy = c[:, 0] + ps // 2, c[:, 1] + ps // 2
        k, n = np.unique(np.c_[cy // WIN, cx // WIN], axis=0, return_counts=True)
        cand = {(int(a), int(b)): int(m) for (a, b), m in zip(k, n) if m >= min_patches}
    else:
        th, s = thumbnail(im["nuc"], 4096)
        lg = np.log1p(np.clip(th, 0, None))
        u8 = cv2.normalize(lg, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        _, m = cv2.threshold(cv2.GaussianBlur(u8, (0, 0), 2), 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        for wy in range(H // WIN):
            for wx in range(W // WIN):
                a = m[int(wy * WIN * s):int((wy + 1) * WIN * s), int(wx * WIN * s):int((wx + 1) * WIN * s)]
                if a.size and a.mean() >= 0.5:
                    cand[(wy, wx)] = int(a.mean() * 100)
    keep = []
    for (wy, wx) in sorted(cand):
        y, x = wy * WIN, wx * WIN
        if y - MARGIN >= 0 and x - MARGIN >= 0 and y + WIN + MARGIN <= H and x + WIN + MARGIN <= W:
            keep.append((y, x))
    if len(keep) > max_windows:
        keep = [keep[i] for i in np.linspace(0, len(keep) - 1, max_windows).astype(int)]
    return keep


def read_zmap(path):
    """Plane label per 1/4-res pixel from the builder's z-map PNG (value = plane * 255 / (n - 1))."""
    z = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if z is None:
        return None
    u = np.unique(z)
    if len(u) < 2:
        return np.zeros(z.shape, np.int16)
    step = float(np.min(np.diff(u.astype(np.float64))))
    return np.rint(z / step).astype(np.int16)


# ---------------------------------------------------------------- per-window measures
def offset(ref_reader, mov_reader, y, x):
    ref = ref_reader.region(y, x, WIN, WIN)
    mov = mov_reader.region(y - MARGIN, x - MARGIN, WIN + 2 * MARGIN, WIN + 2 * MARGIN)
    if ref.std() < 1e-3 or mov.std() < 1e-3:
        return None
    return P.match_window(ref, mov, MARGIN)


def accepted(m):
    return m is not None and np.isfinite(m["r"]) and m["r"] >= R_ACCEPT and not m["edge"] and m["ratio4"] < RATIO_MAX


def sharpness(img):
    g = cv2.GaussianBlur(img, (0, 0), 1.0)
    gx, gy = cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1)
    mu = float(img.mean())
    return float(np.mean(gx * gx + gy * gy) / max(mu * mu, 1e-6))


def seam_index(img, labels_q, oy=0.0, ox=0.0, rng=None):
    """Cross-seam continuity of `img` (full res) given plane labels at 1/4 res (labels_q covers img; pixel (i, j) of
    labels_q is centred at full-res (4 i + 1.5 + oy, 4 j + 1.5 + ox)).
    Seam points sit midway between 4-neighbours with different labels (where the builder's upsampled z-surface
    switches plane), the normal is the neighbour direction. Pixels d px either side of the seam are correlated
    (cs) and compared with the same correlation along the same normals SEAM_CTRL px inside each plane (cc).
    Returns (cs / cc, n_seam_points); NaN when there are too few seam points or no texture."""
    pts = []
    i, j = np.nonzero(labels_q[:, :-1] != labels_q[:, 1:])          # horizontal neighbours
    pts.append(np.c_[4 * i + 1.5, 4 * j + 3.5, np.zeros(len(i)), np.ones(len(i))])
    i, j = np.nonzero(labels_q[:-1, :] != labels_q[1:, :])          # vertical neighbours
    pts.append(np.c_[4 * i + 3.5, 4 * j + 1.5, np.ones(len(i)), np.zeros(len(i))])
    pts = np.concatenate(pts).astype(np.float32)
    pts[:, 0] += oy
    pts[:, 1] += ox
    H, W = img.shape
    lim = SEAM_CTRL + max(SEAM_D) + 2
    pts = pts[(pts[:, 0] > lim) & (pts[:, 0] < H - lim) & (pts[:, 1] > lim) & (pts[:, 1] < W - lim)]
    if len(pts) < SEAM_MIN_PTS:
        return np.nan, len(pts)
    rng = rng or np.random.default_rng(0)
    if len(pts) > 6000:
        pts = pts[rng.choice(len(pts), 6000, replace=False)]
    py, px, ny, nx = pts.T
    f = np.asarray(img, np.float32)
    b = cv2.GaussianBlur(f, (0, 0), 0.8) - cv2.GaussianBlur(f, (0, 0), 16.0)

    def samp(yy, xx):
        return cv2.remap(b, xx.astype(np.float32)[None], yy.astype(np.float32)[None], cv2.INTER_LINEAR)[0]

    def lab(yy, xx):
        i = np.clip(np.floor((yy - oy) / 4).astype(int), 0, labels_q.shape[0] - 1)
        j = np.clip(np.floor((xx - ox) / 4).astype(int), 0, labels_q.shape[1] - 1)
        return labels_q[i, j]

    def corr(a, c):
        if len(a) < SEAM_MIN_PTS or a.std() < 1e-6 or c.std() < 1e-6:
            return np.nan
        return float(np.corrcoef(a, c)[0, 1])

    A, B, CA, CB = [], [], [], []
    for d in SEAM_D:
        A.append(samp(py - d * ny, px - d * nx))
        B.append(samp(py + d * ny, px + d * nx))
        for sgn in (-1, 1):
            cy, cx = py + sgn * SEAM_CTRL * ny, px + sgn * SEAM_CTRL * nx
            l0 = lab(cy, cx)
            same = (lab(cy - d * ny, cx - d * nx) == l0) & (lab(cy + d * ny, cx + d * nx) == l0)
            CA.append(samp(cy - d * ny, cx - d * nx)[same])
            CB.append(samp(cy + d * ny, cx + d * nx)[same])
    cs = corr(np.concatenate(A), np.concatenate(B))
    cc = corr(np.concatenate(CA), np.concatenate(CB))
    if not np.isfinite(cs) or not np.isfinite(cc) or cc <= 0.05:
        return np.nan, len(pts)
    return cs / cc, len(pts)


def _profile(a, c0, c1):
    p = a[:, c0:c1].mean(1, dtype=np.float32)[:, None]
    return cv2.GaussianBlur(p, (1, 0), sigmaX=0.1, sigmaY=1.5) - cv2.GaussianBlur(p, (1, 0), sigmaX=0.1, sigmaY=12)


def boundary(reader, xb, y):
    """Offset across the strip boundary at column xb for rows y .. y + WIN. Returns dict(dy, r, r0, ctrl) or None.
    Columns xb-3..xb-1 (left) vs xb..xb+2 (right); control = columns xb-9..xb-7 vs xb-6..xb-4 inside the strip."""
    S = BOUND_S
    a = reader.region(y - S, xb - 12, WIN + 2 * S, 24)
    L, R = _profile(a, 9, 12), _profile(a, 12, 15)
    C1, C2 = _profile(a, 3, 6), _profile(a, 6, 9)
    t = L[S:S + WIN]
    if t.std() < 1e-6 or R.std() < 1e-6:
        return None
    r = cv2.matchTemplate(R, t, cv2.TM_CCOEFF_NORMED)
    py, _, v, edge = P.peak(r)
    c1, c2 = C1[S:S + WIN, 0], C2[S:S + WIN, 0]
    ctrl = float(np.corrcoef(c1, c2)[0, 1]) if c1.std() > 1e-6 and c2.std() > 1e-6 else np.nan
    return dict(dy=float(py - S), r=float(v), r0=float(r[S, 0]), ctrl=ctrl, edge=bool(edge))


# ---------------------------------------------------------------- one image
def measure(im, windows=None, workers=16, max_windows=600, log=print, strip_w=STRIP_W):
    rn, rs = TiledReader(im["nuc"]), TiledReader(im["stro"])
    H, W = rn.h, rn.w
    if (rs.h, rs.w) != (H, W):
        raise ValueError(f"{im['id']}: channel sizes differ ({H}x{W} vs {rs.h}x{rs.w})")
    if windows is None:
        windows = tissue_windows(im, H, W, max_windows=max_windows)
    zt = TiledReader(im["z_top"]) if im.get("z_top") else None
    zb = TiledReader(im["z_bot"]) if im.get("z_bot") else None
    zl = read_zmap(im["zmap"]) if im.get("zmap") else None
    log(f"{im['id']}: {H}x{W}, {len(windows)} tissue windows; z variants {'yes' if zt else 'no'}, "
        f"z-map {'yes' if zl is not None else 'no'}")

    def one(yx):
        y, x = yx
        cv2.setNumThreads(1)
        row = dict(y=y, x=x)
        c = offset(rn, rs, y, x)
        row.update({f"chan_{k}": (np.nan if c is None else float(c[k])) for k in ("dy", "dx", "r", "ratio4")})
        row["chan_ok"] = bool(accepted(c))
        if zt is not None and zb is not None and (zt.h, zt.w) == (H, W):
            z = offset(zt, zb, y, x)
            row.update({f"plane_{k}": (np.nan if z is None else float(z[k])) for k in ("dy", "dx", "r", "ratio4")})
            row["plane_ok"] = bool(accepted(z))
        img = rn.region(y, x, WIN, WIN)
        row["sharp"] = sharpness(img)
        if zl is not None:
            q = zl[y // 4:(y + WIN) // 4, x // 4:(x + WIN) // 4]
            if q.shape == (WIN // 4, WIN // 4):
                oy, ox = 4 * (y // 4) - y, 4 * (x // 4) - x
                row["seam"], row["seam_pts"] = seam_index(img, q, oy, ox)
                row["seam_stro"], _ = seam_index(rs.region(y, x, WIN, WIN), q, oy, ox)
        return row

    # strip boundaries crossing tissue rows of the window grid
    bjobs = []
    if strip_w and W >= 2 * strip_w:
        for xb in range(strip_w, W - 16, strip_w):
            ys = sorted({y for y, x in windows if x - WIN <= xb <= x + 2 * WIN})
            bjobs += [(xb, y) for y in ys if y - BOUND_S >= 0 and y + WIN + BOUND_S <= H]

    def bone(job):
        xb, y = job
        cv2.setNumThreads(1)
        out = dict(xb=xb, y=y)
        for tag, rd in (("nuc", rn), ("stro", rs)):
            b = boundary(rd, xb, y)
            ok = b is not None and b["r"] >= 0.3 and not b["edge"] and np.isfinite(b["ctrl"]) and b["ctrl"] > 0.05
            out[f"{tag}_ok"] = bool(ok)
            out[f"{tag}_dy"] = b["dy"] if b else np.nan
            out[f"{tag}_cont"] = b["r0"] / b["ctrl"] if ok else np.nan
        return out

    with ThreadPoolExecutor(workers) as ex:
        rows = list(ex.map(one, windows))
        brows = list(ex.map(bone, bjobs))
    return rows, (H, W), brows


def _dist(rows, pre, mpp):
    ok = [r for r in rows if r.get(f"{pre}_ok")]
    n_meas = sum(1 for r in rows if f"{pre}_ok" in r)
    if not n_meas:
        return None
    d = np.array([np.hypot(r[f"{pre}_dy"], r[f"{pre}_dx"]) for r in ok])
    out = dict(n_windows=n_meas, n_accepted=len(ok))
    if len(d):
        out.update(median_px=round(float(np.median(d)), 2), p90_px=round(float(np.percentile(d, 90)), 2),
                   median_um=round(float(np.median(d)) * mpp, 2), p90_um=round(float(np.percentile(d, 90)) * mpp, 2),
                   frac_gt_2px=round(float((d > 2).mean()), 3), frac_gt_5px=round(float((d > 5).mean()), 3))
    return out


def _seam(rows, key):
    v = np.array([r[key] for r in rows if np.isfinite(r.get(key, np.nan))])
    return (dict(n_windows=int(len(v)), median=round(float(np.median(v)), 3),
                 p10=round(float(np.percentile(v, 10)), 3)) if len(v) else None)


def _strips(brows):
    if not brows:
        return None
    out = dict(n_boundary_windows=len(brows))
    for tag in ("nuc", "stro"):
        ok = [b for b in brows if b[f"{tag}_ok"]]
        d = np.abs(np.array([b[f"{tag}_dy"] for b in ok]))
        c = np.array([b[f"{tag}_cont"] for b in ok])
        out[tag] = dict(n_accepted=len(ok))
        if len(ok):
            out[tag].update(median_dy_px=round(float(np.median(d)), 2), p90_dy_px=round(float(np.percentile(d, 90)), 2),
                            frac_gt_2px=round(float((d > 2).mean()), 3),
                            continuity_median=round(float(np.median(c)), 3))
    both = [b for b in brows if b["nuc_ok"] and b["stro_ok"]]
    if both:
        out["nuc_minus_stro_dy_median_px"] = round(float(np.median([abs(b["nuc_dy"] - b["stro_dy"]) for b in both])), 2)
    return out


def summarize(im, rows, shape, mpp=0.7, brows=None):
    sh = np.array([r["sharp"] for r in rows])
    return dict(image=im["id"], files={k: im.get(k) for k in ("nuc", "stro", "z_top", "z_bot", "zmap", "patches_h5")},
                size=list(shape), mpp=mpp, n_windows=len(rows),
                channel=_dist(rows, "chan", mpp), planes=_dist(rows, "plane", mpp),
                seam=_seam(rows, "seam"), seam_stro=_seam(rows, "seam_stro"), strips=_strips(brows),
                sharpness=dict(median=round(float(np.median(sh)), 5)) if len(sh) else None)


# ---------------------------------------------------------------- outputs
COLS = ["y", "x", "chan_dy", "chan_dx", "chan_r", "chan_ratio4", "chan_ok", "plane_dy", "plane_dx", "plane_r",
        "plane_ratio4", "plane_ok", "seam", "seam_stro", "seam_pts", "sharp"]
BCOLS = ["xb", "y", "nuc_dy", "nuc_cont", "nuc_ok", "stro_dy", "stro_cont", "stro_ok"]


def _tsv(path, rows, cols):
    with open(path, "w") as f:
        f.write("\t".join(cols) + "\n")
        for r in rows:
            f.write("\t".join("" if r.get(c) is None or (isinstance(r.get(c), float) and np.isnan(r[c]))
                              else (f"{r[c]:.4f}" if isinstance(r[c], float) else str(r[c])) for c in cols) + "\n")


def write_outputs(out_dir, s, rows, im, brows=None):
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.join(out_dir, s["image"])
    with open(f"{base}_align.json", "w") as f:
        json.dump(s, f, indent=1)
    _tsv(f"{base}_align_windows.tsv", rows, COLS)
    if brows:
        _tsv(f"{base}_align_strip_boundaries.tsv", brows, BCOLS)
    _map(f"{base}_align_channel.png", rows, im, "chan", "470-528 offset (px)", 10.0)
    if s["planes"]:
        _map(f"{base}_align_planes.png", rows, im, "plane", "top-bottom plane offset (px)", 20.0)


def _map(path, rows, im, pre, title, vmax):
    th, s = thumbnail(im["nuc"], 2048)
    g = cv2.normalize(np.log1p(np.clip(th, 0, None)), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    canvas = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
    for r in rows:
        if not r.get(f"{pre}_ok"):
            col = (90, 90, 90)
        else:
            v = min(1.0, np.hypot(r[f"{pre}_dy"], r[f"{pre}_dx"]) / vmax)
            col = tuple(int(c) for c in cv2.applyColorMap(np.uint8([[int(v * 255)]]), cv2.COLORMAP_TURBO)[0, 0])
        p0 = (int(r["x"] * s), int(r["y"] * s))
        p1 = (int((r["x"] + WIN) * s) - 1, int((r["y"] + WIN) * s) - 1)
        cv2.rectangle(canvas, p0, p1, col, -1)
    canvas = cv2.addWeighted(canvas, 0.55, cv2.cvtColor(g, cv2.COLOR_GRAY2BGR), 0.45, 0)
    f = 2048.0 / max(canvas.shape[:2])
    if f < 1:
        canvas = cv2.resize(canvas, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    canvas = cv2.copyMakeBorder(canvas, 56, 0, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    bar = cv2.applyColorMap(np.tile(np.linspace(0, 255, 256).astype(np.uint8), (16, 1)), cv2.COLORMAP_TURBO)
    canvas[8:24, 8:264] = bar
    cv2.putText(canvas, f"{title}: 0 .. {vmax:g}+ (grey = no confident match)", (8, 44), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 2)
    cv2.imwrite(path, canvas)


def read_windows(tsv):
    with open(tsv) as f:
        hdr = f.readline().rstrip("\n").split("\t")
        iy, ix = hdr.index("y"), hdr.index("x")
        return [(int(l.split("\t")[iy]), int(l.split("\t")[ix])) for l in f if l.strip()]


def compare(a, b):
    """Rows (metric, A, B, delta, verdict) for two summaries measured on the same windows."""
    out = []
    for key, lab in (("channel", "470-528 offset"), ("planes", "top-bottom plane offset")):
        A, B = a.get(key), b.get(key)
        if not A or not B or "median_px" not in A or "median_px" not in B:
            out.append((lab, "-", "-", "-", "not measured in both"))
            continue
        for stat, tol in (("median_px", 0.1), ("p90_px", 0.2), ("frac_gt_2px", 0.01), ("frac_gt_5px", 0.01)):
            d = B[stat] - A[stat]
            out.append((f"{lab} {stat}", A[stat], B[stat], round(d, 3),
                        "same" if abs(d) <= tol else ("better" if d < 0 else "worse")))
        out.append((f"{lab} accepted windows", A["n_accepted"], B["n_accepted"], B["n_accepted"] - A["n_accepted"], ""))
    for key, lab in (("seam", "seam continuity 470"), ("seam_stro", "seam continuity 528")):
        if a.get(key) and b.get(key):
            d = b[key]["median"] - a[key]["median"]
            out.append((f"{lab} median", a[key]["median"], b[key]["median"], round(d, 3),
                        "same" if abs(d) <= 0.01 else ("better" if d > 0 else "worse")))
        else:
            out.append((lab, "-", "-", "-", "needs a z-map for both"))
    sa, sb = a.get("strips") or {}, b.get("strips") or {}
    for tag, lab in (("nuc", "strip boundary 470"), ("stro", "strip boundary 528")):
        A, B = sa.get(tag) or {}, sb.get(tag) or {}
        if "median_dy_px" in A and "median_dy_px" in B:
            for stat, tol, lower in (("median_dy_px", 0.1, True), ("p90_dy_px", 0.2, True),
                                     ("continuity_median", 0.01, False)):
                d = B[stat] - A[stat]
                good = d < 0 if lower else d > 0
                out.append((f"{lab} {stat}", A[stat], B[stat], round(d, 3),
                            "same" if abs(d) <= tol else ("better" if good else "worse")))
        else:
            out.append((lab, "-", "-", "-", "no strip boundaries measured in both"))
    if a.get("sharpness") and b.get("sharpness"):
        ra = b["sharpness"]["median"] / a["sharpness"]["median"]
        out.append(("sharpness ratio B/A", a["sharpness"]["median"], b["sharpness"]["median"], round(ra, 3),
                    "ok" if ra >= 0.97 else "B blurrier"))
    return out
