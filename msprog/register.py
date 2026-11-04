"""Per-strip registration of z-planes and channels before EDOF fusion (drop-in for fm_build/ms_register.py).

register_strip(ph, pe, ref_idx=None, workers=16, log=print, return_fields=False) has the same inputs and outputs as
ms_register.register_strip, so build_ms.py (fm builder) and recolor_MS_v2.4 (colour builder) can call it unchanged
(see build_registered.py).

Method, one strip of N planes x 2 channels = 2N passes:
  1. Edges = every adjacent pair: 470 z<->z+1, 528 z<->z+1, 470<->528 at the same z (3N - 2 edges).
  2. Each edge is matched with coarse-to-fine Pearson (pearson.match_chunks) on 256 px sub-windows and fitted per
     1,024 px window with a robust offset + stretch model (pearson.fit_chunks); the model is read at the sub-window
     centres (a 256 px grid).
  3. Joint solve per grid node: unknown displacement u_p of every pass relative to the reference pass (470 of the
     reference plane, u = 0); every edge a->b says u_b - u_a = m_ab. Least squares over all edges, the worst edge
     dropped while its residual > DROP_PX and the system stays determined. Using every edge at once instead of
     chaining planes outward spreads errors instead of accumulating them; the leftover residual (loop closure) is a
     per-node error estimate.
  4. Field cleaning per pass: nodes > OUTLIER_PX from the 3x3 median of their valid neighbours dropped, then
     filled by normalised Gaussian smoothing of the valid nodes (measured nodes are kept as they are).
  5. Each original plane is warped once (cubic): registered(y, x) = plane(y + dy(y, x), x + dx(y, x)), field
     interpolated bilinearly between node centres.
  6. Verification: the adjacent edges are re-measured on the registered planes (every VERIFY_EVERY-th block); the
     report gives the residual shifts. Strips whose residual median exceeds RESID_WARN_UM are flagged.
Shift convention (same as ms_register.py): registered = plane(y + dy, x + dx), i.e. (dy, dx) is where the content
of the pass sits relative to the reference.
"""
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

from . import pearson as P

VERSION = "pearson_c2f_joint_2026-11-04"
MPP = 0.7
BLOCK = 4096              # rows per work unit (4 chunks)
DROP_PX = 1.5             # joint solve: drop the worst edge while its residual exceeds this
OUTLIER_PX = 3.0          # field cleaning: deviation from the 3x3 median of valid neighbours
MAX_SHIFT_PX = 400.0      # larger shifts are failed matches (largest real plane shift seen: ~150 px)
SIGMA_Y, SIGMA_X = 0.7, 1.0
RESID_WARN_UM = 1.0
VERIFY_EVERY = 4


def _edges(n):
    """Pass index: 2k = nuclear plane k, 2k + 1 = stromal plane k. Edges (a, b, kind)."""
    E = [(2 * k, 2 * k + 1, "c") for k in range(n)]
    for k in range(n - 1):
        E += [(2 * k, 2 * k + 2, "z"), (2 * k + 1, 2 * k + 3, "z")]
    return E


def _measure_block(passes, E, b0, b1, H, W, destripe=True):
    """Fitted sub-window shifts of every edge for rows [b0, b1). Returns {edge_index: (n, 7) array, global rows}."""
    y0, y1 = max(0, b0 - P.MARGIN), min(H, b1 + P.MARGIN)
    chunks = [c - y0 for c in range(b0, b1, P.CHUNK) if c + P.CHUNK <= H]
    if not chunks:
        return {}
    cache, out = {}, {}
    order = sorted(range(len(E)), key=lambda e: (min(E[e][0], E[e][1]) // 2, E[e][2]))
    for k, e in enumerate(order):
        a, b, _ = E[e]
        for p in (a, b):
            if p not in cache:
                cache[p] = P.pyramid(passes[p][y0:y1], destripe)
        f = P.fit_chunks(P.match_chunks(cache[a], cache[b], chunks, W))
        f[:, 0] += y0
        out[e] = f
        need = {q for e2 in order[k + 1:] for q in E[e2][:2]}
        for p in [p for p in cache if p not in need]:
            del cache[p]
    return out


def _grid(H, W):
    nwin, x0w = P.window_layout(W)
    ny, nx = H // P.CHUNK * P.NSUB, nwin * P.NSUB
    yc = P.SUB / 2 + P.SUB * np.arange(ny)
    xc = x0w + P.SUB / 2 + P.SUB * np.arange(nx)
    return ny, nx, yc, xc


def _scatter(meas, ny, nx):
    """Edge measurements -> (ny, nx, 2) grids of (dy, dx) and (ny, nx) window rms."""
    M = np.full((ny, nx, 2), np.nan, np.float32)
    R = np.full((ny, nx), np.nan, np.float32)
    for blk in meas:
        if not len(blk):
            continue
        i = (blk[:, 0] // P.CHUNK * P.NSUB + blk[:, 2]).astype(int)
        j = (blk[:, 1] * P.NSUB + blk[:, 3]).astype(int)
        ok = (i < ny) & (j < nx)
        M[i[ok], j[ok], 0], M[i[ok], j[ok], 1], R[i[ok], j[ok]] = blk[ok, 4], blk[ok, 5], blk[ok, 6]
    big = np.hypot(M[..., 0], M[..., 1]) > MAX_SHIFT_PX
    M[big] = np.nan
    return M, R


def _solve(Mg, E, npass, ref):
    """Joint least squares per node. Mg: (n_edges, ny, nx, 2). Returns U (npass, ny, nx, 2) and residual grids
    per edge kind {"z": (ny, nx), "c": (ny, nx)} (rms px of the kept edges of that kind; NaN where none)."""
    ne, ny, nx, _ = Mg.shape
    U = np.full((npass, ny, nx, 2), np.nan, np.float32)
    res = {k: np.full((ny, nx), np.nan, np.float32) for k in ("z", "c")}
    unk = [p for p in range(npass) if p != ref]
    col = {p: i for i, p in enumerate(unk)}
    A_full = np.zeros((ne, len(unk)))
    for e, (a, b, _) in enumerate(E):
        if b in col:
            A_full[e, col[b]] += 1
        if a in col:
            A_full[e, col[a]] -= 1
    kinds = np.array([k for _, _, k in E])
    valid = np.isfinite(Mg[..., 0])
    for i in range(ny):
        for j in range(nx):
            keep = valid[:, i, j].copy()
            if not keep.any():
                continue
            while True:
                A, m = A_full[keep], Mg[keep, i, j]
                sol, *_ = np.linalg.lstsq(A, m, rcond=None)
                r = np.hypot(*(A @ sol - m).T)
                if r.max() <= DROP_PX:
                    break
                # drop the worst edge whose removal keeps every pass as constrained as before
                rank = np.linalg.matrix_rank(A)
                idx = np.flatnonzero(keep)
                dropped = False
                for q in np.argsort(-r):
                    if r[q] <= DROP_PX:
                        break
                    trial = keep.copy()
                    trial[idx[q]] = False
                    if np.linalg.matrix_rank(A_full[trial]) == rank:
                        keep = trial
                        dropped = True
                        break
                if not dropped:
                    break
            reach = _reachable(A_full, keep, len(unk))
            for p in unk:
                if reach[col[p]]:
                    U[p, i, j] = sol[col[p]]
            U[ref, i, j] = 0.0
            ks = kinds[keep]
            for k in ("z", "c"):
                if (ks == k).any():
                    res[k][i, j] = float(np.sqrt(np.mean(r[ks == k] ** 2)))
    return U, res


def _reachable(A_full, keep, nunk):
    """Unknowns connected to the reference (fixed at 0) through the kept edges."""
    A = A_full[keep]
    ref_edges = np.abs(A).sum(1) == 1                  # edges touching the reference
    reach = np.zeros(nunk, bool)
    reach[np.flatnonzero(np.abs(A[ref_edges]).sum(0))] = True
    changed = True
    while changed:
        changed = False
        for row in A[~ref_edges]:
            nz = np.flatnonzero(row)
            if len(nz) == 2 and reach[nz].any() and not reach[nz].all():
                reach[nz] = True
                changed = True
    return reach


def _clean(F):
    """(ny, nx, 2) field with NaN holes -> outliers removed, smoothed, holes filled."""
    ny, nx, _ = F.shape
    v = np.isfinite(F[..., 0])
    if v.sum() == 0:
        return np.zeros_like(F), v
    med = np.full_like(F, np.nan)
    for i in range(ny):
        for j in range(nx):
            nb = F[max(0, i - 1):i + 2, max(0, j - 1):j + 2].reshape(-1, 2)
            nb = nb[np.isfinite(nb[:, 0])]
            if len(nb) >= 3:
                med[i, j] = np.median(nb, axis=0)
    dev = np.hypot(*(F - med).transpose(2, 0, 1))
    v &= ~(np.isfinite(dev) & (dev > OUTLIER_PX))
    if v.sum() == 0:                       # every node rejected: no correction for this pass
        return np.zeros_like(F), v
    out = np.zeros_like(F)
    w = v.astype(np.float32)
    for c in range(2):
        val = np.where(v, F[..., c], 0).astype(np.float32)
        sy, sx = SIGMA_Y, SIGMA_X
        num = cv2.GaussianBlur(val, (0, 0), sigmaX=sx, sigmaY=sy, borderType=cv2.BORDER_REPLICATE)
        den = cv2.GaussianBlur(w, (0, 0), sigmaX=sx, sigmaY=sy, borderType=cv2.BORDER_REPLICATE)
        f = np.where(den > 1e-3, num / np.maximum(den, 1e-6), np.nan)
        while np.isnan(f).any() and sx < max(ny, nx):           # widen until every hole is filled
            sy, sx = sy * 2, sx * 2
            num = cv2.GaussianBlur(val, (0, 0), sigmaX=sx, sigmaY=sy, borderType=cv2.BORDER_REPLICATE)
            den = cv2.GaussianBlur(w, (0, 0), sigmaX=sx, sigmaY=sy, borderType=cv2.BORDER_REPLICATE)
            f = np.where(np.isnan(f) & (den > 1e-3), num / np.maximum(den, 1e-6), f)
        f = np.nan_to_num(f, nan=float(np.median(F[v, c])))
        out[..., c] = np.where(v, F[..., c], f)     # measured nodes are kept; smoothing only fills holes
    return out, v


def warp(img, yc, xc, G, band_rows=2048):
    """registered(y, x) = img(y + dy(y, x), x + dx(y, x)); G (ny, nx, 2) = (dy, dx) at node centres (yc, xc),
    bilinear between nodes, constant beyond the outer nodes; cubic resampling."""
    H, W = img.shape
    out = np.empty((H, W), np.float32)
    sy = float(yc[1] - yc[0]) if len(yc) > 1 else 1.0
    sx = float(xc[1] - xc[0]) if len(xc) > 1 else 1.0
    gx = ((np.arange(W, dtype=np.float32) - xc[0]) / sx)[None, :]
    Gy, Gx = np.ascontiguousarray(G[..., 0]), np.ascontiguousarray(G[..., 1])
    for y0 in range(0, H, band_rows):
        y1 = min(H, y0 + band_rows)
        rows = np.arange(y0, y1, dtype=np.float32)
        gy = ((rows - yc[0]) / sy)[:, None]
        mgx = np.ascontiguousarray(np.broadcast_to(gx, (y1 - y0, W)), np.float32)
        mgy = np.ascontiguousarray(np.broadcast_to(gy, (y1 - y0, W)), np.float32)
        dy = np.nan_to_num(np.clip(cv2.remap(Gy, mgx, mgy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE), -MAX_SHIFT_PX, MAX_SHIFT_PX))
        dx = np.nan_to_num(np.clip(cv2.remap(Gx, mgx, mgy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE), -MAX_SHIFT_PX, MAX_SHIFT_PX))
        s0 = int(max(0, np.floor(y0 + dy.min()) - 3))
        s1 = int(min(H, np.ceil(y1 + dy.max()) + 3))
        map_x = np.arange(W, dtype=np.float32)[None, :] + dx
        map_y = rows[:, None] + dy - s0
        src = np.ascontiguousarray(img[s0:s1], np.float32)
        out[y0:y1] = cv2.remap(src, map_x, map_y, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
    np.maximum(out, 0, out=out)          # cubic overshoot; the builders' planes are non-negative
    return out


def _stats(a):
    a = np.asarray(a, np.float64)
    a = a[np.isfinite(a)]
    return {"median": round(float(np.median(a)), 2), "p90": round(float(np.percentile(a, 90)), 2),
            "max": round(float(np.max(a)), 2), "n": int(a.size)} if a.size else None


def _measure_all(passes, E, H, W, workers, blocks):
    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(lambda b: _measure_block(passes, E, b, min(b + BLOCK, H), H, W), blocks))


def register_strip(ph, pe, ref_idx=None, workers=16, log=print, return_fields=False):
    """ph, pe: lists of 2-D float planes (H, SW) sorted by z (nuclear, stromal).
    Returns (ph_registered, pe_registered, report[, fields]) like ms_register.register_strip."""
    t0 = time.time()
    cv2.setNumThreads(1)
    n = len(ph)
    H = min(min(a.shape[0] for a in ph), min(a.shape[0] for a in pe))
    W = min(min(a.shape[1] for a in ph), min(a.shape[1] for a in pe))
    passes = []
    for k in range(n):
        passes += [np.asarray(ph[k])[:H, :W], np.asarray(pe[k])[:H, :W]]
    r = n // 2 if ref_idx is None else ref_idx
    ref = 2 * r
    ny, nx, yc, xc = _grid(H, W)
    if n < 1 or ny == 0 or nx == 0:
        rep = dict(version=VERSION, skipped=f"strip too small ({H}x{W})", flag=["skipped"])
        return (list(ph), list(pe), rep, None) if return_fields else (list(ph), list(pe), rep)
    E = _edges(n)
    blocks = list(range(0, H, BLOCK))

    # 1-2. measure every edge
    meas = _measure_all(passes, E, H, W, workers, blocks)
    Mg = np.full((len(E), ny, nx, 2), np.nan, np.float32)
    nconf = {}
    for e in range(len(E)):
        Mg[e], _ = _scatter([m[e] for m in meas if e in m], ny, nx)
        nconf[f"{E[e][0]}-{E[e][1]}"] = int(np.isfinite(Mg[e, ..., 0]).sum())
    t_meas = time.time() - t0
    log(f"    register[pearson]: measured {len(E)} pairs on {len(blocks)} blocks in {t_meas:.0f}s")
    # per-edge statistics before registration: shift size and stretch (the misalignment as acquired)
    edge_stats = {}
    for e, (a_, b_, kind) in enumerate(E):
        blks = [m[e] for m in meas if e in m and len(m[e])]
        f = np.concatenate(blks) if blks else np.zeros((0, 9), np.float32)
        f = f[np.isfinite(f[:, 4])]
        edge_stats[f"{a_}-{b_}"] = dict(kind=kind, before_um=_stats(np.hypot(f[:, 4], f[:, 5]) * MPP),
                                        before_dy_median_px=round(float(np.median(f[:, 4])), 2) if len(f) else None,
                                        before_dx_median_px=round(float(np.median(f[:, 5])), 2) if len(f) else None,
                                        stretch_x_median_pct=round(float(np.nanmedian(f[:, 7])) * 100, 4) if len(f) else None,
                                        stretch_y_median_pct=round(float(np.nanmedian(f[:, 8])) * 100, 4) if len(f) else None)

    # 3. joint solve, 4. clean
    ts = time.time()
    U, closure = _solve(Mg, E, 2 * n, ref)
    U[np.hypot(U[..., 0], U[..., 1]) > MAX_SHIFT_PX] = np.nan
    log(f"    register[pearson]: joint solve {ny}x{nx} nodes in {time.time() - ts:.0f}s")
    fields, valid = np.zeros_like(U), np.zeros(U.shape[:3], bool)
    for p in range(2 * n):
        fields[p], valid[p] = (np.zeros((ny, nx, 2), np.float32), np.ones((ny, nx), bool)) if p == ref \
            else _clean(U[p])

    bad = {p: int((~np.isfinite(fields[p])).sum()) for p in range(2 * n) if not np.isfinite(fields[p]).all()}
    if bad:
        log(f"    register[pearson]: non-finite field values replaced by 0 in passes {bad}")
        for p in bad:
            fields[p] = np.nan_to_num(fields[p], nan=0.0, posinf=0.0, neginf=0.0)

    # 5. warp every pass except the reference
    def _w(p):
        return passes[p] if p == ref else warp(passes[p], yc, xc, fields[p])
    with ThreadPoolExecutor(min(workers, 2 * n)) as ex:
        reg = list(ex.map(_w, range(2 * n)))
    t_warp = time.time() - t0 - t_meas
    log(f"    register[pearson]: warped in {t_warp:.0f}s; largest applied shift per pass (px) "
        f"{[round(float(np.abs(fields[p]).max()), 1) for p in range(2 * n)]}")

    # 6. verification on the registered planes (subset of blocks)
    vb = blocks[::VERIFY_EVERY] or blocks[:1]
    vmeas = _measure_all(reg, E, H, W, workers, vb)
    resid = {"z": [], "c": []}
    per_edge = {e: [] for e in range(len(E))}
    for m in vmeas:
        for e, f in m.items():
            g = f[np.isfinite(f[:, 4])]
            d = list(np.hypot(g[:, 4], g[:, 5]) * MPP)
            resid[E[e][2]] += d
            per_edge[e] += d
    for e, (a_, b_, _) in enumerate(E):
        edge_stats[f"{a_}-{b_}"]["after_um"] = _stats(per_edge[e])
    rz, rc = _stats(resid["z"]), _stats(resid["c"])
    applied = {k: _stats(np.hypot(fields[2 * k][..., 0], fields[2 * k][..., 1]) * MPP) for k in range(n)}
    chan = {k: _stats(np.hypot(*(fields[2 * k + 1] - fields[2 * k]).transpose(2, 0, 1)) * MPP) for k in range(n)}
    rep = dict(version=VERSION, ref_plane=r, n_planes=n, grid=[int(ny), int(nx)], node_px=P.SUB,
               applied_plane_shift_um=applied, applied_channel_shift_um=chan,
               measured_nodes_per_edge=nconf, edges=edge_stats,
               solved_nodes_per_pass={p: int(np.isfinite(U[p, ..., 0]).sum()) for p in range(2 * n)},
               closure_plane_um=_stats(closure["z"] * MPP), closure_channel_um=_stats(closure["c"] * MPP),
               residual_plane_um=rz, residual_channel_um=rc,
               flag=[w for w, v in (("plane residual", rz), ("channel residual", rc))
                     if v and v["median"] > RESID_WARN_UM],
               seconds={"measure": round(t_meas), "warp": round(t_warp), "total": round(time.time() - t0)})
    log(f"    register[pearson]: planes {[applied[k]['median'] if applied[k] else None for k in range(n)]} um "
        f"(median applied, ref {r}), channel {[chan[k]['median'] if chan[k] else None for k in range(n)]} um; "
        f"residual plane {rz} channel {rc}" + (f"  FLAG {rep['flag']}" if rep["flag"] else "")
        + f"  ({rep['seconds']['total']}s)")
    ph_r, pe_r = reg[0::2], reg[1::2]
    if return_fields:
        return ph_r, pe_r, rep, dict(yc=yc, xc=xc, U=U, fields=fields, valid=valid, closure=closure)
    return ph_r, pe_r, rep
