"""Coarse-to-fine Pearson matching: the core matcher shared by the registration and the alignment check.

All matching is normalised cross-correlation (Pearson, cv2.TM_CCOEFF_NORMED) on band-passed images. No phase
correlation anywhere.

Geometry for one image pair (A = reference, B = moving), processed in 1,024-row chunks:
  L16  chunk-wide shift at 1/16 res, +-16 (= +-256 px)
  L4w  per 1,024 x 1,024 window at 1/4 res, +-12 (= +-48 px) around L16
  L4s  per 256 x 256 sub-window at 1/4 res, +-4 (= +-16 px) around L4w
  L1   per sub-window at full res, +-2 around L4s, parabolic sub-pixel peak
Shift convention: B[r + dy, c + dx] ~= A[r, c] (full-res px), i.e. B's content sits at A's position + (dy, dx).

Per window, the 16 sub-window shifts are fitted with a robust offset + stretch model
  dx = tx + kx (x - xc),  dy = ty + ky (y - yc)
(2 passes, outliers > max(1 px, 3 x 1.4826 MAD) dropped, >= MIN_SUB accepted sub-windows).
"""
import numpy as np
import cv2

CHUNK = 1024          # rows per chunk
WIN = 1024            # window size
SUB = 256             # sub-window size
NSUB = WIN // SUB     # sub-windows per window side
MARGIN = 320          # rows of context needed around a block (> L16 search of 256 + slack)
R_MIN = 0.3           # accepted sub-window: full-res Pearson peak >= R_MIN and not on the search edge
MIN_SUB = 6           # accepted sub-windows needed for a window fit


def bandpass(f, lo, hi):
    return cv2.GaussianBlur(f, (0, 0), lo) - cv2.GaussianBlur(f, (0, 0), hi)


def pyramid(img, destripe=True):
    """Band-passed matching images at full, 1/4 and 1/16 res.
    destripe: subtract the per-column median first (raw sensor column stripes otherwise lock the match at dx = 0)."""
    f = np.asarray(img, np.float32)
    if destripe:
        f = f - np.median(f[::16], axis=0).astype(np.float32)[None]
    f4 = cv2.resize(f, (f.shape[1] // 4, f.shape[0] // 4), interpolation=cv2.INTER_AREA)
    f16 = cv2.resize(f4, (f4.shape[1] // 4, f4.shape[0] // 4), interpolation=cv2.INTER_AREA)
    lo = cv2.resize(cv2.GaussianBlur(f4, (0, 0), 2.0), (f.shape[1], f.shape[0]), interpolation=cv2.INTER_LINEAR)
    b1 = cv2.GaussianBlur(f, (0, 0), 1.5)
    b1 -= lo                                         # ~1.5-8 px band-pass, low-pass taken at 1/4 res
    return {"b1": b1, "b4": bandpass(f4, .7, 4), "b16": bandpass(f16, .7, 4)}


def peak(r):
    """Argmax of a correlation surface with parabolic sub-pixel refinement. Returns (py, px, value, on_edge)."""
    py, px = np.unravel_index(int(np.argmax(r)), r.shape)
    v = float(r[py, px])

    def sub(m1, c, p1):
        d = m1 - 2 * c + p1
        return 0.0 if d >= 0 else 0.5 * (m1 - p1) / d
    fy = sub(r[py - 1, px], v, r[py + 1, px]) if 0 < py < r.shape[0] - 1 else 0.0
    fx = sub(r[py, px - 1], v, r[py, px + 1]) if 0 < px < r.shape[1] - 1 else 0.0
    edge = py in (0, r.shape[0] - 1) or px in (0, r.shape[1] - 1)
    return py + fy, px + fx, v, bool(edge)


def search(ref, cur, y, x, th, tw, S, dy0=0.0, dx0=0.0):
    """Template ref[y:y+th, x:x+tw] searched in cur around (y+dy0, x+dx0) +-S.
    Returns (dy, dx, r, on_edge) or None when template or search area leaves the images."""
    yy, xx = y + int(round(dy0)) - S, x + int(round(dx0)) - S
    if y < 0 or x < 0 or y + th > ref.shape[0] or x + tw > ref.shape[1]:
        return None
    if yy < 0 or xx < 0 or yy + th + 2 * S > cur.shape[0] or xx + tw + 2 * S > cur.shape[1]:
        return None
    t = ref[y:y + th, x:x + tw]
    if float(t.std()) < 1e-6:
        return None
    r = cv2.matchTemplate(cur[yy:yy + th + 2 * S, xx:xx + tw + 2 * S], t, cv2.TM_CCOEFF_NORMED)
    py, px, v, e = peak(r)
    return yy + py - y, xx + px - x, v, e


def window_layout(width):
    """Number of 1,024 windows across `width` and the x of the first one (centred)."""
    nwin = width // WIN
    return nwin, (width - nwin * WIN) // 2


def match_chunks(A, B, chunks, width):
    """Coarse-to-fine match of every sub-window of the given chunks.
    A, B: pyramids of the same row block; chunks: chunk start rows (block-local, multiples of 16).
    Returns float32 array (n, 8): chunk_row, window, si, sj, dy, dx, r, on_edge (dy/dx NaN when not measured)."""
    nwin, x0w = window_layout(width)
    out = []
    gprev = (0.0, 0.0)
    for c0 in chunks:
        g = search(A["b16"], B["b16"], c0 // 16 + 4, 16, CHUNK // 16 - 8, A["b16"].shape[1] - 32, 16)
        if g is not None and g[2] >= 0.2 and not g[3]:
            gdy, gdx = g[0] * 16, g[1] * 16
            gprev = (gdy, gdx)
        else:
            gdy, gdx = gprev
        for j in range(nwin):
            xw = x0w + j * WIN
            w4 = search(A["b4"], B["b4"], c0 // 4 + 6, xw // 4 + 24, CHUNK // 4 - 12, WIN // 4 - 48, 12,
                        gdy / 4, gdx / 4)
            if w4 is not None and w4[2] >= 0.25 and not w4[3]:
                wdy, wdx = w4[0] * 4, w4[1] * 4
            else:
                wdy, wdx = gdy, gdx
            for si in range(NSUB):
                for sj in range(NSUB):
                    y1, x1 = c0 + si * SUB, xw + sj * SUB
                    rec = [c0, j, si, sj, np.nan, np.nan, np.nan, 1.0]
                    s4 = search(A["b4"], B["b4"], y1 // 4, x1 // 4, SUB // 4, SUB // 4, 4, wdy / 4, wdx / 4)
                    if s4 is not None:
                        s1 = search(A["b1"], B["b1"], y1, x1, SUB, SUB, 2, s4[0] * 4, s4[1] * 4)
                        if s1 is not None:
                            rec[4:8] = [s1[0], s1[1], s1[2], float(s1[3])]
                    out.append(rec)
    return np.asarray(out, np.float32).reshape(-1, 8)


def fit_window(y, x, dy, dx, min_n=MIN_SUB):
    """Robust offset + stretch fit of one window's sub-window shifts (x, y centred by the caller).
    Returns (tx, kx, ty, ky, rms, n) or None."""
    keep = np.ones(len(x), bool)
    for _ in range(2):
        if keep.sum() < min_n or np.ptp(x[keep]) < SUB or np.ptp(y[keep]) < SUB:
            return None
        px = np.polyfit(x[keep], dx[keep], 1)
        py = np.polyfit(y[keep], dy[keep], 1)
        res = np.hypot(dx - np.polyval(px, x), dy - np.polyval(py, y))
        keep = res <= max(1.0, 3 * 1.4826 * np.median(res[keep]))
    if keep.sum() < min_n:
        return None
    rms = float(np.sqrt(np.mean(res[keep] ** 2)))
    return px[1], px[0], py[1], py[0], rms, int(keep.sum())


def fit_chunks(m, r_min=R_MIN, reject_px=3.0):
    """Window fits for match_chunks output, used as an outlier filter: an accepted sub-window keeps its own measured
    shift when it lies within max(reject_px, 3 x 1.4826 MAD) of its window's offset + stretch model (so local
    non-linear motion such as a stage jerk inside a window survives); otherwise, or when the window cannot be fitted,
    it is dropped (NaN).
    Returns float32 (n, 7): chunk_row, window, si, sj, dy, dx, window_rms."""
    out = []
    for c0 in np.unique(m[:, 0]):
        mc = m[m[:, 0] == c0]
        for j in np.unique(mc[:, 1]):
            w = mc[mc[:, 1] == j]
            ok = np.isfinite(w[:, 4]) & (w[:, 6] >= r_min) & (w[:, 7] == 0)
            yy = (w[:, 2] - (NSUB - 1) / 2) * SUB
            xx = (w[:, 3] - (NSUB - 1) / 2) * SUB
            f = fit_window(yy[ok], xx[ok], w[ok, 4], w[ok, 5]) if ok.sum() else None
            if f is not None:
                tx, kx, ty, ky, rms, _ = f
                res = np.hypot(w[:, 4] - (ty + ky * yy), w[:, 5] - (tx + kx * xx))
                ok &= res <= max(reject_px, 3 * 1.4826 * rms)
            for k in range(len(w)):
                if f is None or not ok[k]:
                    out.append([c0, j, w[k, 2], w[k, 3], np.nan, np.nan, np.nan])
                else:
                    out.append([c0, j, w[k, 2], w[k, 3], w[k, 4], w[k, 5], rms])
    return np.asarray(out, np.float32).reshape(-1, 7)


def match_window(ref, mov, margin, r4_min=0.15):
    """One-window two-level match for finished images (alignment check): ref is the template window, mov is the
    same window with `margin` px of context on every side. 1/4 res +-margin/4, then full res +-3 with sub-pixel.
    Returns dict(dy, dx, r, r4, ratio4, edge) or None (dy/dx in full-res px, convention as above)."""
    T = ref.shape[0]
    a4 = bandpass(cv2.resize(ref, (T // 4, T // 4), interpolation=cv2.INTER_AREA), .7, 4)
    m = mov.shape[0]
    b4 = bandpass(cv2.resize(mov, (m // 4, m // 4), interpolation=cv2.INTER_AREA), .7, 4)
    if a4.std() < 1e-6 or b4.std() < 1e-6:
        return None
    r = cv2.matchTemplate(b4, a4, cv2.TM_CCOEFF_NORMED)
    py, px, v4, e4 = peak(r)
    masked = r.copy()
    iy, ix = int(round(py)), int(round(px))
    masked[max(0, iy - 3):iy + 4, max(0, ix - 3):ix + 4] = -1
    ratio = float(max(masked.max(), 0) / v4) if v4 > 0 else 1.0
    S4 = (r.shape[0] - 1) // 2
    dy4, dx4 = (py - S4) * 4, (px - S4) * 4
    if v4 < r4_min or e4:
        return dict(dy=dy4, dx=dx4, r=np.nan, r4=v4, ratio4=ratio, edge=True)
    a1 = bandpass(ref, 1.5, 8)
    b1 = bandpass(mov, 1.5, 8)
    s = search(a1, b1, 0, 0, T, T, 3, margin + dy4, margin + dx4)
    if s is None:
        return dict(dy=dy4, dx=dx4, r=np.nan, r4=v4, ratio4=ratio, edge=True)
    return dict(dy=s[0] - margin, dx=s[1] - margin, r=s[2], r4=v4, ratio4=ratio, edge=s[3])
