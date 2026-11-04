"""Self-test of the alignment check on a real image: inject known misalignments into its own tissue windows and
verify each metric sees them.

  channel  528 window shifted by known (dy, dx); error = |measured - baseline - injected|
  planes   nuclear window vs a blurred (defocus-like), shifted copy of itself
  seam     synthetic EDOF: plane labels from smooth random blobs; label-1 pixels taken from a shifted copy;
           the continuity index must fall as the shift grows
"""
import cv2
import numpy as np

from . import check as C
from . import pearson as P
from .tiffio import TiledReader

SHIFTS = [(0, 2), (3, -4), (-6, 5), (10, 0), (0, -12), (15, 15)]
SEAM_SHIFTS = [0, 1, 2, 4, 8]


def shifted(a, dy, dx):
    """Content moved by (+dy, +dx)."""
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(a, M, a.shape[::-1], flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)


def blobs(shape, seed):
    rng = np.random.default_rng(seed)
    n = cv2.GaussianBlur(rng.standard_normal(shape).astype(np.float32), (0, 0), shape[0] / 12)
    return (n > 0).astype(np.int16)


def run(spec, n_windows=6, log=print):
    im = C.resolve(spec)
    rn, rs = TiledReader(im["nuc"]), TiledReader(im["stro"])
    wins = C.tissue_windows(im, rn.h, rn.w, max_windows=10 ** 6)
    if not wins:
        raise SystemExit("no tissue windows found")
    wins = [wins[i] for i in np.linspace(0, len(wins) - 1, min(n_windows, len(wins))).astype(int)]
    M = C.MARGIN
    res = {"channel": [], "planes": [], "seam": {s: [] for s in SEAM_SHIFTS}}
    for (y, x) in wins:
        ref = rn.region(y, x, C.WIN, C.WIN)
        big_n = rn.region(y - M, x - M, C.WIN + 2 * M, C.WIN + 2 * M)
        big_s = rs.region(y - M, x - M, C.WIN + 2 * M, C.WIN + 2 * M)
        base = P.match_window(ref, big_s, M)
        if not C.accepted(base):
            log(f"  window ({y},{x}): no confident baseline channel match, skipped for the channel test")
        for dy, dx in SHIFTS:
            if C.accepted(base):
                m = P.match_window(ref, shifted(big_s, dy, dx), M)
                e = np.hypot(m["dy"] - base["dy"] - dy, m["dx"] - base["dx"] - dx) if C.accepted(m) else np.nan
                res["channel"].append(e)
            m = P.match_window(ref, shifted(cv2.GaussianBlur(big_n, (0, 0), 1.5), dy, dx), M)
            res["planes"].append(np.hypot(m["dy"] - dy, m["dx"] - dx) if C.accepted(m) else np.nan)
        lab = blobs((C.WIN // 4, C.WIN // 4), y * 7 + x)
        up = cv2.resize(lab.astype(np.uint8), (C.WIN, C.WIN), interpolation=cv2.INTER_NEAREST).astype(bool)
        for s in SEAM_SHIFTS:
            other = cv2.GaussianBlur(shifted(big_n, s, s / 2), (0, 0), 0.6)[M:M + C.WIN, M:M + C.WIN]
            same = cv2.GaussianBlur(big_n, (0, 0), 0.6)[M:M + C.WIN, M:M + C.WIN]
            fused = np.where(up, other, same).astype(np.float32)
            res["seam"][s].append(C.seam_index(fused, lab)[0])

    def st(v):
        v = np.asarray(v, float)
        ok = np.isfinite(v)
        return (f"median {np.median(v[ok]):.2f} px, p90 {np.percentile(v[ok], 90):.2f} px, "
                f"{ok.sum()}/{len(v)} confident") if ok.any() else "no confident matches"
    ch, pl = np.asarray(res["channel"], float), np.asarray(res["planes"], float)
    seam = {s: float(np.nanmedian(v)) for s, v in res["seam"].items()}
    log(f"channel offset error (injected {len(SHIFTS)} shifts x {len(wins)} windows): {st(ch)}")
    log(f"plane offset error (defocus 1.5 px + shift):                  {st(pl)}")
    log("seam continuity index vs shift of one plane: " + "  ".join(f"{s}px {v:.3f}" for s, v in seam.items()))
    ok_ch = np.isfinite(ch).any() and np.nanpercentile(ch, 90) <= 0.5
    ok_pl = np.isfinite(pl).any() and np.nanpercentile(pl, 90) <= 0.75
    sv = [seam[s] for s in SEAM_SHIFTS]
    # the index is flat below ~2 px (sampling of the 1/4-res z-map); it must drop clearly from 4 px on
    ok_seam = np.isfinite(sv).all() and sv[0] >= 0.85 and seam[4] <= sv[0] - 0.1 and seam[8] < seam[4]
    verdict = dict(channel=bool(ok_ch), planes=bool(ok_pl), seam=bool(ok_seam))
    log("PASS/FAIL: " + "  ".join(f"{k} {'PASS' if v else 'FAIL'}" for k, v in verdict.items())
        + "   (channel p90 <= 0.5 px; planes p90 <= 0.75 px; seam index >= 0.85 at 0 px, >= 0.1 lower at 4 px, lower still at 8 px)")
    return verdict, dict(channel=ch.tolist(), planes=pl.tolist(), seam=seam)
