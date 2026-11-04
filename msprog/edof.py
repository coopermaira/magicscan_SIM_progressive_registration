"""EDOF fusion as the group's builders run it (fm_build/build_ms.py -> recolor_MS_v1.8 'fullstack' defaults), for
rebuilding slides where those builders are not installed (tools/rebuild_edof.py).

Per strip: gray = (470 + 528) / 2 at 1/4 res; Sum-Modified-Laplacian focus score (9 x 9 box); argmax over planes
(mid plane where the guide has no signal); 129 x 129 median of the plane index; bilinear upsample to full res;
hard nearest-plane select, the same plane for both channels. Destripe = v1.8's column-gain division
(257-column trend, 1,024-row bands). Re-implemented from those functions, not imported, so it runs anywhere.
"""
import cv2
import numpy as np

DS = 4


def _smooth1d(x, k):
    k = int(k) | 1
    return np.convolve(np.pad(x, (k // 2, k // 2), mode="edge"), np.ones(k, np.float32) / k, "valid")[:len(x)]


def destripe(img, hp_win=257, band_rows=1024, row_sub=4):
    """Divide out the per-column gain ripple (v1.8 estimate_colgain/destripe)."""
    h, w = img.shape
    nb = max(1, h // band_rows)
    G = np.empty((nb, w), np.float32)
    for b in range(nb):
        cm = np.median(img[b * band_rows:(b + 1) * band_rows:row_sub], axis=0)
        G[b] = cm / np.maximum(_smooth1d(cm, hp_win), 1e-6)
    G = cv2.resize(G, (w, h), interpolation=cv2.INTER_LINEAR)
    G /= np.maximum(np.median(G, axis=1, keepdims=True), 1e-6)
    return (img / np.clip(G, 0.2, 5.0)).astype(np.float32)


def score_sml(img, win=9):
    k = np.array([[-1, 2, -1]], np.float32)
    ml = np.abs(cv2.filter2D(img, cv2.CV_32F, k)) + np.abs(cv2.filter2D(img, cv2.CV_32F, k.T))
    return cv2.boxFilter(ml, cv2.CV_32F, (win, win), normalize=False)


def z_surface(ph, pe, median_radius=64, min_signal=0.01):
    """Plane index per 1/4-res pixel (float, from a uint8 median) for planes ph, pe (lists, same shape)."""
    H, W = ph[0].shape
    dh, dw = max(1, H // DS), max(1, W // DS)
    gray = [(cv2.resize(a, (dw, dh), interpolation=cv2.INTER_AREA) +
             cv2.resize(b, (dw, dh), interpolation=cv2.INTER_AREA)) * np.float32(0.5) for a, b in zip(ph, pe)]
    scores = np.stack([score_sml(g) for g in gray], 0)
    guide = np.mean(gray, 0)
    n = len(ph)
    z = np.argmax(scores, 0).astype(np.float32)
    z[guide < min_signal] = (n - 1) / 2.0
    z = cv2.medianBlur(z.astype(np.uint8), 2 * median_radius + 1).astype(np.float32)
    return np.clip(z, 0, n - 1)


def fuse(ph, pe, zq):
    """Hard nearest-plane select of both channels from the 1/4-res surface zq."""
    H, W = ph[0].shape
    zf = cv2.resize(zq, (W, H), interpolation=cv2.INTER_LINEAR)
    best = np.clip(np.rint(zf), 0, len(ph) - 1).astype(np.int32)
    fh, fe = np.empty((H, W), np.float32), np.empty((H, W), np.float32)
    for i, (a, b) in enumerate(zip(ph, pe)):
        m = best == i
        np.copyto(fh, a, where=m)
        np.copyto(fe, b, where=m)
    return fh, fe
