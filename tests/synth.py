"""Synthetic MagicScan-like strips with known displacement fields (for the tests and the self-test)."""
import cv2
import numpy as np


def texture(H, W, seed=0):
    """Tissue-like texture: nuclei-sized blobs on smoothed multi-scale noise, positive values."""
    rng = np.random.default_rng(seed)
    t = np.zeros((H, W), np.float32)
    for s, a in ((1.5, 1.0), (4.0, 0.8), (12.0, 0.6)):
        n = cv2.GaussianBlur(rng.standard_normal((H, W)).astype(np.float32), (0, 0), s)
        t += a * n / n.std()
    blobs = np.zeros((H, W), np.float32)
    k = H * W // 2000
    blobs[rng.integers(0, H, k), rng.integers(0, W, k)] = 1
    blobs = cv2.GaussianBlur(blobs, (0, 0), 3.0)
    t += 3 * blobs / blobs.max()
    return t - t.min() + 0.5


def field(H, W, amp, seed):
    """Smooth field (dy, dx) in px: slow bends along the scan, a tilt across, small row-to-row jerks per 1,024 rows."""
    rng = np.random.default_rng(seed)
    y = np.arange(H, dtype=np.float32)[:, None]
    x = np.arange(W, dtype=np.float32)[None, :]
    ph = rng.uniform(0, 2 * np.pi, 4)
    dy = amp * (0.6 * np.sin(2 * np.pi * y / H + ph[0]) + 0.3 * (x / W - 0.5)) + 0 * x
    dx = amp * (0.5 * np.sin(2 * np.pi * 1.5 * y / H + ph[1]) + 0.4 * np.cos(np.pi * x / W + ph[2])) + 0 * y
    jerk = np.repeat(rng.normal(0, 0.15 * amp, H // 1024 + 1), 1024)[:H, None]
    jerk = cv2.GaussianBlur(jerk.astype(np.float32), (1, 0), sigmaX=0.1, sigmaY=64)
    return (dy + jerk).astype(np.float32), dx.astype(np.float32)


def displace(img, dy, dx):
    """Content of the result sits at its source position + (dy, dx): out(q) = img(q - d(q))."""
    H, W = img.shape
    gx, gy = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    return cv2.remap(img, gx - dx, gy - dy, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)


def strip(H=8192, W=4512, n=5, amp=12.0, chan_amp=4.0, seed=0):
    """n planes x 2 channels. Returns ph, pe (lists) and the true fields {pass: (dy, dx)}; pass 2k = nuclear plane k,
    2k + 1 = stromal plane k; the reference is nuclear plane n // 2 (field 0)."""
    base = texture(H, W, seed)
    rng = np.random.default_rng(seed + 1)
    other = cv2.GaussianBlur(rng.standard_normal((H, W)).astype(np.float32), (0, 0), 3.0)
    stro = 0.6 * cv2.GaussianBlur(base, (0, 0), 1.5) + 0.4 * (other / other.std() + 3)
    r = n // 2
    ph, pe, true = [], [], {}
    for k in range(n):
        if k == r:
            pdy, pdx = np.zeros((H, W), np.float32), np.zeros((H, W), np.float32)
        else:
            pdy, pdx = field(H, W, amp * abs(k - r) / max(r, 1), seed + 10 + k)
        cdy, cdx = field(H, W, chan_amp, seed + 50 + k)
        blur = 0.4 + 0.7 * abs(k - r)
        a = cv2.GaussianBlur(displace(base, pdy, pdx), (0, 0), blur)
        b = cv2.GaussianBlur(displace(stro, pdy + cdy, pdx + cdx), (0, 0), blur)
        noise = np.random.default_rng(seed + 100 + k)
        ph.append(a + 0.05 * noise.standard_normal((H, W)).astype(np.float32))
        pe.append(b + 0.05 * noise.standard_normal((H, W)).astype(np.float32))
        true[2 * k], true[2 * k + 1] = (pdy, pdx), (pdy + cdy, pdx + cdx)
    return ph, pe, true
