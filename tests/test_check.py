"""Alignment check on synthetic images: recovers injected channel shifts, runs end to end on tiled TIFFs, and the
seam index falls when the fused planes are misaligned."""
import os
import sys
import tempfile

import cv2
import numpy as np
import tifffile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from msprog import check as C, pearson as P, selftest as S  # noqa: E402
import synth  # noqa: E402


def _pair(H=6144, W=6144, dy=0.0, dx=0.0, seed=3):
    base = synth.texture(H, W, seed)
    rng = np.random.default_rng(seed + 1)
    other = cv2.GaussianBlur(rng.standard_normal((H, W)).astype(np.float32), (0, 0), 3.0)
    stro = 0.6 * cv2.GaussianBlur(base, (0, 0), 1.5) + 0.4 * (other / other.std() + 3)
    return base, S.shifted(stro, dy, dx)


def test_match_window_recovers_shift():
    nuc, stro = _pair(2048, 2048)
    M = C.MARGIN
    ref = nuc[512:1536, 512:1536]
    for dy, dx in [(0, 0), (2.5, -1.25), (-7, 9), (13, 4)]:
        mov = S.shifted(stro, dy, dx)[512 - M:1536 + M, 512 - M:1536 + M]
        m = P.match_window(ref, mov, M)
        assert C.accepted(m), m
        assert np.hypot(m["dy"] - dy, m["dx"] - dx) < 0.3, (dy, dx, m)


def test_end_to_end_tiled_tiff():
    nuc, stro = _pair(dy=4.0, dx=-3.0)
    with tempfile.TemporaryDirectory() as d:
        for name, a in (("n.tif", nuc), ("s.tif", stro)):
            tifffile.imwrite(os.path.join(d, name), (a * 2000).astype(np.uint16), tile=(512, 512),
                             compression="zlib")
        im = C.resolve(f"{d}/n.tif,{d}/s.tif")
        rows, shape = C.measure(im, workers=4, log=print)
        s = C.summarize(im, rows, shape)
        C.write_outputs(os.path.join(d, "out"), s, rows, im)
        print(s["channel"])
        assert s["channel"]["n_accepted"] >= 0.8 * s["channel"]["n_windows"]
        assert abs(s["channel"]["median_px"] - 5.0) < 0.3
        assert os.path.exists(os.path.join(d, "out", "n_align.json"))


def test_seam_index_falls_with_misalignment():
    img = synth.texture(1024 + 2 * C.MARGIN, 1024 + 2 * C.MARGIN, 9)
    lab = S.blobs((256, 256), 1)
    up = cv2.resize(lab.astype(np.uint8), (1024, 1024), interpolation=cv2.INTER_NEAREST).astype(bool)
    M = C.MARGIN
    vals = []
    for s in S.SEAM_SHIFTS:
        other = S.shifted(img, s, s / 2)[M:M + 1024, M:M + 1024]
        fused = np.where(up, other, img[M:M + 1024, M:M + 1024]).astype(np.float32)
        vals.append(C.seam_index(fused, lab)[0])
    print(dict(zip(S.SEAM_SHIFTS, np.round(vals, 3))))
    assert vals[0] >= 0.9
    assert vals[3] <= vals[0] - 0.1


if __name__ == "__main__":
    test_match_window_recovers_shift()
    test_seam_index_falls_with_misalignment()
    test_end_to_end_tiled_tiff()
    print("ok")
