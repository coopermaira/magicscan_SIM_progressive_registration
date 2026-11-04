"""Registration recovers known fields on a synthetic strip (5 planes x 2 channels, defocused outer planes)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from msprog import register as R  # noqa: E402
import synth  # noqa: E402


def field_errors(fields, yc, xc, true, interior=1):
    """|estimated - true| at interior nodes, per pass (px)."""
    iy, ix = yc.astype(int), xc.astype(int)
    err = {}
    for p, (tdy, tdx) in true.items():
        t = np.stack([tdy[np.ix_(iy, ix)], tdx[np.ix_(iy, ix)]], -1)
        e = np.hypot(*(fields[p] - t).transpose(2, 0, 1))
        err[p] = e[interior:-interior, interior:-interior] if interior else e
    return err


def test_recovers_known_fields():
    ph, pe, true = synth.strip()
    ph_r, pe_r, rep, F = R.register_strip(ph, pe, workers=8, log=print, return_fields=True)
    err = field_errors(F["fields"], F["yc"], F["xc"], true)
    for p, e in err.items():
        print(f"pass {p}: median {np.median(e):.2f} px  p90 {np.percentile(e, 90):.2f} px")
    allerr = np.concatenate([e.ravel() for e in err.values()])
    assert np.median(allerr) < 0.25, np.median(allerr)
    assert np.percentile(allerr, 90) < 1.0, np.percentile(allerr, 90)
    assert rep["residual_plane_um"]["median"] < 1.0 and rep["residual_channel_um"]["median"] < 1.0, rep


def test_clean_and_warp_never_produce_nan():
    F = np.full((6, 6, 2), np.nan, np.float32)
    F[0, 0] = (50, 50)
    F[5, 5] = (-50, -50)                   # isolated, mutually inconsistent nodes
    out, v = R._clean(F)
    assert np.isfinite(out).all()
    img = np.random.default_rng(0).random((600, 600)).astype(np.float32)
    G = np.full((3, 3, 2), np.nan, np.float32)
    res = R.warp(img, np.array([100., 300., 500.]), np.array([100., 300., 500.]), G)
    assert np.isfinite(res).all()


def test_warp_shift_past_the_end():
    """A shift larger than the rows left at the end of the plane (gave an empty source slice and a hang)."""
    import signal
    img = np.random.default_rng(1).random((3000, 1200)).astype(np.float32)
    G = np.zeros((11, 4, 2), np.float32)
    G[..., 0] = 330.0
    signal.alarm(60)
    try:
        out = R.warp(img, 128.0 + 256 * np.arange(11), 168.0 + 256 * np.arange(4), G)
    finally:
        signal.alarm(0)
    assert out.shape == img.shape and np.isfinite(out).all()
    assert np.allclose(out[:2000, 10:-10], img[330:2330, 10:-10], atol=1e-4)


if __name__ == "__main__":
    test_warp_shift_past_the_end()
    test_clean_and_warp_never_produce_nan()
    test_recovers_known_fields()
    print("ok")
