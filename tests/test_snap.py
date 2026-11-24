"""E1 tile snapping: the tile mapping matches the extractor's geometry, every E1 tile of the snapped arm holds one plane,
and through stand-ins with build_ms.py's hook points the production arm is unchanged while the snapped arm shares the
production strip gains, normalisation and mask."""
import os
import subprocess
import sys
import tempfile
import textwrap

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from msprog import snap as SN  # noqa: E402

FAKE_V18 = textwrap.dedent('''
    import numpy as np
    def _select_edof_dual(planes_h, planes_e, z_surface, feather=False, n_workers=8):
        n = len(planes_h)
        best_z = np.clip(np.rint(z_surface), 0, n - 1).astype(np.int32)
        fh = np.choose(best_z, planes_h).astype(np.float32)
        fe = np.choose(best_z, planes_e).astype(np.float32)
        return fh, fe, best_z
''')

FAKE_MSM = textwrap.dedent('''
    import numpy as np
    def apply_gain(canvas, g, SW):
        for s, v in enumerate(g):
            canvas[:, s * SW:(s + 1) * SW] *= v
    def apply_affine(canvas, b, g, b0, SW):
        raise AssertionError("not used")
    def strip_match(canvas, SW, S, mode="gain", n_bands=48):
        g = [1.0 + 0.25 * s for s in range(S)]
        apply_gain(canvas, g, SW)
        return {"glass": None, "gain": g}
''')

FAKE_BD = textwrap.dedent('''
    import numpy as np
    def tissue_anchors(cn, ce, region, thr):
        return (float(np.percentile(cn, 90)), float(np.percentile(ce, 90))), int(cn.size)
''')

FAKE_BP = textwrap.dedent('''
    import os, sys
    import numpy as np
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import build_dual as bd
    def norm_channel(x, valid):
        lo, hi = np.percentile(x[valid], [1.0, 99.9])
        return np.clip((x - lo) / max(hi - lo, 1e-6), 0, None).astype(np.float32), float(lo), float(hi)
    def finish(a, n_raw, e_raw, t0=None, dirs=None, extra_meta=None):
        valid = (n_raw > 0) | (e_raw > 0)
        cn, lo_n, hi_n = norm_channel(n_raw, valid)
        ce, lo_e, hi_e = norm_channel(e_raw, valid)
        thr = 0.1
        region = np.ones((2, 2), bool)
        fixed = getattr(a, "fixed_region", None)
        if fixed is not None:
            region = fixed
        fac, n_px = bd.tissue_anchors(cn, ce, region, thr)
        a._region = region
        d = os.path.join(a.dataset_root, a.dataset, "grayscale"); os.makedirs(d, exist_ok=True)
        np.save(os.path.join(d, a.image_id + "_470.npy"), cn / fac[0])
        np.save(os.path.join(d, a.image_id + "_528.npy"), ce / fac[1])
''')

FAKE_MS = textwrap.dedent('''
    import argparse, importlib.util, os, sys, types
    import numpy as np
    import cv2
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import build_prestitched as bp
    _ZS = os.environ["FAKE_ZS"]
    _spec = importlib.util.spec_from_file_location("recolor_ms_v18", f"{_ZS}/recolor_MS_v1.8.py")
    v18 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(v18)
    import ms_strip_match as msm

    def main(a):
        nuc, stro = "470", "528"
        strips = [0, 1, 2]; SW = 700; H = 900
        W = SW * (max(strips) + 1)
        outs = {"edof": {c: np.zeros((H, W), np.float32) for c in (nuc, stro)}}
        rng = np.random.default_rng(1)
        for s in strips:
            x0 = s * SW
            ph = [np.full((H, SW), 10.0 * (i + 1), np.float32) for i in range(4)]
            pe = [p * 2 for p in ph]
            if a.register == "on":
                import ms_register
                ph, pe, _ = ms_register.register_strip(ph, pe)
            zs = cv2.GaussianBlur(rng.random((H // 4, SW // 4)).astype(np.float32), (0, 0), 6)
            zs = (zs - zs.min()) / (zs.max() - zs.min()) * 3
            zf = cv2.resize(zs, (SW, H), interpolation=cv2.INTER_LINEAR)
            fh, fe, _ = v18._select_edof_dual(ph, pe, zf, feather=False)
            outs["edof"][nuc][:, x0:x0 + SW] = fh; outs["edof"][stro][:, x0:x0 + SW] = fe
            np.save(os.path.join(a.dataset_root, f"zf_{s}.npy"), zf)
        S_ = max(strips) + 1
        for c in (nuc, stro):
            res = msm.strip_match(outs["edof"][c], SW, S_, mode=a.strip_match)
        for k, cv in outs.items():
            b = types.SimpleNamespace(**vars(a))
            extra = dict(ms_variant=k)
            bp.finish(b, cv[nuc], cv[stro], extra_meta=extra)
        print(f"{a.image_id}: all variants done", flush=True)

    p = argparse.ArgumentParser()
    p.add_argument("slice_dir"); p.add_argument("--image_id", required=True)
    p.add_argument("--dataset", default="Prostate_VA_MS_SYBR_EO"); p.add_argument("--dataset_root")
    p.add_argument("--register", default="off", choices=["off", "on"]); p.add_argument("--mpp", type=float, default=0.7)
    p.add_argument("--strip_match", default="gain")
    main(p.parse_args())
''')


def test_tile_geometry():
    j = np.arange(300000)
    assert (SN.tile_of(j) == np.floor((j + 0.5) * 1.4 / 256)).all()
    f, l = SN.tile_canvas_span(np.arange(1500))
    t = SN.tile_of(np.arange(l[-1] + 2))
    for k in range(1500):
        assert (t[f[k]:l[k]] == k).all() and t[l[k]] == k + 1
    # extractor (extract_fm_bench.fm_tile): dst pixel i of the tile at benchmark x samples canvas x/1.4 + (i+0.5)/1.4
    # - 0.5 (pixel-index space); its nearest canvas pixel must belong to the tile (bleed only from the cubic kernel)
    for x in (0, 256 * 7, 256 * 123, 256 * 400):
        src = x / 1.4 + (np.arange(256) + 0.5) / 1.4 - 0.5
        assert (SN.tile_of(np.rint(src).astype(int)) == x // 256).all()


def _fake_tree(d):
    fm, zs = os.path.join(d, "fm"), os.path.join(d, "zs")
    os.makedirs(fm); os.makedirs(zs)
    for name, src in (("build_ms.py", FAKE_MS), ("build_prestitched.py", FAKE_BP), ("build_dual.py", FAKE_BD),
                      ("ms_strip_match.py", FAKE_MSM)):
        open(os.path.join(fm, name), "w").write(src)
    open(os.path.join(zs, "recolor_MS_v1.8.py"), "w").write(FAKE_V18)
    # build_ms.py preflight strings the hook relies on, present in the stand-in
    src = open(os.path.join(fm, "build_ms.py")).read()
    open(os.path.join(fm, "build_ms.py"), "w").write(
        src + "\n# hook points: import ms_register ms_register.register_strip( \"--register\" "
              "fh, fe, _ = v18._select_edof_dual(ph, pe, zf, feather=False) "
              "res = msm.strip_match(outs[\"edof\"][c], SW, S_, mode=a.strip_match) "
              "bp.finish(b, cv[nuc], cv[stro], extra_meta=extra) ms_variant=k\n")
    bp = open(os.path.join(fm, "build_prestitched.py")).read()
    open(os.path.join(fm, "build_prestitched.py"), "w").write(bp)
    return fm, zs


def test_snap_hooks_end_to_end():
    with tempfile.TemporaryDirectory() as d:
        fm, zs = _fake_tree(d)
        root_a, root_b = os.path.join(d, "plain"), os.path.join(d, "snap")
        os.makedirs(root_a); os.makedirs(root_b)
        env = dict(os.environ, FAKE_ZS=zs)
        # the builder on its own
        r = subprocess.run([sys.executable, os.path.join(fm, "build_ms.py"), "x", "--image_id", "T_S1", "--dataset",
                            "T_ds", "--dataset_root", root_a], capture_output=True, text=True, env=env)
        assert r.returncode == 0, r.stderr[-3000:]
        cells = os.path.join(d, "cells.tsv")
        open(cells, "w").write("image_id\tcol\trow\tlabel\nT_S1\t0\t0\tcancer\nOTHER\t0\t0\tbenign\n")
        r = subprocess.run([sys.executable, os.path.join(REPO, "build_registered.py"), "--builder", "fm",
                            "--builder_path", os.path.join(fm, "build_ms.py"), "--register", "off", "--snap", "e1",
                            "--snap_probe_cells", cells, "--", "x", "--image_id", "T_S1", "--dataset", "T_ds",
                            "--dataset_root", root_b], capture_output=True, text=True, env=env)
        assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
        assert "[snapE1] v1.8 _select_edof_dual wrapped" in r.stdout and "all variants done" in r.stdout
        g = lambda root, i, c: np.load(os.path.join(root, "T_ds", "grayscale", f"{i}_{c}.npy"))
        # arm A is exactly the builder's own output
        for c in ("470", "528"):
            assert np.array_equal(g(root_a, "T_S1", c), g(root_b, "T_S1", c))
        A, B = g(root_b, "T_S1", "470"), g(root_b, "T_S1__snapE1", "470")
        H, W = A.shape
        SW = 700
        tx, ty = SN.tile_of(np.arange(W)), SN.tile_of(np.arange(H))
        zf = np.concatenate([np.load(os.path.join(root_b, f"zf_{s}.npy")) for s in range(3)], 1)
        zp, multi, const = np.rint(zf), 0, True
        for s in range(3):
            for t in np.unique(tx[s * SW:(s + 1) * SW]):
                cols = np.flatnonzero(tx == t)
                cols = cols[(cols >= s * SW) & (cols < (s + 1) * SW)]
                for u in np.unique(ty):
                    blk = B[np.ix_(ty == u, cols)]
                    const &= blk.min() == blk.max()
                    multi += len(np.unique(zp[np.ix_(ty == u, cols)])) > 1
        assert const, "a snapped tile cell holds more than one plane"
        assert multi > 0, "test map produced no production boundaries inside tiles"
        # same plane -> identical value: the snapped arm shares strip gains, normalisation and anchors
        snap_z = np.zeros_like(zp)
        for s in range(3):
            zz, _ = SN.snap_strip(zf[:, s * SW:(s + 1) * SW], s * SW, 4)
            snap_z[:, s * SW:(s + 1) * SW] = zz
        same = zp == snap_z
        assert same.mean() > 0.5 and np.array_equal(A[same], B[same])
        assert not np.array_equal(A, B)
        qc = os.path.join(root_b, "_qc", "T_ds", "snap")
        st = np.load(os.path.join(qc, "T_S1_snap_tiles.npz"))
        assert (st["n_planes"] > 1).sum() == multi
        probes = os.listdir(os.path.join(qc, "probe", "T_S1"))
        assert sorted(probes) == ["cell_c0_r0_s0.npz", "cell_c0_r0_s1.npz", "cell_c0_r0_s2.npz"], probes
        p = np.load(os.path.join(qc, "probe", "T_S1", "cell_c0_r0_s0.npz"))
        assert p["planes_nuc"].shape[0] == 4 and p["prod_nuc"].shape == p["snap_nuc"].shape
        assert np.array_equal(np.choose(p["z_snap"], list(p["planes_nuc"])), p["snap_nuc"])


if __name__ == "__main__":
    test_tile_geometry()
    test_snap_hooks_end_to_end()
    print("ok")
