"""The launcher plugs our registration into stand-ins with the same hook points as build_ms.py and
recolor_MS_v2.4.py, and refuses builders that lost those hook points."""
import os
import subprocess
import sys
import tempfile
import textwrap

import numpy as np
import tifffile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TESTS = os.path.dirname(os.path.abspath(__file__))

FAKE_FM = textwrap.dedent('''
    import argparse, json, sys
    sys.path.insert(0, %r)
    import synth
    p = argparse.ArgumentParser()
    p.add_argument("--register", default="off", choices=["off", "on"])
    p.add_argument("--dataset", default="Prostate_VA_MS_SYBR_EO")
    p.add_argument("--out")
    p.add_argument("--workers", type=int, default=2)
    a = p.parse_args()
    ph, pe, true = synth.strip(H=4096, W=2464, n=3, amp=6.0)
    rep = {}
    if a.register == "on":
        import ms_register
        ph, pe, rep = ms_register.register_strip(ph, pe, workers=a.workers)
    json.dump(dict(register=a.register, version=rep.get("version")), open(a.out, "w"))
''') % TESTS

FAKE_COLOR = textwrap.dedent('''
    import argparse, concurrent.futures, sys
    import numpy as np
    sys.path.insert(0, %r)
    import synth

    class _V18:
        @staticmethod
        def estimate_z_surface_argmax(scores, guide, *a, **k):
            return np.ones(guide.shape, np.float32)
    v18 = _V18()

    def _colorize_strip_cpu(fh, fe, prev, args):
        return np.zeros(fh.shape + (3,), np.uint8), False

    class StreamColorizer:
        def __init__(self, full_h, full_w, n, args):
            self.full_h, self.full_w, self.n, self.args = full_h, full_w, n, args
            self.xr = {0: (0, full_w)}
            self.use_edof, self.buf, self.prev, self.eff_h = True, {}, None, full_h
        def feed_plane(self, strip_idx, z_idx, channel, raw):
            tag = "H" if channel == self.args.channel_h else "E"
            f = concurrent.futures.Future(); f.set_result(raw)
            self.buf.setdefault(strip_idx, {}).setdefault(z_idx, {})[tag] = f
        def _finalize(self, strip_idx):
            x0, x1 = self.xr[strip_idx]
            planes = self.buf.pop(strip_idx)
            zs = sorted(planes)
            dhs = [planes[z]["H"].result() for z in zs]
            des = [planes[z]["E"].result() for z in zs]
            g = dhs[0][::4, ::4]
            z = v18.estimate_z_surface_argmax([g], g, 0.01)
            self.fused = (dhs[1], des[1])
            rgb, fb = _colorize_strip_cpu(dhs[1], des[1], self.prev, self.args)
        def close(self, out_path):
            np.save(out_path, np.array([1]))

    def main():
        p = argparse.ArgumentParser()
        p.add_argument("--output", required=True)
        a = p.parse_args()
        a.channel_h, a.channel_e = 470, 528
        ph, pe, true = synth.strip(H=4096, W=2464, n=3, amp=6.0)
        sc = StreamColorizer(4096, 2464, 3, a)
        for z in range(3):
            sc.feed_plane(0, z, 470, ph[z]); sc.feed_plane(0, z, 528, pe[z])
        sc._finalize(0)
        sc.close(a.output)
        return 0

    # hook points the launcher checks for: def _finalize(self, strip_idx) / planes = self.buf.pop(strip_idx) /
    # _colorize_strip_cpu( / v18.estimate_z_surface_argmax( / def close(self, out_path) / class StreamColorizer
''') % TESTS


def _run(args):
    return subprocess.run([sys.executable, os.path.join(REPO, "build_registered.py")] + args,
                          capture_output=True, text=True)


def test_fm_hook_uses_pearson():
    import json
    with tempfile.TemporaryDirectory() as d:
        b = os.path.join(d, "build_ms.py")
        open(b, "w").write(FAKE_FM)
        out = os.path.join(d, "o.json")
        r = _run(["--builder", "fm", "--builder_path", b, "--", "--dataset", "X_pearson", "--out", out])
        assert r.returncode == 0, r.stderr[-2000:]
        j = json.load(open(out))
        assert j["register"] == "on" and j["version"].startswith("pearson"), j
        r = _run(["--builder", "fm", "--builder_path", b, "--", "--out", out])          # default dataset: refused
        assert r.returncode != 0 and "existing dataset" in r.stderr + r.stdout


def test_fm_refuses_changed_builder():
    with tempfile.TemporaryDirectory() as d:
        b = os.path.join(d, "build_ms.py")
        open(b, "w").write("print('no hook here')\n")
        r = _run(["--builder", "fm", "--builder_path", b, "--", "--dataset", "X"])
        assert r.returncode != 0 and "no longer has what the hook needs" in r.stderr + r.stdout


def test_color_hook_registers_and_writes_gray():
    with tempfile.TemporaryDirectory() as d:
        b = os.path.join(d, "recolor_MS_v2.4.py")
        open(b, "w").write(FAKE_COLOR)
        out = os.path.join(d, "slide_v24_edof_he.npy")
        r = _run(["--builder", "color", "--builder_path", b, "--gray_dir", os.path.join(d, "gray"),
                  "--", "--output", out])
        assert r.returncode == 0, r.stderr[-3000:]
        assert "register[pearson]" in r.stdout + r.stderr
        g = os.path.join(d, "gray", "slide_v24_edof_he_470.tif")
        assert os.path.exists(g) and os.path.exists(g.replace("_470.tif", "_528.tif"))
        assert os.path.exists(os.path.join(d, "gray", "slide_v24_edof_he_edof_zmap.png"))
        with tifffile.TiffFile(g) as t:
            assert len(t.series[0].levels) == 3
        assert os.path.exists(os.path.join(d, "slide_v24_edof_he_register.json"))


if __name__ == "__main__":
    test_fm_hook_uses_pearson()
    test_fm_refuses_changed_builder()
    test_color_hook_registers_and_writes_gray()
    print("ok")
