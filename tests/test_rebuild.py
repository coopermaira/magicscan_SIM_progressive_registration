"""End to end on a synthetic raw slice: rebuild (off + pearson), then compare with the check. Registration must
reduce the channel offset."""
import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import tifffile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import synth  # noqa: E402


def test_discover_cartridge_naming():
    sys.path.insert(0, os.path.join(REPO, "tools"))
    import rebuild_edof as RB
    with tempfile.TemporaryDirectory() as d:
        a = np.zeros((8, 8), np.uint16)
        for s, zs in ((0, ("79.900000", "80.100000")), (1, ("80.010000", "80.210000"))):     # z differs per strip
            for z in zs:
                for ch in ("470", "528"):
                    tifffile.imwrite(os.path.join(d, f"V_cartridge1_strip{s}_z{z}_{ch}nm.tiff"), a)
                    for k in range(3):
                        tifffile.imwrite(os.path.join(d, f"V_cartridge1_strip{s}_z{z}_{ch}nm_{k}.tiff"), a)
        tifffile.imwrite(os.path.join(d, "V_cartridge1_strip0_pseudo_color.tiff"), a)
        open(os.path.join(d, "V_cartridge1_strip2_z80.0_470nm_0.tiff"), "w").close()          # empty file: ignored
        P = RB.plan(RB.discover(d), "auto")
        assert sorted(P) == [0, 1], P
        assert [len(P[s][0]) for s in (0, 1)] == [2, 2] and P[0][1] == "recon"
        assert RB.plan(RB.discover(d), "phases")[1][1] == "phases"
    with tempfile.TemporaryDirectory() as d:                                                 # VA033: loc<N> + strip
        a = np.zeros((8, 8), np.uint16)
        for z in (1, 2):
            for ch in ("470", "528"):
                for k in range(3):
                    tifffile.imwrite(os.path.join(d, f"V_Slice2_loc{z}_cartridge1_strip0_{ch}nm_{k}.tiff"), a)
        assert len(RB.plan(RB.discover(d), "auto")[0][0]) == 2


def test_rebuild_and_compare():
    with tempfile.TemporaryDirectory() as d:
        raw = os.path.join(d, "raw")
        os.makedirs(raw)
        for s in range(2):
            ph, pe, _ = synth.strip(H=4096, W=2464, n=3, amp=8.0, chan_amp=5.0, seed=20 + s)
            for z in range(3):
                for ch, a in (("470", ph[z]), ("528", pe[z])):
                    for p in range(3):
                        tifffile.imwrite(os.path.join(raw, f"T_S1_location{z + 1}_{ch}_20mm_s_{s}_0_{p}.tiff"),
                                         (a * 1500).astype(np.uint16))
        out = os.path.join(d, "out")
        r = subprocess.run([sys.executable, os.path.join(REPO, "tools", "rebuild_edof.py"), raw, "--image_id", "T_S1",
                            "--out", out, "--workers", "4"], capture_output=True, text=True)
        assert r.returncode == 0, r.stderr[-3000:]
        specs = dict(l.rstrip("\n").split("\t") for l in open(os.path.join(out, "specs.tsv")))
        r = subprocess.run([sys.executable, os.path.join(REPO, "edof_align_check.py"), "compare", specs["off"],
                            specs["pearson"], "--out", os.path.join(d, "cmp"), "--workers", "4",
                            "--strip_width", "2464"], capture_output=True, text=True)
        assert r.returncode == 0, r.stderr[-3000:]
        print(r.stdout)
        j = json.load(open(os.path.join(d, "cmp", "compare.json")))
        a, b = j["A"]["channel"], j["B"]["channel"]
        assert b["median_px"] < a["median_px"] and b["median_px"] < 1.0, (a, b)
        r = subprocess.run([sys.executable, os.path.join(REPO, "tools", "preview_pair.py"), os.path.join(d, "cmp"),
                            "--out", os.path.join(d, "prev")], capture_output=True, text=True)
        assert r.returncode == 0, r.stderr[-3000:]
        assert any(f.endswith("_crop00.jpg") for f in os.listdir(os.path.join(d, "prev")))
        os.makedirs(os.path.join(out, "compare"), exist_ok=True)
        import shutil
        shutil.copy(os.path.join(d, "cmp", "compare.json"), os.path.join(out, "compare", "compare.json"))
        r = subprocess.run([sys.executable, os.path.join(REPO, "tools", "summarize_eval.py"), d, "--out",
                            os.path.join(d, "sum")], capture_output=True, text=True)
        assert r.returncode == 0, r.stderr[-3000:]
        print(r.stdout)
        assert "T_S1" in open(os.path.join(d, "sum", "slides.tsv")).read()


if __name__ == "__main__":
    test_rebuild_and_compare()
    print("ok")
