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


if __name__ == "__main__":
    test_rebuild_and_compare()
    print("ok")
