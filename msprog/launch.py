"""Run an existing MagicScan EDOF builder with a chosen registration, without editing the builder's files.

fm     fm_build/build_ms.py (FM dataset builder). It calls `ms_register.register_strip(...)` per strip when run with
       `--register on`, importing `ms_register` at that point. Putting our module into sys.modules under that name
       first makes the builder use it; everything else in the builder runs unchanged.
color  z_stack_focussing/recolor_MS_v2.4.py (streaming pseudo-H&E). It has no registration step, so
       StreamColorizer._finalize is wrapped: the strip's destriped planes are registered before the original
       _finalize scores, fuses and colours them. Optionally the fused grayscale channels and the z-map are written
       too (--gray_dir), so edof_align_check.py can score colour-builder output.
"""
import concurrent.futures
import hashlib
import importlib.util
import json
import os
import runpy
import subprocess
import sys
import time
import types

import numpy as np

from . import register as REG

FM_PATHS = ["/sci-it/projects/cedmav/data/ARPA-H/SIM_coloring/fm_build/build_ms.py",
            "/uufs/sci.utah.edu/projects/cedmav/data/ARPA-H/SIM_coloring/fm_build/build_ms.py"]
COLOR_PATHS = ["/sci-it/projects/cedmav/data/ARPA-H/SIM_coloring/z_stack_focussing/recolor_MS_v2.4.py",
               "/uufs/sci.utah.edu/projects/cedmav/data/ARPA-H/SIM_coloring/z_stack_focussing/recolor_MS_v2.4.py"]
# what our hooks rely on; checked before running so a changed builder fails loudly instead of silently
FM_NEEDS = ["import ms_register", "ms_register.register_strip(", '"--register"']
COLOR_NEEDS = ["class StreamColorizer", "def _finalize(self, strip_idx)", "planes = self.buf.pop(strip_idx)",
               "_colorize_strip_cpu(", "v18.estimate_z_surface_argmax(", "def close(self, out_path)"]
PROTECTED_DATASETS = {"Prostate_VA_MS_SYBR_EO"}


def find(paths, given):
    for p in ([given] if given else paths):
        if p and os.path.exists(p):
            return p
    raise SystemExit(f"builder not found (tried {[given] if given else paths}); pass --builder_path")


def preflight(path, needs):
    src = open(path).read()
    missing = [n for n in needs if n not in src]
    if missing:
        raise SystemExit(f"{path} no longer has what the hook needs: {missing}. The builder changed; update "
                         f"msprog/launch.py (or pin --builder_path to a known-good copy).")
    return hashlib.sha256(src.encode()).hexdigest()[:16]


def repo_commit():
    try:
        return subprocess.check_output(["git", "-C", os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                        "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def _reg_fn(workers):
    def register_strip(ph, pe, ref_idx=None, workers_=None, log=print, return_fields=False, **kw):
        w = workers or kw.get("workers") or workers_ or 16
        return REG.register_strip(ph, pe, ref_idx=ref_idx, workers=w, log=log, return_fields=return_fields)
    return register_strip


def _set_flag(args, flag, value):
    """Set `flag value` in a builder argument list (replacing an existing value)."""
    out, i = [], 0
    while i < len(args):
        if args[i] == flag:
            i += 2
            continue
        if args[i].startswith(flag + "="):
            i += 1
            continue
        out.append(args[i])
        i += 1
    return out + [flag, value]


def _get_flag(args, flag, default=None):
    for i, a in enumerate(args):
        if a == flag and i + 1 < len(args):
            return args[i + 1]
        if a.startswith(flag + "="):
            return a.split("=", 1)[1]
    return default


# ---------------------------------------------------------------- fm builder
def run_fm(a, bargs):
    path = find(FM_PATHS, a.builder_path)
    h = preflight(path, FM_NEEDS)
    ds = _get_flag(bargs, "--dataset", "Prostate_VA_MS_SYBR_EO")
    if ds in PROTECTED_DATASETS and not a.allow_protected_dataset:
        raise SystemExit(f"--dataset {ds} is the existing dataset; write the rebuild to a new one "
                         f"(e.g. --dataset {ds}_pearson) or pass --allow_protected_dataset")
    if a.register == "pearson":
        mod = types.ModuleType("ms_register")
        mod.__file__ = os.path.abspath(REG.__file__)
        mod.register_strip = _reg_fn(a.reg_workers)
        mod.VERSION = REG.VERSION
        sys.modules["ms_register"] = mod
        bargs = _set_flag(bargs, "--register", "on")
    elif a.register == "builtin":
        bargs = _set_flag(bargs, "--register", "on")
    else:
        bargs = _set_flag(bargs, "--register", "off")
    print(f"[build_registered] fm builder {path} (sha256 {h}), registration={a.register}"
          f" ({REG.VERSION if a.register == 'pearson' else 'builder default'}), repo {repo_commit()}", flush=True)
    print("[build_registered] note: build_ms.py writes a fixed 'ms_register' description into the manifest; the "
          "per-strip <id>_register.json in _qc/<dataset>/flatfield has the real version", flush=True)
    sys.path.insert(0, os.path.dirname(path))
    sys.argv = [path] + bargs
    runpy.run_path(path, run_name="__main__")


# ---------------------------------------------------------------- colour builder
class _Gray:
    """Fused grayscale channels + z-map captured strip by strip (uint16 memmaps, scaled by the first strip)."""

    def __init__(self, out_dir):
        self.dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        self.mm, self.scale, self.zmap = {}, {}, None
        self.cur = {}

    def put(self, sc, fused_h, fused_e):
        x0 = sc.xr[self.cur["strip"]][0]
        for tag, f in (("h", fused_h), ("e", fused_e)):
            if tag not in self.mm:
                self.mm[tag] = np.lib.format.open_memmap(os.path.join(self.dir, f".tmp_{tag}.npy"), "w+",
                                                         np.uint16, (sc.full_h, sc.full_w))
                v = f[f > 0]
                self.scale[tag] = 40000.0 / max(float(np.percentile(v, 99.9)) if v.size else 1.0, 1e-6)
            sh, sw = f.shape
            self.mm[tag][:sh, x0:x0 + sw] = np.clip(f * self.scale[tag], 0, 65535).astype(np.uint16)
        z = self.cur.get("z")
        if z is not None:
            if self.zmap is None:
                self.zmap = np.zeros((sc.full_h // 4 + 1, sc.full_w // 4 + 1), np.uint8)
            n = max(1, sc.n - 1)
            zz = np.clip(np.rint(z) * (255.0 / n), 0, 255).astype(np.uint8)
            self.zmap[:zz.shape[0], x0 // 4:x0 // 4 + zz.shape[1]] = zz

    def write(self, stem, eff_h, chans):
        import cv2
        import tifffile
        outs = []
        for tag, ch in zip(("h", "e"), chans):
            if tag not in self.mm:
                continue
            a = self.mm[tag][:eff_h]
            levels, cur = [], a
            for _ in range(2):
                rows = []
                for y in range(0, cur.shape[0], 4096):
                    b = np.asarray(cur[y:y + 4096])
                    rows.append(cv2.resize(b, (b.shape[1] // 4, max(1, b.shape[0] // 4)), interpolation=cv2.INTER_AREA))
                cur = np.concatenate(rows, 0)
                levels.append(cur)
            p = os.path.join(self.dir, f"{stem}_{ch}.tif")
            with tifffile.TiffWriter(p, bigtiff=True) as tw:
                tw.write(a, tile=(512, 512), compression="zlib", subifds=len(levels), photometric="minisblack")
                for lv in levels:
                    tw.write(lv, tile=(512, 512), compression="zlib", subfiletype=1, photometric="minisblack")
            outs.append(p)
            del self.mm[tag]
            os.remove(os.path.join(self.dir, f".tmp_{tag}.npy"))
        if self.zmap is not None:
            p = os.path.join(self.dir, f"{stem}_edof_zmap.png")
            cv2.imwrite(p, self.zmap[:eff_h // 4 + 1])
            outs.append(p)
        return outs


def run_color(a, bargs):
    path = find(COLOR_PATHS, a.builder_path)
    h = preflight(path, COLOR_NEEDS)
    spec = importlib.util.spec_from_file_location("recolor_ms_v24", path)
    m = importlib.util.module_from_spec(spec)
    sys.modules["recolor_ms_v24"] = m
    spec.loader.exec_module(m)
    SC = m.StreamColorizer
    reg = _reg_fn(a.reg_workers or 16)
    gray = _Gray(a.gray_dir) if a.gray_dir else None
    reports = {}
    orig_finalize, orig_close = SC._finalize, SC.close
    orig_colorize, orig_zsurf = m._colorize_strip_cpu, m.v18.estimate_z_surface_argmax

    def _finalize(self, strip_idx):
        if gray:
            gray.cur = {"strip": strip_idx, "sc": self}
        if a.register == "pearson" and self.use_edof:
            planes = self.buf[strip_idx]
            zs = sorted(planes)
            dh = [planes[z]["H"].result() for z in zs]
            de = [planes[z]["E"].result() for z in zs]
            ph, pe, rep = reg(dh, de)
            for z, x, y in zip(zs, ph, pe):
                for tag, arr in (("H", x), ("E", y)):
                    f = concurrent.futures.Future()
                    f.set_result(arr)
                    planes[z][tag] = f
            reports[str(strip_idx)] = rep
        return orig_finalize(self, strip_idx)

    def _colorize(fused_h, fused_e, prev, args):
        if gray and gray.cur.get("sc") is not None:
            gray.put(gray.cur["sc"], fused_h, fused_e)
        return orig_colorize(fused_h, fused_e, prev, args)

    def _zsurf(*args, **kw):
        z = orig_zsurf(*args, **kw)
        if gray and "z" not in gray.cur:
            gray.cur["z"] = z
        return z

    def _close(self, out_path):
        r = orig_close(self, out_path)
        stem = os.path.basename(out_path).rsplit(".", 1)[0]
        meta = dict(builder=path, builder_sha256=h, registration=a.register,
                    version=REG.VERSION if a.register == "pearson" else None, repo=repo_commit(),
                    time=time.strftime("%Y-%m-%dT%H:%M:%S"), strips=reports)
        with open(os.path.join(os.path.dirname(out_path), f"{stem}_register.json"), "w") as f:
            json.dump(meta, f, indent=1)
        if gray:
            outs = gray.write(stem, self.eff_h, (str(self.args.channel_h), str(self.args.channel_e)))
            print(f"[build_registered] grayscale + z-map for edof_align_check: {outs}", flush=True)
        return r

    SC._finalize = _finalize
    SC.close = _close
    m._colorize_strip_cpu = _colorize
    m.v18.estimate_z_surface_argmax = _zsurf
    print(f"[build_registered] colour builder {path} (sha256 {h}), registration={a.register}, repo {repo_commit()}",
          flush=True)
    sys.argv = [path] + bargs
    return m.main()
