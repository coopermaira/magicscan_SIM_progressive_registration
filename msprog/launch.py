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


# ---------------------------------------------------------------- fm builder: E1 tile snapping (second arm)
SNAP_NEEDS = {
    "build_ms.py": ["x0 = s * SW", "fh, fe, _ = v18._select_edof_dual(ph, pe, zf, feather=False)",
                    'res = msm.strip_match(outs["edof"][c], SW, S_, mode=a.strip_match)',
                    "bp.finish(b, cv[nuc], cv[stro], extra_meta=extra)", "ms_variant=k",
                    'spec_from_file_location("recolor_ms_v18"', "W = SW * (max(strips) + 1)",
                    "import build_prestitched as bp", "import ms_strip_match as msm"],
    "build_prestitched.py": ["cn, lo_n, hi_n = norm_channel(n_raw, valid)", "ce, lo_e, hi_e = norm_channel(e_raw, valid)",
                             "fac, n_px = bd.tissue_anchors(cn, ce, region, thr)", "if fixed is not None:",
                             "a._region = region"],
    "ms_strip_match.py": ["def apply_gain(canvas, g, SW)", "def apply_affine(canvas, b, g, b0, SW)",
                          'def strip_match(canvas, SW, S, mode="gain"'],
}
SNAP_SUFFIX = "__snapE1"


class SnapHook:
    """Builds a second EDOF canvas with plane choices snapped to the E1 tile grid, from the SAME registered planes and
    the SAME production plane map, inside one build_ms.py run:
      v18._select_edof_dual   production selection is returned unchanged (arm A = the builder's own output); the
                              snapped map (msprog.snap) is selected into our canvas (arm B); per-tile stats and the
                              probe-cell crops (all planes, both fused results) are saved
      msm.strip_match         the gains measured on the production canvas are applied to arm B (as the builder does
                              for its z variants)
      bp.finish               after the production EDOF is finished, arm B goes through finish() with the production
                              tissue mask (fixed_region), normalisation (P1 / P99.9) and colour anchors, so the arms
                              differ only in the plane choice. Output id <image_id>__snapE1."""

    def __init__(self, qc_dir, probe_cells, mpp):
        if abs(mpp - 0.7) > 1e-9:
            raise SystemExit(f"--snap e1 assumes the 0.7 um/px canvas of the E1 benchmark (got --mpp {mpp})")
        self.qc, self.cells = qc_dir, probe_cells
        os.makedirs(qc_dir, exist_ok=True)
        self.cv, self.done, self.stats, self.state = {}, set(), [], None
        self.norm_rec, self.norm_i, self.anchor = [], 0, None
        self.image_id = None

    # -- selection
    def wrap_select(self, orig):
        from . import snap as SN

        def _select_edof_dual(planes_h, planes_e, z_surface, feather=False, n_workers=8):
            res = orig(planes_h, planes_e, z_surface, feather=feather, n_workers=n_workers)
            f = sys._getframe(1).f_locals
            x0, s, H, W, nuc, stro = f["x0"], f["s"], f["H"], f["W"], f["nuc"], f["stro"]
            self.image_id = f["a"].image_id
            if not self.cv:
                self.nuc, self.stro = nuc, stro
                self.cv = {c: np.zeros((H, W), np.float32) for c in (nuc, stro)}
            zsn, st = SN.snap_strip(z_surface, x0, len(planes_h))
            fh, fe, _ = orig(planes_h, planes_e, zsn, feather=False, n_workers=n_workers)
            SW = self.SW = z_surface.shape[1]
            self.cv[nuc][:, x0:x0 + SW] = fh
            self.cv[stro][:, x0:x0 + SW] = fe
            self.done.add(int(x0))
            st["strip"] = np.full(len(st["ty"]), s, np.int16)
            self.stats.append(st)
            changed = float((np.rint(z_surface) != zsn).mean())
            print(f"    [snapE1] strip {s}: {int((st['n_planes'] > 1).sum())} of {len(st['ty'])} tile cells had >1 plane; "
                  f"{100 * changed:.2f} % of pixels changed plane", flush=True)
            for cell in self.cells:
                if cell["image_id"] != self.image_id:
                    continue
                cx0, cx1, cy0, cy1 = cell["rect"]
                a0, a1 = max(cx0, x0), min(cx1, x0 + SW)
                if a0 >= a1:
                    continue
                cy1 = min(cy1, H)
                sl = (slice(cy0, cy1), slice(a0 - x0, a1 - x0))
                d = os.path.join(self.qc, "probe", self.image_id)
                os.makedirs(d, exist_ok=True)
                np.savez(os.path.join(d, f"cell_c{cell['col']}_r{cell['row']}_s{s}.npz"),
                         x0=a0, y0=cy0, strip=s, label=cell["label"], col=cell["col"], row=cell["row"],
                         planes_nuc=np.stack([p[sl] for p in planes_h]), planes_stro=np.stack([p[sl] for p in planes_e]),
                         prod_nuc=res[0][sl], prod_stro=res[1][sl], snap_nuc=fh[sl], snap_stro=fe[sl],
                         z_prod=np.clip(np.rint(z_surface[sl]), 0, 255).astype(np.uint8), z_snap=zsn[sl].astype(np.uint8))
            return res
        return _select_edof_dual

    def _fill(self, c, src):
        """Strips the builder fused without calling the selection (single-plane strips) are copied from production."""
        if not self.cv:
            return
        SW = self.SW
        for x0 in range(0, src.shape[1], SW):
            if x0 not in self.done:
                self.cv[c][:, x0:x0 + SW] = src[:, x0:x0 + SW]

    # -- strip gains
    def wrap_strip_match(self, msm):
        orig = msm.strip_match

        def strip_match(canvas, SW, S, mode="gain", **kw):
            c = sys._getframe(1).f_locals.get("c")
            if self.cv and c in self.cv:
                self._fill(c, canvas)
            res = orig(canvas, SW, S, mode=mode, **kw)
            if self.cv and c in self.cv:
                if res.get("glass") is not None:
                    msm.apply_affine(self.cv[c], res["glass"], res["gain"], res["b0"], SW)
                else:
                    msm.apply_gain(self.cv[c], res["gain"], SW)
                self.matched = getattr(self, "matched", set()) | {c}
            return res
        msm.strip_match = strip_match

    # -- final stage
    def wrap_finish(self, bp, bd):
        orig_finish, orig_norm, orig_anch = bp.finish, bp.norm_channel, bd.tissue_anchors

        def norm_channel(x, valid):
            if self.state == "reuse":
                lo, hi = self.norm_rec[self.norm_i]
                self.norm_i += 1
                return np.clip((x - lo) / max(hi - lo, 1e-6), 0, None).astype(np.float32), float(lo), float(hi)
            out = orig_norm(x, valid)
            if self.state == "record":
                self.norm_rec.append(out[1:])
            return out

        def tissue_anchors(cn, ce, region, thr):
            if self.state == "reuse":
                return self.anchor
            out = orig_anch(cn, ce, region, thr)
            if self.state == "record":
                self.anchor = out              # the last call is the final anchor (line 131)
            return out

        def finish(b, n_raw, e_raw, t0=None, dirs=None, extra_meta=None):
            if (extra_meta or {}).get("ms_variant") != "edof" or not self.cv:
                return orig_finish(b, n_raw, e_raw, t0=t0, dirs=dirs, extra_meta=extra_meta)
            for c, src in ((self.nuc, n_raw), (self.stro, e_raw)):
                if c not in getattr(self, "matched", set()):     # --strip_match off: copy single-plane strips now
                    self._fill(c, src)
            self.state, self.norm_rec, self.anchor = "record", [], None
            r = orig_finish(b, n_raw, e_raw, t0=t0, dirs=dirs, extra_meta=extra_meta)
            b2 = types.SimpleNamespace(**vars(b))
            b2.image_id = b.image_id + SNAP_SUFFIX
            b2.fixed_region = b._region
            b2.source_note = getattr(b, "source_note", "") + "; plane choice snapped to the E1 tile grid"
            st = {k: np.concatenate([s[k] for s in self.stats]) for k in self.stats[0]} if self.stats else {}
            np.savez_compressed(os.path.join(self.qc, f"{b.image_id}_snap_tiles.npz"), **st)
            from . import snap as SN
            ex2 = dict(extra_meta, ms_variant="edof_snapE1",
                       ms_snap=dict(version=SN.VERSION, grid="E1: 256 px @ 0.5 um/px, origin (0,0), = canvas x 1.4",
                                    tile_cells=int(len(st.get("ty", []))),
                                    tile_cells_multiplane_prod=int((st["n_planes"] > 1).sum()) if st else 0,
                                    normalisation="reused from the production EDOF", mask="reused from the production EDOF"))
            self.state, self.norm_i = "reuse", 0
            print(f"[snapE1] finishing {b2.image_id} with the production mask, normalisation and anchors", flush=True)
            try:
                orig_finish(b2, self.cv[self.nuc], self.cv[self.stro], extra_meta=ex2)
            finally:
                self.state = None
                self.cv = {}
            return r

        bp.finish, bp.norm_channel, bd.tissue_anchors = finish, norm_channel, tissue_anchors


def _install_snap(a, path, bargs):
    fm_dir = os.path.dirname(path)
    for f, needs in SNAP_NEEDS.items():
        preflight(os.path.join(fm_dir, f), needs)
    root = _get_flag(bargs, "--dataset_root", "/sci-it/hosts/olympus/magicscan/SIM/0_foundation_model_datasets")
    ds = _get_flag(bargs, "--dataset", "Prostate_VA_MS_SYBR_EO")
    cells = []
    if a.snap_probe_cells:
        from . import snap as SN
        iid = _get_flag(bargs, "--image_id")
        for line in open(a.snap_probe_cells):
            f = line.rstrip("\n").split("\t")
            if not f[0] or f[0].startswith("#") or f[0] == "image_id" or f[0] != iid:
                continue
            col, row = int(f[1]), int(f[2])
            cells.append(dict(image_id=f[0], col=col, row=row, label=f[3], rect=SN.cell_rect(col, row)))
        print(f"[snapE1] {len(cells)} probe cells for {iid}", flush=True)
    hook = SnapHook(os.path.join(root, "_qc", ds, "snap"), cells, float(_get_flag(bargs, "--mpp", "0.7")))
    sys.path.insert(0, fm_dir)
    import build_prestitched as bp      # same module objects build_ms.py imports next
    import build_dual as bd
    import ms_strip_match as msm
    hook.wrap_finish(bp, bd)
    hook.wrap_strip_match(msm)
    # v1.8 is loaded by spec_from_file_location inside build_ms.py: patch it right after its exec_module
    orig_spec = importlib.util.spec_from_file_location

    def spec_from_file_location(name, location=None, *args, **kw):
        spec = orig_spec(name, location, *args, **kw)
        if spec is not None and str(location).endswith("recolor_MS_v1.8.py"):
            ld = spec.loader
            orig_exec = ld.exec_module

            def exec_module(module):
                orig_exec(module)
                if not hasattr(module, "_select_edof_dual"):
                    raise SystemExit("recolor_MS_v1.8.py has no _select_edof_dual; --snap e1 needs updating")
                module._select_edof_dual = hook.wrap_select(module._select_edof_dual)
                print("[snapE1] v1.8 _select_edof_dual wrapped", flush=True)
            ld.exec_module = exec_module
        return spec
    importlib.util.spec_from_file_location = spec_from_file_location
    return hook


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
    if getattr(a, "snap", "off") == "e1":
        _install_snap(a, path, bargs)
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
        from .tiffio import write_pyramid
        outs = []
        for tag, ch in zip(("h", "e"), chans):
            if tag not in self.mm:
                continue
            p = os.path.join(self.dir, f"{stem}_{ch}.tif")
            write_pyramid(p, self.mm[tag][:eff_h])
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
