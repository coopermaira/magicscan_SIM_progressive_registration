"""Snap EDOF plane choices to the E1 benchmark tile grid, so that no inference tile contains a z-plane boundary.

E1 tiles (eval_suite/build_e1.py, distill/extract_fm_bench.py) are 256 px at 0.5 um/px with origin (0, 0) on the
0.5 um/px benchmark image, which is the 0.7 um/px FM canvas scaled by exactly 1.4. A tile at benchmark (x, y) therefore
reads canvas [x/1.4, (x+256)/1.4) = 182.857 canvas px; extract_fm_bench maps destination pixel centre i + 0.5 to canvas
edge coordinate x/1.4 + (i + 0.5)/1.4. Canvas pixel j (centre j + 0.5) belongs to tile floor((j + 0.5) * 1.4 / 256),
computed exactly in integers as ((2j + 1) * 7) // 2560.

snap_strip takes the builder's full-resolution float plane map of one strip (what v1.8 _select_edof_dual receives),
rounds it the way _select_edof_dual does, and gives every (tile row, tile column) cell of the strip the plane most of its
pixels already chose (ties: the lower plane). The builder's own smoothing and flat-focus fallback decisions are kept;
boundaries only move to the nearest tile edge. A tile that straddles a strip boundary is snapped per strip (its two parts
come from different scan passes either way).

Limits that snapping cannot remove: the cubic resampling of the extractor reads ~2 canvas px past a tile edge, so the
outermost 1-2 px of a tile can see the neighbouring tile's plane; reference-fill bands set inside the registration are
not plane choices and are untouched.
"""
import numpy as np

VERSION = "snap_e1_2026-11-24"
NUM, DEN = 7, 2560          # tile = floor((2j + 1) * 7 / 2560)  <=>  floor((j + 0.5) * 1.4 / 256)
E1_TILE, E1_SCALE = 256, 1.4


def tile_of(j):
    """E1 tile index of canvas pixel index j (array or int)."""
    return ((2 * np.asarray(j, np.int64) + 1) * NUM) // DEN


def tile_canvas_span(t):
    """First and one-past-last canvas pixel of E1 tile t (pixels whose centre falls inside the tile)."""
    t = np.asarray(t, np.int64)
    # smallest j with (2j + 1) * 7 >= t * 2560  ->  j >= (t * 2560 / 7 - 1) / 2
    first = -(-(t * DEN - NUM) // (2 * NUM))
    first = np.maximum(first, 0)
    last = -(-((t + 1) * DEN - NUM) // (2 * NUM))
    return first, last


def snap_strip(zf, x0, n_planes):
    """zf: (H, SW) float plane map of the strip whose first canvas column is x0.
    Returns (snapped float32 map of the same shape, stats) where stats is a dict of 1-D arrays over the strip's tile
    cells: ty, tx (global E1 tile indices), n_px (pixels of the cell in this strip), n_planes (distinct production
    planes), plane (snapped plane), maj_frac (fraction of the cell's pixels that already had that plane),
    straddle (the tile also has pixels in a neighbouring strip)."""
    H, SW = zf.shape
    cols = np.arange(x0, x0 + SW)
    txg = tile_of(cols)
    tx = txg - txg[0]
    ntx = int(tx[-1]) + 1
    ty_rows = tile_of(np.arange(H))
    starts = np.flatnonzero(np.r_[True, ty_rows[1:] != ty_rows[:-1]])
    ends = np.r_[starts[1:], H]
    out = np.empty((H, SW), np.float32)
    # tiles cut by the strip's left or right edge
    straddle = np.zeros(ntx, bool)
    straddle[0] = x0 > 0 and tile_of(x0 - 1) == txg[0]
    straddle[-1] = straddle[-1] or tile_of(x0 + SW) == txg[-1]
    S = {k: [] for k in ("ty", "tx", "n_px", "n_planes", "plane", "maj_frac", "straddle")}
    for r0, r1 in zip(starts, ends):
        z = np.clip(np.rint(zf[r0:r1]), 0, n_planes - 1).astype(np.int64)
        cnt = np.bincount((tx[None, :] * n_planes + z).ravel(), minlength=ntx * n_planes).reshape(ntx, n_planes)
        maj = cnt.argmax(1)
        out[r0:r1] = maj[tx].astype(np.float32)[None, :]
        tot = cnt.sum(1)
        S["ty"].append(np.full(ntx, ty_rows[r0], np.int32))
        S["tx"].append((np.arange(ntx) + txg[0]).astype(np.int32))
        S["n_px"].append(tot.astype(np.int32))
        S["n_planes"].append((cnt > 0).sum(1).astype(np.int8))
        S["plane"].append(maj.astype(np.int8))
        S["maj_frac"].append((cnt.max(1) / np.maximum(tot, 1)).astype(np.float32))
        S["straddle"].append(straddle.copy())
    return out, {k: np.concatenate(v) for k, v in S.items()}


def cell_rect(col, row, margin=8):
    """Canvas rectangle (x0, x1, y0, y1) covering every E1 tile of 2 mm cell (col, row) (cell = coords // 4000 at
    0.5 um/px) plus `margin` canvas px for the extractor's resampling kernel."""
    t0x, t1x = -(-col * 4000 // E1_TILE), -(-(col + 1) * 4000 // E1_TILE)   # tiles whose x // 4000 == col
    t0y, t1y = -(-row * 4000 // E1_TILE), -(-(row + 1) * 4000 // E1_TILE)
    x0 = int(np.floor(t0x * E1_TILE / E1_SCALE)) - margin
    x1 = int(np.ceil(t1x * E1_TILE / E1_SCALE)) + margin
    y0 = int(np.floor(t0y * E1_TILE / E1_SCALE)) - margin
    y1 = int(np.ceil(t1y * E1_TILE / E1_SCALE)) + margin
    return max(x0, 0), x1, max(y0, 0), y1
