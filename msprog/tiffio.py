"""Region reads from large (tiled, compressed, pyramidal) TIFFs without loading the whole image."""
import os
import threading

import numpy as np
import tifffile


class TiledReader:
    """Read rectangular regions of one pyramid level. Tiled pages of any tifffile codec decode only the tiles a
    region touches; uncompressed strip pages are memory-mapped; anything else is loaded whole (small files only).
    Thread-safe (pread on a private descriptor)."""

    def __init__(self, path, level=0):
        self.path = path
        self.tf = tifffile.TiffFile(path)
        levels = self.tf.series[0].levels
        self.n_levels = len(levels)
        page = levels[min(level, len(levels) - 1)].keyframe
        self.page = page
        self.h, self.w = page.shape[:2]
        self.spp = page.samplesperpixel
        self._lock = threading.Lock()
        self.mm = self.full = None
        if page.is_tiled:
            self.tw, self.tl = page.tilewidth, page.tilelength
            self.ntx = -(-self.w // self.tw)
            self.offs = np.asarray(page.dataoffsets, np.int64)
            self.cnts = np.asarray(page.databytecounts, np.int64)
            self.fd = os.open(path, os.O_RDONLY)
        elif page.is_contiguous and page.compression == 1:
            self.mm = np.memmap(path, np.dtype(self.tf.byteorder + page.dtype.char), "r",
                                int(page.dataoffsets[0]), page.shape)
        else:
            self.full = page.asarray()

    def _tile(self, ty, tx):
        i = ty * self.ntx + tx
        data = os.pread(self.fd, int(self.cnts[i]), int(self.offs[i]))
        seg, _, _ = self.page.decode(data, i, jpegtables=self.page.jpegtables)
        seg = np.asarray(seg).reshape(self.tl, self.tw, -1)
        return seg[..., 0] if seg.shape[-1] == 1 or self.spp == 1 else seg.mean(-1)

    def region(self, y0, x0, h, w):
        """float32 (h, w); zeros outside the image."""
        out = np.zeros((h, w), np.float32)
        ya, yb, xa, xb = max(0, y0), min(self.h, y0 + h), max(0, x0), min(self.w, x0 + w)
        if ya >= yb or xa >= xb:
            return out
        if self.mm is not None or self.full is not None:
            src = self.mm if self.mm is not None else self.full
            a = np.asarray(src[ya:yb, xa:xb], np.float32)
            out[ya - y0:yb - y0, xa - x0:xb - x0] = a if a.ndim == 2 else a.mean(-1)
            return out
        for ty in range(ya // self.tl, (yb - 1) // self.tl + 1):
            for tx in range(xa // self.tw, (xb - 1) // self.tw + 1):
                t = self._tile(ty, tx)
                ty0, tx0 = ty * self.tl, tx * self.tw
                sy0, sy1 = max(ya, ty0), min(yb, ty0 + self.tl)
                sx0, sx1 = max(xa, tx0), min(xb, tx0 + self.tw)
                out[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = t[sy0 - ty0:sy1 - ty0, sx0 - tx0:sx1 - tx0]
        return out


def thumbnail(path, max_side=4096):
    """(image, scale) from the smallest pyramid level whose long side is >= max_side (scale = level px per level-0
    px); falls back to strided reads of level 0."""
    with tifffile.TiffFile(path) as tf:
        levels = tf.series[0].levels
        h0, w0 = levels[0].shape[:2]
        pick = 0
        for i, lv in enumerate(levels):
            if max(lv.shape[:2]) >= max_side:
                pick = i
        if pick > 0 or max(h0, w0) <= 2 * max_side:
            a = levels[pick].asarray()
            a = a if a.ndim == 2 else a.mean(-1)
            return np.asarray(a, np.float32), a.shape[1] / w0
    r = TiledReader(path)
    step = max(1, int(np.ceil(max(r.h, r.w) / max_side)))
    rows = []
    for y in range(0, r.h, step * 64):
        band = r.region(y, 0, min(step * 64, r.h - y), r.w)
        rows.append(band[::step, ::step])
    a = np.concatenate(rows, 0)
    return a, a.shape[1] / r.w
