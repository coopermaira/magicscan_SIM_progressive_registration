# MagicScan SIM progressive registration

Registration of the MagicScan z-planes and channels **before** EDOF fusion, plus a tool that scores how well any
finished EDOF image is aligned. Both plug into the group's existing builders without editing them.

| What | File |
|---|---|
| Score existing or new EDOF images; compare two builds of a slide | `edof_align_check.py` |
| Rebuild with registration (fm dataset builder or the v2.4 colour builder) | `build_registered.py` |
| Try the registration on one raw strip, before/after numbers + overlays | `tools/try_strip.py` |
| Core: coarse-to-fine Pearson matcher | `msprog/pearson.py` |
| Core: per-strip registration, same call as `fm_build/ms_register.register_strip` | `msprog/register.py` |

Matching is Pearson correlation (normalised cross-correlation) everywhere; there is no phase correlation.

## Setup

```bash
git clone https://github.com/coopermaira/magicscan_SIM_progressive_registration.git
cd magicscan_SIM_progressive_registration
# any Python >= 3.9 with numpy, opencv-python(-headless), tifffile, h5py, e.g. the builders' prostate_env
python -m pytest tests -q        # synthetic tests, ~30 s, no data needed
```

## 1. Score the images you already have

```bash
DS=/sci-it/hosts/olympus/magicscan/SIM/0_foundation_model_datasets/Prostate_VA_MS_SYBR_EO
python edof_align_check.py run $DS:VA026_S4 $DS:VA029_S1 --out results/baseline
python edof_align_check.py selftest $DS:VA026_S4        # optional: proves the metrics on this image
```

An image is named `<dataset_dir>:<image_id>` (fm-builder layout) or, for any other pair of grayscale TIFFs,
`<nuclear.tif>,<stromal.tif>[,<zmap.png>]`.

Per image it writes `<id>_align.json` (summary), `<id>_align_windows.tsv` (every window) and maps
`<id>_align_channel.png` / `<id>_align_planes.png`. Measured on a fixed grid of 1,024 px tissue windows
(up to 600 per image, `--max_windows`):

| Metric | Meaning | Needs |
|---|---|---|
| `channel` | 470 <-> 528 offset of the EDOF (median, p90, share of windows > 2 px and > 5 px). What a model sees as colour fringing. | EDOF only |
| `planes` | Top <-> bottom z-plane offset of the nuclear channel. How well the passes were registered before fusion. | the `__z<k>` variants the fm builder writes |
| `seam` | Cross-seam continuity where the EDOF switches source plane: correlation across the seam divided by the same correlation inside a plane. 1 = seamless; it falls when misaligned planes were fused. Insensitive below ~2 px of seam offset. | the builder's z-map (`_qc/<dataset>/flatfield/<id>_edof_zmap.png`) |
| `sharpness` | Tenengrad focus measure of the nuclear EDOF; guards against registration that blurs. | EDOF only |

Offsets are in native pixels (0.7 um). A window counts only when the Pearson peak is confident (r >= 0.2, not
on the search edge, no near-equal second peak); `n_accepted` says how many did.

## 2. Rebuild with registration

**fm dataset builder** (`fm_build/build_ms.py`). Everything after `--` is passed to the builder unchanged; write to
a new dataset name, the launcher refuses to write into `Prostate_VA_MS_SYBR_EO`.

```bash
python build_registered.py --builder fm --register pearson -- \
    "/sci-it/hosts/olympus/magicscan/SIM/Prostate study - VA/VA0026/VA026_Slice4_9z_steps" \
    --image_id VA026_S4 --dataset Prostate_VA_MS_SYBR_EO_pearson \
    --ff select --strip_match gain --mask clipseg --workers 6
```

`--register builtin` runs the builder's own `ms_register.py` instead, `--register off` a plain rebuild.

**Colour builder** (`z_stack_focussing/recolor_MS_v2.4.py`, replay of location composites). `--gray_dir` also writes
the fused 470/528 grayscale and the z-map, which is what `edof_align_check.py` reads.

```bash
python build_registered.py --builder color --register pearson --gray_dir results/va026_s4_color/gray -- \
    --replay --edof --input_dir "<folder of location composites>" --name VA026_S4 --output_dir results/va026_s4_color
```

Run it once with `--register off` as well, to have a baseline built the same way.

How the plug-in works: `build_ms.py --register on` imports a module named `ms_register` and calls
`register_strip(planes_470, planes_528, workers=...)` per strip before fusing. The launcher puts this repo's
module into Python's module table under that name first, so the builder uses it; the builder file is not touched.
For the colour builder the launcher wraps `StreamColorizer._finalize` so each strip's planes are registered before
the original code scores, fuses and colours them. Before running, the launcher checks that the builder still has
these hook points and stops with an error if it does not.

## 3. Compare

```bash
python edof_align_check.py compare $DS:VA026_S4 /sci-it/.../Prostate_VA_MS_SYBR_EO_pearson:VA026_S4 --out results/cmp_VA026_S4
```

B is measured on A's windows. Prints and saves (`compare.txt`, `compare.json`) every metric for A and B with the
difference and a verdict (better / same / worse, with small tolerances; sharpness flags B if it lost > 3 %).

## Method (`msprog/register.py`)

Per strip of N planes x 2 channels (2N passes):

1. **Pairs**: every adjacent pair, 470 z <-> z+1, 528 z <-> z+1, 470 <-> 528 at the same z.
2. **Matching** (`msprog/pearson.py`), per 1,024-row chunk: chunk-wide shift at 1/16 res (+-256 px), per 1,024 px
   window at 1/4 res (+-48 px), per 256 px sub-window at 1/4 res (+-16 px), then full res (+-2 px) with a
   sub-pixel peak. Column destripe first (sensor stripes otherwise pin the match at dx = 0). Each window's 16
   sub-window shifts are fitted with an offset + stretch model; sub-windows far from their window's fit are dropped,
   the others keep their own measured shift (so stage jerks inside a window survive).
3. **Joint solve** per 256 px node: the displacement of every pass relative to the reference (470 of the middle
   plane) from all pairs at once by least squares, dropping the worst pair while it disagrees by > 1.5 px and the
   system stays constrained. Errors spread out instead of piling up as they do when planes are chained outward.
4. **Cleaning**: nodes > 3 px from the median of their neighbours are dropped; holes are filled from the valid
   neighbours. Measured nodes are kept as they are.
5. **Warp**: each original plane is resampled once (cubic), field interpolated bilinearly between nodes.
6. **Verification**: the pairs are re-measured on the registered planes (every 4th block); the report gives the
   residual shifts and flags strips whose median residual exceeds 1 um.

The report has the same fields as the builder's own registration report (applied plane and channel shifts in um,
residuals, flags) plus the closure numbers from the joint solve and timings; the fm builder saves it as
`_qc/<dataset>/flatfield/<id>_register.json`.

## Validation so far

* Synthetic strip (`tests/test_register.py`): 5 planes x 2 channels, blurred outer planes, known smooth fields up to
  12 px plus row-to-row stage jerks: recovered field error 0.04–0.13 px median, p90 <= 0.56 px, per pass.
* Check self-test on the real VA026_S4 EDOF: injected channel and plane shifts recovered (p90 <= 0.01 px, integer
  shifts); seam index 1.00 at 0 px, 0.81 at 4 px, 0.21 at 8 px.
* Real raw strip (`tools/try_strip.py`, VA026_S4 strip 3, rows 20,480–32,768, 9 planes, reference plane 5 of 9),
  offsets on 48 tissue windows before -> after registration (median / p90, px):
  470 <-> 528 per plane 3.7–13.4 / 5–36 -> 0.13–1.4 / 0.3–3.8; planes 6–9 vs reference 15–54 / 21–75 -> 0.14–0.78 /
  0.5–1.6; planes 1–4 (the defocused side) 7–44 / 36–108 -> 1.6–3.6 / 4–16. The defocused side is not yet solved
  as well as the in-focus side. 390 s for this 12,288-row section on 8 threads of a busy host.
* Baseline check, existing images: VA026_S4 channel median 3.1 px (p90 10.4), top–bottom planes median 30.6 px,
  seam 0.08; VA029_S1 channel median 2.3 px (p90 20.5), planes 5.6 px, seam 0.92.

## Known limits

* The fm builder holds a whole strip's planes in memory, and the registration adds one registered copy plus
  per-thread working buffers (~1 GB per thread).
* Content pulled in from beyond the strip's left/right edge by a shift is mirrored, not real.
* Windows where a plane is badly out of focus do not match; the joint solve fills them from the other pairs and
  neighbouring nodes.
* `build_ms.py` writes a fixed text describing its own registration into the manifest's `ms_register` field even
  when this module ran; the `_register.json` next to it has the real version (`pearson_c2f_joint_...`).
