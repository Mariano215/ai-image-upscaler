---
name: upscale-for-print
description: >-
  Upscale an image to print resolution using AI super-resolution (Real-ESRGAN). Use whenever the
  user wants to enlarge or upscale an image for printing, make a poster, increase an image's
  resolution or DPI for print, prepare artwork/photos for large-format printing, or turn a small
  or low-res image into a print-ready high-res file at a target inch size and DPI. Triggers on
  phrases like "upscale this for print", "make this printable at 24x36", "increase the resolution",
  "super-resolution", "enhance this image for a poster", "300 dpi for print", or "this is too
  low-res to print" — even if the user doesn't name Real-ESRGAN or a specific tool.
---

# Upscale for Print

Turn a small/low-res image into a print-ready high-resolution file using AI super-resolution.
Plain interpolation (Photoshop "resize", `Image.resize`) gets blurry past ~2x; a super-resolution
model reconstructs clean edges and plausible texture, which is what makes a 5–8x enlargement look
professional at poster sizes.

The bundled script `scripts/upscale_for_print.py` does the whole job: tiled GPU super-resolution
→ resample to the exact target pixels → write a print-ready TIFF (target DPI, sRGB, LZW) plus a
JPEG proof. Model weights auto-download on first run.

## Requirements

- Python with: `spandrel`, `torch`, `pillow`, `numpy` (`pip install spandrel torch pillow numpy`;
  a venv is recommended since torch is large). Works on Windows / macOS / Linux.
- The device auto-detects (CUDA → Apple MPS → CPU); override with `--device {auto,cuda,mps,cpu}`.
  A GPU is strongly recommended — CPU works but is slow on large canvases.
- `spandrel` is used to run the model weights directly. Do **not** reach for the `realesrgan` pip
  package: it depends on `basicsr`, which is broken on torchvision ≥0.17 (`functional_tensor` was
  removed). `spandrel` avoids that entirely.

## Workflow

Follow these steps. The thinking happens here; the script does the pixels.

### 1. Inspect the source
Get dimensions, mode, and DPI before doing anything:
```python
from PIL import Image; Image.MAX_IMAGE_PIXELS=None
im = Image.open(src); print(im.size, im.mode, im.info.get("dpi"))
```

### 2. Pick the best source file
Quality is capped by the source, so choose deliberately:
- **Prefer a lossless original** — PSD, PNG, or TIFF. The script reads a PSD's merged composite
  directly (no flatten needed).
- **Avoid JPEGs** when a lossless sibling exists — super-resolution amplifies JPEG block artifacts.
- If several versions exist, confirm which is current/correct with the user. Stale exports are a
  real trap (e.g. an old PNG with an outdated logo). When in doubt, diff candidate files or show
  the user a crop of any region they care about.

### 3. Define the target
- From physical size: `target_px = inches × dpi`. 24×36" @ 300 dpi = 7200×10800.
- Check aspect ratio against the source. If they differ, decide with the user how to reconcile:
  `--fit cover` (fill + center-crop, default), `contain` (pad), or `stretch` (distort). Posters
  usually match already; flag a mismatch rather than silently cropping.
- Note honestly: a large upscale (≳6x) produces clean, sharp results at poster viewing distance
  but cannot invent true native detail. Set expectations.

### 4. Run the upscale
```bash
python scripts/upscale_for_print.py --src "<source>" --inches 24x36 --dpi 300
# or an exact pixel target:
python scripts/upscale_for_print.py --src "<source>" --pixels 7200x10800
```
The script auto-chooses how many 4x passes are needed to reach the target, then Lanczos-resamples
to the exact size (downsampling from a higher pass yields crisper edges than interpolating up).

Useful flags:
- `--model realesrgan-x4plus-anime` — for anime/illustration/flat art (see `references/models.md`).
- `--two-pass` — force extra crispness when one pass already reaches the target.
- `--out path.tif` — choose the output; `.tif` triggers LZW, otherwise honors the extension.
- `--no-proof`, `--tile`, `--overlap`, `--fit`, `--passes` — see `--help`.

### 5. Verify before declaring done
- Confirm the output is exactly the target pixels, mode RGB, and `dpi == (target, target)`
  (so it physically prints at the requested inches): `px ÷ dpi` should equal the inch size.
- Open the proof, and crop a few **100% (native-pixel)** regions — the most detailed areas, any
  text, and faces — to confirm sharpness and no tiling seams. Show the user.

### 6. Text and fine detail (optional, advanced)
Super-resolution usually renders baked-in text and logos cleanly. If a specific element must be
razor-sharp and the original vector/text is available, re-render that element at full resolution
and composite it over the upscaled base, rather than relying on the upscale. This is extra work
with font-matching risk — only do it when the gain is real and the user asks.

### 7. Print-shop notes
- Default deliverable is **RGB TIFF @ target DPI** (works for most large-format digital printers).
- If the vendor requires **CMYK** or a specific ICC profile, convert with `PIL.ImageCms` using
  their profile (e.g. US Web Coated SWOP) — ask them for it.

## Model choice
Default `realesrgan-x4plus` (neutral, great for photos and photoreal art). Use the anime model for
illustration/line art. Details and how to add more models: `references/models.md`.

## Examples

**Example 1 — movie poster from a layered PSD**
Input: "Upscale my poster PSD to 24×36 at 300 dpi for the printer."
Output: inspect PSD composite → `--src poster.psd --inches 24x36 --dpi 300` → verify
7200×10800 @ 300dpi RGB TIFF + proof → show 100% crops.

**Example 2 — low-res photo for a 16×20 print**
Input: "This 1500px photo is too small to print at 16×20, can you fix it?"
Output: warn it's a big enlargement → `--src photo.png --inches 16x20 --dpi 300` (cover-fit if
aspect differs) → verify + proof.

**Example 3 — pixel-exact target**
Input: "I need this exactly 7200×10800."
Output: `--src art.png --pixels 7200x10800` → verify exact size.
