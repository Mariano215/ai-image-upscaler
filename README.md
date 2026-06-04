# ai-image-upscaler

**AI super-resolution upscaling for print.** Turn a small or low-resolution image into a
print-ready high-resolution file — a TIFF at an exact inch size and DPI — using a Real-ESRGAN
super-resolution model. Ships as both a **standalone Python CLI** and a **[Claude Code](https://claude.com/claude-code) skill**.

Plain interpolation (Photoshop "Image Size", `PIL.Image.resize`) goes soft past ~2×. A
super-resolution model reconstructs clean edges and plausible texture, which is what makes a 5–8×
enlargement look professional at poster sizes.

## Features

- **Target by inches + DPI** (`--inches 24x36 --dpi 300`) or by **exact pixels** (`--pixels 7200x10800`).
- **Tiled GPU inference** with feathered blending — handles huge canvases without VRAM spikes or seams.
- **Auto multi-pass**: runs as many 4× passes as needed to reach the target, then Lanczos-resamples
  to the exact size (downsampling from a higher pass = sharper edges than interpolating up).
- **Print-ready output**: RGB TIFF tagged at the target DPI, sRGB profile embedded, LZW-compressed,
  plus a JPEG proof for quick review.
- **PSD aware**: reads a Photoshop file's merged composite directly (no manual flatten).
- **Aspect handling**: `--fit cover|contain|stretch` for size mismatches.
- **Weights auto-download** on first run; swap models with `--model`.

## Requirements

- Python 3.9+
- `torch` (a CUDA build is strongly recommended — CPU works but is slow on large canvases)
- `spandrel`, `pillow`, `numpy`

```bash
pip install -r requirements.txt
```

> **Why spandrel and not the `realesrgan` pip package?** The official package depends on `basicsr`,
> which is broken on torchvision ≥ 0.17 (`torchvision.transforms.functional_tensor` was removed).
> [`spandrel`](https://github.com/chaiNNer-org/spandrel) loads the model weights directly and avoids
> the whole problem — and supports many other super-resolution architectures too.

## Usage

```bash
# 24x36 inch poster at 300 dpi (7200x10800 px)
python scripts/upscale_for_print.py --src poster.psd --inches 24x36 --dpi 300

# exact pixel target
python scripts/upscale_for_print.py --src in.png --pixels 7200x10800

# illustration / anime art (smoother model)
python scripts/upscale_for_print.py --src art.png --inches 18x24 --model realesrgan-x4plus-anime
```

Output is written next to the source as `<name>_<W>x<H>.tif` (override with `--out`), plus a
`*_proof.jpg`.

### Flags

| Flag | Default | Purpose |
|------|---------|---------|
| `--src` | (required) | Source image: png / jpg / tif / **psd** / … |
| `--inches WxH` | — | Print size in inches (combine with `--dpi`) |
| `--pixels WxH` | — | Exact pixel target (alternative to `--inches`) |
| `--dpi` | `300` | DPI for `--inches`, and the value tagged into the output |
| `--out` | `<src>_<W>x<H>.tif` | Output path; `.tif` → LZW, otherwise honors the extension |
| `--model` | `realesrgan-x4plus` | `realesrgan-x4plus` (photo) or `realesrgan-x4plus-anime` |
| `--fit` | `cover` | Aspect mismatch: `cover` (fill+crop), `contain` (pad), `stretch` |
| `--passes` | auto | Force N 4× passes (0 = auto: enough to reach target) |
| `--two-pass` | off | Force ≥ 2 passes for extra crispness |
| `--tile` / `--overlap` | `512` / `32` | Tiling controls for VRAM / seam tuning |
| `--no-proof` | off | Skip the JPEG proof |

## How it works

1. Load the source (RGB; PSD → merged composite).
2. Run the model in overlapping tiles, blending with a feathered window so there are no seams.
3. Repeat 4× passes until the result meets or exceeds the target in both dimensions.
4. Lanczos-resample to the exact target size, applying the chosen `--fit` for any aspect mismatch.
5. Save an sRGB, DPI-tagged TIFF (LZW) so it physically prints at the requested inches, plus a proof.

## Models

Default `realesrgan-x4plus` (neutral, great for photos and photoreal art); `realesrgan-x4plus-anime`
for illustration/line art. spandrel auto-detects many architectures, so most community `.pth`
upscalers can be added with a one-line entry in the `MODELS` dict. See
[`references/models.md`](references/models.md).

## Use as a Claude Code skill

This repo is also a Claude Code skill. Install it by cloning into your skills directory:

```bash
git clone https://github.com/Mariano215/ai-image-upscaler.git ~/.claude/skills/upscale-for-print
```

Then just ask Claude things like *"upscale this poster to 24×36 at 300 dpi for print"* and it will
follow [`SKILL.md`](SKILL.md): inspect the source, pick the best file, check aspect, run the script,
and verify the output.

## Limitations

- A large upscale (≳ 6×) produces clean, sharp results at normal poster viewing distance but **cannot
  invent true native detail** — it is not equivalent to a natively-captured high-res image.
- JPEG sources amplify compression artifacts; prefer a lossless original (PSD / PNG / TIFF).
- Default output is **RGB**. If your print vendor requires **CMYK** with a specific ICC profile,
  convert with `PIL.ImageCms` using their profile.

## License

[MIT](LICENSE) © 2026 Mariano215
