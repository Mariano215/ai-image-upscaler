"""Upscale an image to print resolution with AI super-resolution (Real-ESRGAN).

One self-contained CLI: super-resolve (tiled, GPU if available) then resample to the
exact target pixel size, and write a print-ready TIFF (target DPI, sRGB, LZW) plus a
JPEG proof. Model weights auto-download on first use.

Examples
--------
# 24x36 inch poster at 300 dpi
python upscale_for_print.py --src "poster.psd" --inches 24x36 --dpi 300

# exact pixel target
python upscale_for_print.py --src in.png --pixels 7200x10800

# illustration / anime art
python upscale_for_print.py --src art.png --inches 18x24 --model realesrgan-x4plus-anime
"""
import argparse
import math
import os
import sys
import urllib.request

import numpy as np
import torch
from PIL import Image, ImageCms
from spandrel import ModelLoader

Image.MAX_IMAGE_PIXELS = None  # allow very large canvases

MODELS = {
    # neutral, robust general/photo model — the default
    "realesrgan-x4plus": (
        "RealESRGAN_x4plus.pth",
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
    ),
    # smoother model tuned for anime / flat illustration / line art
    "realesrgan-x4plus-anime": (
        "RealESRGAN_x4plus_anime_6B.pth",
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
    ),
}


def resolve_weights(name, models_dir):
    if name not in MODELS:
        sys.exit(f"unknown model '{name}'. choices: {', '.join(MODELS)}")
    fname, url = MODELS[name]
    os.makedirs(models_dir, exist_ok=True)
    path = os.path.join(models_dir, fname)
    if not os.path.exists(path):
        print(f"downloading {name} weights -> {path}")
        urllib.request.urlretrieve(url, path)
    return path


def load_image(src):
    """Open any raster (incl. PSD via its merged composite) as RGB uint8."""
    im = Image.open(src).convert("RGB")
    return np.asarray(im).copy()  # .copy() -> writable, silences torch warning


def feather(h, w):
    wy = np.minimum(np.arange(h), np.arange(h)[::-1]) + 1
    wx = np.minimum(np.arange(w), np.arange(w)[::-1]) + 1
    f = np.minimum.outer(wy, wx).astype(np.float32)
    return (f / f.max())[..., None]


@torch.inference_mode()
def upscale_once(img, model, scale, device, tile, overlap):
    """One model pass over HxWx3 uint8 -> (H*scale)x(W*scale)x3 uint8, tiled + feather-blended."""
    H, W, _ = img.shape
    out = np.zeros((H * scale, W * scale, 3), np.float32)
    acc = np.zeros((H * scale, W * scale, 1), np.float32)
    step = max(1, tile - overlap)
    ys = sorted(set(list(range(0, H, step)) + [max(0, H - tile)]))
    xs = sorted(set(list(range(0, W, step)) + [max(0, W - tile)]))
    for y in ys:
        for x in xs:
            y1, x1 = min(y + tile, H), min(x + tile, W)
            y0, x0 = max(0, y1 - tile), max(0, x1 - tile)
            patch = img[y0:y1, x0:x1]
            t = torch.from_numpy(patch).permute(2, 0, 1).unsqueeze(0).float().div_(255).to(device)
            r = model(t).clamp_(0, 1)[0].permute(1, 2, 0).cpu().numpy()
            ph, pw = r.shape[:2]
            f = feather(ph, pw)
            oy, ox = y0 * scale, x0 * scale
            out[oy:oy + ph, ox:ox + pw] += r * f
            acc[oy:oy + ph, ox:ox + pw] += f
    out /= np.maximum(acc, 1e-6)
    return (out * 255.0 + 0.5).clip(0, 255).astype(np.uint8)


def fit_to_target(im, tw, th, mode):
    """Resample PIL image to exactly (tw,th). cover=fill+center-crop, contain=pad, stretch=distort."""
    if mode == "stretch" or im.size == (tw, th):
        return im.resize((tw, th), Image.LANCZOS)
    sw, sh = im.size
    if mode == "contain":
        s = min(tw / sw, th / sh)
        rw, rh = max(1, round(sw * s)), max(1, round(sh * s))
        r = im.resize((rw, rh), Image.LANCZOS)
        canvas = Image.new("RGB", (tw, th), (0, 0, 0))
        canvas.paste(r, ((tw - rw) // 2, (th - rh) // 2))
        return canvas
    # cover (default)
    s = max(tw / sw, th / sh)
    rw, rh = max(tw, round(sw * s)), max(th, round(sh * s))
    r = im.resize((rw, rh), Image.LANCZOS)
    left, top = (rw - tw) // 2, (rh - th) // 2
    return r.crop((left, top, left + tw, top + th))


def parse_pair(s):
    a, b = s.lower().replace(" ", "").split("x")
    return int(round(float(a))), int(round(float(b)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="source image (png/jpg/tif/psd/...)")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--inches", help="print size WxH, e.g. 24x36 (combine with --dpi)")
    g.add_argument("--pixels", help="exact pixel target WxH, e.g. 7200x10800")
    ap.add_argument("--dpi", type=int, default=300, help="dots per inch for --inches (default 300)")
    ap.add_argument("--out", help="output path (default <src>_<W>x<H>.tif next to source)")
    ap.add_argument("--model", default="realesrgan-x4plus", choices=list(MODELS))
    ap.add_argument("--models-dir", default=os.path.join(os.path.dirname(__file__), "..", "models"))
    ap.add_argument("--passes", type=int, default=0, help="force N 4x passes (0=auto: enough to reach target)")
    ap.add_argument("--two-pass", action="store_true", help="force >=2 passes for extra crispness")
    ap.add_argument("--fit", choices=["cover", "contain", "stretch"], default="cover",
                    help="how to reconcile aspect mismatch (default cover)")
    ap.add_argument("--tile", type=int, default=512)
    ap.add_argument("--overlap", type=int, default=32)
    ap.add_argument("--no-proof", action="store_true")
    args = ap.parse_args()

    # target pixels + dpi
    if args.pixels:
        tw, th = parse_pair(args.pixels)
        out_dpi = args.dpi
        size_note = f"{tw}x{th}px"
    else:
        iw, ih = parse_pair(args.inches)
        tw, th = iw * args.dpi, ih * args.dpi
        out_dpi = args.dpi
        size_note = f"{iw}x{ih}in @ {args.dpi}dpi = {tw}x{th}px"

    img = load_image(args.src)
    sh, sw = img.shape[:2]
    print(f"source: {sw}x{sh}  target: {size_note}")
    if args.src.lower().endswith((".jpg", ".jpeg")):
        print("WARNING: JPEG source — compression artifacts get amplified. Prefer a lossless original (PSD/PNG/TIFF).")

    src_ar, tgt_ar = sw / sh, tw / th
    if abs(src_ar - tgt_ar) / tgt_ar > 0.005:
        print(f"NOTE: aspect mismatch (src {src_ar:.3f} vs target {tgt_ar:.3f}) -> --fit {args.fit}")

    # how many 4x passes
    need = max(tw / sw, th / sh)
    passes = args.passes or max(1, math.ceil(math.log(need, 4))) if need > 1 else 1
    if args.two_pass:
        passes = max(passes, 2)
    passes = min(passes, 3)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: no CUDA GPU found — running on CPU, this will be slow.")
    model = ModelLoader().load_from_file(resolve_weights(args.model, args.models_dir)).to(device).eval()
    scale = model.scale
    print(f"model: {args.model} ({scale}x)  device: {device}  passes: {passes}")

    up = img
    for i in range(passes):
        up = upscale_once(up, model, scale, device, args.tile, args.overlap)
        print(f"  pass {i + 1}: {up.shape[1]}x{up.shape[0]}")

    im = fit_to_target(Image.fromarray(up), tw, th, args.fit)

    out = args.out or os.path.splitext(args.src)[0] + f"_{tw}x{th}.tif"
    icc = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    save_kw = {"dpi": (out_dpi, out_dpi), "icc_profile": icc}
    if out.lower().endswith((".tif", ".tiff")):
        save_kw["compression"] = "tiff_lzw"
    im.save(out, **save_kw)
    print(f"saved {out}  {im.size[0]}x{im.size[1]}  dpi={out_dpi}  = {im.size[0]/out_dpi:g}x{im.size[1]/out_dpi:g}in")

    if not args.no_proof:
        proof = os.path.splitext(out)[0] + "_proof.jpg"
        pw = 2000
        p = im.resize((pw, round(pw * th / tw)), Image.LANCZOS)
        p.save(proof, quality=92)
        print(f"proof: {proof}")


if __name__ == "__main__":
    main()
