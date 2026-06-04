# Super-resolution models

The script loads ESRGAN-architecture `.pth` weights with **spandrel** and runs them tiled on the
GPU. Weights auto-download to `../models/` on first use, keyed by the `--model` name.

## Built-in choices

| `--model` value            | Best for                                  | Notes |
|----------------------------|-------------------------------------------|-------|
| `realesrgan-x4plus`        | Photos, photoreal art, cinematic posters  | Default. Neutral, robust. Preserves skin/grain texture. |
| `realesrgan-x4plus-anime`  | Anime, illustration, flat art, line art   | Smooths gradients and cleans lines; **wrong for photos** (erases pore/skin texture). |

Match the model to the content. Running the anime model on a photoreal image makes skin look
plasticky; running the photo model on flat illustration can leave compression-like noise in flat
areas.

## Adding more models

`spandrel` auto-detects many super-resolution architectures (ESRGAN, Real-ESRGAN, SwinIR,
HAT, DAT, SPAN, and more), so most community `.pth` upscalers "just work". To add one, drop a new
entry in the `MODELS` dict in `scripts/upscale_for_print.py`:

```python
MODELS = {
    ...
    "4x-ultrasharp": ("4x-UltraSharp.pth", "https://.../4x-UltraSharp.pth"),
    "4x-remacri":    ("4x_foolhardy_Remacri.pth", "https://.../4x_foolhardy_Remacri.pth"),
}
```

Popular community photo upscalers worth trying for posters:
- **4x-UltraSharp** — crisp, slightly aggressive sharpening.
- **4x_foolhardy_Remacri** — natural, less over-sharpened; good for skin.
- **4x_NMKD-Siax_200k** — strong general detail.

These are distributed on the OpenModelDB community site and various mirrors; verify the source and
license before bundling a download URL into a public/shared copy of this skill.

## Scale and passes

All the models above are 4x. The script reaches larger factors by running multiple 4x passes
(auto-chosen to reach the target), then Lanczos-resampling down to the exact size — downsampling
from a higher resolution gives sharper edges than interpolating upward. Override with `--passes N`
or force a minimum of two with `--two-pass`.
