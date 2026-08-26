"""Generate map-source style augmentations of FSC training images.

Diagnosis (docs/fsc_diagnosis.md) shows FSC misses concentrate on map-source
domains absent from training (yandex etc.). This script creates 3 styled copies
per FSC training image (hsv jitter + gamma, contrast+sharpen, blur+brightness)
to inject rendering-style diversity. Copies carry the original labels.

Usage (run where data_v2 lives, e.g. the server):
    python scripts/augment_fsc_style.py [--copies 3] [--seed 2026]
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

from PIL import Image, ImageEnhance, ImageFilter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA = PROJECT_ROOT / "data_v2"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--copies", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--data-root", default=None, help="Override data root (for testing); default data_v2 under project.")
    return parser.parse_args()


def hsv_jitter(im: Image.Image, rng: random.Random) -> Image.Image:
    im = im.convert("HSV")
    h, s, v = im.split()
    h = h.point(lambda x: (x + rng.randint(-25, 25)) % 256)
    s = s.point(lambda x: max(0, min(255, int(x * rng.uniform(0.6, 1.4)))))
    v = v.point(lambda x: max(0, min(255, int(x * rng.uniform(0.7, 1.3)))))
    return Image.merge("HSV", (h, s, v)).convert("RGB")


def gamma_adjust(im: Image.Image, gamma: float) -> Image.Image:
    lut = [max(0, min(255, int(255 * (i / 255) ** gamma))) for i in range(256)]
    if im.mode == "L":
        return im.point(lut)
    return Image.merge(im.mode, [band.point(lut) for band in im.split()])


def style_a(im: Image.Image, rng: random.Random) -> Image.Image:
    """Map-source style A: strong color jitter + gamma (yandex-like rendering)."""
    im = hsv_jitter(im, rng)
    return gamma_adjust(im, rng.uniform(0.75, 1.25))


def style_b(im: Image.Image, rng: random.Random) -> Image.Image:
    """Map-source style B: contrast boost + sharpen (Google-like crisp rendering)."""
    im = ImageEnhance.Contrast(im).enhance(rng.uniform(1.2, 1.8))
    return ImageEnhance.Sharpness(im).enhance(rng.uniform(1.2, 2.0))


def style_c(im: Image.Image, rng: random.Random) -> Image.Image:
    """Map-source style C: slight blur + brightness shift (Bing-like softer rendering)."""
    im = im.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.5, 1.5)))
    return ImageEnhance.Brightness(im).enhance(rng.uniform(0.75, 1.25))


def main() -> int:
    args = parse_args()
    data_root = Path(args.data_root) if args.data_root else DATA
    img_dir = data_root / "images/train"
    label_dir = data_root / "labels/train"
    fsc_images = sorted(img_dir.glob("fsc_*"))
    rng = random.Random(args.seed)
    styles = (style_a, style_b, style_c)

    made = 0
    for image_path in fsc_images:
        label_path = label_dir / (image_path.stem + ".txt")
        if not label_path.exists():
            print(f"skip (no label): {image_path.name}", flush=True)
            continue
        label_text = label_path.read_text(encoding="utf-8")
        with Image.open(image_path) as src:
            im = src.convert("RGB")
        for copy_index in range(args.copies):
            style = styles[copy_index % len(styles)]
            aug = style(im, rng)
            out_name = f"fsc_aug{copy_index}_{image_path.stem}.jpg"
            aug.save(img_dir / out_name, quality=95)
            (label_dir / (out_name.rsplit(".", 1)[0] + ".txt")).write_text(label_text, encoding="utf-8")
            made += 1

    print(f"augmented {len(fsc_images)} FSC images -> {made} new samples", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
