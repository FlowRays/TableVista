from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
from io import BytesIO
from typing import Tuple, Optional


def apply_low_resolution(img: Image.Image, scale: float = 0.8) -> Image.Image:
    scale = float(scale)
    if scale <= 0 or scale > 1:
        raise ValueError("scale must be in (0, 1]")

    w, h = img.size
    if w <= 1 or h <= 1:
        return img

    down_w = max(1, int(round(w * scale)))
    down_h = max(1, int(round(h * scale)))

    resampling = getattr(Image, "Resampling", Image)
    down = img.resize((down_w, down_h), resample=resampling.BILINEAR)
    up = down.resize((w, h), resample=resampling.BILINEAR)
    return up


def add_gaussian_noise(
    img: Image.Image,
    sigma: int = 15,
    rng: Optional[np.random.Generator] = None,
) -> Image.Image:
    img_array = np.array(img)

    if rng is None:
        noise = np.random.normal(0, sigma, img_array.shape)
    else:
        noise = rng.normal(0, sigma, img_array.shape)

    noisy = img_array + noise
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)

    return Image.fromarray(noisy)


def apply_jpeg_compression(img: Image.Image, quality: int = 25) -> Image.Image:
    quality = int(max(1, min(95, quality)))
    buffer = BytesIO()
    img.convert("RGB").save(buffer, format="JPEG", quality=quality, optimize=False)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def apply_motion_blur(
    img: Image.Image, kernel_size: int = 11, angle: float = 0.0
) -> Image.Image:
    k = max(3, int(kernel_size))
    if k % 2 == 0:
        k += 1

    kernel = np.zeros((k, k), dtype=np.float32)
    kernel[k // 2, :] = 1.0

    center = (k / 2, k / 2)
    rot = cv2.getRotationMatrix2D(center, angle, 1.0)
    kernel = cv2.warpAffine(kernel, rot, (k, k))
    kernel_sum = float(kernel.sum())
    if kernel_sum > 0:
        kernel /= kernel_sum

    img_array = np.array(img.convert("RGB"))
    blurred = cv2.filter2D(img_array, -1, kernel)
    return Image.fromarray(blurred)


def apply_rotate(img: Image.Image, angle: float = 2.0) -> Image.Image:
    src = img.convert("RGB")
    return src.rotate(angle, fillcolor=(255, 255, 255), expand=True)


def add_tiled_watermark(
    img: Image.Image,
    text: str = "DRAFT",
    *,
    opacity: int = 40,
    angle: float = -25.0,
    spacing: int = 220,
    color: Tuple[int, int, int] = (80, 80, 80),
) -> Image.Image:
    img_rgba = img.copy().convert("RGBA")
    overlay = Image.new("RGBA", img_rgba.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)

    try:
        font_size = int(min(img_rgba.size) * 0.08)
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", font_size)
    except (OSError, IOError, TypeError):
        font = ImageFont.load_default()

    w, h = img_rgba.size
    step = max(120, int(spacing))
    fill = (int(color[0]), int(color[1]), int(color[2]), int(max(0, min(255, opacity))))

    for y in range(-h, h * 2, step):
        for x in range(-w, w * 2, step):
            draw.text((x, y), text, fill=fill, font=font)

    overlay = overlay.rotate(float(angle), expand=False)
    out = Image.alpha_composite(img_rgba, overlay)
    return out.convert("RGB")
