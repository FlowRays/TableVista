from __future__ import annotations

import random
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from . import operations


def apply_noise(
    img: Image.Image, rng: random.Random, np_rng: np.random.Generator
) -> Tuple[Image.Image, Dict[str, Any]]:
    cosmetic = ["tilt", "watermark"]

    first = "lowres" if (rng.random() < 0.72) else "speckle"
    if first == "speckle":
        if rng.random() < 0.42:
            second = "lowres"
        else:
            second = rng.choice(cosmetic)
    else:
        second = rng.choice(cosmetic) if (rng.random() < 0.82) else "speckle"
    effects_applied = [first, second]
    rng.shuffle(effects_applied)

    apply_lowres = "lowres" in effects_applied
    apply_tilt = "tilt" in effects_applied
    apply_watermark = "watermark" in effects_applied
    apply_speckle = "speckle" in effects_applied

    lowres_scale: Optional[float] = None
    if apply_lowres:
        lowres_scale = rng.uniform(0.80, 0.86)

    rotate_angle: Optional[float] = None
    if apply_tilt:
        sign = -1.0 if (rng.random() < 0.5) else 1.0
        rotate_angle = float(sign * rng.uniform(0.7, 3.0))

    gaussian_sigma: Optional[int] = None
    if apply_speckle:
        gaussian_sigma = (
            int(rng.choice([2, 2, 3])) if apply_lowres else int(rng.choice([3, 4, 5]))
        )

    watermark_text = (
        rng.choice(["DRAFT", "SAMPLE", "COPY"]) if apply_watermark else None
    )
    watermark_opacity = rng.randint(18, 34) if apply_watermark else None
    watermark_spacing = rng.randint(200, 320) if apply_watermark else None
    watermark_angle = rng.uniform(-35.0, -15.0) if apply_watermark else None

    out = img.convert("RGB")
    if apply_lowres and lowres_scale is not None:
        out = operations.apply_low_resolution(out, scale=lowres_scale)
    if apply_tilt and rotate_angle is not None:
        out = operations.apply_rotate(out, angle=rotate_angle)
    if (
        apply_watermark
        and watermark_text is not None
        and watermark_opacity is not None
        and watermark_spacing is not None
        and watermark_angle is not None
    ):
        out = operations.add_tiled_watermark(
            out,
            text=watermark_text,
            opacity=watermark_opacity,
            angle=watermark_angle,
            spacing=watermark_spacing,
            color=(80, 80, 80),
        )
    if apply_speckle and gaussian_sigma is not None:
        out = operations.add_gaussian_noise(out, sigma=gaussian_sigma, rng=np_rng)

    meta = {
        "effects_applied": effects_applied,
        "lowres_scale": lowres_scale,
        "rotate_angle": rotate_angle,
        "blur_applied": False,
        "blur_radius": None,
        "gaussian_applied": apply_speckle,
        "gaussian_sigma": gaussian_sigma,
        "watermark_text": watermark_text,
        "watermark_opacity": watermark_opacity,
        "watermark_spacing": watermark_spacing,
        "watermark_angle": watermark_angle,
    }
    return out, meta


def apply_photo(
    img: Image.Image, rng: random.Random, np_rng: np.random.Generator
) -> Tuple[Image.Image, Dict[str, Any]]:
    out = img.convert("RGB")
    w, h = out.size

    meta: Dict[str, Any] = {"effects": []}
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)

    jitter = int(round(min(w, h) * rng.uniform(0.008, 0.020)))
    src = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    dst = np.float32(
        [
            [rng.randint(0, jitter), rng.randint(0, jitter)],
            [w - 1 - rng.randint(0, jitter), rng.randint(0, jitter)],
            [w - 1 - rng.randint(0, jitter), h - 1 - rng.randint(0, jitter)],
            [rng.randint(0, jitter), h - 1 - rng.randint(0, jitter)],
        ]
    )
    M = cv2.getPerspectiveTransform(src, dst)
    bg = int(rng.randint(6, 26))
    arr = cv2.cvtColor(np.array(out), cv2.COLOR_RGB2BGR)
    warped = cv2.warpPerspective(
        arr, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(bg, bg, bg)
    )

    k1 = rng.uniform(0.006, 0.020)
    k2 = rng.uniform(-0.002, 0.010)
    cx = (w - 1) / 2.0
    cy = (h - 1) / 2.0
    xn = (xx - cx) / cx
    yn = (yy - cy) / cy
    r2 = xn * xn + yn * yn
    factor = 1.0 + k1 * r2 + k2 * r2 * r2
    mapx = (xn * factor * cx + cx).astype(np.float32)
    mapy = (yn * factor * cy + cy).astype(np.float32)
    warped = cv2.remap(
        warped,
        mapx,
        mapy,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(bg, bg, bg),
    )

    out = Image.fromarray(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
    meta["effects"].append({"type": "perspective", "jitter": jitter, "bg": bg})
    meta["effects"].append({"type": "lens", "k1": k1, "k2": k2})

    border_ratio = float(rng.uniform(0.012, 0.030))
    pad_x = int(round(w * border_ratio))
    pad_y = int(round(h * border_ratio))
    if pad_x > 0 or pad_y > 0:
        resampling = getattr(Image, "Resampling", Image)
        inner_w = max(1, w - 2 * pad_x)
        inner_h = max(1, h - 2 * pad_y)
        scaled = out.resize((inner_w, inner_h), resample=resampling.BICUBIC)
        canvas = Image.new("RGB", (w, h), (bg, bg, bg))
        canvas.paste(scaled, (pad_x, pad_y))
        out = canvas
        meta["effects"].append(
            {"type": "border", "pad_x": pad_x, "pad_y": pad_y, "bg": bg}
        )

    arr_f = np.array(out, dtype=np.float32) / 255.0
    r_gain = rng.uniform(0.98, 1.06)
    g_gain = rng.uniform(0.99, 1.03)
    b_gain = rng.uniform(0.96, 1.05)
    contrast = rng.uniform(0.98, 1.06)
    gamma = rng.uniform(0.97, 1.06)
    brightness = rng.uniform(-0.015, 0.035)
    gains = np.array([r_gain, g_gain, b_gain], dtype=np.float32).reshape(1, 1, 3)
    arr_f = np.clip(arr_f * gains, 0.0, 1.0)
    arr_f = np.clip((arr_f - 0.5) * contrast + 0.5 + brightness, 0.0, 1.0)
    arr_f = np.clip(arr_f, 0.0, 1.0) ** gamma
    meta["effects"].append(
        {
            "type": "color",
            "r_gain": r_gain,
            "g_gain": g_gain,
            "b_gain": b_gain,
            "contrast": contrast,
            "gamma": gamma,
            "brightness": brightness,
        }
    )

    theta = rng.uniform(-np.pi, np.pi)
    u = np.cos(theta) * xx + np.sin(theta) * yy
    v = -np.sin(theta) * xx + np.cos(theta) * yy

    shade_applied = rng.random() < 0.70
    shade_sx = rng.uniform(-0.05, 0.05)
    shade_sy = rng.uniform(-0.04, 0.04)
    if shade_applied:
        shade = (
            1.0 + shade_sx * (xx / float(w) - 0.5) + shade_sy * (yy / float(h) - 0.5)
        )
        arr_f = np.clip(arr_f * shade[..., None], 0.0, 1.0)
        meta["effects"].append({"type": "shade", "sx": shade_sx, "sy": shade_sy})

    period1 = rng.uniform(7.5, 14.5)
    period2 = period1 * rng.uniform(0.90, 1.12)
    phase1 = rng.uniform(0.0, 2 * np.pi)
    phase2 = rng.uniform(0.0, 2 * np.pi)
    amp = rng.uniform(0.018, 0.050)
    rgb_phase = rng.uniform(0.25, 0.95)

    s1 = np.sin(2 * np.pi * u / period1 + phase1)
    s2 = np.sin(2 * np.pi * (u * 0.98 + v * 0.08) / period2 + phase2)

    amp_mod_period = rng.uniform(140.0, 320.0)
    amp_mod_phase = rng.uniform(0.0, 2 * np.pi)
    amp_map = amp * (
        0.70 + 0.30 * np.sin(2 * np.pi * v / amp_mod_period + amp_mod_phase)
    )

    base = (s1 + 0.8 * s2) / 1.8
    luma = (
        arr_f[..., 0] * 0.299 + arr_f[..., 1] * 0.587 + arr_f[..., 2] * 0.114
    ).astype(np.float32)
    luma2 = np.clip(luma * (1.0 + amp_map * base), 0.0, 1.0)
    ratio = luma2 / (luma + 1e-3)
    arr_f = np.clip(arr_f * ratio[..., None], 0.0, 1.0)

    chroma_scale = float(rng.uniform(0.06, 0.16))
    chroma_amp = amp_map * chroma_scale
    c = np.sin(2 * np.pi * u / period1 + phase1 + rgb_phase) + 0.35 * s2
    arr_f[..., 0] = np.clip(arr_f[..., 0] * (1.0 + chroma_amp * c), 0.0, 1.0)
    arr_f[..., 2] = np.clip(arr_f[..., 2] * (1.0 - chroma_amp * c), 0.0, 1.0)
    meta["effects"].append(
        {
            "type": "moire",
            "theta": theta,
            "period1": period1,
            "period2": period2,
            "amp": amp,
            "rgb_phase": rgb_phase,
            "amp_mod_period": amp_mod_period,
            "amp_mod_phase": amp_mod_phase,
            "chroma_scale": chroma_scale,
        }
    )

    band_amp = rng.uniform(0.010, 0.024)
    band_period = rng.uniform(80.0, 220.0)
    band_phase = rng.uniform(0.0, 2 * np.pi)
    y = yy[:, :1]
    sine = 1.0 + band_amp * np.sin(2 * np.pi * y / band_period + band_phase)
    bar_count = int(rng.choice([0, 1, 1, 2]))
    bars = np.zeros_like(y, dtype=np.float32)
    bar_min = 1.0
    bar_max = 1.0
    for _ in range(bar_count):
        y0 = float(rng.uniform(0.08, 0.92) * h)
        sigma = float(rng.uniform(0.04 * h, 0.12 * h))
        depth = float(rng.uniform(0.02, 0.07))
        band0 = -depth * np.exp(-((y - y0) ** 2) / (2.0 * sigma**2)).astype(np.float32)
        bars = bars + band0
        bar_min = min(bar_min, 1.0 - depth)
        bar_max = max(bar_max, 1.0)

    band = np.clip(sine * (1.0 + bars), 0.85, 1.10)
    arr_f = np.clip(arr_f * band[..., None], 0.0, 1.0)

    color_banding = rng.random() < 0.55
    if color_banding:
        dr = float(rng.uniform(-0.055, 0.055))
        db = float(rng.uniform(-0.055, 0.055))
        m = (band - 1.0).astype(np.float32)
        arr_f[..., 0] = np.clip(arr_f[..., 0] * (1.0 + dr * m), 0.0, 1.0)
        arr_f[..., 2] = np.clip(arr_f[..., 2] * (1.0 + db * m), 0.0, 1.0)
        meta["effects"].append({"type": "color_banding", "dr": dr, "db": db})
    meta["effects"].append(
        {
            "type": "banding",
            "amp": band_amp,
            "period": band_period,
            "phase": band_phase,
            "bar_count": bar_count,
            "bar_min": bar_min,
            "bar_max": bar_max,
        }
    )

    scanline_applied = rng.random() < 0.45
    if scanline_applied:
        sl_period = float(rng.uniform(3.0, 6.5))
        sl_amp = float(rng.uniform(0.004, 0.012))
        sl_phase = float(rng.uniform(0.0, 2 * np.pi))
        y = yy[:, :1]
        sl = 1.0 + sl_amp * np.sin(2 * np.pi * y / sl_period + sl_phase)
        arr_f = np.clip(arr_f * sl[..., None], 0.0, 1.0)
        meta["effects"].append(
            {"type": "scanline", "period": sl_period, "amp": sl_amp, "phase": sl_phase}
        )

    subpixel_applied = rng.random() < 0.70
    if subpixel_applied:
        sp_period = float(rng.uniform(2.6, 4.2))
        sp_eps = float(rng.uniform(0.004, 0.012))
        sp_phase = float(rng.uniform(0.0, 2 * np.pi))
        t = (2 * np.pi * xx / sp_period + sp_phase).astype(np.float32)
        arr_f[..., 0] = np.clip(arr_f[..., 0] * (1.0 + sp_eps * np.sin(t)), 0.0, 1.0)
        arr_f[..., 1] = np.clip(
            arr_f[..., 1] * (1.0 + sp_eps * np.sin(t + 2 * np.pi / 3.0)), 0.0, 1.0
        )
        arr_f[..., 2] = np.clip(
            arr_f[..., 2] * (1.0 + sp_eps * np.sin(t + 4 * np.pi / 3.0)), 0.0, 1.0
        )
        meta["effects"].append(
            {"type": "subpixel", "period": sp_period, "eps": sp_eps, "phase": sp_phase}
        )

    glare_applied = rng.random() < 0.55
    if glare_applied:
        cx = rng.uniform(0.10 * w, 0.85 * w)
        cy = rng.uniform(0.05 * h, 0.55 * h)
        sx = rng.uniform(0.22 * w, 0.65 * w)
        sy = rng.uniform(0.18 * h, 0.55 * h)
        intensity = rng.uniform(0.03, 0.10)
        glow = np.exp(
            -(((xx - cx) ** 2) / (2 * sx**2) + ((yy - cy) ** 2) / (2 * sy**2))
        ).astype(np.float32)
        glow = glow[..., None]
        warm = np.array([1.0, 0.98, 0.92], dtype=np.float32).reshape(1, 1, 3)
        arr_f = np.clip(arr_f + intensity * glow * warm, 0.0, 1.0)
        meta["effects"].append(
            {
                "type": "glare",
                "cx": cx,
                "cy": cy,
                "sx": sx,
                "sy": sy,
                "intensity": intensity,
            }
        )

    streak_applied = rng.random() < 0.38
    if streak_applied:
        phi = rng.uniform(-np.pi / 2.0, np.pi / 2.0)
        cx = rng.uniform(0.15 * w, 0.85 * w)
        cy = rng.uniform(0.10 * h, 0.85 * h)
        t = np.cos(phi) * (xx - cx) + np.sin(phi) * (yy - cy)
        d = -np.sin(phi) * (xx - cx) + np.cos(phi) * (yy - cy)
        sigma_d = rng.uniform(10.0, 26.0)
        sigma_t = rng.uniform(0.55 * max(w, h), 1.10 * max(w, h))
        intensity = rng.uniform(0.012, 0.045)
        streak = np.exp(
            -(d * d) / (2 * sigma_d * sigma_d) - (t * t) / (2 * sigma_t * sigma_t)
        ).astype(np.float32)
        streak = streak[..., None]
        cold = np.array([0.98, 0.99, 1.0], dtype=np.float32).reshape(1, 1, 3)
        arr_f = np.clip(arr_f + intensity * streak * cold, 0.0, 1.0)
        meta["effects"].append(
            {
                "type": "streak",
                "phi": phi,
                "cx": cx,
                "cy": cy,
                "sigma_d": sigma_d,
                "intensity": intensity,
            }
        )

    vignette_strength = rng.uniform(0.08, 0.22)
    cx0 = (w / 2.0) + rng.uniform(-0.05 * w, 0.05 * w)
    cy0 = (h / 2.0) + rng.uniform(-0.05 * h, 0.05 * h)
    xn = (xx - cx0) / (w / 2.0)
    yn = (yy - cy0) / (h / 2.0)
    r2 = xn**2 + yn**2
    vignette = 1.0 - vignette_strength * np.clip(r2, 0.0, 1.4)
    vignette = np.clip(vignette, 0.55, 1.0).astype(np.float32)
    arr_f = np.clip(arr_f * vignette[..., None], 0.0, 1.0)
    meta["effects"].append({"type": "vignette", "strength": vignette_strength})

    chroma_applied = rng.random() < 0.45
    chroma_shift = int(rng.choice([1, 1, 2]))
    if chroma_applied:
        rgb = (arr_f * 255.0).astype(np.uint8)
        r = rgb[..., 0]
        g = rgb[..., 1]
        b = rgb[..., 2]
        dx = int(rng.choice([-chroma_shift, chroma_shift]))
        dy = int(rng.choice([-1, 0, 1]))
        M1 = np.float32([[1, 0, dx], [0, 1, dy]])
        M2 = np.float32([[1, 0, -dx], [0, 1, -dy]])
        r2 = cv2.warpAffine(r, M1, (w, h), borderMode=cv2.BORDER_REFLECT_101)
        b2 = cv2.warpAffine(b, M2, (w, h), borderMode=cv2.BORDER_REFLECT_101)
        rgb2 = np.stack([r2, g, b2], axis=2).astype(np.float32) / 255.0
        arr_f = np.clip(rgb2, 0.0, 1.0)
        meta["effects"].append(
            {"type": "chroma", "shift": chroma_shift, "dx": dx, "dy": dy}
        )

    out = Image.fromarray((arr_f * 255.0).astype(np.uint8), mode="RGB")

    defocus_applied = rng.random() < 0.10
    defocus_sigma = float(rng.uniform(0.06, 0.18))
    if defocus_applied:
        arr2 = cv2.cvtColor(np.array(out), cv2.COLOR_RGB2BGR)
        arr2 = cv2.GaussianBlur(
            arr2, (0, 0), sigmaX=defocus_sigma, sigmaY=defocus_sigma
        )
        out = Image.fromarray(cv2.cvtColor(arr2, cv2.COLOR_BGR2RGB))
        meta["effects"].append({"type": "defocus", "sigma": defocus_sigma})

    motion_applied = rng.random() < 0.03
    if motion_applied:
        k = int(rng.choice([3, 5, 5]))
        angle = rng.uniform(-8.0, 8.0)
        out = operations.apply_motion_blur(out, kernel_size=k, angle=angle)
        meta["effects"].append({"type": "motion_blur", "kernel": k, "angle": angle})

    sharpen_applied = rng.random() < 0.80
    sharpen_amount = float(rng.uniform(0.25, 0.60))
    sharpen_sigma = float(rng.uniform(0.9, 1.8))
    if sharpen_applied:
        arr2 = cv2.cvtColor(np.array(out), cv2.COLOR_RGB2BGR)
        blur = cv2.GaussianBlur(
            arr2, (0, 0), sigmaX=sharpen_sigma, sigmaY=sharpen_sigma
        )
        arr2 = cv2.addWeighted(arr2, 1.0 + sharpen_amount, blur, -sharpen_amount, 0.0)
        arr2 = np.clip(arr2, 0, 255).astype(np.uint8)
        out = Image.fromarray(cv2.cvtColor(arr2, cv2.COLOR_BGR2RGB))
        meta["effects"].append(
            {"type": "sharpen", "amount": sharpen_amount, "sigma": sharpen_sigma}
        )

    sensor_applied = rng.random() < 0.90
    shot_scale = float(rng.uniform(220.0, 600.0))
    read_sigma = float(rng.uniform(0.0015, 0.0045))
    if sensor_applied:
        arr_s = np.array(out, dtype=np.float32) / 255.0
        lam = np.clip(arr_s, 0.0, 1.0) * shot_scale
        noisy = np_rng.poisson(lam).astype(np.float32) / shot_scale
        noisy = noisy + np_rng.normal(0.0, read_sigma, noisy.shape).astype(np.float32)
        out = Image.fromarray(
            (np.clip(noisy, 0.0, 1.0) * 255.0).astype(np.uint8), mode="RGB"
        )
        meta["effects"].append(
            {"type": "sensor", "shot_scale": shot_scale, "read_sigma": read_sigma}
        )

    lowres_applied = rng.random() < 0.08
    lowres_scale = rng.uniform(0.97, 0.995)
    if lowres_applied:
        out = operations.apply_low_resolution(out, scale=lowres_scale)
        meta["effects"].append({"type": "lowres", "scale": lowres_scale})

    jpeg_quality = int(rng.randint(78, 95))
    out = operations.apply_jpeg_compression(out, quality=jpeg_quality)
    meta["effects"].append({"type": "jpeg", "quality": jpeg_quality})

    noise_sigma = int(rng.randint(0, 1))
    if noise_sigma > 0:
        out = operations.add_gaussian_noise(out, sigma=noise_sigma, rng=np_rng)
        meta["effects"].append({"type": "gaussian", "sigma": noise_sigma})

    return out, meta


def apply_structural(
    img: Image.Image,
    rng: random.Random,
    np_rng: np.random.Generator,
) -> Tuple[Image.Image, Dict[str, Any]]:
    from scipy.interpolate import RectBivariateSpline

    arr = cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)
    h, w = arr.shape[:2]

    grid_size = rng.randint(5, 9)
    amplitude = 10.0

    gx = np.linspace(0, w - 1, grid_size, dtype=np.float32)
    gy = np.linspace(0, h - 1, grid_size, dtype=np.float32)

    dx_grid = np_rng.uniform(-amplitude, amplitude, (grid_size, grid_size)).astype(
        np.float32
    )
    dy_grid = np_rng.uniform(-amplitude, amplitude, (grid_size, grid_size)).astype(
        np.float32
    )

    dx_grid[0, :] = 0.0
    dx_grid[-1, :] = 0.0
    dx_grid[:, 0] = 0.0
    dx_grid[:, -1] = 0.0
    dy_grid[0, :] = 0.0
    dy_grid[-1, :] = 0.0
    dy_grid[:, 0] = 0.0
    dy_grid[:, -1] = 0.0

    spline_x = RectBivariateSpline(gy, gx, dx_grid)
    spline_y = RectBivariateSpline(gy, gx, dy_grid)

    yi = np.arange(h, dtype=np.float32)
    xi = np.arange(w, dtype=np.float32)
    dx_full = spline_x(yi, xi).astype(np.float32)
    dy_full = spline_y(yi, xi).astype(np.float32)

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    mapx = np.clip(xx + dx_full, 0, w - 1)
    mapy = np.clip(yy + dy_full, 0, h - 1)

    warped = cv2.remap(
        arr, mapx, mapy, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
    )
    out = Image.fromarray(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
    return out, {"grid_size": grid_size, "amplitude": amplitude}
