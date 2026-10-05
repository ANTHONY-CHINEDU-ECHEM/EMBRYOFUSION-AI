"""Procedural blastocyst renderer.

Renders microscopy style day 5 blastocyst images from Gardner grades (expansion,
inner cell mass, trophectoderm). Used to create the image half of the paired
multimodal dataset, because paired clinical and embryo image data cannot be
shared publicly. Illumination, relief contrast, blur and noise are randomised.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class RenderResult:
    image: np.ndarray           # uint8 grayscale, shape (size, size)
    icm_x: float                # inner cell mass centre, as a fraction of the image width
    icm_y: float
    icm_r: float                # inner cell mass radius, as a fraction of the image width
    embryo_r: float


def _disc(size: int, cx: float, cy: float, r: float, soften: float = 1.2) -> np.ndarray:
    """Anti aliased filled disc as a float mask in [0, 1]."""
    mask = np.zeros((size, size), np.uint8)
    cv2.circle(mask, (int(round(cx * 16)), int(round(cy * 16))), int(round(max(r, 0.5) * 16)), 255, -1, cv2.LINE_AA, shift=4)
    mask = mask.astype(np.float32) / 255.0
    return cv2.GaussianBlur(mask, (0, 0), soften) if soften > 0 else mask


def _blend(canvas: np.ndarray, mask: np.ndarray, value: float, alpha: float = 1.0) -> None:
    canvas *= 1 - alpha * mask
    canvas += alpha * mask * value


def _cells_on_arc(canvas, rng, cx, cy, radius, n, cell_len, cell_thick, coverage, tone, edge):
    """Draw trophectoderm cells as tangent ellipses spaced around a circle."""
    size = canvas.shape[0]
    layer = np.zeros((size, size), np.uint8)
    rim = np.zeros((size, size), np.uint8)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False) + rng.uniform(0, 2 * np.pi)
    for a in angles:
        if rng.random() > coverage:
            continue
        a += rng.normal(0, 0.25 * 2 * np.pi / n)
        r = radius + rng.normal(0, 0.15 * cell_thick)
        centre = (int(round((cx + r * np.cos(a)) * 16)), int(round((cy + r * np.sin(a)) * 16)))
        axes = (int(max(1, cell_len * rng.uniform(0.8, 1.2)) * 16), int(max(1, cell_thick * rng.uniform(0.8, 1.2)) * 16))
        tilt = np.degrees(a) + 90 + rng.normal(0, 8)
        cv2.ellipse(layer, centre, axes, tilt, 0, 360, 255, -1, cv2.LINE_AA, shift=4)
        cv2.ellipse(rim, centre, axes, tilt, 0, 360, 255, 1, cv2.LINE_AA, shift=4)
    _blend(canvas, cv2.GaussianBlur(layer.astype(np.float32) / 255, (0, 0), 0.8), tone, 0.75)
    _blend(canvas, cv2.GaussianBlur(rim.astype(np.float32) / 255, (0, 0), 0.6), edge, 0.6)


def render_blastocyst(rng: np.random.Generator, expansion: int, icm: int, te: int, size: int = 192) -> RenderResult:
    """Render one blastocyst.

    Parameters use zero based class indices: ``expansion`` 0 to 5 (Gardner 1
    to 6), ``icm`` and ``te`` 0 to 2 (grades A to C).
    """
    grade = expansion + 1
    s = float(size)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    canvas = np.full((size, size), rng.uniform(0.50, 0.56), np.float32)
    tilt = rng.uniform(0, 2 * np.pi)
    canvas += rng.uniform(0.03, 0.09) * ((xx - s / 2) * np.cos(tilt) + (yy - s / 2) * np.sin(tilt)) / s

    base_r = s * rng.uniform(0.25, 0.29)
    radius = base_r * {1: 1.0, 2: 1.0, 3: 1.04, 4: 1.2, 5: 1.2, 6: 1.3}[grade] * rng.uniform(0.96, 1.04)
    cx, cy = s / 2 + rng.normal(0, 0.025 * s, 2)
    zona_t = base_r * {1: 0.17, 2: 0.17, 3: 0.14, 4: 0.07, 5: 0.06, 6: 0.0}[grade] * rng.uniform(0.85, 1.15)
    hatch_angle = rng.uniform(0, 2 * np.pi)

    # Zona pellucida: a bright ring with darker borders, breached at the hatching site.
    if zona_t > 0:
        ring = _disc(size, cx, cy, radius + zona_t + 1.5) - _disc(size, cx, cy, radius + 1.5)
        if grade == 5:
            gap = _disc(size, cx + (radius + zona_t) * np.cos(hatch_angle), cy + (radius + zona_t) * np.sin(hatch_angle), 0.38 * radius, 2.0)
            ring = ring * (1 - gap)
        _blend(canvas, np.clip(ring, 0, 1), 0.66, 0.8)
        outline = _disc(size, cx, cy, radius + zona_t + 2.5, 0.8) - _disc(size, cx, cy, radius + zona_t + 0.8, 0.8)
        _blend(canvas, np.clip(outline, 0, 1), 0.36, 0.5)
    elif rng.random() < 0.5:
        # A hatched embryo often sits next to its empty zona shell.
        away = hatch_angle + np.pi
        ex, ey = cx + 1.9 * radius * np.cos(away), cy + 1.9 * radius * np.sin(away)
        shell = _disc(size, ex, ey, 0.8 * base_r) - _disc(size, ex, ey, 0.66 * base_r)
        _blend(canvas, np.clip(shell, 0, 1), 0.64, 0.6)

    # Embryo body, plus the herniating lobe for a hatching blastocyst.
    body = _disc(size, cx, cy, radius)
    lobe_r = 0.0
    if grade == 5:
        lobe_r = radius * rng.uniform(0.38, 0.6)
        lx = cx + (radius + 0.55 * lobe_r) * np.cos(hatch_angle)
        ly = cy + (radius + 0.55 * lobe_r) * np.sin(hatch_angle)
        body = np.clip(body + _disc(size, lx, ly, lobe_r), 0, 1)
    _blend(canvas, body, 0.47, 0.9)

    # Blastocoel cavity: partial for early grades, filling the embryo from grade 3.
    te_thick = radius * 0.075
    if grade == 1:
        cav_r, offset = radius * rng.uniform(0.38, 0.55), radius * rng.uniform(0.25, 0.4)
    elif grade == 2:
        cav_r, offset = radius * rng.uniform(0.7, 0.8), radius * rng.uniform(0.08, 0.16)
    else:
        cav_r, offset = radius - te_thick, 0.0
    cav_angle = rng.uniform(0, 2 * np.pi)
    cav_x, cav_y = cx + offset * np.cos(cav_angle), cy + offset * np.sin(cav_angle)
    cavity = _disc(size, cav_x, cav_y, cav_r)
    if grade == 5:
        cavity = np.clip(cavity + _disc(size, lx, ly, lobe_r - te_thick), 0, 1)

    # Cellular texture in the solid part of an early embryo.
    grain = cv2.GaussianBlur(rng.normal(0, 1, (size, size)).astype(np.float32), (0, 0), 2.2)
    canvas += 0.16 * grain * body * (1 - cavity)
    _blend(canvas, cavity, 0.585, 0.85)

    # Trophectoderm lining the cavity.
    te_grade = te if grade >= 3 else 1
    n_cells, coverage, length = [(rng.integers(36, 46), 0.97, 0.085), (rng.integers(18, 25), 0.85, 0.13), (rng.integers(7, 12), 0.7, 0.2)][te_grade]
    scale = radius / base_r
    _cells_on_arc(canvas, rng, cx, cy, radius - te_thick * 0.6, int(n_cells * scale), radius * length / scale ** 0.5, te_thick * 0.75,
                  coverage, 0.43, 0.33)
    if grade == 5:
        _cells_on_arc(canvas, rng, lx, ly, lobe_r - te_thick * 0.6, max(4, int(n_cells * lobe_r / radius)), radius * length,
                      te_thick * 0.7, coverage, 0.43, 0.33)

    # Inner cell mass on the cavity wall.
    icm_angle = rng.uniform(0, 2 * np.pi)
    icm_x, icm_y, icm_r = cx, cy, 0.0
    if grade >= 3:
        icm_r, n_blobs, spread, tone, alpha = [
            (radius * rng.uniform(0.32, 0.38), rng.integers(16, 22), 0.55, 0.30, 0.85),
            (radius * rng.uniform(0.22, 0.27), rng.integers(7, 11), 0.85, 0.35, 0.7),
            (radius * rng.uniform(0.10, 0.14), rng.integers(2, 4), 1.0, 0.36, 0.65),
        ][icm]
        distance = cav_r - icm_r * 0.75
        icm_x, icm_y = cx + distance * np.cos(icm_angle), cy + distance * np.sin(icm_angle)
        cluster = np.zeros((size, size), np.float32)
        for _ in range(int(n_blobs)):
            bx, by = icm_x + rng.normal(0, icm_r * spread * 0.45), icm_y + rng.normal(0, icm_r * spread * 0.45)
            cluster = np.maximum(cluster, _disc(size, bx, by, icm_r * rng.uniform(0.22, 0.34), 1.0))
        _blend(canvas, cluster * cavity.clip(0, 1) ** 0.3, tone, alpha)

    # Optics: relief contrast, focus blur, sensor noise, debris, exposure.
    shift = cv2.warpAffine(canvas, np.float32([[1, 0, 1.5 * np.cos(tilt)], [0, 1, 1.5 * np.sin(tilt)]]), (size, size), borderMode=cv2.BORDER_REFLECT)
    canvas += rng.uniform(0.5, 1.1) * (canvas - shift)
    for _ in range(rng.integers(0, 4)):
        dx, dy = rng.uniform(0, s, 2)
        if (dx - cx) ** 2 + (dy - cy) ** 2 > (radius * 1.5) ** 2:
            _blend(canvas, _disc(size, dx, dy, rng.uniform(1.5, 4.0), 1.0), rng.uniform(0.3, 0.42), 0.6)
    canvas = cv2.GaussianBlur(canvas, (0, 0), rng.uniform(0.7, 1.6))
    canvas += rng.normal(0, rng.uniform(0.012, 0.03), canvas.shape).astype(np.float32)
    canvas = (canvas - 0.5) * rng.uniform(1.25, 1.75) + 0.5 + rng.uniform(-0.05, 0.05)
    image = (np.clip(canvas, 0, 1) * 255).astype(np.uint8)
    return RenderResult(image, icm_x / s, icm_y / s, icm_r / s, radius / s)
