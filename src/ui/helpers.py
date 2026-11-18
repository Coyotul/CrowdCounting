"""Functii ajutatoare de procesare vizuala si calcul statistic pentru aplicatia Streamlit."""

from __future__ import annotations

import matplotlib
import numpy as np
from PIL import Image, ImageDraw


def make_colored_heatmap(
    density: np.ndarray,
    colormap_name: str = "jet",
    percentile: float = 99.5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Genereaza o harta colorata RGB (H, W, 3) din matricea 2D de densitate.

    Returns:
        rgb_heatmap: Array uint8 (H, W, 3) cu harta colorata.
        normalized: Array float (H, W) cu valorile normalizate in [0, 1].
    """
    d = np.maximum(density, 0.0)
    pos = d[d > 0]
    vmax = float(np.percentile(pos, percentile)) if pos.size > 0 else 1.0
    vmax = max(vmax, 1e-12)

    norm = np.power(np.clip(d / vmax, 0.0, 1.0), 0.5)

    cmap = matplotlib.colormaps.get(colormap_name, matplotlib.colormaps["jet"])
    rgb = (cmap(norm)[..., :3] * 255.0).astype(np.uint8)
    return rgb, norm


def make_blended_overlay(
    img: np.ndarray,
    heatmap: np.ndarray,
    norm: np.ndarray,
    alpha: float = 0.6,
) -> np.ndarray:
    """Suprapune harta de densitate peste imaginea originala ponderat cu densitatea normalizata."""
    alpha_clamped = float(np.clip(alpha, 0.0, 1.0))
    w = np.clip(norm * 1.4, 0.0, 1.0)[..., np.newaxis] * alpha_clamped
    out = img.astype(np.float32) * (1.0 - w) + heatmap.astype(np.float32) * w
    return np.clip(out, 0, 255).astype(np.uint8)


def draw_crowd_points(
    img: np.ndarray,
    points: np.ndarray,
    radius: int = 2,
    color: tuple[int, int, int] = (0, 255, 0),
) -> np.ndarray:
    """Deseneaza adnotarile de capete peste imagine."""
    pil_img = Image.fromarray(img.copy())
    draw = ImageDraw.Draw(pil_img)
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2) if points.size > 0 else np.empty((0, 2))

    for x, y in pts:
        ix, iy = int(round(float(x))), int(round(float(y)))
        draw.ellipse((ix - radius, iy - radius, ix + radius, iy + radius), fill=color)

    return np.asarray(pil_img, dtype=np.uint8)


def compute_density_quadrants(density: np.ndarray) -> dict[str, float]:
    """
    Calculeaza distributia spatiala a multimii pe 4 cadrane (Top-Left, Top-Right, Bottom-Left, Bottom-Right).
    Util pentru analiza distributiei spatiale si a metricii GAME(1).
    """
    h, w = density.shape
    mid_h = h // 2
    mid_w = w // 2

    tl = float(density[0:mid_h, 0:mid_w].sum())
    tr = float(density[0:mid_h, mid_w:w].sum())
    bl = float(density[mid_h:h, 0:mid_w].sum())
    br = float(density[mid_h:h, mid_w:w].sum())

    total = max(tl + tr + bl + br, 1e-12)
    return {
        "top_left": tl,
        "top_right": tr,
        "bottom_left": bl,
        "bottom_right": br,
        "pct_top_left": (tl / total) * 100.0,
        "pct_top_right": (tr / total) * 100.0,
        "pct_bottom_left": (bl / total) * 100.0,
        "pct_bottom_right": (br / total) * 100.0,
    }
