"""Generare harti de densitate bazate pe kernel-uri Gaussiene (fix si adaptiv k-NN)."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import zoom
from scipy.spatial import cKDTree


def _add_normalized_gaussian_patch(
    density: np.ndarray,
    x: float,
    y: float,
    sigma: float,
    truncate: float = 3.5,
) -> None:
    """
    Adauga un kernel Gaussian 2D normalizat centrat in (x, y) cu deviatia standard `sigma`.
    Normalizarea pe fereastra vizibila asigura conservarea masei (suma = 1.0 per persoana).
    """
    h, w = density.shape
    if sigma <= 0.0:
        cx = int(np.clip(round(x), 0, w - 1))
        cy = int(np.clip(round(y), 0, h - 1))
        density[cy, cx] += 1.0
        return

    radius = max(1, int(np.ceil(truncate * sigma)))
    cx = int(np.clip(round(x), 0, w - 1))
    cy = int(np.clip(round(y), 0, h - 1))

    x0 = max(0, cx - radius)
    x1 = min(w, cx + radius + 1)
    y0 = max(0, cy - radius)
    y1 = min(h, cy + radius + 1)

    xs = np.arange(x0, x1, dtype=np.float64) - x
    ys = np.arange(y0, y1, dtype=np.float64) - y

    inv_two_sigma2 = 1.0 / (2.0 * sigma * sigma)
    gx = np.exp(-(xs * xs) * inv_two_sigma2)
    gy = np.exp(-(ys * ys) * inv_two_sigma2)

    kernel = np.outer(gy, gx)
    k_sum = kernel.sum()
    if k_sum > 0.0:
        kernel /= k_sum
        density[y0:y1, x0:x1] += kernel


def compute_adaptive_sigmas(
    points: np.ndarray,
    k: int = 3,
    beta: float = 0.3,
    sigma_min: float = 1.5,
    sigma_max: float = 25.0,
    fallback_sigma: float = 15.0,
) -> np.ndarray:
    """
    Calculeaza sigma adaptiv pentru fiecare cap adnotat:
        sigma_i = clip(beta * mean(d_{i,1}, ..., d_{i,k}), sigma_min, sigma_max)
    unde d_{i,j} sunt distantele euclidiene catre cei mai apropiati k vecini.
    """
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    n = len(pts)
    if n == 0:
        return np.empty((0,), dtype=np.float64)
    if n == 1:
        return np.array([fallback_sigma], dtype=np.float64)

    k_eff = min(k, n - 1)
    tree = cKDTree(pts)
    # query k_eff + 1 deoarece primul vecin (dist=0) este punctul insusi
    distances, _ = tree.query(pts, k=k_eff + 1)
    mean_nn_dist = distances[:, 1:].mean(axis=1)
    sigmas = beta * mean_nn_dist
    return np.clip(sigmas, sigma_min, sigma_max)


def generate_fixed_density_map(
    image_shape: tuple[int, int],
    points: np.ndarray,
    sigma: float = 15.0,
    dtype: np.dtype | type = np.float32,
) -> np.ndarray:
    """
    Genereaza harta de densitate cu kernel Gaussian fix (recomandat pentru Part B).

    Args:
        image_shape: (height, width)
        points: array (N, 2) cu coordonate (x, y)
        sigma: deviatia standard a kernel-ului Gaussian (pixeli)
        dtype: tipul numeric al hartii finale (implicit float32)
    """
    h, w = image_shape
    density = np.zeros((h, w), dtype=np.float64)
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2) if len(points) > 0 else np.empty((0, 2))

    for x, y in pts:
        if 0.0 <= x <= w - 1 and 0.0 <= y <= h - 1:
            _add_normalized_gaussian_patch(density, float(x), float(y), sigma=sigma)

    return density.astype(dtype, copy=False)


def generate_adaptive_density_map(
    image_shape: tuple[int, int],
    points: np.ndarray,
    k: int = 3,
    beta: float = 0.3,
    sigma_min: float = 1.5,
    sigma_max: float = 25.0,
    fallback_sigma: float = 15.0,
    dtype: np.dtype | type = np.float32,
) -> np.ndarray:
    """
    Genereaza harta de densitate cu kernel Gaussian adaptiv geometric (k-NN),
    conform lucrarilor MCNN (Zhang et al., CVPR 2016) si CSRNet (Li et al., CVPR 2018).
    Recomandat pentru ShanghaiTech Part A (aglomerari mari, perspectiva variabila).
    """
    h, w = image_shape
    density = np.zeros((h, w), dtype=np.float64)
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2) if len(points) > 0 else np.empty((0, 2))

    if len(pts) == 0:
        return density.astype(dtype, copy=False)

    sigmas = compute_adaptive_sigmas(
        pts,
        k=k,
        beta=beta,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        fallback_sigma=fallback_sigma,
    )

    for (x, y), sigma in zip(pts, sigmas):
        if 0.0 <= x <= w - 1 and 0.0 <= y <= h - 1:
            _add_normalized_gaussian_patch(density, float(x), float(y), sigma=float(sigma))

    return density.astype(dtype, copy=False)


def generate_density_map(
    image_shape: tuple[int, int],
    points: np.ndarray,
    method: str = "adaptive",
    **kwargs: float | int,
) -> np.ndarray:
    """Functie unificata pentru generarea hartii de densitate ('adaptive' sau 'fixed')."""
    method_norm = method.strip().lower()
    if method_norm == "adaptive":
        return generate_adaptive_density_map(
            image_shape=image_shape,
            points=points,
            k=int(kwargs.get("knn_k", kwargs.get("k", 3))),
            beta=float(kwargs.get("beta", 0.3)),
            sigma_min=float(kwargs.get("sigma_min", 1.5)),
            sigma_max=float(kwargs.get("sigma_max", 25.0)),
            fallback_sigma=float(kwargs.get("fallback_sigma", 15.0)),
        )
    if method_norm == "fixed":
        return generate_fixed_density_map(
            image_shape=image_shape,
            points=points,
            sigma=float(kwargs.get("fixed_sigma", kwargs.get("sigma", 15.0))),
        )
    raise ValueError(f"Metoda necunoscuta pentru harta de densitate: '{method}'. Foloseste 'adaptive' sau 'fixed'.")


def downsample_density_map(density_map: np.ndarray, factor: int = 8) -> np.ndarray:
    """
    Redimensioneaza harta de densitate la rezolutie 1/factor (ex. 1/8 pentru CSRNet)
    pastrand EXACT integrala (numarul total de persoane).
    """
    if factor == 1:
        return density_map.astype(np.float32, copy=False)

    h, w = density_map.shape
    target_h = max(1, h // factor)
    target_w = max(1, w // factor)

    orig_sum = float(density_map.sum())
    resized = zoom(
        density_map.astype(np.float64),
        zoom=(target_h / h, target_w / w),
        order=1,
    )
    resized = np.maximum(resized, 0.0)
    new_sum = float(resized.sum())
    if orig_sum > 0.0 and new_sum > 0.0:
        resized *= orig_sum / new_sum

    return resized.astype(np.float32)
