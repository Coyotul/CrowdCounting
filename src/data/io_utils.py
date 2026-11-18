"""Utilitare I/O pentru citirea dataset-ului ShanghaiTech si salvarea hartilor de densitate."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import scipy.io as sio
import yaml


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Incarca fisierul de configuratie YAML."""
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _natural_sort_key(path: Path) -> list[int | str]:
    """Cheie de sortare naturala (IMG_2.jpg inainte de IMG_10.jpg)."""
    return [
        int(tok) if tok.isdigit() else tok.lower()
        for tok in re.split(r"(\d+)", path.name)
    ]


def discover_shanghaitech_pairs(
    raw_root: str | Path,
    part: str,
    split: str,
) -> tuple[list[tuple[Path, Path]], list[Path], list[Path]]:
    """
    Descopera perechile (imagine, fisier .mat) pentru un subset ShanghaiTech.

    Returns:
        pairs: lista sortata de tupluri (image_path, mat_path)
        orphan_images: imagini fara fisier GT corespondent
        orphan_mats: fisiere GT fara imagine corespondenta
    """
    split_dir = Path(raw_root) / part / split
    images_dir = split_dir / "images"
    gt_dir = split_dir / "ground-truth"

    if not images_dir.exists():
        raise FileNotFoundError(f"Lipseste directorul de imagini: {images_dir}")
    if not gt_dir.exists():
        raise FileNotFoundError(f"Lipseste directorul ground-truth: {gt_dir}")

    image_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    image_map: dict[str, Path] = {}
    for p in sorted(images_dir.iterdir(), key=_natural_sort_key):
        if p.is_file() and p.suffix.lower() in image_exts:
            image_map[p.stem] = p

    mat_map: dict[str, Path] = {}
    for p in sorted(gt_dir.iterdir(), key=_natural_sort_key):
        if p.is_file() and p.suffix.lower() == ".mat":
            # ShanghaiTech foloseste conventia GT_IMG_1.mat <-> IMG_1.jpg
            stem = p.stem[3:] if p.stem.startswith("GT_") else p.stem
            mat_map[stem] = p

    matched_keys = sorted(
        set(image_map.keys()) & set(mat_map.keys()),
        key=lambda s: [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)],
    )
    pairs = [(image_map[k], mat_map[k]) for k in matched_keys]
    orphan_images = [image_map[k] for k in sorted(set(image_map) - set(mat_map))]
    orphan_mats = [mat_map[k] for k in sorted(set(mat_map) - set(image_map))]

    return pairs, orphan_images, orphan_mats


def load_shanghaitech_mat(mat_path: str | Path) -> tuple[np.ndarray, int]:
    """
    Citeste coordonatele capetelor (x, y) si numarul declarat dintr-un fisier .mat ShanghaiTech.

    Returns:
        points: array float64 cu forma (N, 2) in format (x, y) = (coloana, rand)
        declared_count: int, numarul de adnotari raportat in structura .mat
    """
    mat = sio.loadmat(str(mat_path))
    if "image_info" not in mat:
        raise KeyError(f"Fisierul {mat_path} nu contine cheia 'image_info'.")

    info = mat["image_info"][0, 0][0, 0]
    raw_points = np.asarray(info[0], dtype=np.float64)

    if raw_points.size == 0:
        points = np.empty((0, 2), dtype=np.float64)
    else:
        points = raw_points.reshape(-1, 2)

    declared_count = int(np.asarray(info[1]).squeeze()) if len(info) > 1 else len(points)
    return points, declared_count


def save_density_map_h5(
    output_path: str | Path,
    density_map: np.ndarray,
    points: np.ndarray | None = None,
    metadata: dict[str, Any] | None = None,
    compression: str | None = "gzip",
    compression_opts: int | None = 4,
) -> None:
    """
    Salveaza harta de densitate in format HDF5 (.h5), compatibil cu CSRNet/PyTorch.

    Cheia principala este 'density' (standardul din repo-ul oficial CSRNet),
    impreuna cu 'points' (coordonatele curatate) si atribute de audit.
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    dset_kwargs: dict[str, Any] = {}
    if compression:
        dset_kwargs["compression"] = compression
        if compression == "gzip" and compression_opts is not None:
            dset_kwargs["compression_opts"] = compression_opts

    with h5py.File(out, "w") as hf:
        hf.create_dataset("density", data=density_map, **dset_kwargs)
        if points is not None:
            hf.create_dataset("points", data=np.asarray(points, dtype=np.float32))
        if metadata:
            for key, value in metadata.items():
                if value is not None:
                    hf.attrs[key] = value


def load_density_map_h5(h5_path: str | Path) -> tuple[np.ndarray, dict[str, Any]]:
    """Incarca harta de densitate si atributele sale dintr-un fisier .h5."""
    with h5py.File(h5_path, "r") as hf:
        density = np.asarray(hf["density"], dtype=np.float32)
        attrs = dict(hf.attrs)
    return density, attrs
