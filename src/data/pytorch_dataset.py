"""Dataset PyTorch pentru incarcarea imaginilor si a hartilor de densitate."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

from .density_map import downsample_density_map
from .io_utils import load_density_map_h5

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _load_sample_numpy(
    image_path: Path,
    density_path: Path,
    augment: bool = False,
    crop_scale: float = 0.5,
    downsample_factor: int = 8,
    rng: random.Random | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Incarca si preproceseaza un esantion in format NumPy (C, H, W) + (1, H/f, W/f).
    Poate fi testat local fara PyTorch instalat, iar in Colab/Kaggle este invelit in Tensor.
    """
    r = rng if rng is not None else random
    img = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.float32) / 255.0
    density, _ = load_density_map_h5(density_path)

    h, w, _ = img.shape
    if augment:
        # 1. Random crop (stil CSRNet: 1/2 din latime x 1/2 din inaltime)
        if 0.0 < crop_scale < 1.0:
            ch = max(32, (int(h * crop_scale) // downsample_factor) * downsample_factor)
            cw = max(32, (int(w * crop_scale) // downsample_factor) * downsample_factor)
            if ch < h and cw < w:
                y0 = r.randint(0, h - ch)
                x0 = r.randint(0, w - cw)
                img = img[y0 : y0 + ch, x0 : x0 + cw]
                density = density[y0 : y0 + ch, x0 : x0 + cw]

        # 2. Random horizontal flip sincronizat
        if r.random() < 0.5:
            img = np.ascontiguousarray(img[:, ::-1, :])
            density = np.ascontiguousarray(density[:, ::-1])

    # Asiguram dimensiuni multiplu de downsample_factor pentru a evita neconcordante de shape in CNN
    h, w, _ = img.shape
    h_aligned = max(downsample_factor, (h // downsample_factor) * downsample_factor)
    w_aligned = max(downsample_factor, (w // downsample_factor) * downsample_factor)
    if h_aligned != h or w_aligned != w:
        img = img[:h_aligned, :w_aligned]
        density = density[:h_aligned, :w_aligned]

    # Normalizare ImageNet -> CHW
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    img_chw = np.transpose(img, (2, 0, 1)).astype(np.float32)

    gt_count = float(density.sum())
    density_ds = downsample_density_map(density, factor=downsample_factor)
    density_chw = density_ds[np.newaxis, :, :].astype(np.float32)
    return img_chw, density_chw, gt_count


class ShanghaiTechCrowdDataset:
    """
    Dataset compatibil cu `torch.utils.data.DataLoader` pentru ShanghaiTech.

    Exemplu de utilizare:
        ds = ShanghaiTechCrowdDataset(
            manifest_csv="data/processed/manifests/part_A_train.csv",
            project_root=".",
            augment=True,
            downsample_factor=8,
            as_torch=True,
        )
    """

    def __init__(
        self,
        manifest_csv: str | Path,
        project_root: str | Path = ".",
        augment: bool = False,
        crop_scale: float = 0.5,
        downsample_factor: int = 8,
        as_torch: bool = True,
    ) -> None:
        self.project_root = Path(project_root)
        self.df = pd.read_csv(manifest_csv)
        self.augment = augment
        self.crop_scale = crop_scale
        self.downsample_factor = downsample_factor
        self.as_torch = as_torch

        if self.as_torch:
            try:
                import torch  # noqa: F401
            except ImportError as exc:
                raise ImportError(
                    "PyTorch nu este instalat in acest mediu. Seteaza as_torch=False "
                    "pentru rulare NumPy sau instaleaza torch in Colab/Kaggle."
                ) from exc

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.df.iloc[idx]
        img_path = self.project_root / row["clean_image_path"]
        density_path = self.project_root / row["density_path"]

        img_chw, density_chw, gt_count = _load_sample_numpy(
            image_path=img_path,
            density_path=density_path,
            augment=self.augment,
            crop_scale=self.crop_scale,
            downsample_factor=self.downsample_factor,
        )

        if self.as_torch:
            import torch

            return {
                "image": torch.from_numpy(img_chw),
                "density": torch.from_numpy(density_chw),
                "count": torch.tensor(gt_count, dtype=torch.float32),
                "sample_id": str(row["sample_id"]),
            }

        return {
            "image": img_chw,
            "density": density_chw,
            "count": gt_count,
            "sample_id": str(row["sample_id"]),
        }
