"""Pipeline de audit, validare si curatare a datelor pentru Crowd Counting."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree

from .io_utils import load_shanghaitech_mat


@dataclass
class PointCleaningStats:
    """Statistici rezultate in urma curatarii coordonatelor adnotate."""

    raw_count: int
    declared_count: int
    count_mismatch: bool
    nan_or_inf_dropped: int
    far_oob_dropped: int
    border_oob_clipped: int
    duplicates_dropped: int
    clean_count: int


@dataclass
class SampleAudit:
    """Rezultatul complet de audit si curatare pentru o imagine din dataset."""

    sample_id: str
    part: str
    original_split: str
    assigned_split: str
    image_path: str
    mat_path: str
    clean_image_path: str
    density_path: str
    width: int
    height: int
    original_mode: str
    converted_to_rgb: bool
    raw_count: int
    declared_count: int
    count_mismatch: bool
    nan_or_inf_dropped: int
    far_oob_dropped: int
    border_oob_clipped: int
    duplicates_dropped: int
    clean_count: int
    density_sum: float
    abs_count_error: float
    is_valid: bool
    error_message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def clean_points(
    points: np.ndarray,
    width: int,
    height: int,
    declared_count: int | None = None,
    clip_oob: bool = True,
    drop_outside_margin_px: float = 10.0,
    deduplicate_radius_px: float = 0.5,
) -> tuple[np.ndarray, PointCleaningStats]:
    """
    Curata coordonatele adnotate (x, y) ale persoanelor dintr-o imagine.

    Pasi aplicati:
    1. Elimina coordonatele NaN / Inf.
    2. Elimina punctele aflate mult in afara imaginii (> drop_outside_margin_px) ca erori de adnotare.
    3. Aduce in interiorul [0, W-1] x [0, H-1] punctele aflate usor pe marginea cadrului (daca clip_oob=True).
    4. Elimina adnotarile duplicate (distanta euclidiana <= deduplicate_radius_px), pastrand prima aparitie.
    """
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2) if points.size > 0 else np.empty((0, 2), dtype=np.float64)
    raw_count = int(len(pts))
    decl = raw_count if declared_count is None else int(declared_count)
    count_mismatch = raw_count != decl

    # 1. Filtrare NaN / Inf
    finite_mask = np.isfinite(pts).all(axis=1)
    nan_or_inf_dropped = int((~finite_mask).sum())
    pts = pts[finite_mask]

    # 2. Puncte mult in afara imaginii vs puncte usor pe margine
    far_oob_mask = (
        (pts[:, 0] < -drop_outside_margin_px)
        | (pts[:, 0] > (width - 1) + drop_outside_margin_px)
        | (pts[:, 1] < -drop_outside_margin_px)
        | (pts[:, 1] > (height - 1) + drop_outside_margin_px)
    )
    far_oob_dropped = int(far_oob_mask.sum())
    pts = pts[~far_oob_mask]

    # 3. Puncte usor in afara [0, W-1] x [0, H-1]
    oob_mask = (
        (pts[:, 0] < 0.0)
        | (pts[:, 0] > width - 1)
        | (pts[:, 1] < 0.0)
        | (pts[:, 1] > height - 1)
    )
    border_oob_clipped = int(oob_mask.sum())
    if border_oob_clipped > 0:
        if clip_oob:
            pts = pts.copy()
            pts[:, 0] = np.clip(pts[:, 0], 0.0, float(width - 1))
            pts[:, 1] = np.clip(pts[:, 1], 0.0, float(height - 1))
        else:
            pts = pts[~oob_mask]
            far_oob_dropped += border_oob_clipped
            border_oob_clipped = 0

    # 4. Eliminare duplicate foarte apropiate
    duplicates_dropped = 0
    if deduplicate_radius_px > 0.0 and len(pts) > 1:
        tree = cKDTree(pts)
        pairs = tree.query_pairs(r=deduplicate_radius_px)
        if pairs:
            to_drop: set[int] = set()
            for i, j in sorted(pairs):
                if i not in to_drop:
                    to_drop.add(j)
            keep_mask = np.ones(len(pts), dtype=bool)
            keep_mask[list(to_drop)] = False
            duplicates_dropped = int(len(to_drop))
            pts = pts[keep_mask]

    stats = PointCleaningStats(
        raw_count=raw_count,
        declared_count=decl,
        count_mismatch=count_mismatch,
        nan_or_inf_dropped=nan_or_inf_dropped,
        far_oob_dropped=far_oob_dropped,
        border_oob_clipped=border_oob_clipped,
        duplicates_dropped=duplicates_dropped,
        clean_count=int(len(pts)),
    )
    return pts, stats


def audit_and_clean_sample(
    image_path: Path,
    mat_path: Path,
    part: str,
    original_split: str,
    convert_grayscale_to_rgb: bool = True,
    clip_oob: bool = True,
    drop_outside_margin_px: float = 10.0,
    deduplicate_radius_px: float = 0.5,
) -> tuple[Image.Image | None, np.ndarray, SampleAudit]:
    """
    Verifica integritatea unei imagini si curata adnotarile asociate.

    Returns:
        img: imaginea PIL (convertita la RGB daca a fost cerut), sau None daca e corupta
        cleaned_points: coordonatele curatate (N, 2)
        audit: obiect SampleAudit cu toate metricile de calitate
    """
    sample_id = f"{part}_{original_split}_{image_path.stem}"
    try:
        with Image.open(image_path) as probe:
            probe.verify()

        with Image.open(image_path) as raw_img:
            raw_img.load()
            original_mode = raw_img.mode
            width, height = raw_img.size
            converted_to_rgb = False

            if convert_grayscale_to_rgb and original_mode != "RGB":
                img = raw_img.convert("RGB")
                converted_to_rgb = True
            else:
                img = raw_img.copy()

        raw_points, declared_count = load_shanghaitech_mat(mat_path)
        cleaned_points, pstats = clean_points(
            raw_points,
            width=width,
            height=height,
            declared_count=declared_count,
            clip_oob=clip_oob,
            drop_outside_margin_px=drop_outside_margin_px,
            deduplicate_radius_px=deduplicate_radius_px,
        )

        audit = SampleAudit(
            sample_id=sample_id,
            part=part,
            original_split=original_split,
            assigned_split="test" if original_split == "test_data" else "train",
            image_path=str(image_path),
            mat_path=str(mat_path),
            clean_image_path="",
            density_path="",
            width=width,
            height=height,
            original_mode=original_mode,
            converted_to_rgb=converted_to_rgb,
            raw_count=pstats.raw_count,
            declared_count=pstats.declared_count,
            count_mismatch=pstats.count_mismatch,
            nan_or_inf_dropped=pstats.nan_or_inf_dropped,
            far_oob_dropped=pstats.far_oob_dropped,
            border_oob_clipped=pstats.border_oob_clipped,
            duplicates_dropped=pstats.duplicates_dropped,
            clean_count=pstats.clean_count,
            density_sum=0.0,
            abs_count_error=0.0,
            is_valid=True,
        )
        return img, cleaned_points, audit

    except Exception as exc:
        audit = SampleAudit(
            sample_id=sample_id,
            part=part,
            original_split=original_split,
            assigned_split="invalid",
            image_path=str(image_path),
            mat_path=str(mat_path),
            clean_image_path="",
            density_path="",
            width=0,
            height=0,
            original_mode="UNKNOWN",
            converted_to_rgb=False,
            raw_count=0,
            declared_count=0,
            count_mismatch=False,
            nan_or_inf_dropped=0,
            far_oob_dropped=0,
            border_oob_clipped=0,
            duplicates_dropped=0,
            clean_count=0,
            density_sum=0.0,
            abs_count_error=0.0,
            is_valid=False,
            error_message=str(exc),
        )
        return None, np.empty((0, 2), dtype=np.float64), audit


def create_stratified_train_val_split(
    audits: list[SampleAudit],
    val_ratio: float = 0.15,
    n_bins: int = 5,
    seed: int = 42,
) -> None:
    """
    Imparte in-place esantioanele valide din 'train_data' in 'train' si 'val',
    stratificat dupa cuantilele numarului de persoane, pentru a pastra aceeasi
    distributie a densitatii in ambele subseturi.

    Esantioanele din 'test_data' raman strict 'test'.
    """
    rng = np.random.default_rng(seed)

    train_candidates = [
        a for a in audits if a.is_valid and a.original_split == "train_data"
    ]
    if not train_candidates:
        return

    counts = np.array([a.clean_count for a in train_candidates], dtype=np.float64)
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.unique(np.quantile(counts, quantiles))

    if len(edges) <= 2:
        bin_ids = np.zeros(len(train_candidates), dtype=int)
    else:
        bin_ids = np.digitize(counts, edges[1:-1], right=True)

    val_indices: set[int] = set()
    for b in np.unique(bin_ids):
        members = np.where(bin_ids == b)[0]
        shuffled = rng.permutation(members)
        n_val = max(1, int(round(len(members) * val_ratio))) if len(members) > 1 else 0
        val_indices.update(shuffled[:n_val].tolist())

    for idx, audit in enumerate(train_candidates):
        audit.assigned_split = "val" if idx in val_indices else "train"
