"""Teste unitare pentru curatarea datelor, generarea hartilor de densitate si loader-ul PyTorch."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from src.data.cleaning import SampleAudit, clean_points, create_stratified_train_val_split
from src.data.density_map import (
    compute_adaptive_sigmas,
    downsample_density_map,
    generate_adaptive_density_map,
    generate_fixed_density_map,
)
from src.data.io_utils import load_density_map_h5, save_density_map_h5
from src.data.pytorch_dataset import _load_sample_numpy


class TestPointCleaning(unittest.TestCase):
    def test_cleans_nan_oob_and_duplicates(self) -> None:
        pts = np.array(
            [
                [10.0, 20.0],
                [10.2, 20.1],      # duplicat (< 0.5px distanta) -> eliminat
                [-2.0, 50.0],      # usor in afara marginii -> clipat la x=0
                [102.0, 50.0],     # usor in afara marginii (W=100) -> clipat la x=99
                [-50.0, 50.0],     # mult in afara (>10px) -> eliminat
                [np.nan, 10.0],    # NaN -> eliminat
                [50.0, 50.0],      # valid
            ],
            dtype=np.float64,
        )
        cleaned, stats = clean_points(
            pts,
            width=100,
            height=80,
            declared_count=7,
            clip_oob=True,
            drop_outside_margin_px=10.0,
            deduplicate_radius_px=0.5,
        )
        self.assertEqual(stats.raw_count, 7)
        self.assertEqual(stats.nan_or_inf_dropped, 1)
        self.assertEqual(stats.far_oob_dropped, 1)
        self.assertEqual(stats.border_oob_clipped, 2)
        self.assertEqual(stats.duplicates_dropped, 1)
        self.assertEqual(stats.clean_count, 4)
        self.assertTrue((cleaned[:, 0] >= 0.0).all() and (cleaned[:, 0] <= 99.0).all())
        self.assertTrue((cleaned[:, 1] >= 0.0).all() and (cleaned[:, 1] <= 79.0).all())

    def test_empty_points(self) -> None:
        cleaned, stats = clean_points(np.empty((0, 2)), width=100, height=100)
        self.assertEqual(len(cleaned), 0)
        self.assertEqual(stats.clean_count, 0)


class TestDensityMaps(unittest.TestCase):
    def test_fixed_density_conserves_mass_including_borders(self) -> None:
        pts = np.array([[0.0, 0.0], [99.0, 79.0], [50.0, 40.0], [0.5, 78.5]], dtype=np.float64)
        dmap = generate_fixed_density_map((80, 100), pts, sigma=15.0)
        self.assertEqual(dmap.shape, (80, 100))
        self.assertEqual(dmap.dtype, np.float32)
        self.assertGreaterEqual(float(dmap.min()), 0.0)
        self.assertAlmostEqual(float(dmap.sum()), float(len(pts)), places=4)

    def test_adaptive_density_conserves_mass_and_respects_bounds(self) -> None:
        rng = np.random.default_rng(0)
        pts = np.column_stack([rng.uniform(0, 199, size=40), rng.uniform(0, 149, size=40)])
        sigmas = compute_adaptive_sigmas(pts, k=3, beta=0.3, sigma_min=1.5, sigma_max=25.0)
        self.assertTrue((sigmas >= 1.5).all() and (sigmas <= 25.0).all())

        dmap = generate_adaptive_density_map((150, 200), pts, k=3, beta=0.3)
        self.assertEqual(dmap.shape, (150, 200))
        self.assertAlmostEqual(float(dmap.sum()), 40.0, places=3)

    def test_adaptive_single_and_zero_points(self) -> None:
        d0 = generate_adaptive_density_map((64, 64), np.empty((0, 2)))
        self.assertEqual(float(d0.sum()), 0.0)

        d1 = generate_adaptive_density_map((64, 64), np.array([[32.0, 32.0]]))
        self.assertAlmostEqual(float(d1.sum()), 1.0, places=5)

    def test_downsample_conserves_mass(self) -> None:
        pts = np.array([[10.0, 10.0], [40.0, 60.0], [110.0, 80.0]], dtype=np.float64)
        dmap = generate_fixed_density_map((96, 128), pts, sigma=8.0)
        ds = downsample_density_map(dmap, factor=8)
        self.assertEqual(ds.shape, (12, 16))
        self.assertAlmostEqual(float(ds.sum()), float(dmap.sum()), places=4)


class TestStratifiedSplitAndIO(unittest.TestCase):
    def test_stratified_split_preserves_test_and_ratio(self) -> None:
        audits: list[SampleAudit] = []
        for i in range(100):
            audits.append(
                SampleAudit(
                    sample_id=f"tr_{i}",
                    part="part_A",
                    original_split="train_data",
                    assigned_split="train",
                    image_path="",
                    mat_path="",
                    clean_image_path="",
                    density_path="",
                    width=100,
                    height=100,
                    original_mode="RGB",
                    converted_to_rgb=False,
                    raw_count=i * 10 + 5,
                    declared_count=i * 10 + 5,
                    count_mismatch=False,
                    nan_or_inf_dropped=0,
                    far_oob_dropped=0,
                    border_oob_clipped=0,
                    duplicates_dropped=0,
                    clean_count=i * 10 + 5,
                    density_sum=float(i * 10 + 5),
                    abs_count_error=0.0,
                    is_valid=True,
                )
            )
        for i in range(20):
            audits.append(
                SampleAudit(
                    sample_id=f"te_{i}",
                    part="part_A",
                    original_split="test_data",
                    assigned_split="test",
                    image_path="",
                    mat_path="",
                    clean_image_path="",
                    density_path="",
                    width=100,
                    height=100,
                    original_mode="RGB",
                    converted_to_rgb=False,
                    raw_count=50,
                    declared_count=50,
                    count_mismatch=False,
                    nan_or_inf_dropped=0,
                    far_oob_dropped=0,
                    border_oob_clipped=0,
                    duplicates_dropped=0,
                    clean_count=50,
                    density_sum=50.0,
                    abs_count_error=0.0,
                    is_valid=True,
                )
            )

        create_stratified_train_val_split(audits, val_ratio=0.15, n_bins=5, seed=42)
        n_train = sum(a.assigned_split == "train" for a in audits)
        n_val = sum(a.assigned_split == "val" for a in audits)
        n_test = sum(a.assigned_split == "test" for a in audits)
        self.assertEqual(n_test, 20)
        self.assertEqual(n_val, 15)
        self.assertEqual(n_train, 85)

    def test_h5_roundtrip_and_loader(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            img_path = tmp_dir / "sample.jpg"
            h5_path = tmp_dir / "sample.h5"

            Image.fromarray(np.full((64, 96, 3), 128, dtype=np.uint8)).save(img_path)
            pts = np.array([[20.0, 20.0], [70.0, 40.0]], dtype=np.float64)
            dmap = generate_fixed_density_map((64, 96), pts, sigma=6.0)
            save_density_map_h5(h5_path, dmap, points=pts, metadata={"clean_count": 2})

            loaded_dmap, attrs = load_density_map_h5(h5_path)
            self.assertEqual(attrs["clean_count"], 2)
            np.testing.assert_allclose(loaded_dmap, dmap, atol=1e-6)

            img_chw, d_chw, gt_count = _load_sample_numpy(
                img_path, h5_path, augment=False, downsample_factor=8
            )
            self.assertEqual(img_chw.shape, (3, 64, 96))
            self.assertEqual(d_chw.shape, (1, 8, 12))
            self.assertAlmostEqual(float(d_chw.sum()), gt_count, places=4)


if __name__ == "__main__":
    unittest.main()
