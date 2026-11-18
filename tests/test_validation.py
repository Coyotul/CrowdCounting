"""Teste unitare pentru modulul de validare a integritatii datelor procesate."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from src.data.validate_processed import (
    validate_density_file,
    validate_split_integrity,
)


class TestDensityFileValidation(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.test_dir = Path(self.tmp_dir.name)

    def tearDown(self) -> None:
        self.tmp_dir.cleanup()

    def test_valid_density_file(self) -> None:
        h5_path = self.test_dir / "valid.h5"
        density = np.ones((50, 60), dtype=np.float32) * 0.1  # suma = 300.0
        points = np.array([[10.0, 10.0], [20.0, 20.0]], dtype=np.float32)

        with h5py.File(h5_path, "w") as hf:
            hf.create_dataset("density", data=density)
            hf.create_dataset("points", data=points)

        ok, errs, stats = validate_density_file(
            h5_path=h5_path,
            expected_shape=(50, 60),
            expected_count=300.0,
            tol=0.01,
        )
        self.assertTrue(ok)
        self.assertEqual(len(errs), 0)
        self.assertTrue(stats["mass_conserved"])
        self.assertTrue(stats["no_negative_values"])
        self.assertTrue(stats["is_finite"])
        self.assertTrue(stats["points_in_bounds"])

    def test_shape_mismatch_detected(self) -> None:
        h5_path = self.test_dir / "wrong_shape.h5"
        density = np.ones((40, 60), dtype=np.float32)

        with h5py.File(h5_path, "w") as hf:
            hf.create_dataset("density", data=density)

        ok, errs, _ = validate_density_file(
            h5_path=h5_path,
            expected_shape=(50, 60),
            expected_count=2400.0,
        )
        self.assertFalse(ok)
        self.assertTrue(any("Dimensiune gresita" in e for e in errs))

    def test_negative_and_nan_detected(self) -> None:
        h5_path_neg = self.test_dir / "negative.h5"
        density_neg = np.ones((10, 10), dtype=np.float32)
        density_neg[0, 0] = -0.5

        with h5py.File(h5_path_neg, "w") as hf:
            hf.create_dataset("density", data=density_neg)

        ok, errs, stats = validate_density_file(
            h5_path=h5_path_neg,
            expected_shape=(10, 10),
            expected_count=99.5,
        )
        self.assertFalse(ok)
        self.assertFalse(stats["no_negative_values"])
        self.assertTrue(any("valori negative" in e for e in errs))

        h5_path_nan = self.test_dir / "nan.h5"
        density_nan = np.ones((10, 10), dtype=np.float32)
        density_nan[0, 0] = np.nan

        with h5py.File(h5_path_nan, "w") as hf:
            hf.create_dataset("density", data=density_nan)

        ok, errs, stats = validate_density_file(
            h5_path=h5_path_nan,
            expected_shape=(10, 10),
            expected_count=100.0,
        )
        self.assertFalse(ok)
        self.assertFalse(stats["is_finite"])
        self.assertTrue(any("NaN sau Inf" in e for e in errs))

    def test_mass_conservation_violation(self) -> None:
        h5_path = self.test_dir / "mass_viol.h5"
        density = np.ones((10, 10), dtype=np.float32)  # suma = 100.0

        with h5py.File(h5_path, "w") as hf:
            hf.create_dataset("density", data=density)

        ok, errs, stats = validate_density_file(
            h5_path=h5_path,
            expected_shape=(10, 10),
            expected_count=105.0,  # diff = 5.0 > tol
            tol=0.1,
        )
        self.assertFalse(ok)
        self.assertFalse(stats["mass_conserved"])
        self.assertTrue(any("Eroare de conservare a masei" in e for e in errs))


class TestSplitIntegrityValidation(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.test_dir = Path(self.tmp_dir.name)

    def tearDown(self) -> None:
        self.tmp_dir.cleanup()

    def test_disjoint_and_complete_splits(self) -> None:
        part = "part_test"
        df_train = pd.DataFrame({"sample_id": ["s1", "s2", "s3"]})
        df_val = pd.DataFrame({"sample_id": ["s4"]})
        df_test = pd.DataFrame({"sample_id": ["s5", "s6"]})
        df_main = pd.DataFrame({"sample_id": ["s1", "s2", "s3", "s4", "s5", "s6"]})

        df_train.to_csv(self.test_dir / f"{part}_train.csv", index=False)
        df_val.to_csv(self.test_dir / f"{part}_val.csv", index=False)
        df_test.to_csv(self.test_dir / f"{part}_test.csv", index=False)
        df_main.to_csv(self.test_dir / f"{part}_manifest.csv", index=False)

        ok, errs = validate_split_integrity(self.test_dir, part)
        self.assertTrue(ok)
        self.assertEqual(len(errs), 0)

    def test_overlapping_splits_detected(self) -> None:
        part = "part_test"
        df_train = pd.DataFrame({"sample_id": ["s1", "s2"]})
        df_val = pd.DataFrame({"sample_id": ["s2", "s3"]})
        df_test = pd.DataFrame({"sample_id": ["s4"]})
        df_main = pd.DataFrame({"sample_id": ["s1", "s2", "s3", "s4"]})

        df_train.to_csv(self.test_dir / f"{part}_train.csv", index=False)
        df_val.to_csv(self.test_dir / f"{part}_val.csv", index=False)
        df_test.to_csv(self.test_dir / f"{part}_test.csv", index=False)
        df_main.to_csv(self.test_dir / f"{part}_manifest.csv", index=False)

        ok, errs = validate_split_integrity(self.test_dir, part)
        self.assertFalse(ok)
        self.assertTrue(any("Suprapunere intre Train si Val" in e for e in errs))


if __name__ == "__main__":
    unittest.main()
