"""Teste unitare pentru modulul de evaluare si calcul al metricilor (MAE, MSE, RMSE, NAE, GAME)."""

from __future__ import annotations

import math
import unittest

import numpy as np

from src.evaluation.metrics import (
    compute_game,
    compute_mae,
    compute_mse,
    compute_nae,
    compute_rmse,
    evaluate_predictions,
)


class TestCrowdMetrics(unittest.TestCase):
    def test_mae_exact_and_zero_error(self) -> None:
        preds = [10.0, 20.0, 30.0]
        gts = [10.0, 20.0, 30.0]
        self.assertEqual(compute_mae(preds, gts), 0.0)

        preds = [10.0, 20.0, 30.0]
        gts = [12.0, 18.0, 33.0]
        # Erori absolute: |10-12|=2, |20-18|=2, |30-33|=3 -> media = 7/3
        self.assertAlmostEqual(compute_mae(preds, gts), 7.0 / 3.0, places=6)

    def test_mse_and_rmse(self) -> None:
        preds = np.array([10.0, 20.0])
        gts = np.array([12.0, 16.0])
        # Erori patratice: (10-12)^2 = 4, (20-16)^2 = 16 -> media = 20/2 = 10
        self.assertAlmostEqual(compute_mse(preds, gts), 10.0, places=6)
        self.assertAlmostEqual(compute_rmse(preds, gts), math.sqrt(10.0), places=6)

    def test_nae(self) -> None:
        preds = [10.0, 20.0]
        gts = [10.0, 30.0]
        # Suma erori: 0 + 10 = 10. Suma GT: 40 -> NAE = 10/40 = 0.25
        self.assertAlmostEqual(compute_nae(preds, gts), 0.25, places=6)

        # Cazul fara persoane
        self.assertEqual(compute_nae([0.0, 0.0], [0.0, 0.0]), 0.0)

    def test_input_validation(self) -> None:
        with self.assertRaises(ValueError):
            compute_mae([], [])

        with self.assertRaises(ValueError):
            compute_mae([1.0, 2.0], [1.0])

        with self.assertRaises(ValueError):
            compute_mse([1.0], [1.0, 2.0])

    def test_evaluate_predictions_summary(self) -> None:
        preds = [10.0, 20.0]
        gts = [12.0, 16.0]
        res = evaluate_predictions(preds, gts)
        self.assertEqual(res["num_samples"], 2)
        self.assertIn("mae", res)
        self.assertIn("mse", res)
        self.assertIn("rmse", res)
        self.assertIn("nae", res)
        self.assertAlmostEqual(res["mae"], 3.0, places=2)
        self.assertAlmostEqual(res["mse"], 10.0, places=2)


class TestGAMEMetric(unittest.TestCase):
    def test_game_level_0_matches_global_count_error(self) -> None:
        pred_map = np.ones((10, 10), dtype=np.float32) * 2.0  # Suma = 200
        gt_map = np.ones((10, 10), dtype=np.float32) * 1.5    # Suma = 150
        # GAME(0) = |200 - 150| = 50.0
        self.assertAlmostEqual(compute_game(pred_map, gt_map, grid_level=0), 50.0, places=5)

    def test_game_level_1_quadrants(self) -> None:
        # Harta 4x4 impartita in 4 cadrane de 2x2
        pred = np.zeros((4, 4), dtype=np.float32)
        gt = np.zeros((4, 4), dtype=np.float32)

        # Cadran top-left (0:2, 0:2)
        pred[0:2, 0:2] = 5.0 / 4.0   # suma pred = 5
        gt[0:2, 0:2] = 4.0 / 4.0     # suma gt = 4 -> diff = 1

        # Cadran top-right (0:2, 2:4)
        pred[0:2, 2:4] = 5.0 / 4.0   # suma pred = 5
        gt[0:2, 2:4] = 6.0 / 4.0     # suma gt = 6 -> diff = 1

        # Cadran bottom-left (2:4, 0:2)
        pred[2:4, 0:2] = 5.0 / 4.0   # suma pred = 5
        gt[2:4, 0:2] = 3.0 / 4.0     # suma gt = 3 -> diff = 2

        # Cadran bottom-right (2:4, 2:4)
        pred[2:4, 2:4] = 5.0 / 4.0   # suma pred = 5
        gt[2:4, 2:4] = 7.0 / 4.0     # suma gt = 7 -> diff = 2

        # Total GAME(1) = 1 + 1 + 2 + 2 = 6.0
        # Notam ca global: pred_sum = 20, gt_sum = 20, deci GAME(0) ar fi 0, dar GAME(1) detecteaza eroarea locala!
        self.assertAlmostEqual(compute_game(pred, gt, grid_level=0), 0.0, places=5)
        self.assertAlmostEqual(compute_game(pred, gt, grid_level=1), 6.0, places=5)

    def test_game_validation_errors(self) -> None:
        with self.assertRaises(ValueError):
            compute_game(np.ones(10), np.ones(10), grid_level=0)

        with self.assertRaises(ValueError):
            compute_game(np.ones((10, 10)), np.ones((8, 8)), grid_level=0)

        with self.assertRaises(ValueError):
            compute_game(np.ones((10, 10)), np.ones((10, 10)), grid_level=-1)


if __name__ == "__main__":
    unittest.main()
