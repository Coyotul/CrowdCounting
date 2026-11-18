"""Teste unitare pentru functiile ajutatoare de procesare vizuala (UI Helpers)."""

from __future__ import annotations

import unittest

import numpy as np

from src.ui.helpers import (
    compute_density_quadrants,
    draw_crowd_points,
    make_blended_overlay,
    make_colored_heatmap,
)


class TestUIHelpers(unittest.TestCase):
    def test_make_colored_heatmap(self) -> None:
        density = np.zeros((20, 30), dtype=np.float32)
        density[5:15, 10:20] = 0.5

        heatmap, norm = make_colored_heatmap(density, colormap_name="jet")
        self.assertEqual(heatmap.shape, (20, 30, 3))
        self.assertEqual(heatmap.dtype, np.uint8)
        self.assertEqual(norm.shape, (20, 30))
        self.assertGreaterEqual(float(norm.min()), 0.0)
        self.assertLessEqual(float(norm.max()), 1.0)

    def test_make_blended_overlay(self) -> None:
        img = np.ones((20, 30, 3), dtype=np.uint8) * 128
        heatmap = np.ones((20, 30, 3), dtype=np.uint8) * 255
        norm = np.ones((20, 30), dtype=np.float32) * 0.5

        overlay = make_blended_overlay(img, heatmap, norm, alpha=0.5)
        self.assertEqual(overlay.shape, (20, 30, 3))
        self.assertEqual(overlay.dtype, np.uint8)

    def test_draw_crowd_points(self) -> None:
        img = np.zeros((50, 50, 3), dtype=np.uint8)
        points = np.array([[10.0, 10.0], [25.0, 25.0]], dtype=np.float32)

        out = draw_crowd_points(img, points, radius=2, color=(0, 255, 0))
        self.assertEqual(out.shape, (50, 50, 3))
        # Verifica daca pixelul punctului a fost colorat cu verde
        self.assertEqual(tuple(out[10, 10]), (0, 255, 0))

        # Cazul fara puncte
        out_empty = draw_crowd_points(img, np.empty((0, 2)))
        self.assertEqual(out_empty.shape, (50, 50, 3))

    def test_compute_density_quadrants(self) -> None:
        # Matrice 4x4: punem cate 10 in fiecare cadran de 2x2 -> total = 40
        density = np.zeros((4, 4), dtype=np.float32)
        density[0:2, 0:2] = 2.5   # top-left suma = 10
        density[0:2, 2:4] = 2.5   # top-right suma = 10
        density[2:4, 0:2] = 2.5   # bottom-left suma = 10
        density[2:4, 2:4] = 2.5   # bottom-right suma = 10

        quads = compute_density_quadrants(density)
        self.assertAlmostEqual(quads["top_left"], 10.0, places=4)
        self.assertAlmostEqual(quads["top_right"], 10.0, places=4)
        self.assertAlmostEqual(quads["bottom_left"], 10.0, places=4)
        self.assertAlmostEqual(quads["bottom_right"], 10.0, places=4)
        self.assertAlmostEqual(quads["pct_top_left"], 25.0, places=2)
        self.assertAlmostEqual(quads["pct_top_right"], 25.0, places=2)


if __name__ == "__main__":
    unittest.main()
