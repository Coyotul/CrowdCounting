"""Calculul metricilor standard de performanta pentru Crowd Counting: MAE, MSE, RMSE, NAE, GAME."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np


def _validate_1d_inputs(
    predictions: Sequence[float] | np.ndarray,
    ground_truths: Sequence[float] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Valideaza vectorii 1D de predictii si ground truth."""
    preds = np.asarray(predictions, dtype=np.float64).ravel()
    gts = np.asarray(ground_truths, dtype=np.float64).ravel()

    if preds.size == 0 or gts.size == 0:
        raise ValueError("Vectorii de intrare nu pot fi vizi.")
    if preds.size != gts.size:
        raise ValueError(
            f"Dimensiunile nu coincid: predictions are {preds.size} elemente, "
            f"iar ground_truths are {gts.size} elemente."
        )
    return preds, gts


def compute_mae(
    predictions: Sequence[float] | np.ndarray,
    ground_truths: Sequence[float] | np.ndarray,
) -> float:

    preds, gts = _validate_1d_inputs(predictions, ground_truths)
    return float(np.mean(np.abs(preds - gts)))


def compute_mse(
    predictions: Sequence[float] | np.ndarray,
    ground_truths: Sequence[float] | np.ndarray,
) -> float:

    preds, gts = _validate_1d_inputs(predictions, ground_truths)
    return float(np.mean((preds - gts) ** 2))


def compute_rmse(
    predictions: Sequence[float] | np.ndarray,
    ground_truths: Sequence[float] | np.ndarray,
) -> float:

    return float(np.sqrt(compute_mse(predictions, ground_truths)))


def compute_nae(
    predictions: Sequence[float] | np.ndarray,
    ground_truths: Sequence[float] | np.ndarray,
) -> float:

    preds, gts = _validate_1d_inputs(predictions, ground_truths)
    gt_sum = float(np.sum(gts))
    err_sum = float(np.sum(np.abs(preds - gts)))

    if gt_sum == 0.0:
        return 0.0 if err_sum == 0.0 else float("inf")
    return float(err_sum / gt_sum)


def compute_game(
    pred_density: np.ndarray,
    gt_density: np.ndarray,
    grid_level: int = 0,
) -> float:

    if grid_level < 0:
        raise ValueError(f"grid_level trebuie sa fie >= 0, primit: {grid_level}")

    pred = np.asarray(pred_density, dtype=np.float64)
    gt = np.asarray(gt_density, dtype=np.float64)

    if pred.ndim != 2 or gt.ndim != 2:
        raise ValueError(
            f"Hartile de densitate trebuie sa fie 2D (H, W). Formate primite: "
            f"pred={pred.shape}, gt={gt.shape}"
        )
    if pred.shape != gt.shape:
        raise ValueError(
            f"Formatele hartilor trebuie sa coincida. Formate primite: "
            f"pred={pred.shape}, gt={gt.shape}"
        )

    h, w = pred.shape

    if grid_level == 0:
        return float(abs(pred.sum() - gt.sum()))

    n_blocks = 2**grid_level
    row_splits = np.array_split(np.arange(h), n_blocks)
    col_splits = np.array_split(np.arange(w), n_blocks)

    total_error = 0.0
    for r_idx in row_splits:
        if len(r_idx) == 0:
            continue
        r_start, r_end = r_idx[0], r_idx[-1] + 1
        for c_idx in col_splits:
            if len(c_idx) == 0:
                continue
            c_start, c_end = c_idx[0], c_idx[-1] + 1

            cell_pred_sum = pred[r_start:r_end, c_start:c_end].sum()
            cell_gt_sum = gt[r_start:r_end, c_start:c_end].sum()
            total_error += abs(cell_pred_sum - cell_gt_sum)

    return float(total_error)


def evaluate_predictions(
    predictions: Sequence[float] | np.ndarray,
    ground_truths: Sequence[float] | np.ndarray,
) -> dict[str, Any]:

    preds, gts = _validate_1d_inputs(predictions, ground_truths)
    mae = compute_mae(preds, gts)
    mse = compute_mse(preds, gts)
    rmse = compute_rmse(preds, gts)
    nae = compute_nae(preds, gts)

    return {
        "num_samples": int(preds.size),
        "mae": round(mae, 4),
        "mse": round(mse, 4),
        "rmse": round(rmse, 4),
        "nae": round(nae, 4),
    }
