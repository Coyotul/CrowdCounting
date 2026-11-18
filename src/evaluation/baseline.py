"""Calculul predictorilor naivi de referinta (Baseline) pe seturile de testare ShanghaiTech."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .metrics import evaluate_predictions


def run_baseline_on_part(
    manifest_dir: Path,
    part: str,
) -> dict[str, Any]:
    """
    Calculeaza metricile de baseline pentru o parte a dataset-ului (part_A sau part_B).

    Strategii de baseline evaluate pe setul de test:
    1. Mean Predictor: prezice constant valoarea medie a numarului de persoane din setul de antrenare.
    2. Median Predictor: prezice constant mediana din setul de antrenare.
    3. Zero Predictor: prezice constant 0 (util pentru calibrarea erorii maxime / total masa).
    """
    train_csv = manifest_dir / f"{part}_train.csv"
    test_csv = manifest_dir / f"{part}_test.csv"

    if not train_csv.exists() or not test_csv.exists():
        raise FileNotFoundError(
            f"Fisierele de manifest nu au fost gasite in {manifest_dir}. "
            f"Asigurati-va ca {train_csv.name} si {test_csv.name} exista."
        )

    df_train = pd.read_csv(train_csv)
    df_test = pd.read_csv(test_csv)

    y_train = df_train["clean_count"].to_numpy(dtype=np.float64)
    y_test = df_test["clean_count"].to_numpy(dtype=np.float64)

    mean_train = float(np.mean(y_train))
    median_train = float(np.median(y_train))

    preds_mean = np.full_like(y_test, fill_value=mean_train)
    preds_median = np.full_like(y_test, fill_value=median_train)
    preds_zero = np.zeros_like(y_test)

    eval_mean = evaluate_predictions(preds_mean, y_test)
    eval_median = evaluate_predictions(preds_median, y_test)
    eval_zero = evaluate_predictions(preds_zero, y_test)

    return {
        "part": part,
        "train_samples": len(df_train),
        "test_samples": len(df_test),
        "train_stats": {
            "mean": round(mean_train, 2),
            "median": round(median_train, 2),
            "min": round(float(np.min(y_train)), 2),
            "max": round(float(np.max(y_train)), 2),
            "std": round(float(np.std(y_train)), 2),
        },
        "test_stats": {
            "mean": round(float(np.mean(y_test)), 2),
            "median": round(float(np.median(y_test)), 2),
            "min": round(float(np.min(y_test)), 2),
            "max": round(float(np.max(y_test)), 2),
            "std": round(float(np.std(y_test)), 2),
        },
        "baselines": {
            "mean_predictor": {
                "predicted_constant": round(mean_train, 2),
                **eval_mean,
            },
            "median_predictor": {
                "predicted_constant": round(median_train, 2),
                **eval_median,
            },
            "zero_predictor": {
                "predicted_constant": 0.0,
                **eval_zero,
            },
        },
    }


def print_benchmark_table(results: dict[str, Any]) -> None:
    """Afiseaza tabelul cu rezultatele de baseline."""
    print("\nRezultate baseline pe test set:")
    print("-" * 75)

    for part_name, res in results.items():
        print(f"\n{part_name.upper()} ({res['test_samples']} imagini test):")
        print(f"  Train: medie = {res['train_stats']['mean']:.1f}, mediana = {res['train_stats']['median']:.1f}")
        print(f"  Test:  medie = {res['test_stats']['mean']:.1f}, mediana = {res['test_stats']['median']:.1f}")
        print("-" * 75)
        print(f"  {'Metoda':<20} | {'Valoare':<10} | {'MAE':<10} | {'RMSE':<10} | {'NAE':<10}")
        print("-" * 75)
        for strat, data in res["baselines"].items():
            print(
                f"  {strat:<20} | {data['predicted_constant']:<10.1f} | "
                f"{data['mae']:<10.2f} | {data['rmse']:<10.2f} | {data['nae']:<10.4f}"
            )
        print("-" * 75)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calculeaza scorurile de baseline pe setul de test ShanghaiTech."
    )
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        default=Path("data/processed/manifests"),
        help="Directorul unde se afla fisierele de manifest CSV.",
    )
    parser.add_argument(
        "--parts",
        nargs="+",
        default=["part_A", "part_B"],
        choices=["part_A", "part_B"],
        help="Partile de evaluat (implicit: part_A part_B).",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("data/processed/manifests/baseline_benchmark.json"),
        help="Calea unde se salveaza rezultatele in format JSON.",
    )
    args = parser.parse_args()

    results: dict[str, Any] = {}
    for part in args.parts:
        results[part] = run_baseline_on_part(args.manifest_dir, part)

    print_benchmark_table(results)

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"[OK] Rezultate salvate in: {args.output_json}")


if __name__ == "__main__":
    main()
