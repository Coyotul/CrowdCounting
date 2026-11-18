"""Pipeline si utilitar CLI pentru validarea integritatii datelor procesate (HDF5, imagini, split-uri)."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm


@dataclass
class SampleValidationResult:
    """Rezultatul validarii individuale a unui esantion din dataset."""

    sample_id: str
    part: str
    split: str
    is_valid: bool
    image_exists: bool
    h5_exists: bool
    shape_matches: bool
    expected_shape: tuple[int, int]  # (height, width)
    actual_shape: tuple[int, int]
    expected_count: float
    actual_density_sum: float
    abs_count_error: float
    mass_conserved: bool
    no_negative_values: bool
    is_finite: bool
    points_in_bounds: bool
    errors: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_density_file(
    h5_path: Path,
    expected_shape: tuple[int, int],
    expected_count: float,
    tol: float = 0.05,
) -> tuple[bool, list[str], dict[str, Any]]:
    """
    Valideaza un fisier HDF5 (.h5) de densitate:
    - Existenta cheii 'density'
    - Corespondenta dimensionala (height, width)
    - Finitudinea valorilor (fara NaN / Inf)
    - Non-negativitatea valorilor (densitate >= 0)
    - Conservarea masei (|sum(D) - expected_count| <= tol)
    - Validitatea coordonatelor din cheia 'points' (daca este prezenta)
    """
    errors: list[str] = []
    stats: dict[str, Any] = {
        "actual_shape": (0, 0),
        "actual_density_sum": 0.0,
        "abs_count_error": 0.0,
        "mass_conserved": False,
        "no_negative_values": False,
        "is_finite": False,
        "points_in_bounds": True,
    }

    if not h5_path.exists():
        errors.append(f"Fisierul HDF5 nu exista: {h5_path}")
        return False, errors, stats

    try:
        with h5py.File(h5_path, "r") as hf:
            if "density" not in hf:
                errors.append(f"Lipseste dataset-ul 'density' in {h5_path.name}")
                return False, errors, stats

            dmap = np.asarray(hf["density"], dtype=np.float32)
            actual_shape = (int(dmap.shape[0]), int(dmap.shape[1]))
            stats["actual_shape"] = actual_shape

            # Verificare shape
            if actual_shape != expected_shape:
                errors.append(
                    f"Dimensiune gresita a hartii de densitate: asteptat {expected_shape}, "
                    f"gasit {actual_shape}"
                )

            # Verificare finite
            stats["is_finite"] = bool(np.isfinite(dmap).all())
            if not stats["is_finite"]:
                errors.append("Harta contine valori NaN sau Inf")

            # Verificare non-negativ
            stats["no_negative_values"] = bool((dmap >= 0.0).all())
            if not stats["no_negative_values"]:
                min_val = float(dmap.min())
                errors.append(f"Harta contine valori negative (min: {min_val})")

            # Verificare conservare masa
            density_sum = float(dmap.sum())
            stats["actual_density_sum"] = density_sum
            err = abs(density_sum - expected_count)
            stats["abs_count_error"] = err
            stats["mass_conserved"] = err <= tol
            if not stats["mass_conserved"]:
                errors.append(
                    f"Eroare de conservare a masei: suma={density_sum:.4f}, "
                    f"asteptat={expected_count:.4f}, diff={err:.4f} > tol={tol}"
                )

            # Verificare puncte daca exista
            if "points" in hf:
                pts = np.asarray(hf["points"], dtype=np.float32)
                if pts.ndim == 2 and pts.size > 0:
                    h, w = expected_shape
                    in_bounds = (
                        (pts[:, 0] >= 0.0)
                        & (pts[:, 0] <= w - 1)
                        & (pts[:, 1] >= 0.0)
                        & (pts[:, 1] <= h - 1)
                    ).all()
                    stats["points_in_bounds"] = bool(in_bounds)
                    if not stats["points_in_bounds"]:
                        errors.append("Exista coordonate de puncte in afara cadrului imaginii")
    except Exception as exc:
        errors.append(f"Exceptie la citirea HDF5: {exc}")
        return False, errors, stats

    is_valid = len(errors) == 0
    return is_valid, errors, stats


def validate_split_integrity(
    manifest_dir: Path,
    part: str,
) -> tuple[bool, list[str]]:
    """
    Verifica disjunctia si completitudinea split-urilor Train, Val si Test:
    - Train ∩ Val = ∅
    - Train ∩ Test = ∅
    - Val ∩ Test = ∅
    - (Train ∪ Val ∪ Test) == Total Manifest
    """
    errors: list[str] = []
    main_manifest = manifest_dir / f"{part}_manifest.csv"
    train_csv = manifest_dir / f"{part}_train.csv"
    val_csv = manifest_dir / f"{part}_val.csv"
    test_csv = manifest_dir / f"{part}_test.csv"

    for p in (main_manifest, train_csv, val_csv, test_csv):
        if not p.exists():
            errors.append(f"Fisierul de manifest nu exista: {p.name}")
            return False, errors

    df_main = pd.read_csv(main_manifest)
    df_train = pd.read_csv(train_csv)
    df_val = pd.read_csv(val_csv)
    df_test = pd.read_csv(test_csv)

    set_main = set(df_main["sample_id"])
    set_train = set(df_train["sample_id"])
    set_val = set(df_val["sample_id"])
    set_test = set(df_test["sample_id"])

    # Disjunctie
    if not set_train.isdisjoint(set_val):
        overlap = set_train.intersection(set_val)
        errors.append(f"Suprapunere intre Train si Val: {len(overlap)} elemente (ex: {list(overlap)[:3]})")
    if not set_train.isdisjoint(set_test):
        overlap = set_train.intersection(set_test)
        errors.append(f"Suprapunere intre Train si Test: {len(overlap)} elemente (ex: {list(overlap)[:3]})")
    if not set_val.isdisjoint(set_test):
        overlap = set_val.intersection(set_test)
        errors.append(f"Suprapunere intre Val si Test: {len(overlap)} elemente (ex: {list(overlap)[:3]})")

    # Reuniune
    union_splits = set_train.union(set_val).union(set_test)
    if union_splits != set_main:
        missing = set_main - union_splits
        extra = union_splits - set_main
        if missing:
            errors.append(f"{len(missing)} esantioane lipsesc din split-uri fata de manifestul principal")
        if extra:
            errors.append(f"{len(extra)} esantioane din split-uri nu sunt in manifestul principal")

    return len(errors) == 0, errors


def validate_part_dataset(
    manifest_dir: Path,
    part: str,
    project_root: Path = Path("."),
    tol: float = 0.05,
    show_progress: bool = True,
) -> tuple[dict[str, Any], list[SampleValidationResult]]:
    """
    Valideaza toate esantioanele dintr-o parte (part_A sau part_B).
    """
    manifest_path = manifest_dir / f"{part}_manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifestul nu a fost gasit: {manifest_path}")

    split_ok, split_errors = validate_split_integrity(manifest_dir, part)

    df = pd.read_csv(manifest_path)
    results: list[SampleValidationResult] = []

    iterator = df.itertuples(index=False)
    if show_progress:
        iterator = tqdm(iterator, total=len(df), desc=f"Validare {part}", unit="img")

    for row in iterator:
        sample_id = str(row.sample_id)
        assigned_split = str(row.assigned_split)
        expected_shape = (int(row.height), int(row.width))
        expected_count = float(row.clean_count)

        img_path = project_root / str(row.clean_image_path)
        if not img_path.exists():
            img_path = project_root / str(row.image_path)

        image_exists = img_path.exists()
        h5_path = project_root / str(row.density_path)
        h5_exists = h5_path.exists()

        sample_errors: list[str] = []
        if not image_exists:
            sample_errors.append(f"Imaginea nu a fost gasita: {img_path}")
        if not h5_exists:
            sample_errors.append(f"Fisierul H5 nu a fost gasit: {h5_path}")

        if h5_exists:
            h5_ok, h5_errs, stats = validate_density_file(
                h5_path=h5_path,
                expected_shape=expected_shape,
                expected_count=expected_count,
                tol=tol,
            )
            sample_errors.extend(h5_errs)
        else:
            stats = {
                "actual_shape": (0, 0),
                "actual_density_sum": 0.0,
                "abs_count_error": expected_count,
                "mass_conserved": False,
                "no_negative_values": False,
                "is_finite": False,
                "points_in_bounds": False,
            }

        is_valid = len(sample_errors) == 0
        res = SampleValidationResult(
            sample_id=sample_id,
            part=part,
            split=assigned_split,
            is_valid=is_valid,
            image_exists=image_exists,
            h5_exists=h5_exists,
            shape_matches=stats["actual_shape"] == expected_shape,
            expected_shape=expected_shape,
            actual_shape=stats["actual_shape"],
            expected_count=expected_count,
            actual_density_sum=stats["actual_density_sum"],
            abs_count_error=stats["abs_count_error"],
            mass_conserved=stats["mass_conserved"],
            no_negative_values=stats["no_negative_values"],
            is_finite=stats["is_finite"],
            points_in_bounds=stats["points_in_bounds"],
            errors=sample_errors,
        )
        results.append(res)

    total_samples = len(results)
    valid_samples = sum(1 for r in results if r.is_valid)
    max_count_err = max(r.abs_count_error for r in results) if results else 0.0
    mean_count_err = (sum(r.abs_count_error for r in results) / total_samples) if results else 0.0

    summary = {
        "part": part,
        "total_samples": total_samples,
        "valid_samples": valid_samples,
        "invalid_samples": total_samples - valid_samples,
        "all_images_exist": all(r.image_exists for r in results),
        "all_h5_exist": all(r.h5_exists for r in results),
        "all_shapes_matched": all(r.shape_matches for r in results),
        "all_mass_conserved": all(r.mass_conserved for r in results),
        "all_non_negative": all(r.no_negative_values for r in results),
        "all_finite": all(r.is_finite for r in results),
        "all_points_in_bounds": all(r.points_in_bounds for r in results),
        "split_integrity_ok": split_ok,
        "split_errors": split_errors,
        "max_abs_count_error": round(float(max_count_err), 6),
        "mean_abs_count_error": round(float(mean_count_err), 8),
    }

    return summary, results


def run_full_validation(
    manifest_dir: Path,
    project_root: Path = Path("."),
    parts: list[str] | None = None,
    output_json: Path | None = None,
    tol: float = 0.05,
) -> dict[str, Any]:
    """Valideaza toate partile configurate si afiseaza sumarul de integritate."""
    parts = parts or ["part_A", "part_B"]
    full_summary: dict[str, Any] = {"parts": {}}

    print("\nValidare integritate date procesate:")
    print("-" * 60)

    all_passed = True
    for part in parts:
        summary, _ = validate_part_dataset(
            manifest_dir=manifest_dir,
            part=part,
            project_root=project_root,
            tol=tol,
            show_progress=True,
        )
        full_summary["parts"][part] = summary

        status_str = "OK" if summary["invalid_samples"] == 0 and summary["split_integrity_ok"] else "FAIL"
        if summary["invalid_samples"] > 0 or not summary["split_integrity_ok"]:
            all_passed = False

        print(f"\n{part}: {status_str}")
        print(f"  Esantioane:        {summary['valid_samples']} / {summary['total_samples']}")
        print(f"  Imagini prezente:  {'DA' if summary['all_images_exist'] else 'NU'}")
        print(f"  Fisiere H5:        {'DA' if summary['all_h5_exist'] else 'NU'}")
        print(f"  Dimensiuni corecte:{'DA' if summary['all_shapes_matched'] else 'NU'}")
        print(f"  Conservare masa:   {'DA' if summary['all_mass_conserved'] else 'NU'} (eroare max: {summary['max_abs_count_error']:.6f})")
        print(f"  Valori non-negative / finite: {'DA' if summary['all_non_negative'] and summary['all_finite'] else 'NU'}")
        print(f"  Split-uri disjuncte:          {'DA' if summary['split_integrity_ok'] else 'NU'}")

    full_summary["overall_status"] = "PASSED" if all_passed else "FAILED"
    print("\n" + "-" * 60)
    print(f"Status: {'Datele sunt valide si gata de antrenare.' if all_passed else 'Au fost gasite erori in date.'}")
    print("=" * 80 + "\n")

    if output_json:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(full_summary, f, indent=2)
        print(f"[OK] Raport complet salvat in: {output_json}")

    return full_summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Valideaza integritatea fisierelor HDF5 si a split-urilor procesate."
    )
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        default=Path("data/processed/manifests"),
        help="Directorul cu fisierele CSV de manifest.",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path("."),
        help="Directorul radacina al proiectului.",
    )
    parser.add_argument(
        "--parts",
        nargs="+",
        default=["part_A", "part_B"],
        choices=["part_A", "part_B"],
        help="Partile de validat (implicit: part_A part_B).",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("data/processed/manifests/validation_report.json"),
        help="Calea fisierului JSON de output pentru raport.",
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=0.05,
        help="Toleranta maxima acceptata pentru conservarea masei (implicit: 0.05).",
    )
    args = parser.parse_args()

    run_full_validation(
        manifest_dir=args.manifest_dir,
        project_root=args.project_root,
        parts=args.parts,
        output_json=args.output_json,
        tol=args.tol,
    )


if __name__ == "__main__":
    main()
