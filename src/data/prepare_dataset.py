"""Pipeline de curatare a datelor si generare a hartilor de densitate (M1).

Exemplu de rulare:
    python -m src.data.prepare_dataset --config configs/dataset_config.yaml
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from .cleaning import SampleAudit, audit_and_clean_sample, create_stratified_train_val_split
from .density_map import generate_density_map
from .io_utils import discover_shanghaitech_pairs, load_config, save_density_map_h5


def _to_rel_posix(path: Path, project_root: Path) -> str:
    """Converteste o cale in cale relativa cu slash-uri POSIX (portabila Windows/Linux/Colab)."""
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def process_part(
    part: str,
    part_cfg: dict[str, Any],
    cfg: dict[str, Any],
    project_root: Path,
    limit: int | None = None,
) -> tuple[list[SampleAudit], dict[str, Any]]:
    """Ruleaza intregul pipeline de curatare si generare GT pentru un subset (part_A / part_B)."""
    raw_root = project_root / cfg["paths"]["raw_root"]
    processed_root = project_root / cfg["paths"]["processed_root"]
    clean_cfg = cfg["cleaning"]
    storage_cfg = cfg["storage"]
    split_cfg = cfg["splits"]

    audits: list[SampleAudit] = []
    orphan_summary: dict[str, dict[str, int]] = {}
    method = part_cfg["density_method"]

    for orig_split in ("train_data", "test_data"):
        pairs, orphan_imgs, orphan_mats = discover_shanghaitech_pairs(raw_root, part, orig_split)
        orphan_summary[orig_split] = {
            "orphan_images": len(orphan_imgs),
            "orphan_mats": len(orphan_mats),
        }
        if limit is not None:
            pairs = pairs[:limit]

        density_dir = processed_root / part / orig_split / "density_maps"
        rgb_dir = processed_root / part / orig_split / "images_rgb"
        density_dir.mkdir(parents=True, exist_ok=True)

        pbar = tqdm(pairs, desc=f"[{part} | {orig_split}] ({method})", unit="img")
        for img_path, mat_path in pbar:
            img, clean_pts, audit = audit_and_clean_sample(
                image_path=img_path,
                mat_path=mat_path,
                part=part,
                original_split=orig_split,
                convert_grayscale_to_rgb=bool(clean_cfg["convert_grayscale_to_rgb"]),
                clip_oob=bool(clean_cfg["clip_out_of_bounds_points"]),
                drop_outside_margin_px=float(clean_cfg["drop_outside_margin_px"]),
                deduplicate_radius_px=float(clean_cfg["deduplicate_radius_px"]),
            )
            audit.image_path = _to_rel_posix(img_path, project_root)
            audit.mat_path = _to_rel_posix(mat_path, project_root)

            if not audit.is_valid or img is None:
                audits.append(audit)
                continue

            # Daca imaginea a fost convertita din Grayscale/RGBA in RGB, o salvam curata
            if audit.converted_to_rgb:
                rgb_dir.mkdir(parents=True, exist_ok=True)
                clean_img_out = rgb_dir / img_path.name
                img.save(clean_img_out, quality=95)
                audit.clean_image_path = _to_rel_posix(clean_img_out, project_root)
            else:
                audit.clean_image_path = audit.image_path

            # Generam harta de densitate Ground-Truth (H, W)
            density = generate_density_map(
                image_shape=(audit.height, audit.width),
                points=clean_pts,
                method=method,
                **{k: v for k, v in part_cfg.items() if k not in ("description", "density_method")},
            )

            density_sum = float(density.sum())
            abs_err = abs(density_sum - float(audit.clean_count))
            audit.density_sum = round(density_sum, 6)
            audit.abs_count_error = round(abs_err, 6)

            h5_path = density_dir / f"{img_path.stem}.h5"
            save_density_map_h5(
                output_path=h5_path,
                density_map=density,
                points=clean_pts,
                metadata={
                    "sample_id": audit.sample_id,
                    "part": part,
                    "original_split": orig_split,
                    "method": method,
                    "clean_count": audit.clean_count,
                    "density_sum": density_sum,
                    "width": audit.width,
                    "height": audit.height,
                },
                compression=storage_cfg.get("h5_compression", "gzip"),
                compression_opts=storage_cfg.get("h5_compression_opts", 4),
            )
            audit.density_path = _to_rel_posix(h5_path, project_root)
            audits.append(audit)

    # Cream split stratificat train / val din train_data (test_data ramane intact)
    create_stratified_train_val_split(
        audits,
        val_ratio=float(split_cfg["val_ratio"]),
        n_bins=int(split_cfg["stratify_bins"]),
        seed=int(split_cfg["seed"]),
    )

    # Salvam manifesturile CSV
    manifests_dir = processed_root / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame([a.to_dict() for a in audits])
    df.to_csv(manifests_dir / f"{part}_manifest.csv", index=False)

    valid_df = df[df["is_valid"]].copy()
    for split_name in ("train", "val", "test"):
        sub = valid_df[valid_df["assigned_split"] == split_name]
        sub.to_csv(manifests_dir / f"{part}_{split_name}.csv", index=False)

    # Construim sumarul statistic
    summary = {
        "part": part,
        "description": part_cfg.get("description", ""),
        "density_method": method,
        "total_pairs": int(len(df)),
        "valid_samples": int(len(valid_df)),
        "invalid_samples": int((~df["is_valid"]).sum()),
        "orphans": orphan_summary,
        "splits": {
            s: {
                "count": int((valid_df["assigned_split"] == s).sum()),
                "mean_people": round(float(valid_df.loc[valid_df["assigned_split"] == s, "clean_count"].mean()), 2)
                if (valid_df["assigned_split"] == s).any()
                else 0.0,
                "median_people": round(float(valid_df.loc[valid_df["assigned_split"] == s, "clean_count"].median()), 2)
                if (valid_df["assigned_split"] == s).any()
                else 0.0,
                "min_people": int(valid_df.loc[valid_df["assigned_split"] == s, "clean_count"].min())
                if (valid_df["assigned_split"] == s).any()
                else 0,
                "max_people": int(valid_df.loc[valid_df["assigned_split"] == s, "clean_count"].max())
                if (valid_df["assigned_split"] == s).any()
                else 0,
            }
            for s in ("train", "val", "test")
        },
        "cleaning_totals": {
            "converted_to_rgb": int(valid_df["converted_to_rgb"].sum()),
            "count_mismatches": int(valid_df["count_mismatch"].sum()),
            "nan_or_inf_dropped": int(valid_df["nan_or_inf_dropped"].sum()),
            "far_oob_dropped": int(valid_df["far_oob_dropped"].sum()),
            "border_oob_clipped": int(valid_df["border_oob_clipped"].sum()),
            "duplicates_dropped": int(valid_df["duplicates_dropped"].sum()),
        },
        "crowd_stats": {
            "total_people_annotations": int(valid_df["clean_count"].sum()),
            "min_people": int(valid_df["clean_count"].min()),
            "mean_people": round(float(valid_df["clean_count"].mean()), 2),
            "median_people": round(float(valid_df["clean_count"].median()), 2),
            "std_people": round(float(valid_df["clean_count"].std()), 2),
            "max_people": int(valid_df["clean_count"].max()),
        },
        "resolution_stats": {
            "unique_resolutions": int(valid_df[["width", "height"]].drop_duplicates().shape[0]),
            "min_width": int(valid_df["width"].min()),
            "max_width": int(valid_df["width"].max()),
            "min_height": int(valid_df["height"].min()),
            "max_height": int(valid_df["height"].max()),
        },
        "density_verification": {
            "max_abs_count_error": float(valid_df["abs_count_error"].max()),
            "mean_abs_count_error": float(valid_df["abs_count_error"].mean()),
        },
    }
    return audits, summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Curatare date si generare harti de densitate ShanghaiTech (M1)."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/dataset_config.yaml",
        help="Calea catre fisierul de configuratie YAML.",
    )
    parser.add_argument(
        "--parts",
        nargs="+",
        default=None,
        help="Subseturile de procesat (ex: part_A part_B). Implicit: toate din config.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limiteaza numarul de imagini per split (util pentru teste rapide).",
    )
    args = parser.parse_args()

    project_root = Path.cwd()
    cfg = load_config(project_root / args.config)
    parts_cfg = cfg["dataset"]["parts"]
    selected_parts = args.parts if args.parts else list(parts_cfg.keys())

    t0 = time.perf_counter()
    all_summaries: dict[str, Any] = {}

    for part in selected_parts:
        if part not in parts_cfg:
            raise ValueError(f"Subset necunoscut: {part}. Disponibile: {list(parts_cfg.keys())}")
        _, summary = process_part(
            part=part,
            part_cfg=parts_cfg[part],
            cfg=cfg,
            project_root=project_root,
            limit=args.limit,
        )
        all_summaries[part] = summary

    elapsed = time.perf_counter() - t0
    manifests_dir = project_root / cfg["paths"]["processed_root"] / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)

    output_summary = {
        "elapsed_seconds": round(elapsed, 2),
        "parts": all_summaries,
    }
    summary_path = manifests_dir / "audit_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(output_summary, f, indent=2, ensure_ascii=False)

    print(f"\n[OK] Pipeline finalizat in {elapsed:.2f}s. Sumar salvat in: {summary_path}")
    for part, s in all_summaries.items():
        print(
            f"  - {part}: {s['valid_samples']}/{s['total_pairs']} valide | "
            f"train={s['splits']['train']['count']}, val={s['splits']['val']['count']}, test={s['splits']['test']['count']} | "
            f"RGB conv={s['cleaning_totals']['converted_to_rgb']}, "
            f"OOB clip={s['cleaning_totals']['border_oob_clipped']}, "
            f"far OOB drop={s['cleaning_totals']['far_oob_dropped']}, "
            f"dup drop={s['cleaning_totals']['duplicates_dropped']} | "
            f"max err={s['density_verification']['max_abs_count_error']:.6f}"
        )


if __name__ == "__main__":
    main()
