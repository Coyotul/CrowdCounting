"""Inspector simplu pentru vizualizarea hartilor de densitate (.h5).

Se poate rula direct din butonul Run din PyCharm sau cu:
    python -m streamlit run src/data/inspect_density_app.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import streamlit as st
from streamlit import runtime as st_runtime

if __name__ == "__main__" and not st_runtime.exists():
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(Path(__file__).resolve()), *sys.argv[1:]],
        check=False,
    )
    raise SystemExit(0)

import h5py
import matplotlib
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@st.cache_data(show_spinner=False)
def load_manifest(part: str) -> pd.DataFrame:
    path = PROJECT_ROOT / "data" / "processed" / "manifests" / f"{part}_manifest.csv"
    df = pd.read_csv(path)
    return df[df["is_valid"]].reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_sample(img_rel: str, h5_rel: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    img = np.asarray(Image.open(PROJECT_ROOT / img_rel).convert("RGB"), dtype=np.uint8)
    with h5py.File(PROJECT_ROOT / h5_rel, "r") as hf:
        density = np.asarray(hf["density"], dtype=np.float32)
        points = np.asarray(hf["points"], dtype=np.float32) if "points" in hf else np.empty((0, 2), dtype=np.float32)
    return img, density, points


def draw_points(img: np.ndarray, points: np.ndarray, radius: int = 2) -> np.ndarray:
    pil_img = Image.fromarray(img.copy())
    draw = ImageDraw.Draw(pil_img)
    for x, y in points:
        ix, iy = int(round(float(x))), int(round(float(y)))
        draw.ellipse((ix - radius, iy - radius, ix + radius, iy + radius), fill=(0, 255, 0))
    return np.asarray(pil_img, dtype=np.uint8)


def make_heatmap(density: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    d = np.maximum(density, 0.0)
    pos = d[d > 0]
    vmax = float(np.percentile(pos, 99.5)) if pos.size > 0 else 1.0
    norm = np.power(np.clip(d / max(vmax, 1e-12), 0.0, 1.0), 0.5)
    rgb = (matplotlib.colormaps["jet"](norm)[..., :3] * 255.0).astype(np.uint8)
    return rgb, norm


def make_overlay(img: np.ndarray, heatmap: np.ndarray, norm: np.ndarray, alpha: float) -> np.ndarray:
    w = np.clip(norm * 1.4, 0.0, 1.0)[..., np.newaxis] * alpha
    out = img.astype(np.float32) * (1.0 - w) + heatmap.astype(np.float32) * w
    return np.clip(out, 0, 255).astype(np.uint8)


def main() -> None:
    st.set_page_config(page_title="Density Map Inspector", layout="wide")
    st.title("Density Map Inspector")

    with st.sidebar:
        part = st.radio("Subset", ["part_A", "part_B"], horizontal=True)
        split = st.selectbox("Split", ["all", "train", "val", "test"])

        df = load_manifest(part)
        if split != "all":
            df = df[df["assigned_split"] == split].reset_index(drop=True)

        if "idx" not in st.session_state:
            st.session_state.idx = 0
        st.session_state.idx = min(st.session_state.idx, len(df) - 1)

        c_prev, c_next = st.columns(2)
        if c_prev.button("Prev", width="stretch"):
            st.session_state.idx = (st.session_state.idx - 1) % len(df)
        if c_next.button("Next", width="stretch"):
            st.session_state.idx = (st.session_state.idx + 1) % len(df)

        st.session_state.idx = st.selectbox(
            f"Image ({len(df)})",
            options=range(len(df)),
            index=st.session_state.idx,
            format_func=lambda i: f"{Path(df.iloc[i]['image_path']).name} ({int(df.iloc[i]['clean_count'])})",
        )

        alpha = st.slider("Overlay opacity", 0.0, 1.0, 0.65, 0.05)
        show_pts = st.checkbox("Show GT points", value=True)

    rec = df.iloc[st.session_state.idx]
    img, density, points = load_sample(rec["clean_image_path"], rec["density_path"])

    heatmap, norm = make_heatmap(density)
    overlay = make_overlay(img, heatmap, norm, alpha)
    img_view = draw_points(img, points) if show_pts else img

    st.caption(
        f"{Path(rec['image_path']).name}  |  "
        f"Split: {rec['assigned_split']}  |  "
        f"GT: {int(rec['clean_count'])}  |  "
        f"H5 sum: {float(density.sum()):.2f}  |  "
        f"{int(rec['width'])}x{int(rec['height'])}"
    )

    col1, col2, col3 = st.columns(3)
    col1.image(img_view, caption="Original", width="stretch")
    col2.image(heatmap, caption="Density map (.h5)", width="stretch")
    col3.image(overlay, caption="Overlay", width="stretch")


if __name__ == "__main__":
    main()
