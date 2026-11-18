
from __future__ import annotations

import json
from pathlib import Path
import random
import subprocess
import sys

import streamlit as st
from streamlit import runtime as st_runtime

if __name__ == "__main__" and not st_runtime.exists():
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(Path(__file__).resolve()), *sys.argv[1:]],
        check=False,
    )
    raise SystemExit(0)

import h5py
import numpy as np
import pandas as pd
from PIL import Image

from src.ui.helpers import (
    compute_density_quadrants,
    draw_crowd_points,
    make_blended_overlay,
    make_colored_heatmap,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@st.cache_data(show_spinner=False)
def load_manifest(part: str) -> pd.DataFrame:
    path = PROJECT_ROOT / "data" / "processed" / "manifests" / f"{part}_manifest.csv"
    if not path.exists():
        st.error(f"Nu s-a gasit fisierul: {path}")
        return pd.DataFrame()
    df = pd.read_csv(path)
    return df[df["is_valid"]].reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_benchmark_data() -> dict:
    path = PROJECT_ROOT / "data" / "processed" / "manifests" / "baseline_benchmark.json"
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


@st.cache_data(show_spinner=False)
def load_sample(img_rel: str, h5_rel: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    img_path = PROJECT_ROOT / img_rel
    h5_path = PROJECT_ROOT / h5_rel

    img = np.asarray(Image.open(img_path).convert("RGB"), dtype=np.uint8)
    with h5py.File(h5_path, "r") as hf:
        density = np.asarray(hf["density"], dtype=np.float32)
        points = np.asarray(hf["points"], dtype=np.float32) if "points" in hf else np.empty((0, 2), dtype=np.float32)
    return img, density, points


def main() -> None:
    st.set_page_config(
        page_title="Inspectie Date Crowd Counting",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("Inspectie Date si Estimare Densitate")
    st.caption("Analiza Continutului Imaginilor | Constantin Florin-George & Jugaru Cosmin-Alexandru")

    tabs = st.tabs([
        "Inspectie Date",
        "Metrici si Baseline",
        "Testare Imagine",
    ])

    # -------------------------------------------------------------------------
    # TAB 1: INSPECTIE DATE SI HARTI
    # -------------------------------------------------------------------------
    with tabs[0]:
        with st.sidebar:
            st.subheader("Selectie Date")

            part = st.radio(
                "Subset",
                ["part_A", "part_B"],
                format_func=lambda x: "Part A (aglomerat, rezolutii diferite)" if x == "part_A" else "Part B (strada, rezolutie fixa)",
            )

            df = load_manifest(part)
            if df.empty:
                st.warning("Nu exista date preprocesate.")
                st.stop()

            split = st.selectbox("Split", ["all", "train", "val", "test"])
            if split != "all":
                df = df[df["assigned_split"] == split].reset_index(drop=True)

            if "sample_idx" not in st.session_state:
                st.session_state.sample_idx = 0

            c_prev, c_rand, c_next = st.columns(3)
            if c_prev.button("Inapoi", use_container_width=True):
                st.session_state.sample_idx = (st.session_state.sample_idx - 1) % len(df)
            if c_rand.button("Aleator", use_container_width=True):
                st.session_state.sample_idx = random.randint(0, len(df) - 1)
            if c_next.button("Inainte", use_container_width=True):
                st.session_state.sample_idx = (st.session_state.sample_idx + 1) % len(df)

            st.session_state.sample_idx = min(st.session_state.sample_idx, len(df) - 1)
            selected_idx = st.selectbox(
                f"Imagine ({len(df)} in lista)",
                options=range(len(df)),
                index=st.session_state.sample_idx,
                format_func=lambda i: f"{Path(df.iloc[i]['image_path']).name} ({int(df.iloc[i]['clean_count'])} pers)",
            )
            st.session_state.sample_idx = selected_idx

            st.markdown("---")
            st.subheader("Afisare")
            colormap_name = st.selectbox(
                "Colormap",
                ["jet", "viridis", "magma", "plasma"],
                index=0,
            )
            alpha = st.slider("Transparenta overlay", 0.0, 1.0, 0.65, 0.05)
            show_pts = st.checkbox("Afiseaza puncte GT", value=True)
            dot_radius = st.slider("Raza puncte (px)", 1, 4, 2) if show_pts else 2

        rec = df.iloc[st.session_state.sample_idx]
        img, density, points = load_sample(rec["clean_image_path"], rec["density_path"])

        heatmap, norm = make_colored_heatmap(density, colormap_name=colormap_name)
        overlay = make_blended_overlay(img, heatmap, norm, alpha=alpha)
        img_with_points = draw_crowd_points(img, points, radius=dot_radius) if show_pts else img

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Numar persoane GT", f"{int(rec['clean_count'])}")
        c2.metric("Integrala H5", f"{float(density.sum()):.2f}")
        diff = abs(float(density.sum()) - float(rec["clean_count"]))
        c3.metric("Diferenta masa", f"{diff:.5f}")
        c4.metric("Rezolutie", f"{int(rec['width'])}x{int(rec['height'])}")

        col1, col2, col3, col4 = st.columns(4)
        col1.image(img, caption="Imagine originala", use_container_width=True)
        col2.image(img_with_points, caption=f"Puncte GT ({len(points)})", use_container_width=True)
        col3.image(heatmap, caption=f"Harta densitate ({colormap_name})", use_container_width=True)
        col4.image(overlay, caption=f"Suprapunere (alpha={alpha:.2f})", use_container_width=True)

        with st.expander("Distributie pe cadrane (grila 2x2)"):
            quads = compute_density_quadrants(density)
            q1, q2, q3, q4 = st.columns(4)
            q1.metric("Stanga-Sus", f"{quads['top_left']:.1f}", f"{quads['pct_top_left']:.1f}%")
            q2.metric("Dreapta-Sus", f"{quads['top_right']:.1f}", f"{quads['pct_top_right']:.1f}%")
            q3.metric("Stanga-Jos", f"{quads['bottom_left']:.1f}", f"{quads['pct_bottom_left']:.1f}%")
            q4.metric("Dreapta-Jos", f"{quads['bottom_right']:.1f}", f"{quads['pct_bottom_right']:.1f}%")

    # -------------------------------------------------------------------------
    # TAB 2: METRICI SI BASELINE
    # -------------------------------------------------------------------------
    with tabs[1]:
        st.subheader("Rezultate Baseline pe setul de test")
        st.write(
            "Pentru a compara modelul din Sprint 2 (CSRNet), am calculat erorile "
            "obtinute de niste predictori simpli (media si mediana din train) aplicati pe test."
        )

        b_data = load_benchmark_data()
        if b_data:
            c_p1, c_p2 = st.columns(2)

            with c_p1:
                st.write("**Part A (182 imagini test)**")
                pA = b_data.get("part_A", {})
                bA = pA.get("baselines", {})
                rows_A = []
                for name, d in bA.items():
                    rows_A.append({
                        "Metoda": name,
                        "Valoare": d.get("predicted_constant", 0.0),
                        "MAE": d.get("mae", 0.0),
                        "RMSE": d.get("rmse", 0.0),
                        "NAE": d.get("nae", 0.0),
                    })
                df_A = pd.DataFrame(rows_A)
                st.dataframe(df_A, use_container_width=True, hide_index=True)

            with c_p2:
                st.write("**Part B (316 imagini test)**")
                pB = b_data.get("part_B", {})
                bB = pB.get("baselines", {})
                rows_B = []
                for name, d in bB.items():
                    rows_B.append({
                        "Metoda": name,
                        "Valoare": d.get("predicted_constant", 0.0),
                        "MAE": d.get("mae", 0.0),
                        "RMSE": d.get("rmse", 0.0),
                        "NAE": d.get("nae", 0.0),
                    })
                df_B = pd.DataFrame(rows_B)
                st.dataframe(df_B, use_container_width=True, hide_index=True)

    # -------------------------------------------------------------------------
    # TAB 3: TESTARE IMAGINE PROPRIE
    # -------------------------------------------------------------------------
    with tabs[2]:
        st.subheader("Incarcare imagine de test")
        st.write("Incarca o imagine pentru a verifica dimensiunile si formatul inainte de integrarea modelului.")

        up = st.file_uploader("Fisier imagine", type=["jpg", "jpeg", "png"])
        if up is not None:
            user_img = Image.open(up).convert("RGB")
            arr = np.asarray(user_img)

            col_u1, col_u2 = st.columns([1, 2])
            with col_u1:
                st.image(user_img, caption="Imagine incarcata", use_container_width=True)
            with col_u2:
                st.write(f"Rezolutie: {arr.shape[1]}x{arr.shape[0]} pixeli")
                st.write(f"Canale: {arr.shape[2]} (RGB)")



if __name__ == "__main__":
    main()
