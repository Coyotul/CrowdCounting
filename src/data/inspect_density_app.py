"""Punct de intrare pentru aplicatia interactiva de inspectie Streamlit.

Redirectioneaza catre modulul centralizat dezvoltat in src.ui.app.
Poate fi rulat direct cu:
    python -m streamlit run src/data/inspect_density_app.py
sau:
    python -m streamlit run src/ui/app.py
"""

from __future__ import annotations

from src.ui.app import main

if __name__ == "__main__":
    main()
