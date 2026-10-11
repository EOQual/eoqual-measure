#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Génère deux mires synthétiques à bord incliné (verticale et horizontale)
à FTM théorique connue, et exporte image (TIFF float32), courbe FTM
théorique (CSV) et visualisation (HTML Plotly) dans ``OUTPUT/``.

Nécessite l'extra ``plot`` : ``pip install "eoqual-measure[plot]"``.
"""
from pathlib import Path

from eoqual.measure.synthetic import SyntheticSlantedEdge

# Création du répertoire de sortie si nécessaire
OUTPUT_DIR = Path("OUTPUT")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

edge = SyntheticSlantedEdge(
    nx=256,
    ny=256,
    angle_deg=5.0,
    orientation="vertical",
    psf_type="gaussian",
    sigma_psf=1.2,
    noise_type="poisson_gaussian",
    noise_level=0.005,
    poisson_scale=500.0,
    seed=0,
)
img, mtf_func = edge.generate(
    csv_path=OUTPUT_DIR / "mtf_curve_gaussian.csv",
    html_path=OUTPUT_DIR / "mtf_visualisation_gaussian.html",
    image_path=OUTPUT_DIR / "mire_gaussian_vertical.tiff",
)

edge_horizontal = SyntheticSlantedEdge(
    nx=256,
    ny=256,
    angle_deg=5.0,
    orientation="horizontal",
    psf_type="gaussian",
    sigma_psf=1.2,
    noise_type="gaussian",
    noise_level=0.005,
    seed=0,
)
edge_horizontal.generate(
    html_path=OUTPUT_DIR / "mtf_visualisation_horizontal.html",
    image_path=OUTPUT_DIR / "mire_gaussian_horizontal.tiff",
)

#EOF
