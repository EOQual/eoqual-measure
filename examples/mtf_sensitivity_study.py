#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Étude de sensibilité complète, branchée sur `SyntheticSlantedEdge` :
effet de l'angle, effet du bruit, comparaison des algorithmes
d'orientation et de MTF (§12.1 de la partie I de REFERENCE_TECHNIQUE.md).

Usage
-----
python examples/mtf_sensitivity_study.py
python examples/mtf_sensitivity_study.py --quick          # grille réduite, pour vérifier que tout tourne
python examples/mtf_sensitivity_study.py --with-tikhonov  # inclut esf_lsf_tikhonov (coûteux)
python examples/mtf_sensitivity_study.py --orientation horizontal --psf-type lorentzian --sigma-psf 1.5
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from eoqual.measure.mtf.slanted_edge.estimators.base import MTFComparator
from eoqual.measure.mtf.slanted_edge.sensitivity import (
    SensitivityGrid,
    SensitivityStudy,
    fold_angle_deg,
    plot_heatmap,
    plot_sensitivity,
)
from eoqual.measure.mtf.slanted_edge.estimators.spatial import SpatialMTFEstimator
from eoqual.measure.mtf.slanted_edge.estimators.spectral_correlation import SpectralCorrelationMTFEstimator
from eoqual.measure.mtf.slanted_edge.estimators.spectral import SpectralMTFEstimator
from eoqual.measure.mtf.slanted_edge.estimators.esf_lsf import EsfLsfMTFEstimator
from eoqual.measure.synthetic.slanted_edge import SyntheticSlantedEdge

def build_mire_generator(
    orientation: str,
    psf_type: str,
    sigma_psf: float,
    nx: int,
    ny: int,
    noise_type: str,
    poisson_scale: float,
    fe: float,
):
    """
    Construit le générateur de mire respectant le contrat de
    `sensitivity.py` (`(angle_deg, noise_level, seed) ->
    (image, mtf_theoretical)`), à partir de `SyntheticSlantedEdge` avec
    tous les paramètres fixes de l'étude fermés par cette fonction --
    seuls `angle_deg` (le paramètre `angle_deg` de `SyntheticSlantedEdge`)
    et `noise_level` (son paramètre `noise_level`) varient d'un essai à
    l'autre.
    """

    def mire_generator(angle_deg: float, noise_level: float, seed: int):
        edge = SyntheticSlantedEdge(
            nx=nx,
            ny=ny,
            angle_deg=angle_deg,
            orientation=orientation,
            psf_type=psf_type,
            sigma_psf=sigma_psf,
            fe=fe,
            noise_type=noise_type if noise_level > 0 else "none",
            noise_level=noise_level,
            poisson_scale=poisson_scale,
            seed=seed,
        )
        image = edge.generate_image().astype(np.float64)
        return image, edge.mtf_theoretical

    return mire_generator


def angle_conversion_for(orientation: str):
    """
    Conversion angle_deg (paramètre `SyntheticSlantedEdge`) -> convention
    des estimateurs (`atan2`, repliée dans [-90, 90]) -- voir docstring
    de `SensitivityGrid.true_angle_to_convention`, vérifiée
    empiriquement (voir REFERENCE_TECHNIQUE.md, partie I / l'historique de
    cette conversation).
    """
    if orientation == "vertical":
        return lambda a: fold_angle_deg(a - 90.0)
    if orientation == "horizontal":
        return lambda a: fold_angle_deg(-a)
    raise ValueError(f"orientation inconnue : {orientation!r}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quick", action="store_true",
        help="Grille réduite (3 angles x 2 bruits x 3 répétitions), pour "
        "vérifier rapidement que tout fonctionne avant un run complet.",
    )
    parser.add_argument(
        "--with-tikhonov", action="store_true",
        help="Inclut esf_lsf_tikhonov dans la comparaison MTF (exclu par "
        "défaut : nettement plus coûteux, voir sa docstring).",
    )
    parser.add_argument("--orientation", choices=["vertical", "horizontal"], default="vertical")
    parser.add_argument("--psf-type", choices=["gaussian", "lorentzian", "pillbox"], default="gaussian")
    parser.add_argument("--sigma-psf", type=float, default=1.2)
    parser.add_argument("--nx", type=int, default=300)
    parser.add_argument("--ny", type=int, default=150)
    parser.add_argument("--noise-type", choices=["gaussian", "poisson_gaussian"], default="gaussian")
    parser.add_argument("--poisson-scale", type=float, default=500.0)
    parser.add_argument("--fe", type=float, default=1.0)
    parser.add_argument("--n-repetitions", type=int, default=10)
    parser.add_argument("--output-dir", type=str, default="sensitivity_output")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mire_generator = build_mire_generator(
        orientation=args.orientation,
        psf_type=args.psf_type,
        sigma_psf=args.sigma_psf,
        nx=args.nx,
        ny=args.ny,
        noise_type=args.noise_type,
        poisson_scale=args.poisson_scale,
        fe=args.fe,
    )

    if args.quick:
        angles_deg = [1.0, 5.0, 20.0]
        noise_levels = [0.0, 0.01]
        n_repetitions = 3
    else:
        angles_deg = [0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 15.0, 20.0, 25.0, 30.0]
        noise_levels = [0.0, 0.002, 0.005, 0.01, 0.02, 0.05]
        n_repetitions = args.n_repetitions

    grid = SensitivityGrid(
        angles_deg=angles_deg,
        noise_levels=noise_levels,
        n_repetitions=n_repetitions,
        # Bande de fréquence pour le RMSE de MTF : au-delà, le signal
        # théorique est proche de 0 et seul le bruit de mesure domine
        # (voir §9.3 de la partie I de REFERENCE_TECHNIQUE.md) -- resserrer encore
        # si sigma_psf est grand (MTF déjà proche de 0 à plus basse
        # fréquence).
        mtf_frequency_band=(0.02, min(0.35, 0.6 / max(args.sigma_psf, 0.5))),
        true_angle_to_convention=angle_conversion_for(args.orientation),
    )

    mtf_estimators = {
        "spatial": SpatialMTFEstimator(),
        "spectral": SpectralMTFEstimator(),
        "esf_lsf_pchip": EsfLsfMTFEstimator(esf_model="pchip"),
        "spectral_correlation": SpectralCorrelationMTFEstimator(),
    }
    if args.with_tikhonov:
        mtf_estimators["esf_lsf_tikhonov"] = EsfLsfMTFEstimator(esf_model="tikhonov")

    study = SensitivityStudy(
        mire_generator,
        mtf_comparator=MTFComparator(estimators=mtf_estimators),
    )

    n_images = len(angles_deg) * len(noise_levels) * n_repetitions
    print(f"Grille : {len(angles_deg)} angles x {len(noise_levels)} niveaux de bruit x "
          f"{n_repetitions} répétitions = {n_images} images, "
          f"{len(mtf_estimators)} méthodes MTF + 3 méthodes d'orientation.")

    t0 = time.perf_counter()
    df = study.run(grid)
    elapsed = time.perf_counter() - t0
    print(f"\nTerminé en {elapsed:.1f}s ({elapsed/max(n_images,1)*1000:.1f} ms/image en moyenne).")

    df.to_csv(output_dir / "trials.csv", index=False)
    summary = study.summarize(df)
    summary.to_csv(output_dir / "summary.csv", index=False)
    print(f"Résultats bruts et agrégés sauvegardés dans {output_dir}/")

    # -- Quelques graphiques de synthèse ---------------------------------
    mid_noise = noise_levels[len(noise_levels) // 2]
    mid_angle = angles_deg[len(angles_deg) // 2]

    plot_sensitivity(
        summary, x="angle_deg_true", family="orientation",
        fixed={"noise_level": noise_levels[0]},
        output_path=str(output_dir / "orientation_vs_angle_no_noise.html"),
        title="Biais d'angle vs angle du bord (sans bruit)",
    )
    plot_sensitivity(
        summary, x="noise_level", family="orientation",
        fixed={"angle_deg_true": mid_angle},
        output_path=str(output_dir / "orientation_vs_noise.html"),
        title=f"Biais d'angle vs bruit (angle={mid_angle}°)",
    )
    plot_sensitivity(
        summary, x="angle_deg_true", family="mtf",
        fixed={"noise_level": mid_noise},
        output_path=str(output_dir / "mtf_vs_angle.html"),
        title=f"RMSE de MTF vs angle du bord (bruit={mid_noise})",
    )
    plot_sensitivity(
        summary, x="noise_level", family="mtf",
        fixed={"angle_deg_true": mid_angle},
        output_path=str(output_dir / "mtf_vs_noise.html"),
        title=f"RMSE de MTF vs bruit (angle={mid_angle}°)",
    )
    plot_heatmap(
        summary, family="mtf", method="esf_lsf_pchip",
        output_path=str(output_dir / "heatmap_esf_lsf_pchip.html"),
    )

    # -- Tableau de synthèse : erreur systématique + temps, tous angles/bruits confondus
    print("\n--- Comparaison des méthodes (moyenne tous angles/bruits confondus) ---")
    overall = summary.groupby(["family", "method"]).agg(
        mean_abs_bias=("bias", lambda s: np.mean(np.abs(s))),
        mean_std=("std", "mean"),
        mean_time_ms=("mean_time_s", lambda s: np.mean(s) * 1000),
        failure_rate=("failure_rate", "mean"),
    ).reset_index()
    print(overall.to_string(index=False))


if __name__ == "__main__":
    main()

# EOF
