#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pipeline complet bord incliné, multi-zones : estimation de l'orientation
(gradient, PCA, Fourier) puis de la MTF (spatiale, spectrale, ESF/LSF,
corrélation spectrale) pour une ou plusieurs images/zones, avec
comparaison des algorithmes à chaque étape et comparaison des zones
entre elles sur la MTF. Supporte les images en niveaux de gris comme les
images RAW Bayer, quatre motifs (`--image-format`, `--bayer-pattern`).

Une zone = une image ; les résultats sont comparés zone par zone puis
toutes zones confondues, via les comparateurs (`EdgeOrientationComparator`,
`MTFComparator`) et affiché via `SlantedEdgeVisualizer`
(`visualization.py`) plutôt que par une classe monolithique.

Une zone (une image) peut elle-même produire plusieurs résultats si
elle est au format Bayer : `--image-format bayer` analyse les plans R, G
et B séparément (`--image-format bayer-r/-g/-b` n'en analyse qu'un), pour
le motif donné par `--bayer-pattern` (RGGB par défaut ; `rggb*` restent
acceptés comme alias). Le
paramètre `pixel_pitch` (voir `estimators.base.SlantedEdgeMTFEstimator`)
est réglé à `eoqual.core.bayer.BAYER_SAMPLING_FACTOR` pour ces plans, afin que leur
fréquence rapportée reste exprimée par pixel physique du capteur --
directement comparable à une zone en niveaux de gris pleine résolution.
Pour une zone Bayer à plusieurs plans, le décalage sous-pixel latéral
entre plans (aberration chromatique) est aussi calculé et affiché
(`chromatic.py`, `--chromatic-*`).

Usage
-----
eoqual-measure-mtf mire_H0.png
eoqual-measure-mtf mire_H0.png mire_H1.png mire_V0.png --zone-labels H0 H1 V0
eoqual-measure-mtf mire.png --oversample-factor 2 --fourier-angle-source multiharmonic
eoqual-measure-mtf mire_raw.tiff --image-format bayer
eoqual-measure-mtf mire_raw.tiff --image-format bayer-g --bayer-pattern GBRG
"""
from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from eoqual.core.bayer import BAYER_PATTERNS, BAYER_SAMPLING_FACTOR, bayer_offsets, extract_bayer_planes
from eoqual.measure.mtf.slanted_edge.chromatic import compute_chromatic_shifts, print_chromatic_shifts
from eoqual.measure.mtf.slanted_edge.orientation.base import EdgeOrientationComparator, OrientationResult
from eoqual.measure.mtf.slanted_edge.estimators.esf_lsf import EsfLsfMTFEstimator
from eoqual.measure.mtf.slanted_edge.orientation.fourier import FourierOrientationDetector
from eoqual.measure.mtf.slanted_edge.orientation.gradient import GradientOrientationDetector
from eoqual.measure.mtf.slanted_edge.estimators.base import MTFComparator, MTFResult
from eoqual.measure.mtf.slanted_edge.orientation.pca import PCAOrientationDetector
from eoqual.measure.mtf.slanted_edge.preprocess import SlantedEdgePreprocessor
from eoqual.measure.mtf.slanted_edge.estimators.spatial import SpatialMTFEstimator
from eoqual.measure.mtf.slanted_edge.estimators.spectral_correlation import SpectralCorrelationMTFEstimator
from eoqual.measure.mtf.slanted_edge.estimators.spectral import SpectralMTFEstimator
from eoqual.measure.mtf.slanted_edge.visualization import SlantedEdgeVisualizer

#: Plans Bayer effectivement analysés pour chaque valeur de --image-format.
BAYER_PLANE_SELECTION = {
    "bayer": ("R", "G", "B"),
    "bayer-r": ("R",),
    "bayer-g": ("G",),
    "bayer-b": ("B",),
    # Alias historiques (motif donné par --bayer-pattern, RGGB par défaut).
    "rggb": ("R", "G", "B"),
    "rggb-r": ("R",),
    "rggb-g": ("G",),
    "rggb-b": ("B",),
}


def load_image(path: str, image_format: str) -> np.ndarray:
    """
    Charge une image depuis le disque et la convertit en tableau
    float64.

    Parameters
    ----------
    path :
        Chemin de l'image.
    image_format :
        `"gray"` ou l'une des valeurs de `BAYER_PLANE_SELECTION`. Pour
        Bayer, l'image doit être une mosaïque RAW 2D (pas une image déjà
        démosaïquée à 3 canaux).

    Returns
    -------
    np.ndarray
        Image 2D en float64 -- niveaux de gris, ou mosaïque Bayer brute
        selon `image_format`.
    """
    image = imageio.imread(path)

    if image_format == "gray":
        if image.ndim == 3:
            image = image[:, :, :3].mean(axis=2)  # retire l'alpha, moyenne les couleurs
        return image.astype(np.float64)

    if image.ndim != 2:
        raise ValueError(
            f"'{path}' has {image.ndim} dimensions; a Bayer RAW image "
            "(--image-format bayer*/rggb*) must be a 2D mosaic, not an already "
            "demosaiced 3-channel image."
        )
    return image.astype(np.float64)


def get_analysis_planes(
    image: np.ndarray,
    image_format: str,
    bayer_pattern: str = "RGGB",
) -> tuple[dict[str, np.ndarray], float]:
    """
    Détermine les plans à analyser et le facteur d'échantillonnage
    associé (`pixel_pitch` à utiliser pour la MTF).

    Parameters
    ----------
    image :
        Image chargée par `load_image`.
    image_format :
        `"gray"` ou l'une des valeurs de `BAYER_PLANE_SELECTION`.
    bayer_pattern :
        Motif de la mosaïque (`RGGB`, `BGGR`, `GRBG`, `GBRG`).

    Returns
    -------
    tuple[dict[str, np.ndarray], float]
        `(planes, pixel_pitch)` -- `planes` associe un nom de plan (vide
        pour "gray", sinon "R"/"G"/"B") à son image ; `pixel_pitch` est
        1.0 pour "gray", `BAYER_SAMPLING_FACTOR` pour un plan Bayer.
    """
    if image_format == "gray":
        return {"": image}, 1.0

    if image_format not in BAYER_PLANE_SELECTION:
        raise ValueError(f"Unknown --image-format: {image_format!r}.")

    bayer_planes = extract_bayer_planes(image, bayer_pattern)
    selected = BAYER_PLANE_SELECTION[image_format]
    planes = {name: bayer_planes[name] for name in selected}
    return planes, BAYER_SAMPLING_FACTOR


def process_plane(
    image: np.ndarray,
    plane_label: str,
    pixel_pitch: float,
    output_stem: Path,
    args: argparse.Namespace,
) -> tuple[dict[str, OrientationResult], dict[str, MTFResult]]:
    """
    Traite un plan unique (une image en niveaux de gris) : orientation
    puis MTF, affichage inclus.

    Un « plan » est soit une image en niveaux de gris complète
    (`--image-format gray`), soit l'un des plans couleur extraits d'une
    mosaïque Bayer (`--image-format bayer*`).

    Parameters
    ----------
    image :
        Image (ou plan) en niveaux de gris à analyser.
    plane_label :
        Nom complet du plan pour l'affichage (ex. "H0" ou "H0/R").
    pixel_pitch :
        Transmis à chaque estimateur de MTF -- 1.0 pour une image en
        niveaux de gris pleine résolution, `BAYER_SAMPLING_FACTOR` pour
        un plan Bayer (voir `get_analysis_planes`).
    output_stem :
        Base de nom de fichier pour les graphiques produits (un plan =
        un jeu de fichiers, pour ne pas écraser ceux des autres plans
        d'une même zone).
    args :
        Arguments de la ligne de commande (voir `parse_args`).

    Returns
    -------
    tuple[dict[str, OrientationResult], dict[str, MTFResult]]
        Les résultats d'orientation (un par algorithme -- réutilisés
        pour l'aberration chromatique en Bayer multi-plans, voir
        `process_zone`) et de MTF (un par méthode) de ce plan.
    """
    print(f"\n=== {plane_label} ===  shape={image.shape}  "
          f"range=[{image.min():.1f}, {image.max():.1f}]  pixel_pitch={pixel_pitch}")

    if args.show_mire and not args.no_plot:
        SlantedEdgeVisualizer.plot_image(
            image,
            output_stem.with_name(f"{output_stem.name}_mire.html"),
            title=f"Mire -- {plane_label}",
        )

    # ---------------------------------------------------------------- #
    # 1. Orientation : gradient, PCA, Fourier.
    # ---------------------------------------------------------------- #
    orientation_comparator = EdgeOrientationComparator(
        detectors={
            "gradient": GradientOrientationDetector(
                canonicalization_percentile=args.canonicalization_percentile,
            ),
            "pca": PCAOrientationDetector(
                gradient_percentile=args.gradient_percentile,
                canonicalization_percentile=args.canonicalization_percentile,
            ),
            "fourier": FourierOrientationDetector(
                canonicalization_percentile=args.canonicalization_percentile,
                angle_source=args.fourier_angle_source,
                low_freq_ratio=args.fourier_low_freq_ratio,
                block_gap=args.fourier_block_gap,
            ),
        }
    )
    orientation_results = orientation_comparator.compare(image)

    print()
    SlantedEdgeVisualizer.print_orientation_summary(orientation_results)

    if not args.no_plot:
        SlantedEdgeVisualizer.plot_orientation_comparison(
            image,
            orientation_results,
            output_stem.with_name(f"{output_stem.name}_edge_comparison.html"),
        )

    # ---------------------------------------------------------------- #
    # 2. MTF : spatiale, spectrale, ESF/LSF, corrélation spectrale.
    #
    # Chaque estimateur construit lui-même, à partir de l'image
    # canonicalisée, la représentation dont il a besoin (représentation
    # symétrique demi-onde pour spatiale/spectrale ; l'image canonique
    # telle quelle pour ESF/LSF et corrélation spectrale) -- voir
    # docstring de estimators/base.py. `pixel_pitch` est le même pour les
    # quatre, ce qui les rend comparables entre plans de résolutions
    # différentes (voir docstring du module).
    # ---------------------------------------------------------------- #
    frame = SlantedEdgePreprocessor(
        percentile=args.canonicalization_percentile
    ).preprocess(image)

    esf_lsf_estimator = EsfLsfMTFEstimator(
        mode=args.esf_lsf_mode,
        esf_model=args.esf_lsf_esf_model,
        pixel_pitch=pixel_pitch,
        lsf_derivative_method=args.esf_lsf_derivative_method,
        lsf_filter_type=args.esf_lsf_filter_type,
        lsf_window_length=args.esf_lsf_window_length,
        lsf_polyorder=args.esf_lsf_polyorder,
    )
    mtf_comparator = MTFComparator(
        estimators={
            "spatial": SpatialMTFEstimator(
                low_freq_ratio=args.mtf_low_freq_ratio,
                block_gap=args.mtf_block_gap,
                pixel_pitch=pixel_pitch,
            ),
            "spectral": SpectralMTFEstimator(
                low_freq_ratio=args.mtf_low_freq_ratio,
                block_gap=args.mtf_block_gap,
                pixel_pitch=pixel_pitch,
            ),
            esf_lsf_estimator.name: esf_lsf_estimator,
            "spectral_correlation": SpectralCorrelationMTFEstimator(
                apply_hanning=not args.no_spectral_correlation_hanning,
                line_spacing=args.spectral_correlation_line_spacing,
                low_frequency_ratio=args.spectral_correlation_low_freq_ratio,
                pixel_pitch=pixel_pitch,
            ),
        }
    )
    mtf_results = mtf_comparator.compare(frame, oversample_factor=args.oversample_factor)

    print()
    SlantedEdgeVisualizer.print_mtf_summary(
        mtf_results, angle_correction=not args.no_angle_correction
    )

    return orientation_results, mtf_results


def process_zone(
    image_path: str,
    zone_label: str,
    args: argparse.Namespace,
) -> dict[str, MTFResult]:
    """
    Traite une zone (une image, éventuellement Bayer) : détermine ses
    plans (`get_analysis_planes`) puis traite chacun (`process_plane`).
    Pour une zone Bayer multi-plans, calcule aussi le décalage
    chromatique latéral entre plans (`chromatic.py`).

    Parameters
    ----------
    image_path :
        Chemin de l'image de cette zone.
    zone_label :
        Nom de la zone (ex. "H0", "V1", ou le nom de fichier par défaut).
    args :
        Arguments de la ligne de commande (voir `parse_args`).

    Returns
    -------
    dict[str, MTFResult]
        Les résultats de MTF de cette zone, un par (plan, méthode) --
        ex. `{"spatial": ...}` en niveaux de gris, ou
        `{"R/spatial": ..., "G/spatial": ..., "B/spatial": ...}` en
        Bayer complet.
    """
    print(f"\n########## Zone {zone_label} ({image_path}) ##########")

    image = load_image(image_path, args.image_format)
    planes, pixel_pitch = get_analysis_planes(image, args.image_format, args.bayer_pattern)

    zone_mtf_results: dict[str, MTFResult] = {}
    zone_orientation_results: dict[str, dict[str, object]] = {}
    for plane_name, plane_image in planes.items():
        plane_label = f"{zone_label}/{plane_name}" if plane_name else zone_label
        output_stem = Path(image_path).with_name(
            f"{zone_label}_{plane_name}" if plane_name else zone_label
        )
        orientation_results, plane_mtf_results = process_plane(
            plane_image, plane_label, pixel_pitch, output_stem, args
        )
        zone_orientation_results[plane_name] = orientation_results
        prefix = f"{plane_name}/" if plane_name else ""
        for method_name, mtf_result in plane_mtf_results.items():
            zone_mtf_results[f"{prefix}{method_name}"] = mtf_result

    # -- Aberration chromatique latérale (zones Bayer multi-plans) ------
    if not args.no_chromatic and len(zone_orientation_results) > 1:
        try:
            per_plane_results = {
                plane: results[args.chromatic_algorithm]
                for plane, results in zone_orientation_results.items()
            }
            chromatic_result = compute_chromatic_shifts(
                per_plane_results,
                reference_plane=args.chromatic_reference_plane,
                plane_offsets=bayer_offsets(args.bayer_pattern),
            )
            print()
            print_chromatic_shifts(chromatic_result)
        except (KeyError, ValueError) as exc:
            print(f"\nAberration chromatique non calculée : {exc}")

    return zone_mtf_results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate the slant angle (gradient, PCA, Fourier) and MTF "
            "(spatial, spectral, ESF/LSF, spectral correlation) of one "
            "or more slanted-edge zones -- grayscale or Bayer -- "
            "and compare the zones on the same MTF plot."
        )
    )
    parser.add_argument(
        "images",
        nargs="+",
        help="Path(s) to the slanted-edge image(s), one per zone.",
    )
    parser.add_argument(
        "--zone-labels",
        nargs="+",
        default=None,
        help="One label per image (default: each image's filename stem).",
    )
    parser.add_argument(
        "--image-format",
        choices=["gray", *BAYER_PLANE_SELECTION],
        default="gray",
        help="'gray' (default) for a regular grayscale image. 'bayer' "
        "for a Bayer RAW mosaic (pattern given by --bayer-pattern), "
        "analyzing its R, G, and B planes separately; "
        "'bayer-r'/'bayer-g'/'bayer-b' to analyze only one plane. "
        "'rggb*' are kept as aliases of 'bayer*'. Applies to every zone "
        "in this run.",
    )
    parser.add_argument(
        "--bayer-pattern",
        choices=list(BAYER_PATTERNS),
        default="RGGB",
        help="Bayer mosaic pattern, read row by row from the top-left "
        "pixel (default: RGGB). Used by the bayer*/rggb* formats.",
    )
    parser.add_argument(
        "--show-mire",
        action="store_true",
        help="Also plot each zone/plane's raw image.",
    )
    parser.add_argument(
        "--chromatic-algorithm",
        choices=["gradient", "pca", "fourier"],
        default="fourier",
        help="Orientation algorithm whose per-row edge positions are "
        "used to measure lateral chromatic shift between Bayer planes "
        "(default: fourier). Ignored for --image-format gray or a "
        "single-plane bayer-*/rggb-* format.",
    )
    parser.add_argument(
        "--chromatic-reference-plane",
        default="G",
        help="Bayer plane taken as the zero-shift reference for "
        "chromatic aberration measurement (default: G).",
    )
    parser.add_argument(
        "--no-chromatic",
        action="store_true",
        help="Skip lateral chromatic aberration measurement even for "
        "multi-plane Bayer zones.",
    )

    # -- Orientation --------------------------------------------------
    parser.add_argument(
        "--gradient-percentile",
        type=float,
        default=90.0,
        help="Percentile threshold for the gradient/PCA pixel selection "
        "(default: 90).",
    )
    parser.add_argument(
        "--canonicalization-percentile",
        type=float,
        default=10.0,
        help="Percentile of columns used to estimate black/white levels "
        "(default: 10).",
    )
    parser.add_argument(
        "--fourier-angle-source",
        choices=["fundamental", "multiharmonic"],
        default="fundamental",
        help="Which estimate feeds the Fourier detector's angle_deg: "
        "'fundamental' (k=1 phase + per-row fit, default) or "
        "'multiharmonic' (phase-difference between neighbouring rows, "
        "averaged over several low odd harmonics). The one not chosen "
        "stays available in details.",
    )
    parser.add_argument(
        "--fourier-low-freq-ratio",
        type=float,
        default=0.2,
        help="Fraction of the lowest odd harmonics used by the Fourier "
        "'multiharmonic' angle estimate (default: 0.2).",
    )
    parser.add_argument(
        "--fourier-block-gap",
        type=int,
        default=2,
        help="Row gap between the two lines compared by the Fourier "
        "'multiharmonic' angle estimate (default: 2).",
    )

    # -- MTF ------------------------------------------------------------
    parser.add_argument(
        "--oversample-factor",
        type=int,
        choices=[1, 2, 4],
        default=4,
        help="MTF oversampling factor (default: 4).",
    )
    parser.add_argument(
        "--mtf-low-freq-ratio",
        type=float,
        default=0.2,
        help="Fraction of the lowest odd harmonics used to estimate the "
        "edge slope for MTF computation (default: 0.2).",
    )
    parser.add_argument(
        "--mtf-block-gap",
        type=int,
        default=2,
        help="Row gap between the two lines compared to estimate the "
        "edge slope for MTF computation (default: 2).",
    )
    parser.add_argument(
        "--no-angle-correction",
        action="store_true",
        help="Plot/report MTF against raw frequency instead of the "
        "cos(theta)-corrected frequency.",
    )
    parser.add_argument(
        "--esf-lsf-mode",
        choices=["sigmoid", "gaussian"],
        default="sigmoid",
        help="Per-row sub-pixel edge LOCALIZATION used by the ESF/LSF "
        "MTF estimator's ESF builder: 'sigmoid' fits the intensity "
        "profile, 'gaussian' fits its gradient (default: sigmoid). Not "
        "to be confused with --esf-lsf-esf-model, which chooses how the "
        "aggregated ESF is turned into a LSF.",
    )
    parser.add_argument(
        "--esf-lsf-esf-model",
        choices=["numeric", "sigmoid", "pchip", "tikhonov"],
        default="pchip",
        help="How the aggregated ESF is turned into a LSF: 'numeric' "
        "(direct numerical differentiation, see --esf-lsf-derivative-"
        "method/--esf-lsf-filter-type), 'sigmoid' (fit + analytic "
        "derivative -- can overestimate high-frequency MTF if the true "
        "edge PSF isn't tanh-shaped, see fit_esf_sigmoid docstring), "
        "'pchip' (monotone spline interpolation, no assumed shape, FFT "
        "phase-ramp recentered, default), or 'tikhonov' (regularized "
        "ESF fit with per-bin precision weights, more robust to noise, "
        "see fit_esf_tikhonov docstring).",
    )
    parser.add_argument(
        "--esf-lsf-derivative-method",
        choices=["gradient", "savgol_deriv"],
        default="gradient",
        help="How the ESF/LSF estimator differentiates the ESF into a "
        "LSF: 'gradient' (default, central differences) or "
        "'savgol_deriv' (direct Savitzky-Golay derivative, smooths and "
        "derives in one step).",
    )
    parser.add_argument(
        "--esf-lsf-filter-type",
        choices=["tukey", "savgol", "gaussian", "none"],
        default="tukey",
        help="How the ESF/LSF estimator's LSF is filtered/windowed "
        "after differentiation (default: tukey).",
    )
    parser.add_argument(
        "--esf-lsf-window-length",
        type=int,
        default=15,
        help="Window length for 'savgol_deriv'/'savgol' (and, divided "
        "by 6, the sigma for 'gaussian') (default: 15).",
    )
    parser.add_argument(
        "--esf-lsf-polyorder",
        type=int,
        default=3,
        help="Savitzky-Golay polynomial order, for "
        "'savgol_deriv'/'savgol' (default: 3).",
    )
    parser.add_argument(
        "--no-spectral-correlation-hanning",
        action="store_true",
        help="Disable the Hanning window used by the spectral_correlation "
        "estimator's own slope estimation (default: enabled).",
    )
    parser.add_argument(
        "--spectral-correlation-line-spacing",
        type=int,
        default=2,
        help="Row gap between the (non-overlapping) line pairs used by "
        "the spectral_correlation estimator's slope estimation "
        "(default: 2).",
    )
    parser.add_argument(
        "--spectral-correlation-low-freq-ratio",
        type=float,
        default=0.2,
        help="Fraction of the lowest frequencies used by the "
        "spectral_correlation estimator's slope estimation (default: 0.2).",
    )

    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip every plot (per-zone/plane and combined).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    zone_labels = args.zone_labels or [Path(p).stem for p in args.images]
    if len(zone_labels) != len(args.images):
        raise ValueError("--zone-labels must have as many entries as images.")

    # Une zone = une image (un ou plusieurs plans selon --image-format) ;
    # chacune est traitée indépendamment puis combinée pour la
    # comparaison finale toutes zones/plans confondus.
    combined_mtf_results: dict[str, MTFResult] = {}
    for image_path, zone_label in zip(args.images, zone_labels):
        zone_mtf_results = process_zone(image_path, zone_label, args)
        for method_name, mtf_result in zone_mtf_results.items():
            combined_mtf_results[f"{zone_label} ({method_name})"] = mtf_result

    # Comparaison finale : toutes les zones (et tous leurs plans),
    # toutes les méthodes, sur un même graphique.
    if not args.no_plot:
        combined_plot_path = Path(args.images[0]).with_name("mtf_comparison_all_zones.html")
        SlantedEdgeVisualizer.plot_mtf_comparison(
            combined_mtf_results,
            combined_plot_path,
            angle_correction=not args.no_angle_correction,
            title="MTF -- comparaison des zones",
        )


if __name__ == "__main__":
    main()

# EOF
