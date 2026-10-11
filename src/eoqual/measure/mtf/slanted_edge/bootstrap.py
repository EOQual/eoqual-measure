#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Intervalles de confiance sur une courbe de MTF par bootstrap
(ré-échantillonnage des lignes de la ROI, avec remise).

Chaque ligne de la ROI est une mesure indépendante et bruitée du même
bord -- c'est exactement le principe qui permet à la méthode du bord
incliné de fonctionner (combiner de nombreuses lignes). Ré-échantillonner
les lignes avec remise, refaire l'estimation de MTF sur chaque tirage,
et regarder la dispersion des courbes obtenues donne une estimation de
l'incertitude de la MTF -- utile en usage normal (une seule image
réelle), à la différence du harnais de sensibilité
(`sensitivity.py`), qui a besoin de plusieurs images de vérité
terrain connue et répond à une question différente (biais systématique
vs incertitude sur une mesure donnée).

Fonctionne avec n'importe quel `SlantedEdgeMTFEstimator` (même
interface `estimate(frame, oversample_factor)` pour les quatre
méthodes, qu'elles utilisent le socle commun ou qu'elles surchargent
`estimate()` en entier) -- aucun cas particulier nécessaire.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from eoqual.measure.mtf.slanted_edge.estimators.base import CanonicalFrame, SlantedEdgeMTFEstimator


@dataclass(slots=True)
class BootstrapMTFResult:
    """
    Résultat d'un bootstrap de MTF.

    Attributes
    ----------
    method :
        Nom de l'estimateur utilisé.
    frequency :
        Grille de fréquences commune (linéaire, 0 à 0.5) sur laquelle
        toutes les courbes de MTF ont été interpolées avant agrégation
        -- nécessaire car la grille native de certaines méthodes
        (notamment `esf_lsf_*`) peut varier légèrement d'un tirage à
        l'autre (elle dépend des positions de bord détectées, donc des
        lignes effectivement tirées).
    mtf_mean, mtf_std :
        Moyenne et écart-type de la MTF (module), par fréquence, sur
        les tirages réussis.
    mtf_ci_low, mtf_ci_high :
        Bornes de l'intervalle de confiance empirique (percentiles) à
        chaque fréquence.
    n_successful, n_requested :
        Nombre de tirages ayant abouti / demandés (un tirage peut
        échouer, ex. pas assez de lignes valides après ré-échantillonnage
        -- rare mais possible pour de petites ROI).
    """

    method: str
    frequency: NDArray[np.float64]
    mtf_mean: NDArray[np.float64]
    mtf_std: NDArray[np.float64]
    mtf_ci_low: NDArray[np.float64]
    mtf_ci_high: NDArray[np.float64]
    n_successful: int
    n_requested: int


def bootstrap_mtf(
    estimator: SlantedEdgeMTFEstimator,
    frame: CanonicalFrame,
    n_bootstrap: int = 30,
    oversample_factor: int = 4,
    confidence: float = 0.95,
    frequency_grid: NDArray[np.float64] | None = None,
    seed: int = 0,
) -> BootstrapMTFResult:
    """
    Estime un intervalle de confiance sur la courbe de MTF par
    ré-échantillonnage des lignes de `frame.image`, avec remise.

    Parameters
    ----------
    estimator :
        N'importe quel `SlantedEdgeMTFEstimator` (`SpatialMTFEstimator`,
        `SpectralMTFEstimator`, `EsfLsfMTFEstimator`,
        `SpectralCorrelationMTFEstimator`, ...).
    frame :
        Image canonicalisée (voir `SlantedEdgePreprocessor`).
    n_bootstrap :
        Nombre de tirages. Attention au coût : `esf_model="tikhonov"`
        est nettement plus lent que les autres (voir sa docstring dans
        `estimators/esf_lsf.py`) -- réduire `n_bootstrap` en conséquence si on
        l'utilise.
    oversample_factor :
        Transmis à `estimator.estimate()`.
    confidence :
        Niveau de confiance de l'intervalle (ex. 0.95 pour un IC à 95%,
        c'est-à-dire les percentiles 2.5 et 97.5).
    frequency_grid :
        Grille de fréquences commune sur laquelle interpoler chaque
        tirage avant agrégation. Par défaut, 200 points entre 0 et 0.5.
    seed :
        Graine du tirage aléatoire (reproductibilité).

    Returns
    -------
    BootstrapMTFResult
    """
    if n_bootstrap < 2:
        raise ValueError("'n_bootstrap' must be >= 2.")
    if frequency_grid is None:
        frequency_grid = np.linspace(0.0, 0.5, 200)

    rng = np.random.default_rng(seed)
    n_rows = frame.image.shape[0]

    curves: list[NDArray[np.float64]] = []
    for _ in range(n_bootstrap):
        row_idx = rng.integers(0, n_rows, n_rows)
        resampled_image = frame.image[row_idx, :]
        resampled_frame = CanonicalFrame(
            image=resampled_image,
            black_level=frame.black_level,
            white_level=frame.white_level,
            was_transposed=frame.was_transposed,
            was_flipped=frame.was_flipped,
        )
        try:
            result = estimator.estimate(resampled_frame, oversample_factor=oversample_factor)
            interpolated = np.interp(
                frequency_grid,
                result.frequency_corrected,
                np.abs(result.mtf),
                left=np.nan,
                right=np.nan,
            )
            curves.append(interpolated)
        except Exception:  # noqa: BLE001 - un tirage raté ne doit pas interrompre le bootstrap
            continue

    n_successful = len(curves)
    if n_successful < 2:
        raise RuntimeError(
            f"Bootstrap failed: only {n_successful}/{n_bootstrap} draws succeeded."
        )

    stacked = np.array(curves)  # (n_successful, len(frequency_grid))
    alpha = (1.0 - confidence) / 2.0

    # Aux fréquences en bord de grille, certains tirages peuvent ne pas
    # couvrir (interpolation -> NaN, voir ci-dessus) : bénin, mais
    # np.nanmean/nanstd/nanpercentile émettent un avertissement si TOUS
    # les tirages sont NaN à une fréquence donnée -- supprimé ici plutôt
    # que de complexifier l'appelant pour un cas sans conséquence.
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mtf_mean = np.nanmean(stacked, axis=0)
        mtf_std = np.nanstd(stacked, axis=0)
        mtf_ci_low = np.nanpercentile(stacked, 100 * alpha, axis=0)
        mtf_ci_high = np.nanpercentile(stacked, 100 * (1.0 - alpha), axis=0)

    return BootstrapMTFResult(
        method=estimator.name,
        frequency=frequency_grid,
        mtf_mean=mtf_mean,
        mtf_std=mtf_std,
        mtf_ci_low=mtf_ci_low,
        mtf_ci_high=mtf_ci_high,
        n_successful=n_successful,
        n_requested=n_bootstrap,
    )


def plot_bootstrap_mtf(
    results: dict[str, BootstrapMTFResult],
    output_path: str,
    title: str = "MTF avec intervalle de confiance (bootstrap)",
) -> "object":
    """
    Trace la MTF moyenne ± intervalle de confiance (bande remplie), une
    courbe par méthode.

    Parameters
    ----------
    results :
        Un `BootstrapMTFResult` par méthode (ex. sortie de
        `bootstrap_mtf` appelé sur plusieurs estimateurs).
    output_path :
        Chemin du fichier HTML à écrire.
    """
    import plotly.graph_objects as go

    colors = {
        "spatial": "darkorange", "spectral": "seagreen",
        "esf_lsf_numeric": "mediumpurple", "esf_lsf_sigmoid": "crimson",
        "esf_lsf_pchip": "goldenrod", "esf_lsf_tikhonov": "steelblue",
        "spectral_correlation": "teal",
    }

    fig = go.Figure()
    for name, result in results.items():
        color = colors.get(name, None)
        fig.add_trace(go.Scatter(
            x=np.concatenate([result.frequency, result.frequency[::-1]]),
            y=np.concatenate([result.mtf_ci_high, result.mtf_ci_low[::-1]]),
            fill="toself", fillcolor=color, opacity=0.15,
            line=dict(color="rgba(0,0,0,0)"), showlegend=False,
            hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=result.frequency, y=result.mtf_mean, mode="lines",
            line=dict(color=color),
            name=f"{name} (n={result.n_successful}/{result.n_requested})",
        ))

    fig.update_layout(
        title=title,
        xaxis=dict(title="Fréquence normalisée (f / fₑ)", range=[0, 0.5]),
        yaxis=dict(title="MTF (module)", range=[0, 1]),
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
        template="simple_white",
        width=800,
        height=550,
    )
    fig.write_html(output_path)
    print(f"Bootstrap MTF plot saved to: {output_path}")
    return fig


# EOF
