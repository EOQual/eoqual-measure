#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Estimation de MTF par la méthode « spatiale ».

Le bord est d'abord « dé-interlacé » ligne-par-ligne compte tenu de sa
pente (`interlace_1d`), puis ré-échantillonné sur une grille uniforme et
symétrisé (`regularise_signal`). La MTF est obtenue comme le rapport du
spectre mesuré sur le spectre analytique d'un bord idéal.

Voir `SpatialMTFEstimator` pour le détail.
"""
from __future__ import annotations

import numpy as np
import scipy.interpolate
from numpy.typing import NDArray

from eoqual.measure.mtf.slanted_edge.fourier_utils import reference_spectrum
from eoqual.measure.mtf.slanted_edge.estimators.base import SlantedEdgeMTFEstimator


def interlace_1d(
    rect: NDArray[np.float64],
    dx: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Convertit la zone 2D en un signal 1D en tenant compte du décalage
    ligne-à-ligne `dx`.

    Chaque ligne du bord incliné est une version décalée des autres ;
    en recalant chaque ligne de `dx * y` colonnes puis en triant tous
    les échantillons selon leur abscisse recalée, on obtient un unique
    profil de bord sur-échantillonné (le fameux « dé-interlacement »
    des méthodes ISO 12233 / slanted-edge).

    Parameters
    ----------
    rect :
        Zone contenant le bord incliné.
    dx :
        Décalage inter-lignes (pixels).

    Returns
    -------
    x_effective :
        Coordonnées le long du bord (triées).
    signal :
        Intensités correspondantes, réordonnées.
    """
    n_rows, n_cols = rect.shape
    x_grid, y_grid = np.meshgrid(np.arange(n_cols), np.arange(n_rows))

    # Coordonnée "abscisse du bord" dans le repère du bord.
    x_eff = (x_grid - dx * y_grid).ravel() % n_cols

    # Tri des points selon l'abscisse (on veut un signal monotone).
    sort_idx = np.argsort(x_eff)

    return x_eff[sort_idx], rect.ravel()[sort_idx]


def regularise_signal(
    x_eff: NDArray[np.float64],
    signal: NDArray[np.float64],
    rect: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Interpole le signal 1D sur une grille régulière et impose
    l'anti-symétrie attendue pour un bord idéal.

    Parameters
    ----------
    x_eff :
        Coordonnées "effectives" (dé-interlacées).
    signal :
        Intensités correspondantes.
    rect :
        Zone d'origine (seulement pour connaître le nombre d'échantillons).

    Returns
    -------
    x_reg :
        Grille régulière.
    signal_reg :
        Signal ré-échantillonné et anti-symétrique.
    """
    # Interpolation linéaire (ou spline selon le défaut de scipy) sur
    # tout le domaine.
    interp = scipy.interpolate.interp1d(
        x_eff, signal, bounds_error=False, fill_value=0
    )

    # Points réguliers allant de 0 à largeur-1, sur le même nombre
    # d'échantillons que le rectangle complet.
    n_samples = rect.size
    x_reg = np.arange(n_samples) / float(n_samples) * rect.shape[1]
    y_interp = interp(x_reg)

    # Symétrisation : on veut que le signal soit anti-symétrique autour
    # de la moitié.
    signal_reg = (y_interp + 1.0 - np.roll(y_interp, n_samples // 2)) / 2.0

    return x_reg, signal_reg


class SpatialMTFEstimator(SlantedEdgeMTFEstimator):
    """
    Estime la MTF par dé-interlacement spatial du bord.

    Étapes : dé-interlacement (`interlace_1d`) -> ré-échantillonnage sur
    grille régulière + anti-symétrisation (`regularise_signal`) -> FFT
    du signal 1D ré-échantillonné -> rapport au spectre analytique d'un
    bord idéal (`reference_spectrum`).

    Avantage : un seul signal 1D très sur-échantillonné (autant de
    points que de pixels dans `rect`), donc un bon rapport signal/bruit
    sur la forme du bord elle-même. Inconvénient : sensible à la qualité
    de l'interpolation (`regularise_signal`) et suppose le déplacement
    `dx` parfaitement linéaire sur toute la hauteur de `rect`.
    """

    name = "spatial"

    def _compute_mtf(
        self,
        rect: NDArray[np.float64],
        dx: float,
        theta_rad: float,
        oversample_factor: int,
    ) -> tuple[NDArray[np.float64], NDArray[np.complex128], dict]:
        x_eff, signal = interlace_1d(rect, dx)
        _, signal_reg = regularise_signal(x_eff, signal, rect)

        n_cols = rect.shape[1]
        fft_signal = (
            np.fft.rfft(signal_reg)[1 : n_cols * oversample_factor // 2 : 2]
            / float(rect.size)
        )

        ref_spec = reference_spectrum(n_cols * oversample_factor // 4)

        mtf = fft_signal / ref_spec
        frequency = (1 + 2 * np.arange(ref_spec.size)) / n_cols

        return frequency, mtf, {}


# EOF
