#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilitaires Fourier bas niveau, partagés par l'estimation d'orientation
(`orientation/fourier.py`) et l'estimation de MTF (`estimators/base.py` et
ses implémentations).

Les deux s'appuient sur la même représentation symétrique demi-onde
(voir `FourierOrientationDetector._build_symmetric_representation`) et
sur la même estimation de pente ligne-à-ligne par différence de phase
entre lignes voisines -- ce module évite de dupliquer cette logique.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def reference_spectrum(size: int) -> NDArray[np.float64]:
    """
    Spectre analytique d'un bord idéal (échelon) de hauteur 1, aux
    harmoniques impaires.

    C'est le développement en série de Fourier d'un créneau à 50 % de
    rapport cyclique (voir `FourierOrientationDetector` pour la
    justification de la symétrie demi-onde) : seules les harmoniques
    impaires sont non nulles, avec une amplitude en 1/k.

    Parameters
    ----------
    size :
        Nombre de composantes fréquentielles à retourner.

    Returns
    -------
    NDArray[np.float64]
        Spectre du step aux harmoniques impaires (indices 0, 1, 2, ...
        correspondant à k = 1, 3, 5, ...).
    """
    k = np.arange(size)
    return (-1.0) ** (1 + k) / ((1 + 2 * k) * np.pi)


def estimate_slope_multiharmonic(
    rect: NDArray[np.float64],
    low_freq_ratio: float = 0.2,
    block_gap: int = 2,
) -> tuple[float, float]:
    """
    Estime le déplacement moyen ligne-à-ligne (`dx`) et l'angle `theta`
    du bord à partir de la différence de phase entre lignes voisines.

    `rect` doit déjà posséder la symétrie demi-onde requise (voir
    `FourierOrientationDetector._build_symmetric_representation`) :
    cette fonction ne construit pas cette représentation elle-même, elle
    suppose qu'elle lui est fournie.

    Le calcul moyenne l'information de plusieurs harmoniques impaires
    basses (fraction `low_freq_ratio`, les moins bruitées) et de
    nombreuses paires de lignes séparées de `block_gap`, ce qui le rend
    statistiquement plus robuste qu'une estimation reposant sur un seul
    harmonique -- au prix de ne donner qu'un déplacement global, sans
    position par ligne (voir `FourierOrientationDetector` pour
    l'alternative fondée sur la phase de l'harmonique k=1 seule, qui
    fournit elle des positions par ligne).

    Parameters
    ----------
    rect :
        Zone d'image contenant le bord incliné, déjà symétrisée.
    low_freq_ratio :
        Portion des harmoniques impaires les plus basses utilisées pour
        la moyenne.
    block_gap :
        Distance (en lignes) entre les deux lignes comparées.

    Returns
    -------
    tuple[float, float]
        `(dx, theta)` : déplacement moyen en pixels (`dx`) et angle
        d'inclinaison en radians (`theta = arctan(dx)`).
    """
    if block_gap < 1:
        raise ValueError("'block_gap' must be >= 1.")
    if not 0.0 < low_freq_ratio <= 1.0:
        raise ValueError("'low_freq_ratio' must belong to (0, 1].")

    # 1. Transformée de Fourier le long des lignes (axis=1).
    tf_rect = np.fft.rfft(rect, axis=1)

    # 2. On ne garde que les fréquences impaires (les autres sont
    #    nulles par construction -- voir le docstring de la classe
    #    FourierOrientationDetector) et un nombre pair de lignes.
    tf_rect = tf_rect[: 2 * int(rect.shape[0] / 2), 1::2]

    # 3. Découpage en deux ensembles de lignes séparées de block_gap.
    tf_a = tf_rect[:-block_gap]
    tf_b = tf_rect[block_gap:]

    # 4. Différence de phase -> estimation du décalage inter-ligne.
    n_couples = tf_a.shape[0]
    n_freq = tf_a.shape[1]
    phase_diff = np.angle(tf_a / tf_b)
    k = 2 * np.arange(n_freq) + 1
    dx_per_freq = (
        phase_diff.sum(axis=0) * rect.shape[1]
        / (2.0 * np.pi * block_gap * n_couples * k)
    )

    # 5. On ne garde que les plus basses fréquences (moins bruitées).
    n_low = max(1, int(low_freq_ratio * dx_per_freq.size))
    dx = float(dx_per_freq[:n_low].mean())
    theta = float(np.arctan(dx))

    return dx, theta


# EOF
