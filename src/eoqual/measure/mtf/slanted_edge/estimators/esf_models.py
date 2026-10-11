#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fonctions modèles pures pour l'ajustement d'une ESF/LSF -- aucune
logique de fit, aucun accès à l'image. Séparées de `estimators/esf_lsf.py` sur
le même principe que `fourier_utils.py` (utilitaires bas niveau
partagés) : ce sont de simples fonctions mathématiques, interchangeables
comme modèles par `scipy.optimize.curve_fit`, qui n'ont pas de raison
de dépendre du reste du pipeline ni d'être mêlées à l'orchestration.

Chaque fonction a sa dérivée analytique associée (utilisée comme LSF
quand la fonction elle-même modélise une ESF).
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def sigmoid(
    x: NDArray[np.float64],
    a: float,
    b: float,
    x0: float,
    w: float,
) -> NDArray[np.float64]:
    """
    Fonction sigmoïde (paramétrisation tanh) pour modéliser une ESF.

    Utilisée à deux endroits distincts dans `estimators/esf_lsf.py`, avec le
    même modèle mais un rôle différent : par `ParametricEdgeDetector`
    (fit local, par ligne, pour localiser le bord) et par
    `fit_esf_sigmoid` (fit global, sur l'ESF agrégée, pour en tirer la
    LSF analytique).

    Parameters
    ----------
    x :
        Coordonnées.
    a :
        Offset.
    b :
        Amplitude.
    x0 :
        Position du bord (sous-pixel).
    w :
        Largeur de transition.
    """
    return a + b * np.tanh((x - x0) / w)


def sigmoid_derivative(
    x: NDArray[np.float64],
    a: float,
    b: float,
    x0: float,
    w: float,
) -> NDArray[np.float64]:
    """Dérivée analytique de `sigmoid` (= la LSF si `sigmoid` modélise l'ESF)."""
    return (b / w) * (1.0 - np.tanh((x - x0) / w) ** 2)


def gaussian(
    x: NDArray[np.float64],
    amplitude: float,
    x0: float,
    sigma: float,
    offset: float,
) -> NDArray[np.float64]:
    """
    Fonction gaussienne pour modéliser la LSF (dérivée d'une ESF en
    forme de sigmoïde).

    Parameters
    ----------
    x :
        Coordonnées.
    amplitude :
        Amplitude.
    x0 :
        Centre (sous-pixel).
    sigma :
        Écart-type.
    offset :
        Offset.
    """
    return amplitude * np.exp(-((x - x0) ** 2) / (2 * sigma**2)) + offset


# EOF
