#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
Fonctions communes de simulation de bruit, utilisées pour valider les estimateurs de `eoqual.measure.noise`
(injection d'un bruit de variance connue, éventuellement dépendant du
signal, sur une image propre). Voir `examples/noise_intercomparison.py`.
"""

from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray


def generate_clean_image(shape: Tuple[int, int] = (512, 512)) -> NDArray[np.float32]:
    """
    Génère une image synthétique "propre" (sans bruit), avec un relief basse
    fréquence, utilisée comme signal de référence quand aucune image source
    n'est fournie.

    Parameters
    ----------
    shape : Tuple[int, int]
        Dimensions (h, w) de l'image générée.

    Returns
    -------
    NDArray[np.float32]
        Image 2D sans bruit.
    """

    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w]
    image = (
        120.0
        + 40.0 * np.sin(xx / 60.0) * np.cos(yy / 80.0)
        + 0.03 * xx
        + 0.01 * yy
    )
    return image.astype(np.float32)


def add_noise(
    image: NDArray[np.float32],
    a: float,
    b: float,
    seed: int = 0,
    bit_depth: Optional[int] = None,
) -> NDArray[np.float32]:
    """
    Ajoute un bruit gaussien de variance σ²(I) = a·I + b à une image
    (a=0 pour un bruit homoscédastique de variance constante b).

    Si `bit_depth` est fourni, l'image bruitée est en plus quantifiée et
    écrêtée sur la plage [0, 2^bit_depth - 1], reproduisant les effets non
    linéaires réels d'un capteur aux extrêmes de luminance : écrêtage/
    saturation en haute luminance (la variance mesurée y diminue
    artificiellement car les valeurs saturées sont plafonnées à une
    constante) et quantification/plancher de bruit en basse luminance (le
    pas de quantification devient non négligeable devant l'écart-type du
    bruit). Simuler ces effets permet d'étudier explicitement l'écart à la
    linéarité mentionné par l'utilisateur, plutôt que de le supposer.

    Parameters
    ----------
    image : NDArray[np.float32]
        Image 2D d'entrée, sans bruit.
    a : float
        Pente du modèle de bruit (composante de type Poisson, dépendante du signal).
    b : float
        Ordonnée à l'origine du modèle de bruit (bruit de lecture, indépendant du signal).
    seed : int
        Graine du générateur aléatoire, pour reproductibilité.
    bit_depth : int ou None
        Si fourni (8 ou 16 typiquement), quantifie et écrête l'image sur
        [0, 2^bit_depth - 1] pour simuler un capteur réel. Si None, l'image
        bruitée reste en float32 non écrêtée (cas idéal).

    Returns
    -------
    NDArray[np.float32]
        Image bruitée, même forme que `image`. Valeurs entières si
        `bit_depth` est fourni (mais toujours stockées en float32).
    """

    rng = np.random.default_rng(seed)
    local_variance = np.clip(a * image.astype(np.float64) + b, a_min=1e-6, a_max=None)
    noise = rng.normal(0.0, np.sqrt(local_variance))
    noisy = image.astype(np.float64) + noise

    if bit_depth is not None:
        max_val = float(2 ** bit_depth - 1)
        noisy = np.round(np.clip(noisy, 0.0, max_val))

    return noisy.astype(np.float32)

# EOF
