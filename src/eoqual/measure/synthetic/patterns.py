"""
Générateurs aléatoires communs au simulateur.

- `pattern_rng` : générateur dédié à un motif fixe, stable quand on ajoute
  d'autres motifs (un motif = une étiquette, pas une position dans une
  suite de tirages).
- `structured_map` : carte décomposée en composantes pixel, ligne et
  colonne (DSNU, PRNU).
- `power_law_noise` : bruit de spectre de puissance en 1/f^β, en 1D ou 2D
  (β = 0 blanc, 1 rose, 2 brownien), par synthèse spectrale.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


def pattern_rng(seed: int, tag: str) -> np.random.Generator:
    """
    Générateur pour le motif fixe `tag` d'un capteur de graine `seed`.

    Chaque motif a son propre flux : ajouter un motif au simulateur ne
    modifie pas les motifs existants pour une même graine.
    """
    return np.random.default_rng([int(seed), zlib.crc32(tag.encode())])


@dataclass(slots=True)
class StructuredMap:
    """
    Carte décomposée en composantes centrées (moyenne spatiale nulle).

    Attributes
    ----------
    total :
        Somme des trois composantes, forme (H, W).
    pixel :
        Composante pixel à pixel, forme (H, W).
    row :
        Composante par ligne, forme (H,), diffusée sur les colonnes.
    col :
        Composante par colonne, forme (W,), diffusée sur les lignes.
    """

    total: NDArray[np.float64]
    pixel: NDArray[np.float64]
    row: NDArray[np.float64]
    col: NDArray[np.float64]


def structured_map(
    shape: tuple[int, int],
    sigma_pixel: float,
    sigma_row: float,
    sigma_col: float,
    rng: np.random.Generator,
) -> StructuredMap:
    """
    Tire une carte gaussienne pixel + ligne + colonne.

    Chaque composante est centrée, pour que la moyenne spatiale de la carte
    soit exactement nulle : l'offset ou le gain moyen du capteur reste
    celui demandé.
    """
    h, w = shape
    pixel = rng.normal(0.0, sigma_pixel, size=(h, w)) if sigma_pixel > 0 else np.zeros((h, w))
    row = rng.normal(0.0, sigma_row, size=h) if sigma_row > 0 else np.zeros(h)
    col = rng.normal(0.0, sigma_col, size=w) if sigma_col > 0 else np.zeros(w)
    pixel = pixel - pixel.mean()
    row = row - row.mean()
    col = col - col.mean()
    total = pixel + row[:, None] + col[None, :]
    return StructuredMap(total=total, pixel=pixel, row=row, col=col)


def power_law_noise(
    shape: int | tuple[int, ...],
    beta: float,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """
    Bruit gaussien de moyenne nulle, d'écart-type 1, de spectre P(f) ∝ 1/f^β.

    Synthèse spectrale : bruit blanc gaussien filtré dans le domaine de
    Fourier par |f|^(−β/2), composante continue annulée (cf. Downey,
    *Think DSP*, chap. 4). En 2D, f est la fréquence radiale.

    Parameters
    ----------
    shape :
        Longueur (1D) ou forme (2D, ou plus).
    beta :
        Exposant du spectre de puissance (0 blanc, 1 rose, 2 brownien).
    rng :
        Générateur aléatoire.
    """
    shape = (shape,) if isinstance(shape, int) else tuple(shape)
    white = rng.standard_normal(shape)
    if beta == 0:
        return (white - white.mean()) / white.std()
    spectrum = np.fft.fftn(white)
    grids = np.meshgrid(*[np.fft.fftfreq(n) for n in shape], indexing="ij")
    f = np.sqrt(sum(g**2 for g in grids))
    f.flat[0] = 1.0
    spectrum *= f ** (-beta / 2.0)
    spectrum.flat[0] = 0.0
    noise = np.fft.ifftn(spectrum).real
    return noise / noise.std()
