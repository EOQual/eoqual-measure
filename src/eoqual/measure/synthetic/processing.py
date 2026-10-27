"""
Traitements à fonction de transfert connue (S3).

Un traitement linéaire de l'image (rééchantillonnage, dématriçage,
débruitage, compression à faible taux…) corrèle le bruit : un bruit blanc
de variance σ² filtré par un noyau h a pour spectre σ²·|H(f)|². Ce module
applique un noyau par convolution **circulaire** (FFT), pour que la
relation soit exacte, et fournit |H|² sur la grille de fréquences ; c'est
la vérité qui valide `noise_power_spectrum`. Voir REFERENCE_TECHNIQUE.md, partie V, §2.5.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _kernel_otf(kernel: NDArray[np.float64], shape: tuple[int, int]) -> NDArray[np.complex128]:
    """Fonction de transfert du noyau (centré) sur la grille FFT de `shape`."""
    kh, kw = kernel.shape
    if kh > shape[0] or kw > shape[1]:
        raise ValueError("noyau plus grand que l'image")
    pad = np.zeros(shape)
    pad[:kh, :kw] = kernel
    pad = np.roll(pad, (-(kh // 2), -(kw // 2)), axis=(0, 1))   # centre du noyau en (0, 0)
    return np.fft.fft2(pad)


def apply_kernel(image: ArrayLike, kernel: ArrayLike) -> NDArray[np.float64]:
    """
    Convolution circulaire de `image` (H, W) — ou de chaque image d'une pile
    (K, H, W) — par `kernel` (impair de préférence, centré).
    """
    x = np.asarray(image, dtype=np.float64)
    k = np.atleast_2d(np.asarray(kernel, dtype=np.float64))
    shape = x.shape[-2:]
    otf = _kernel_otf(k, shape)
    return np.fft.ifft2(np.fft.fft2(x, axes=(-2, -1)) * otf, axes=(-2, -1)).real


def kernel_power_transfer(kernel: ArrayLike, shape: tuple[int, int]) -> NDArray[np.float64]:
    """
    |H(f)|² du noyau sur la grille FFT de `shape` (ordre `np.fft.fftfreq`).

    NPS attendu d'un bruit blanc σ² filtré : σ²·|H|². Sa moyenne sur la
    grille vaut Σh² (variance transmise).
    """
    return np.abs(_kernel_otf(np.atleast_2d(np.asarray(kernel, dtype=np.float64)), shape)) ** 2


def gaussian_kernel(sigma: float, radius: int | None = None) -> NDArray[np.float64]:
    """Noyau gaussien 2D normé (Σh = 1), de rayon 4σ par défaut."""
    r = int(np.ceil(4 * sigma)) if radius is None else radius
    x = np.arange(-r, r + 1)
    g = np.exp(-(x**2) / (2 * sigma**2))
    k = np.outer(g, g)
    return k / k.sum()
