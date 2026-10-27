"""
Décomposition d'une carte spatiale en composantes ligne, colonne et pixel.

Modèle : M_ij = m + r_i + c_j + p_ij + e_ij, où r, c, p sont les
composantes fixes (écarts-types σ_r, σ_c, σ_p) et e un résidu de bruit
temporel indépendant de variance v connue (par exemple σ²_temp/K pour une
moyenne de K poses).

Estimateurs sans biais (voir REFERENCE_TECHNIQUE.md, partie IV, §2.7) :

    R_i = moyenne de la ligne i,  C_j = moyenne de la colonne j
    P_ij = M_ij − R_i − C_j + moyenne(M)
    s²_pe = var(P) · HW / ((H − 1)(W − 1))        → σ_p² + v
    σ_r² = var(R) − (s²_pe / W)·(H − 1)/H
    σ_c² = var(C) − (s²_pe / H)·(W − 1)/W
    σ_p² = s²_pe − v
    σ_tot² = var(M) − v

Variances spatiales en 1/n : les écarts-types sont ceux des composantes
réalisées de la carte (convention du simulateur, composantes centrées).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(slots=True)
class SpatialDecomposition:
    """
    Décomposition ligne / colonne / pixel d'une carte (unités de la carte).

    Attributes
    ----------
    total, row, col, pixel :
        Écarts-types estimés (racine de la variance, ramenée à 0 si
        l'estimation est négative). `row` et `col` valent `None` si la
        carte n'a qu'une ligne ou qu'une colonne (pas de décomposition
        possible, par exemple un détecteur push broom).
    total_var, row_var, col_var, pixel_var :
        Variances estimées, éventuellement négatives (non tronquées), pour
        les études de biais.
    row_profile, col_profile :
        Moyennes des lignes (H,) et des colonnes (W,), centrées.
    noise_variance :
        Variance du résidu temporel v retirée.
    """

    total: float
    row: float | None
    col: float | None
    pixel: float | None
    total_var: float
    row_var: float | None
    col_var: float | None
    pixel_var: float | None
    row_profile: NDArray[np.float64]
    col_profile: NDArray[np.float64]
    noise_variance: float


def _sigma(var: float | None) -> float | None:
    return None if var is None else float(np.sqrt(max(var, 0.0)))


def decompose_spatial(image: ArrayLike, noise_variance: float = 0.0) -> SpatialDecomposition:
    """
    Décompose une carte 2D en composantes ligne, colonne et pixel.

    Parameters
    ----------
    image :
        Carte (H, W), par exemple une carte d'offset moyennée sur K poses.
    noise_variance :
        Variance v du bruit temporel résiduel de chaque pixel de la carte,
        retirée des composantes pixel et totale (σ²_temp/K pour une
        moyenne de K poses ; 0 si la carte est exacte).
    """
    m = np.asarray(image, dtype=np.float64)
    if m.ndim != 2:
        raise ValueError(f"image doit être 2D, reçu une forme {m.shape}")
    if noise_variance < 0:
        raise ValueError("noise_variance doit être positif ou nul")
    h, w = m.shape
    mean = m.mean()
    rows = m.mean(axis=1)
    cols = m.mean(axis=0)
    total_var = float(m.var() - noise_variance)

    if h == 1 or w == 1:
        # Une seule ligne (ou colonne) : composantes indiscernables.
        return SpatialDecomposition(
            total=_sigma(total_var), row=None, col=None, pixel=None,
            total_var=total_var, row_var=None, col_var=None, pixel_var=None,
            row_profile=rows - mean, col_profile=cols - mean,
            noise_variance=float(noise_variance),
        )

    resid = m - rows[:, None] - cols[None, :] + mean
    s2_pe = float(resid.var() * h * w / ((h - 1) * (w - 1)))
    row_var = float(rows.var() - s2_pe / w * (h - 1) / h)
    col_var = float(cols.var() - s2_pe / h * (w - 1) / w)
    pixel_var = s2_pe - noise_variance
    return SpatialDecomposition(
        total=_sigma(total_var), row=_sigma(row_var), col=_sigma(col_var), pixel=_sigma(pixel_var),
        total_var=total_var, row_var=row_var, col_var=col_var, pixel_var=pixel_var,
        row_profile=rows - mean, col_profile=cols - mean,
        noise_variance=float(noise_variance),
    )
