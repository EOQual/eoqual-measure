"""
`prnu` — non-uniformité de réponse (PRNU) d'un capteur matriciel.

Entrée : une ou plusieurs piles de champs uniformes et une obscurité. La
carte de réponse est la moyenne temporelle corrigée de l'obscurité ; sa
dispersion relative, corrigée du bruit temporel résiduel, est la PRNU.
Une surface polynomiale lisse sépare la composante basse fréquence
(vignettage, éclairement non uniforme) de la PRNU haute fréquence. Voir
REFERENCE_TECHNIQUE.md, partie IV, chapitre 5.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.polynomial import legendre
from numpy.typing import ArrayLike, NDArray

from eoqual.measure.radiometry._stack import as_stack
from eoqual.measure.radiometry.dark import DarkFrameResult, _u_sigma, measure_dark_frame
from eoqual.measure.radiometry.decomposition import SpatialDecomposition, decompose_spatial


@dataclass(slots=True)
class PRNUResult:
    """
    Résultat de `measure_prnu` (relatif, sauf mention).

    Attributes
    ----------
    levels :
        Signal moyen corrigé de l'obscurité de chaque pile (DN).
    response_map :
        Réponse relative par pixel (moyenne 1 sur les pixels retenus),
        combinaison des niveaux pondérée par le signal ; NaN pour les pixels
        exclus.
    prnu, u_prnu :
        PRNU totale (écart-type relatif corrigé du bruit), et incertitude.
    prnu_per_level :
        PRNU totale mesurée à chaque niveau.
    prnu_emva :
        PRNU par l'ajustement EMVA σ²_s = PRNU²·μ² sur les niveaux (si au
        moins deux), sinon `None`.
    decomposition :
        Décomposition ligne / colonne / pixel de la PRNU haute fréquence
        (réponse / surface lisse).
    smooth_map :
        Surface lisse (polynôme de Legendre 2D de degré `smooth_degree`) de
        la réponse relative.
    low_frequency :
        Écart-type relatif de la surface lisse (vignettage, éclairement).
    high_frequency, u_high_frequency :
        PRNU haute fréquence : dispersion de réponse / surface lisse,
        corrigée du bruit.
    noise_variance :
        Variance relative du résidu temporel dans la carte de réponse.
    excluded :
        Pixels exclus (masque fourni, ou non finis).
    """

    levels: NDArray[np.float64]
    response_map: NDArray[np.float64]
    prnu: float
    u_prnu: float
    prnu_per_level: NDArray[np.float64]
    prnu_emva: float | None
    decomposition: SpatialDecomposition
    smooth_map: NDArray[np.float64]
    low_frequency: float
    high_frequency: float
    u_high_frequency: float
    noise_variance: float
    excluded: NDArray[np.bool_]


def _legendre_surface(z: NDArray[np.float64], use: NDArray[np.bool_], degree: int) -> NDArray[np.float64]:
    """Ajuste une surface polynomiale (Legendre 2D, degré total ≤ degree) aux pixels `use`."""
    h, w = z.shape
    y = np.linspace(-1, 1, h)
    x = np.linspace(-1, 1, w)
    terms = [(i, j) for i in range(degree + 1) for j in range(degree + 1 - i)]
    ly = [legendre.legval(y, np.eye(degree + 1)[i]) for i in range(degree + 1)]
    lx = [legendre.legval(x, np.eye(degree + 1)[j]) for j in range(degree + 1)]
    basis = np.stack([np.outer(ly[i], lx[j]) for i, j in terms], axis=-1)       # (H, W, n)
    coef, *_ = np.linalg.lstsq(basis[use], z[use], rcond=None)
    return basis @ coef


def measure_prnu(
    flats: ArrayLike | Sequence[ArrayLike],
    dark: ArrayLike | DarkFrameResult,
    *,
    exclude: ArrayLike | None = None,
    smooth_degree: int = 4,
) -> PRNUResult:
    """
    Mesure la PRNU d'un capteur matriciel.

    Parameters
    ----------
    flats :
        Une pile (K ≥ 2, H, W) de champs uniformes, ou une séquence de piles
        à plusieurs niveaux.
    dark :
        Pile d'obscurité (même temps de pose) ou `DarkFrameResult` ; son
        résidu temporel est propagé.
    exclude :
        Masque (H, W) de pixels à exclure (défectueux, par exemple
        `map_defective_pixels(...).any`).
    smooth_degree :
        Degré total de la surface lisse qui sépare basses et hautes
        fréquences (0 : pas de séparation, la surface est la moyenne).
    """
    seq = [flats] if np.ndim(flats) == 3 else list(flats)
    stacks = [as_stack(f, min_frames=2, name="flats") for f in seq]
    dk = dark if isinstance(dark, DarkFrameResult) else measure_dark_frame(as_stack(dark, min_frames=2, name="dark"))
    shape = stacks[0].shape[1:]
    if any(s.shape[1:] != shape for s in stacks) or dk.offset_map.shape != shape:
        raise ValueError("flats et dark doivent avoir la même taille d'image")
    excl = np.zeros(shape, dtype=bool) if exclude is None else np.asarray(exclude, dtype=bool)
    if excl.shape != shape:
        raise ValueError("exclude doit avoir la forme (H, W)")
    v_dark = dk.temporal_variance / dk.n_frames

    levels, prnu_lvl, s2_lvl, u_s2, maps, noise = [], [], [], [], [], []
    for s in stacks:
        k = s.shape[0]
        m = s.mean(axis=0) - dk.offset_map
        v = float(s.var(axis=0, ddof=1)[~excl].mean()) / k + v_dark
        mu = float(m[~excl].mean())
        if mu <= 0:
            raise ValueError("signal moyen nul ou négatif : flat sous l'obscurité ?")
        s2 = float(m[~excl].var() - v)
        n = int((~excl).sum())
        levels.append(mu)
        s2_lvl.append(s2)
        u_s2.append(np.sqrt((4 * max(s2, 0) * v + 2 * v**2) / n))
        prnu_lvl.append(np.sqrt(max(s2, 0.0)) / mu)
        maps.append(m)
        noise.append(v)
    levels = np.asarray(levels)
    order = np.argsort(levels)

    # Carte de réponse : moyenne des cartes relatives pondérée par μ² (variance relative ∝ v/μ²).
    wts = levels**2 / np.asarray(noise)
    rel_maps = [m / mu for m, mu in zip(maps, levels)]
    response = sum(w * r for w, r in zip(wts, rel_maps)) / wts.sum()
    v_rel = float(1.0 / np.sum(levels**2 / np.asarray(noise)))       # variance relative du résidu combiné
    excl = excl | ~np.isfinite(response)
    response = np.where(excl, np.nan, response / np.nanmean(np.where(excl, np.nan, response)))

    use = ~excl
    filled = np.where(excl, 1.0, response)
    s2_tot = float(response[use].var() - v_rel)
    prnu = float(np.sqrt(max(s2_tot, 0.0)))
    u_prnu = _u_sigma(s2_tot, v_rel, int(use.sum()), np.inf)

    if smooth_degree > 0:
        smooth = _legendre_surface(filled, use, smooth_degree)
    else:
        smooth = np.full(shape, float(response[use].mean()))
    lf = float(smooth[use].std() / smooth[use].mean())
    ratio = filled / smooth
    v_hf = float(np.mean(v_rel / smooth[use] ** 2))
    s2_hf = float(ratio[use].var() - v_hf)
    hf = float(np.sqrt(max(s2_hf, 0.0)))
    u_hf = _u_sigma(s2_hf, v_hf, int(use.sum()), np.inf)
    # Décomposition de la PRNU haute fréquence : sur la surface lisse retirée,
    # sinon les profils moyens de lignes et de colonnes contiennent le vignettage.
    decomposition = decompose_spatial(np.where(use, ratio, 1.0), noise_variance=v_hf)

    prnu_emva = None
    if len(stacks) >= 2:
        # σ²_s = PRNU²·μ² : moindres carrés pondérés passant par l'origine.
        x = levels**2
        y = np.asarray(s2_lvl)
        w = 1.0 / np.asarray(u_s2) ** 2
        slope = float(np.sum(w * x * y) / np.sum(w * x * x))
        prnu_emva = float(np.sqrt(max(slope, 0.0)))

    return PRNUResult(
        levels=levels[order],
        response_map=response,
        prnu=prnu,
        u_prnu=u_prnu,
        prnu_per_level=np.asarray(prnu_lvl)[order],
        prnu_emva=prnu_emva,
        decomposition=decomposition,
        smooth_map=smooth,
        low_frequency=lf,
        high_frequency=hf,
        u_high_frequency=u_hf,
        noise_variance=v_rel,
        excluded=excl,
    )
