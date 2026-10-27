"""
Réponse du capteur (S4, partie) : transmission optique, non-linéarité,
pixels morts et bloqués.

- Transmission T(i, j) : multiplie la lumière avant le capteur (et elle
  seule). Vignettage : T_v = 1 − v·ρ², ρ distance au centre normalisée à 1
  au coin (baisse relative v aux coins). Poussières : anneaux gaussiens
  d'atténuation (ombre d'une poussière défocalisée sur un filtre ou une
  fenêtre), de profondeur d, de rayon R et de largeur R/4.
- Non-linéarité (compression) : e' = e·(1 − β·e/FW), β = β₀ + δβ(i, j),
  appliquée aux électrons collectés ; β₀ commun à tous les pixels, δβ de
  dispersion s_β. La variance est comprimée comme le signal :
  Var(e') ≈ (1 − 2βe/FW)²·e.
- Pixels morts : réponse nulle à la lumière (offset et bruit de lecture
  conservés) ; pixels bloqués : code constant, sans bruit.

Voir REFERENCE_TECHNIQUE.md, partie V, §2.7.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(slots=True)
class Transmission:
    """
    Transmission optique et ses composantes (sans unité, ≤ 1).

    Attributes
    ----------
    total :
        T = vignettage × poussières, forme (H, W).
    vignetting, dust :
        Composantes.
    dust_centers, dust_radii :
        Centres (i, j) et rayons (pixels) des poussières.
    """

    total: NDArray[np.float64]
    vignetting: NDArray[np.float64]
    dust: NDArray[np.float64]
    dust_centers: NDArray[np.float64]
    dust_radii: NDArray[np.float64]


def transmission_map(
    shape: tuple[int, int],
    vignetting: float,
    n_dust: int,
    dust_depth: float,
    dust_radius: tuple[float, float],
    rng: np.random.Generator,
) -> Transmission:
    """Tire la carte de transmission (vignettage quadratique et poussières)."""
    h, w = shape
    i, j = np.mgrid[0:h, 0:w].astype(np.float64)
    ci, cj = (h - 1) / 2, (w - 1) / 2
    rho2 = ((i - ci) ** 2 + (j - cj) ** 2) / (ci**2 + cj**2)
    vig = 1.0 - vignetting * rho2
    dust = np.ones(shape)
    centers = np.zeros((n_dust, 2))
    radii = np.zeros(n_dust)
    for k in range(n_dust):
        r = rng.uniform(*dust_radius)
        y0, x0 = rng.uniform(r, h - r), rng.uniform(r, w - r)
        d = np.hypot(i - y0, j - x0)
        ring = np.exp(-0.5 * ((d - r) / (r / 4)) ** 2)
        dust *= 1.0 - dust_depth * ring
        centers[k] = (y0, x0)
        radii[k] = r
    return Transmission(total=vig * dust, vignetting=vig, dust=dust, dust_centers=centers, dust_radii=radii)


@dataclass(slots=True)
class Defects:
    """
    Pixels défectueux de réponse (les pixels chauds relèvent de S1).

    Attributes
    ----------
    dead :
        Masque des pixels morts (aucune réponse à la lumière).
    stuck :
        Masque des pixels bloqués.
    stuck_value :
        Code de sortie des pixels bloqués (DN), 0 ailleurs.
    """

    dead: NDArray[np.bool_]
    stuck: NDArray[np.bool_]
    stuck_value: NDArray[np.float64]


def defect_maps(shape: tuple[int, int], dead_fraction: float, stuck_fraction: float, max_code: int,
                rng: np.random.Generator) -> Defects:
    """Tire des pixels morts et bloqués disjoints ; code bloqué uniforme dans [0, max_code]."""
    n = shape[0] * shape[1]
    n_dead = int(round(dead_fraction * n))
    n_stuck = int(round(stuck_fraction * n))
    idx = rng.permutation(n)[: n_dead + n_stuck]
    dead = np.zeros(shape, dtype=bool)
    stuck = np.zeros(shape, dtype=bool)
    dead.flat[idx[:n_dead]] = True
    stuck.flat[idx[n_dead:]] = True
    value = np.zeros(shape)
    value.flat[idx[n_dead:]] = rng.integers(0, max_code + 1, size=n_stuck)
    return Defects(dead=dead, stuck=stuck, stuck_value=value)


@dataclass(slots=True)
class ADCTransfer:
    """
    Convertisseur à pas inégaux (S4).

    Le code c est attribué aux tensions de [t_{c−1}, t_c[ (en LSB), avec
    t_{−1} = −∞ et t_{2ⁿ−1} = +∞. Un convertisseur idéal a t_c = c + ½ ; ici
    t_c = c + ½ + δ_c, δ_c gaussien d'écart-type `dnl` (rendu monotone),
    et des codes manquants (largeur nulle).

    Attributes
    ----------
    thresholds :
        Seuils t_c, c = 0 … 2ⁿ − 2 (LSB).
    widths :
        Largeur de chaque code (LSB) ; DNL_c = width_c − 1. Les codes
        extrêmes (0 et 2ⁿ − 1) sont ouverts : largeur NaN.
    missing :
        Codes manquants (largeur nulle).
    """

    thresholds: NDArray[np.float64]
    widths: NDArray[np.float64]
    missing: NDArray[np.int64]

    @property
    def dnl(self) -> NDArray[np.float64]:
        """Non-linéarité différentielle par code (LSB)."""
        return self.widths - 1.0

    @property
    def inl(self) -> NDArray[np.float64]:
        """Non-linéarité intégrale : écart du seuil haut de chaque code au seuil idéal (LSB)."""
        c = np.arange(self.thresholds.size)
        return np.concatenate([self.thresholds - (c + 0.5), [np.nan]])

    def quantize(self, analog: NDArray[np.float64]) -> NDArray[np.float64]:
        """Code de sortie de chaque tension (LSB)."""
        return np.searchsorted(self.thresholds, analog, side="right").astype(np.float64)


def adc_transfer(bit_depth: int, dnl: float, n_missing: int, rng: np.random.Generator) -> ADCTransfer:
    """Tire les seuils d'un convertisseur à pas inégaux et à codes manquants."""
    n_codes = 2**bit_depth
    c = np.arange(n_codes - 1, dtype=np.float64)
    thr = c + 0.5 + (rng.normal(0.0, dnl, size=c.size) if dnl > 0 else 0.0)
    thr = np.maximum.accumulate(thr)                      # monotone
    missing = np.zeros(0, dtype=np.int64)
    if n_missing > 0:
        missing = np.sort(rng.choice(np.arange(2, n_codes - 2), size=n_missing, replace=False))
        for m in missing:                                 # code m de largeur nulle : t_m = t_{m−1}
            thr[m] = thr[m - 1]
        thr = np.maximum.accumulate(thr)
    widths = np.full(n_codes, np.nan)
    widths[1:-1] = np.diff(thr)
    return ADCTransfer(thresholds=thr, widths=widths, missing=missing)
