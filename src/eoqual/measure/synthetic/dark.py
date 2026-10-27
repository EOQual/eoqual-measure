"""
Courant d'obscurité (S1) : carte par pixel, pixels chauds, loi en
température, lueur d'amplificateur.

- Carte : I_d(i, j) = I_d · g(i, j), g de moyenne 1. La dispersion d'un
  pixel à l'autre suit une loi log-normale (asymétrique, à queue vers les
  forts courants, comme les cartes d'obscurité réelles) ; une fraction de
  pixels « chauds » reçoit un facteur log-uniforme dans une plage donnée.
- Température : I_d(T) = I_d(T_réf) · (T/T_réf)³ · exp(−(E_g/2k)(1/T − 1/T_réf)),
  courant de génération dans la zone de déplétion ; pour E_g = 1,12 eV
  (silicium), il double tous les 8,3 K vers 20 °C.
- Lueur d'amplificateur : charge additionnelle linéaire en temps de pose,
  décroissant exponentiellement avec la distance au coin (0, 0) où est
  placé l'amplificateur.

Voir REFERENCE_TECHNIQUE.md, partie V, §2.6.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

BOLTZMANN_EV = 8.617333262e-5          # eV/K
ZERO_CELSIUS = 273.15


def temperature_factor(temperature_c: float, reference_c: float, bandgap_ev: float) -> float:
    """Facteur I_d(T)/I_d(T_réf) de la loi T³·exp(−E_g/2kT)."""
    t = temperature_c + ZERO_CELSIUS
    t0 = reference_c + ZERO_CELSIUS
    if t <= 0 or t0 <= 0:
        raise ValueError("températures en °C supérieures au zéro absolu")
    return float((t / t0) ** 3 * np.exp(-bandgap_ev / (2 * BOLTZMANN_EV) * (1 / t - 1 / t0)))


def doubling_temperature(temperature_c: float, bandgap_ev: float) -> float:
    """
    Écart de température ΔT (K) tel que I_d(T + ΔT) = 2·I_d(T), solution
    exacte de la loi T³·exp(−E_g/2kT) (l'approximation au premier ordre,
    ln 2 / (d ln I/dT), le sous-estime d'environ 2 % vers 20 °C).
    """
    from scipy.optimize import brentq

    return float(brentq(lambda dt: temperature_factor(temperature_c + dt, temperature_c, bandgap_ev) - 2.0,
                        1e-3, 100.0))


@dataclass(slots=True)
class DarkCurrentMap:
    """
    Carte de courant d'obscurité à la température de référence.

    Attributes
    ----------
    current :
        Courant par pixel (e⁻/s), forme (H, W), pixels chauds compris.
    relative :
        Facteur g par pixel hors pixels chauds (moyenne 1).
    hot :
        Masque des pixels chauds.
    hot_factor :
        Facteur multiplicatif des pixels chauds (1 ailleurs).
    glow :
        Lueur d'amplificateur (e⁻/s), forme (H, W), nulle si absente.
    """

    current: NDArray[np.float64]
    relative: NDArray[np.float64]
    hot: NDArray[np.bool_]
    hot_factor: NDArray[np.float64]
    glow: NDArray[np.float64]


def dark_current_map(
    shape: tuple[int, int],
    mean_current: float,
    nonuniformity: float,
    hot_fraction: float,
    hot_range: tuple[float, float],
    glow_rate: float,
    glow_scale: float,
    rng_map: np.random.Generator,
    rng_hot: np.random.Generator,
) -> DarkCurrentMap:
    """
    Tire la carte de courant d'obscurité.

    Parameters
    ----------
    mean_current :
        Courant moyen des pixels ordinaires I_d (e⁻/s).
    nonuniformity :
        Écart-type de ln g (log-normale) ; la DSNU relative du courant vaut
        √(exp(s²) − 1) ≈ s pour s petit.
    hot_fraction, hot_range :
        Fraction de pixels chauds et plage (min, max) de leur facteur,
        tiré log-uniformément.
    glow_rate, glow_scale :
        Lueur au coin (0, 0) (e⁻/s) et longueur de décroissance (pixels).
    """
    h, w = shape
    if nonuniformity > 0:
        s = nonuniformity
        rel = np.exp(rng_map.normal(-0.5 * s**2, s, size=shape))     # moyenne 1
    else:
        rel = np.ones(shape)
    n_hot = int(round(hot_fraction * h * w))
    hot = np.zeros(shape, dtype=bool)
    factor = np.ones(shape)
    if n_hot > 0:
        idx = rng_hot.choice(h * w, size=n_hot, replace=False)
        lo, hi = hot_range
        hot.flat[idx] = True
        factor.flat[idx] = np.exp(rng_hot.uniform(np.log(lo), np.log(hi), size=n_hot))
    if glow_rate > 0:
        i, j = np.mgrid[0:h, 0:w]
        glow = glow_rate * np.exp(-np.hypot(i, j) / glow_scale)
    else:
        glow = np.zeros(shape)
    current = mean_current * rel * factor
    return DarkCurrentMap(current=current, relative=rel, hot=hot, hot_factor=factor, glow=glow)
