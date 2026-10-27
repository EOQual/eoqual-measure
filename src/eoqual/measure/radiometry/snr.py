"""
`snr_model` — SNR aux niveaux spécifiés, à partir d'un modèle de bruit.

SNR(μ) = μ / σ(μ), σ²(μ) = a·μ + b, avec l'incertitude propagée de la
covariance de (a, b). Fonction utilitaire : elle s'applique au résultat
de toute mesure qui fournit a, b et leur covariance (`measure_ptc`,
`measure_pushbroom_noise`). Voir REFERENCE_TECHNIQUE.md, partie IV, chapitre 10.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray


class NoiseModel(Protocol):
    """Tout objet portant un modèle σ² = a·μ + b et sa covariance."""

    a: float
    b: float
    cov: NDArray[np.float64]


@dataclass(slots=True)
class SNRResult:
    """
    SNR aux niveaux demandés.

    Attributes
    ----------
    levels :
        Signaux μ (DN, corrigés de l'obscurité).
    sigma, u_sigma :
        Bruit temporel σ(μ) (DN) et son incertitude-type.
    snr, u_snr :
        Rapport signal sur bruit μ/σ(μ) et son incertitude-type.
    """

    levels: NDArray[np.float64]
    sigma: NDArray[np.float64]
    u_sigma: NDArray[np.float64]
    snr: NDArray[np.float64]
    u_snr: NDArray[np.float64]


def snr_at_levels(model: NoiseModel, levels: ArrayLike, *, gain: float | None = None) -> SNRResult:
    """
    SNR et incertitude aux niveaux `levels`.

    Parameters
    ----------
    model :
        Résultat portant `a`, `b` et `cov` : matrice 2 × 2 sur (a, b), ou
        1 × 1 sur a seul si b a été imposé (b est alors tenu pour exact).
    levels :
        Signaux μ en DN, ou en e⁻ si `gain` est fourni.
    gain :
        Gain K (e⁻/DN) pour exprimer `levels` en électrons ; ses
        incertitudes ne sont pas propagées.
    """
    mu = np.atleast_1d(np.asarray(levels, dtype=np.float64))
    if gain is not None:
        mu = mu / gain
    if np.any(mu <= 0):
        raise ValueError("les niveaux doivent être strictement positifs")
    cov = np.atleast_2d(np.asarray(model.cov, dtype=np.float64))
    if cov.shape == (1, 1):
        cov = np.array([[cov[0, 0], 0.0], [0.0, 0.0]])
    elif cov.shape != (2, 2):
        raise ValueError("cov doit être 1 × 1 ou 2 × 2")
    var = model.a * mu + model.b
    if np.any(var <= 0):
        raise ValueError("modèle de bruit négatif à certains niveaux")
    sigma = np.sqrt(var)
    # ∂σ/∂a = μ/(2σ), ∂σ/∂b = 1/(2σ)
    jac = np.stack([mu / (2 * sigma), 1.0 / (2 * sigma)], axis=1)
    u_sigma = np.sqrt(np.einsum("ni,ij,nj->n", jac, cov, jac))
    snr = mu / sigma
    u_snr = snr * u_sigma / sigma
    return SNRResult(levels=mu, sigma=sigma, u_sigma=u_sigma, snr=snr, u_snr=u_snr)
