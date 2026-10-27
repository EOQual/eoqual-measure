"""
Scènes synthétiques en photons par pixel (S2).

Une scène « naturelle » peu texturée est le produit d'un niveau moyen,
d'un gradient lent (éclairement variant le long de la fauchée, angle
solaire) et d'une texture fractale (fBm) de spectre 1/f^β :

    Φ(i, j) = Φ₀ · (1 + g_along·u_i + g_across·v_j) · (1 + c · T(i, j))

avec u, v ∈ [−½, ½] les coordonnées réduites le long et en travers de la
trace, T un champ `power_law_noise` d'écart-type 1 et c le contraste
relatif de la texture. Voir REFERENCE_TECHNIQUE.md, partie V, §2.3.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from eoqual.measure.synthetic.patterns import power_law_noise


@dataclass(slots=True)
class Scene:
    """
    Scène synthétique et ses composantes.

    Attributes
    ----------
    photons :
        Photons moyens par pixel, forme (M, W), strictement positifs.
    mean :
        Niveau moyen Φ₀ demandé.
    illumination :
        Facteur lent 1 + g_along·u + g_across·v, forme (M, W).
    texture :
        Champ fractal T (écart-type 1), forme (M, W).
    contrast, beta :
        Contraste relatif c et exposant spectral β de la texture.
    """

    photons: NDArray[np.float64]
    mean: float
    illumination: NDArray[np.float64]
    texture: NDArray[np.float64]
    contrast: float
    beta: float


def natural_scene(
    shape: tuple[int, int],
    mean_photons: float,
    rng: np.random.Generator,
    *,
    contrast: float = 0.02,
    beta: float = 3.0,
    gradient_along: float = 0.0,
    gradient_across: float = 0.0,
    psf_sigma: float = 0.0,
    pad: int = 2,
) -> Scene:
    """
    Scène peu texturée (désert, neige, océan) pour les mesures sur scène.

    Parameters
    ----------
    shape :
        (M, W) : lignes le long de la trace, colonnes en travers.
    mean_photons :
        Niveau moyen Φ₀ (photons par pixel et par étage TDI).
    rng :
        Générateur (motif de scène : `pattern_rng(seed, "scene")`).
    contrast :
        Écart-type relatif de la texture (0 : scène uniforme).
    beta :
        Exposant du spectre de puissance de la texture : β ≥ 3 donne une
        texture lisse, dominée par les grandes échelles.
    gradient_along, gradient_across :
        Variation relative totale de l'éclairement sur la hauteur et la
        largeur de la scène (0,1 = 10 % d'un bord à l'autre).
    psf_sigma :
        Écart-type (pixels) d'une PSF optique gaussienne appliquée à la
        texture avant échantillonnage : FTM(f) = exp(−2π²σ²f²), soit 0,17 à
        Nyquist pour σ = 0,6 px. Le bruit du capteur, ajouté après, n'est
        pas filtré. 0 : pas de flou.
    pad :
        La texture est tirée sur un champ `pad` fois plus grand puis
        recadrée, pour qu'elle ne soit pas périodique.

    Notes
    -----
    `contrast` est l'écart-type relatif de la texture *après* flou.
    """
    m, w = shape
    if mean_photons <= 0:
        raise ValueError("mean_photons doit être strictement positif")
    if contrast < 0 or psf_sigma < 0:
        raise ValueError("contrast et psf_sigma doivent être positifs ou nuls")
    if contrast > 0:
        big = power_law_noise((pad * m, pad * w), beta, rng)
        if psf_sigma > 0:
            fy = np.fft.fftfreq(big.shape[0])[:, None]
            fx = np.fft.fftfreq(big.shape[1])[None, :]
            mtf = np.exp(-2.0 * np.pi**2 * psf_sigma**2 * (fx**2 + fy**2))
            big = np.fft.ifft2(np.fft.fft2(big) * mtf).real
        big = big[:m, :w]
        texture = (big - big.mean()) / big.std()
    else:
        texture = np.zeros(shape)
    u = (np.arange(m) + 0.5) / m - 0.5
    v = (np.arange(w) + 0.5) / w - 0.5
    illumination = 1.0 + gradient_along * u[:, None] + gradient_across * v[None, :]
    photons = mean_photons * illumination * (1.0 + contrast * texture)
    if np.any(photons <= 0):
        raise ValueError("scène négative : contraste ou gradient trop forts")
    return Scene(photons=photons, mean=float(mean_photons), illumination=illumination,
                 texture=texture, contrast=float(contrast), beta=float(beta))
