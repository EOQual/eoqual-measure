"""
`equalization_noise` — bruit résiduel d'égalisation d'un push broom.

Entrée : zone uniforme d'une image push broom **égalisée** (corrigée des
offsets et des gains de chaque détecteur). Les statistiques se calculent
le long des lignes, sur la ligne moyenne : résidu de bruit spatial fixe
(FPN), séparé en basses fréquences (BF : vignettage mal corrigé, gains à
grande échelle) et hautes fréquences (HF : résidu détecteur à détecteur),
chaque composante corrigée de sa part de bruit temporel moyenné. Voir
REFERENCE_TECHNIQUE.md, partie IV, chapitre 11.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.ndimage import gaussian_filter1d


@dataclass(slots=True)
class EqualizationNoiseResult:
    """
    Résultat de `measure_equalization_noise` (unités : DN).

    Attributes
    ----------
    n_lines :
        Nombre M de lignes moyennées.
    mean_level :
        Niveau moyen de la zone.
    profile :
        Ligne moyenne centrée (W,).
    temporal_variance :
        Variance temporelle par détecteur σ²_temp (différences de lignes).
    noise_variance :
        Résidu temporel de la ligne moyenne, σ²_temp/M.
    fpn, u_fpn, fpn_relative :
        FPN résiduel total (écart-type corrigé du résidu temporel), son
        incertitude-type, et rapporté au niveau moyen.
    lf_profile, hf_profile :
        Composantes BF (lissage gaussien) et HF (reste) du profil.
    lf, hf, u_lf, u_hf :
        Écarts-types BF et HF corrigés de la part du résidu temporel que
        chaque filtre transmet, et leurs incertitudes-types.
    lf_noise_fraction, hf_noise_fraction :
        Part de la variance d'un bruit blanc transmise par le filtre BF et
        par le filtre HF (somme ≤ 1, le reste étant la covariance négative
        des deux sorties).
    cutoff :
        Fréquence de coupure BF/HF (cycles/détecteur) : celle où le gain du
        filtre gaussien vaut 1/2.
    frequency, spectrum :
        Fréquences (cycles/détecteur) et densité spectrale unilatérale du
        profil, normalisée pour que Σ spectrum·Δf = variance du profil.
    """

    n_lines: int
    mean_level: float
    profile: NDArray[np.float64]
    temporal_variance: float
    noise_variance: float
    fpn: float
    u_fpn: float
    fpn_relative: float | None
    lf_profile: NDArray[np.float64]
    hf_profile: NDArray[np.float64]
    lf: float
    hf: float
    u_lf: float
    u_hf: float
    lf_noise_fraction: float
    hf_noise_fraction: float
    cutoff: float
    frequency: NDArray[np.float64]
    spectrum: NDArray[np.float64]


def _u_component(s2: float, v: float, g: NDArray[np.float64], n: int, m: int) -> float:
    """
    Incertitude-type de √S, S estimé par var(g∗profil) − v·Σg², pour une
    composante filtrée par la réponse impulsionnelle g (g = δ : profil
    entier). Le résidu temporel ε du profil est blanc, de variance v.

    - terme croisé motif × bruit : 4·S·v/n (le motif filtré par g est
      inchangé, à la transmission près, s'il est dans la bande de g) ;
    - variance propre du bruit filtré, corrélé : 2v²·Σ_k (g⋆g)_k² / n ;
    - erreur sur v̂ (estimé sur M·n échantillons) : 2(v·Σg²)²/(M·n).
    """
    acf = np.correlate(g, g, mode="full")
    f = float(np.sum(g**2))
    var_d = (4.0 * max(s2, 0.0) * v + 2.0 * v**2 * float(np.sum(acf**2)) + 2.0 * (v * f) ** 2 / m) / n
    sigma = np.sqrt(max(s2, 0.0))
    return float(np.sqrt(var_d) / (2.0 * sigma)) if sigma > 0 else float(var_d**0.25)


def _gaussian_sigma_for_cutoff(cutoff: float) -> float:
    """Écart-type (détecteurs) du lissage gaussien dont le gain vaut 1/2 à `cutoff`."""
    # Gain exp(−2π²σ²f²) = 1/2  ⇔  σ = √(ln 2 / 2) / (π f)
    return float(np.sqrt(np.log(2.0) / 2.0) / (np.pi * cutoff))


def profile_spectrum(profile: ArrayLike) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Densité spectrale unilatérale d'un profil centré (périodogramme).

    Normalisée pour que Σ S(f)·Δf (f > 0) égale la variance (1/n) du profil.
    """
    p = np.asarray(profile, dtype=np.float64)
    p = p - p.mean()
    n = p.size
    power = np.abs(np.fft.rfft(p)) ** 2 / n**2       # Σ_tous |X|²/n² = var
    one_sided = power.copy()
    one_sided[1:] *= 2.0
    if n % 2 == 0:
        one_sided[-1] /= 2.0                          # Nyquist compté une fois
    freq = np.fft.rfftfreq(n)
    df = 1.0 / n
    return freq[1:], one_sided[1:] / df


def measure_equalization_noise(
    image: ArrayLike,
    *,
    cutoff: float = 1.0 / 32.0,
) -> EqualizationNoiseResult:
    """
    Mesure le bruit résiduel d'égalisation sur une zone uniforme égalisée.

    Parameters
    ----------
    image :
        Zone uniforme égalisée (M ≥ 2, W) : lignes le long de la trace,
        détecteurs en travers.
    cutoff :
        Fréquence de séparation BF/HF en cycles par détecteur (défaut 1/32 :
        structures de plus de ~32 détecteurs en BF).
    """
    img = np.asarray(image, dtype=np.float64)
    if img.ndim != 2 or img.shape[0] < 2 or img.shape[1] < 8:
        raise ValueError("image doit être une zone (M ≥ 2, W ≥ 8)")
    if not 0 < cutoff < 0.5:
        raise ValueError("cutoff doit être dans ]0, 0,5[ cycles/détecteur")
    m, w = img.shape
    line = img.mean(axis=0)
    level = float(line.mean())
    profile = line - level
    temporal = float(np.mean(np.diff(img, axis=0) ** 2) / 2.0)
    v = temporal / m

    # Filtre BF gaussien (bords en miroir) et parts du bruit blanc transmises.
    sigma = _gaussian_sigma_for_cutoff(cutoff)
    lf_profile = gaussian_filter1d(profile, sigma, mode="reflect")
    hf_profile = profile - lf_profile
    # Réponse impulsionnelle au centre : Σh² (BF) et Σ(δ − h)² (HF).
    delta = np.zeros(w)
    delta[w // 2] = 1.0
    h = gaussian_filter1d(delta, sigma, mode="reflect")
    lf_frac = float(np.sum(h**2))
    hf_frac = float(np.sum((delta - h) ** 2))

    total_var = float(profile.var() - v)
    lf_var = float(lf_profile.var() - v * lf_frac)
    hf_var = float(hf_profile.var() - v * hf_frac)
    fpn = float(np.sqrt(max(total_var, 0.0)))
    u_fpn = _u_component(total_var, v, np.array([1.0]), w, m)
    # Réponses impulsionnelles tronquées à leur support utile.
    half = min(w // 2, int(np.ceil(6 * sigma)))
    g_lf = h[w // 2 - half: w // 2 + half + 1]
    g_hf = (delta - h)[w // 2 - half: w // 2 + half + 1]
    u_lf = _u_component(lf_var, v, g_lf, w, m)
    u_hf = _u_component(hf_var, v, g_hf, w, m)
    freq, spec = profile_spectrum(profile)
    return EqualizationNoiseResult(
        n_lines=m,
        mean_level=level,
        profile=profile,
        temporal_variance=temporal,
        noise_variance=v,
        fpn=fpn,
        u_fpn=u_fpn,
        fpn_relative=None if level == 0 else fpn / level,
        lf_profile=lf_profile,
        hf_profile=hf_profile,
        lf=float(np.sqrt(max(lf_var, 0.0))),
        hf=float(np.sqrt(max(hf_var, 0.0))),
        u_lf=u_lf,
        u_hf=u_hf,
        lf_noise_fraction=lf_frac,
        hf_noise_fraction=hf_frac,
        cutoff=cutoff,
        frequency=freq,
        spectrum=spec,
    )
