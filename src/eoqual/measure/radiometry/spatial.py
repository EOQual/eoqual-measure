"""
`spatial_noise` — bruit spatial fixe, colonnage, bandes périodiques.

Entrée : pile matricielle de champs uniformes (K, H, W), zone uniforme
d'une image push broom (M, W), ou zone uniforme d'une image whisk broom
(n_balayages·k, W). Le bruit spatial est la dispersion de la
moyenne temporelle (pile) ou de la ligne moyenne (push broom), corrigée du
bruit temporel résiduel. Voir REFERENCE_TECHNIQUE.md, partie IV, chapitre 11.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from eoqual.measure.radiometry._stack import as_stack
from eoqual.measure.radiometry.dark import DarkFrameResult
from eoqual.measure.radiometry.decomposition import SpatialDecomposition, decompose_spatial


@dataclass(slots=True)
class BandStructure:
    """
    Structure périodique de période k en travers (bandes de whisk broom,
    multiplexage de lecture).

    Attributes
    ----------
    period :
        Période k (détecteurs).
    profile :
        Moyenne de chaque phase j mod k, centrée, forme (k,).
    sigma, variance :
        Écart-type des moyennes de phase corrigé de la dispersion propre
        des détecteurs, et variance brute (éventuellement négative).
    """

    period: int
    profile: NDArray[np.float64]
    sigma: float
    variance: float


@dataclass(slots=True)
class SpatialNoiseResult:
    """
    Résultat de `measure_spatial_noise` (unités : DN).

    Attributes
    ----------
    layout :
        ``"stack"``, ``"pushbroom"`` ou ``"whiskbroom"``.
    n_samples :
        Nombre K de poses (pile) ou M de lignes (push broom).
    mean_signal :
        Niveau moyen (corrigé de l'obscurité si fournie).
    mean_map :
        Carte moyenne (H, W) (pile) ou ligne moyenne (1, W) (push broom),
        corrigée de l'obscurité si fournie.
    temporal_variance :
        Variance temporelle moyenne par pixel σ²_temp.
    noise_variance :
        Variance du résidu temporel retirée de la carte moyenne :
        σ²_temp/K (+ résidu de la carte d'obscurité retirée).
    spatial_variance, spatial_sigma, u_spatial_sigma :
        Bruit spatial fixe total corrigé et son incertitude-type.
    relative :
        spatial_sigma / mean_signal (PRNU apparente si l'obscurité est
        retirée), ou `None` si le signal est nul.
    decomposition :
        Décomposition ligne / colonne / pixel (pile seulement).
    bands :
        Structure de période `period`, si demandée (colonnes pour une pile
        ou un push broom, lignes pour un whisk broom).
    forward_backward, u_forward_backward :
        Whisk broom : moyenne des lignes des balayages retour moins celle
        des balayages aller (DN), et son incertitude ; `None` sinon.
    """

    layout: str
    n_samples: int
    mean_signal: float
    mean_map: NDArray[np.float64]
    temporal_variance: float
    noise_variance: float
    spatial_variance: float
    spatial_sigma: float
    u_spatial_sigma: float
    relative: float | None
    decomposition: SpatialDecomposition | None
    bands: BandStructure | None
    forward_backward: float | None = None
    u_forward_backward: float | None = None


def _u_sigma(s2: float, v: float, n: int, k: int) -> float:
    """Incertitude de √(var(carte) − v) par rapport à la carte réalisée (cf. dark_frame)."""
    var_d = (4.0 * max(s2, 0.0) * v + 2.0 * v**2 * (1.0 + 1.0 / max(k - 1, 1))) / n
    sigma = np.sqrt(max(s2, 0.0))
    return float(np.sqrt(var_d) / (2.0 * sigma)) if sigma > 0 else float(var_d**0.25)


def band_structure(profile: ArrayLike, period: int) -> BandStructure:
    """
    Moyennes par phase j mod k d'un profil (W,) et leur dispersion corrigée.

    Modèle : p_j = m + q_{j mod k} + r_j, r de variance σ_r² ; la variance
    des moyennes de phase vaut σ_q² + (σ_r²/n)·(k − 1)/k, n = W/k, et σ_r²
    s'estime par la variance intra-phase.
    """
    p = np.asarray(profile, dtype=np.float64)
    if p.ndim != 1:
        raise ValueError("profile doit être 1D")
    if period < 2 or p.size < 2 * period:
        raise ValueError("period doit être ≥ 2 et le profil contenir au moins deux périodes")
    n_per = p.size // period
    q = p[: n_per * period].reshape(n_per, period)
    phase = q.mean(axis=0)
    within = float(np.sum((q - phase[None, :]) ** 2) / (period * (n_per - 1)))
    var = float(phase.var() - within / n_per * (period - 1) / period)
    return BandStructure(period=period, profile=phase - phase.mean(),
                         sigma=float(np.sqrt(max(var, 0.0))), variance=var)


def _dark_terms(dark, shape: tuple[int, int]) -> tuple[NDArray[np.float64] | float, float]:
    """Carte (ou niveau) d'obscurité à retirer et variance résiduelle qu'elle apporte."""
    if dark is None:
        return 0.0, 0.0
    if isinstance(dark, DarkFrameResult):
        m = dark.offset_map
        v = dark.temporal_variance / dark.n_frames
    else:
        m = np.asarray(dark, dtype=np.float64)
        v = 0.0
    if m.ndim == 0:
        return float(m), 0.0
    if m.ndim == 1:
        m = m[None, :]
    if m.shape != shape and not (m.shape[0] == 1 and m.shape[1] == shape[1]):
        raise ValueError(f"dark de forme {m.shape} incompatible avec {shape}")
    return m, v


def measure_spatial_noise(
    data: ArrayLike,
    *,
    layout: Literal["stack", "pushbroom", "whiskbroom"] = "stack",
    dark: float | ArrayLike | DarkFrameResult | None = None,
    period: int | None = None,
) -> SpatialNoiseResult:
    """
    Mesure le bruit spatial fixe d'un champ uniforme.

    Parameters
    ----------
    data :
        Pile (K ≥ 2, H, W) (``layout="stack"``) ou zone uniforme push broom
        (M ≥ 2, W) (``layout="pushbroom"``).
    layout :
        Organisation de l'entrée.
    dark :
        Obscurité à retirer : niveau, carte (H, W), profil (W,) ou (1, W),
        ou `DarkFrameResult` (dont le résidu temporel est alors propagé).
        Sans obscurité, le bruit spatial mesuré contient DSNU et PRNU.
    period :
        Période k de bandes à analyser : colonnes (pile, push broom), ou
        lignes (whisk broom : nombre de détecteurs, obligatoire).

    Notes
    -----
    En push broom, la variance temporelle est estimée par la différence de
    lignes successives, var(Δx)/2, insensible à une variation lente de la
    scène le long de la trace.
    """
    if layout == "stack":
        stack = as_stack(data, min_frames=2)
        k = stack.shape[0]
        mean_map = stack.mean(axis=0)
        temporal = float(stack.var(axis=0, ddof=1).mean())
    elif layout == "pushbroom":
        img = np.asarray(data, dtype=np.float64)
        if img.ndim != 2 or img.shape[0] < 2:
            raise ValueError("data doit être une zone push broom (M ≥ 2, W)")
        k = img.shape[0]
        mean_map = img.mean(axis=0, keepdims=True)
        temporal = float(np.mean(np.diff(img, axis=0) ** 2) / 2.0)
    elif layout == "whiskbroom":
        return _whiskbroom(np.asarray(data, dtype=np.float64), dark, period)
    else:
        raise ValueError(f"layout inconnu : {layout!r}")

    dark_map, v_dark = _dark_terms(dark, mean_map.shape)
    mean_map = mean_map - dark_map
    v = temporal / k + v_dark
    n = mean_map.size
    total_var = float(mean_map.var() - v)
    decomposition = decompose_spatial(mean_map, noise_variance=v) if layout == "stack" else None
    mean_signal = float(mean_map.mean())
    sigma = float(np.sqrt(max(total_var, 0.0)))
    bands = band_structure(mean_map.mean(axis=0), period) if period is not None else None
    return SpatialNoiseResult(
        layout=layout,
        n_samples=k,
        mean_signal=mean_signal,
        mean_map=mean_map,
        temporal_variance=temporal,
        noise_variance=v,
        spatial_variance=total_var,
        spatial_sigma=sigma,
        u_spatial_sigma=_u_sigma(float(mean_map.var()) - v, v, n, k),
        relative=None if mean_signal == 0 else sigma / mean_signal,
        decomposition=decomposition,
        bands=bands,
    )


def _whiskbroom(img: NDArray[np.float64], dark, period: int | None) -> SpatialNoiseResult:
    """
    Whisk broom : une ligne par détecteur et par balayage, k lignes par
    balayage, balayages aller et retour alternés. Le bruit temporel se
    mesure le long d'une ligne (pixels successifs d'un même détecteur), le
    bruit spatial sur le profil des moyennes de lignes.
    """
    if period is None or period < 1:
        raise ValueError("whisk broom : `period` (nombre de détecteurs k) est obligatoire")
    if img.ndim != 2 or img.shape[0] < 4 * period or img.shape[1] < 2:
        raise ValueError("data doit être une zone whisk broom (au moins 4 balayages de k lignes, W ≥ 2)")
    k = period
    rows, w = img.shape
    n_scans = rows // k
    img = img[: n_scans * k]
    profile = img.mean(axis=1)                                       # (lignes,)
    if dark is not None:
        level = dark.mean_offset if isinstance(dark, DarkFrameResult) else float(np.mean(dark))
        profile = profile - level
    temporal = float(np.mean(np.diff(img, axis=1) ** 2) / 2.0)
    v = temporal / w
    total_var = float(profile.var() - v)
    sigma = float(np.sqrt(max(total_var, 0.0)))
    # Balayages aller / retour.
    backward = (np.arange(n_scans * k) // k) % 2 == 1
    fb = float(profile[backward].mean() - profile[~backward].mean())
    # Incertitude : la dispersion des détecteurs s'annule (mêmes détecteurs
    # dans les deux sens) ; restent le bruit temporel des lignes et
    # l'incertitude sur la moyenne de chaque sens.
    resid = profile.reshape(n_scans, k) - profile.reshape(n_scans, k).mean(axis=0)
    resid = resid - np.where(backward.reshape(n_scans, k), fb / 2, -fb / 2)
    s2 = float(resid.var(ddof=1))
    u_fb = float(np.sqrt(s2 * (1 / backward.sum() + 1 / (~backward).sum())))
    bands = band_structure(profile, k)
    mean_signal = float(profile.mean())
    return SpatialNoiseResult(
        layout="whiskbroom", n_samples=w, mean_signal=mean_signal, mean_map=profile[:, None],
        temporal_variance=temporal, noise_variance=v, spatial_variance=total_var, spatial_sigma=sigma,
        u_spatial_sigma=_u_sigma(total_var, v, profile.size, w),
        relative=None if mean_signal == 0 else sigma / mean_signal,
        decomposition=None, bands=bands, forward_backward=fb, u_forward_backward=u_fb,
    )
