"""
`photon_transfer_curve` — gain, bruit de lecture, saturation, linéarité.

Entrée : une paire (ou pile) de champs uniformes par niveau d'éclairement,
et une pile d'obscurité. À chaque niveau, `measure_temporal_noise` donne
(μ, σ²) ; on ajuste σ²(μ) = a·μ + b sur la plage linéaire, d'où le gain
K = 1/a (e⁻/DN) et le bruit de lecture. Voir REFERENCE_TECHNIQUE.md, partie IV, chapitre 3.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from eoqual.measure.radiometry._stack import as_stack
from eoqual.measure.radiometry.dark import DarkFrameResult, measure_dark_frame
from eoqual.measure.radiometry.temporal import measure_temporal_noise

QUANTIZATION_VARIANCE = 1.0 / 12.0


@dataclass(slots=True)
class PTCResult:
    """
    Résultat de `measure_ptc`.

    Attributes
    ----------
    signal :
        Signal moyen corrigé de l'obscurité par niveau (DN), trié ; le
        premier point est l'obscurité (0) si ``include_dark``.
    variance, u_variance :
        Variance temporelle par niveau (DN²) et son incertitude-type.
    saturated_fraction :
        Fraction d'échantillons au code de saturation, par niveau (0 si
        ``saturation_dn`` n'est pas fourni).
    used :
        Niveaux retenus pour l'ajustement.
    a, b, cov :
        Pente (1/K), ordonnée à l'origine (DN²) et matrice de covariance
        (2 × 2) de l'ajustement pondéré σ² = a·μ + b.
    u_a, u_b :
        Incertitudes-types de a et b.
    reduced_chi2 :
        χ² réduit de l'ajustement : proche de 1 si le modèle affine et les
        incertitudes sont corrects.
    gain, u_gain :
        Gain de conversion K = 1/a (e⁻/DN) et son incertitude-type.
    read_noise_dn, read_noise_e, u_read_noise_e :
        Bruit à signal nul hors quantification, √(b − 1/12) (ou √b), en DN
        et en e⁻, et l'incertitude en e⁻.
    saturation_signal_dn, saturation_e :
        Signal au maximum de variance (point de saturation au sens EMVA
        1288), en DN et en e⁻ ; `None` si la saturation n'est pas atteinte
        (variance maximale au dernier niveau).
    linearity_error :
        Erreur de linéarité (%) si ``exposures`` est fourni :
        (max δ − min δ)/2, δ écart relatif à la droite μ = s·x + c ajustée
        sur les niveaux retenus ; sinon `None`.
    """

    signal: NDArray[np.float64]
    variance: NDArray[np.float64]
    u_variance: NDArray[np.float64]
    saturated_fraction: NDArray[np.float64]
    used: NDArray[np.bool_]
    a: float
    b: float
    cov: NDArray[np.float64]
    u_a: float
    u_b: float
    reduced_chi2: float
    gain: float
    u_gain: float
    read_noise_dn: float
    read_noise_e: float
    u_read_noise_e: float
    saturation_signal_dn: float | None
    saturation_e: float | None
    linearity_error: float | None


def _weighted_line(x, y, u):
    """Ajustement pondéré y = a·x + b ; renvoie a, b, cov, χ² réduit."""
    w = 1.0 / u**2
    design = np.column_stack([x, np.ones_like(x)])
    normal = design.T @ (design * w[:, None])
    cov = np.linalg.inv(normal)
    a, b = cov @ (design.T @ (w * y))
    resid = y - (a * x + b)
    dof = max(len(x) - 2, 1)
    return float(a), float(b), cov, float(np.sum(w * resid**2) / dof)


def measure_ptc(
    flats: Sequence[ArrayLike],
    dark: ArrayLike | DarkFrameResult,
    *,
    exposures: ArrayLike | None = None,
    saturation_dn: float | None = None,
    max_saturated_fraction: float = 1e-4,
    fit_fraction: float = 0.7,
    include_dark: bool = True,
    quantization_correction: bool = True,
) -> PTCResult:
    """
    Construit et ajuste la courbe de transfert photonique.

    Parameters
    ----------
    flats :
        Une pile (K ≥ 2, H, W) par niveau d'éclairement ; une paire suffit.
    dark :
        Pile d'obscurité (K ≥ 2, H, W) ou résultat de `measure_dark_frame`.
    exposures :
        Grandeur proportionnelle à l'exposition de chaque niveau (temps de
        pose, photons…), même ordre que `flats`, pour l'erreur de linéarité.
    saturation_dn :
        Code de saturation du convertisseur (2ⁿ − 1). Si fourni, un niveau
        dont plus de `max_saturated_fraction` des échantillons sont à ce
        code est exclu de l'ajustement.
    fit_fraction :
        Si la saturation est atteinte (variance maximale avant le dernier
        niveau), l'ajustement se limite aux signaux ≤ fit_fraction ×
        signal de saturation (EMVA 1288 : 70 %).
    include_dark :
        Ajoute l'obscurité comme point (μ = 0, σ² = b̂) : contraint b.
    quantization_correction :
        Retire 1/12 DN² de b avant d'en déduire le bruit de lecture.
    """
    if len(flats) < 2:
        raise ValueError("au moins deux niveaux d'éclairement sont nécessaires")
    if not 0 < fit_fraction <= 1:
        raise ValueError("fit_fraction doit être dans ]0, 1]")
    dark_result = dark if isinstance(dark, DarkFrameResult) else measure_dark_frame(as_stack(dark, min_frames=2, name="dark"))

    signal, variance, u_variance, sat_frac = [], [], [], []
    for flat in flats:
        stack = as_stack(flat, min_frames=2, name="flats[i]")
        r = measure_temporal_noise(stack, dark=dark_result)
        signal.append(r.signal)
        variance.append(r.temporal_variance)
        u_variance.append(r.u_temporal_variance)
        sat_frac.append(0.0 if saturation_dn is None else float(np.mean(stack[: r.n_frames] >= saturation_dn)))
    signal = np.asarray(signal)
    variance = np.asarray(variance)
    u_variance = np.asarray(u_variance)
    sat_frac = np.asarray(sat_frac)
    expo = None if exposures is None else np.asarray(exposures, dtype=np.float64)
    if expo is not None and expo.shape != signal.shape:
        raise ValueError("exposures doit avoir une valeur par niveau")

    order = np.argsort(signal)
    signal, variance, u_variance, sat_frac = signal[order], variance[order], u_variance[order], sat_frac[order]
    if expo is not None:
        expo = expo[order]

    # Saturation : maximum de la variance.
    i_max = int(np.argmax(variance))
    if i_max < len(signal) - 1:
        sat_signal = float(signal[i_max])
        used = signal <= fit_fraction * sat_signal
    else:
        sat_signal = None
        used = np.ones(len(signal), dtype=bool)
    used &= sat_frac <= max_saturated_fraction

    if include_dark:
        signal = np.concatenate([[0.0], signal])
        variance = np.concatenate([[dark_result.temporal_variance], variance])
        u_variance = np.concatenate([[dark_result.u_temporal_variance], u_variance])
        sat_frac = np.concatenate([[0.0], sat_frac])
        used = np.concatenate([[True], used])
        if expo is not None:
            expo = np.concatenate([[0.0], expo])
    if used.sum() < 3:
        raise ValueError("moins de trois niveaux exploitables pour l'ajustement de la PTC")

    a, b, cov, chi2 = _weighted_line(signal[used], variance[used], u_variance[used])
    u_a, u_b = float(np.sqrt(cov[0, 0])), float(np.sqrt(cov[1, 1]))
    gain = 1.0 / a
    u_gain = u_a / a**2

    q = QUANTIZATION_VARIANCE if quantization_correction else 0.0
    r2 = max(b - q, 0.0)
    rn_dn = float(np.sqrt(r2))
    rn_e = rn_dn * gain
    if rn_dn > 0:
        # σ_e = √(b − q)/a : propagation avec la covariance de (a, b).
        jac = np.array([-rn_dn / a**2, 1.0 / (2.0 * a * rn_dn)])
        u_rn_e = float(np.sqrt(jac @ cov @ jac))
    else:
        u_rn_e = float("nan")

    lin = None
    if expo is not None:
        sel = used & (signal > 0) if include_dark else used
        if sel.sum() >= 3:
            s, c = np.polyfit(expo[sel], signal[sel], 1)
            fit = s * expo[sel] + c
            delta = 100.0 * (signal[sel] - fit) / fit
            lin = float((delta.max() - delta.min()) / 2.0)

    return PTCResult(
        signal=signal,
        variance=variance,
        u_variance=u_variance,
        saturated_fraction=sat_frac,
        used=used,
        a=a,
        b=b,
        cov=cov,
        u_a=u_a,
        u_b=u_b,
        reduced_chi2=chi2,
        gain=float(gain),
        u_gain=float(u_gain),
        read_noise_dn=rn_dn,
        read_noise_e=float(rn_e),
        u_read_noise_e=u_rn_e,
        saturation_signal_dn=sat_signal,
        saturation_e=None if sat_signal is None else float(sat_signal * gain),
        linearity_error=lin,
    )
