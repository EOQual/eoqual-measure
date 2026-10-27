"""
`linearity` — non-linéarité commune et dispersion par pixel.

Entrée : piles d'une source stable à plusieurs expositions connues (temps
de pose, ou niveaux de référence), et une obscurité. On mesure :

- l'erreur de linéarité EMVA 1288 sur le signal moyen ;
- la courbure commune, μ = s·x + q·x² + c, exprimée par κ = q/s² (DN⁻¹) :
  pour une compression e' = e(1 − βe/FW), κ = −β·K/FW ;
- la courbure par pixel et sa dispersion, corrigée du bruit.

Une non-linéarité commune à tous les pixels ne se voit pas par
comparaison relative entre pixels (égalisation) : seule une référence
externe d'exposition la révèle. Voir REFERENCE_TECHNIQUE.md, partie IV, chapitre 6.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from eoqual.measure.radiometry._stack import as_stack
from eoqual.measure.radiometry.dark import DarkFrameResult, _u_sigma, measure_dark_frame


@dataclass(slots=True)
class LinearityResult:
    """
    Résultat de `measure_linearity`.

    Attributes
    ----------
    exposures :
        Expositions (temps de pose ou niveau de référence), triées.
    mean_signal, u_mean_signal :
        Signal moyen corrigé de l'obscurité par exposition (DN) et son
        incertitude-type (bruit temporel).
    saturated_fraction :
        Part des échantillons au code de saturation, par exposition.
    used :
        Expositions de la plage d'ajustement (5 %–95 % du signal maximal
        non saturé, par défaut).
    deviation :
        Écart relatif δ (%) à la droite μ = s·x + c ajustée sur la plage.
    linearity_error :
        Erreur de linéarité EMVA 1288 : (max δ − min δ)/2 sur la plage (%).
    slope, curvature, u_curvature :
        Pente s (DN par unité d'exposition) et courbure commune
        κ = q/s² (DN⁻¹) de l'ajustement quadratique, avec son incertitude.
    curvature_map :
        Courbure κ par pixel (DN⁻¹).
    curvature_dispersion, u_curvature_dispersion :
        Dispersion de κ entre pixels, corrigée du bruit de l'ajustement.
    """

    exposures: NDArray[np.float64]
    mean_signal: NDArray[np.float64]
    u_mean_signal: NDArray[np.float64]
    saturated_fraction: NDArray[np.float64]
    used: NDArray[np.bool_]
    deviation: NDArray[np.float64]
    linearity_error: float
    slope: float
    curvature: float
    u_curvature: float
    curvature_map: NDArray[np.float64]
    curvature_dispersion: float
    u_curvature_dispersion: float


def measure_linearity(
    stacks: Sequence[ArrayLike],
    exposures: Sequence[float],
    dark: ArrayLike | DarkFrameResult,
    *,
    saturation_dn: float | None = None,
    fit_range: tuple[float, float] = (0.05, 0.95),
    exclude: ArrayLike | None = None,
) -> LinearityResult:
    """
    Mesure la non-linéarité d'un capteur à partir d'expositions connues.

    Parameters
    ----------
    stacks :
        Piles (K_i ≥ 2, H, W), une par exposition.
    exposures :
        Exposition de chaque pile (temps de pose avec une source stable, ou
        niveau de référence), proportionnelle au nombre de photons.
    dark :
        Pile d'obscurité ou `DarkFrameResult`. Si le temps de pose varie,
        le courant d'obscurité doit être négligeable ou retiré au préalable.
    saturation_dn :
        Code de saturation : une exposition dont plus de 10⁻⁴ des
        échantillons y sont est exclue.
    fit_range :
        Plage d'ajustement en fraction du signal maximal non saturé
        (EMVA 1288 : 5 %–95 %).
    exclude :
        Pixels à exclure (défectueux).
    """
    arrs = [as_stack(s, min_frames=2, name="stacks[i]") for s in stacks]
    x = np.asarray(exposures, dtype=np.float64)
    if len(arrs) != x.size:
        raise ValueError("une exposition par pile")
    if np.unique(x).size < 3:
        raise ValueError("au moins trois expositions distinctes")
    dk = dark if isinstance(dark, DarkFrameResult) else measure_dark_frame(as_stack(dark, min_frames=2, name="dark"))
    shape = arrs[0].shape[1:]
    excl = np.zeros(shape, dtype=bool) if exclude is None else np.asarray(exclude, dtype=bool)
    use_px = ~excl
    n_px = int(use_px.sum())

    order = np.argsort(x)
    x = x[order]
    arrs = [arrs[i] for i in order]
    means = np.stack([a.mean(axis=0) - dk.offset_map for a in arrs])           # (n, H, W)
    var_t = np.array([a.var(axis=0, ddof=1)[use_px].mean() for a in arrs])
    counts = np.array([a.shape[0] for a in arrs], dtype=np.float64)
    mu = np.array([m[use_px].mean() for m in means])
    u_mu = np.sqrt(var_t / (counts * n_px))
    sat = np.zeros(x.size) if saturation_dn is None else np.array([np.mean(a >= saturation_dn) for a in arrs])

    ok = sat <= 1e-4
    if ok.sum() < 3:
        raise ValueError("moins de trois expositions non saturées")
    mu_max = mu[ok].max()
    used = ok & (mu >= fit_range[0] * mu_max) & (mu <= fit_range[1] * mu_max)
    if used.sum() < 3:
        raise ValueError("moins de trois expositions dans la plage d'ajustement")

    # Erreur de linéarité (EMVA 1288) : écarts à la droite sur la plage.
    w = 1.0 / u_mu[used] ** 2
    a1 = np.column_stack([x[used], np.ones(used.sum())])
    s1, c1 = np.linalg.solve(a1.T @ (a1 * w[:, None]), a1.T @ (w * mu[used]))
    fit = s1 * x + c1
    deviation = np.full(x.size, np.nan)
    deviation[used] = 100.0 * (mu[used] - fit[used]) / fit[used]
    lin_err = float((np.nanmax(deviation) - np.nanmin(deviation)) / 2.0)

    # Courbure commune : μ = c + s·x + q·x², κ = q/s².
    a2 = np.column_stack([np.ones(used.sum()), x[used], x[used] ** 2])
    cov2 = np.linalg.inv(a2.T @ (a2 * w[:, None]))
    c, s, q = cov2 @ (a2.T @ (w * mu[used]))
    kappa = float(q / s**2)
    u_kappa = float(np.sqrt(cov2[2, 2]) / s**2)

    # Courbure par pixel : même ajustement (moindres carrés ordinaires), vectorisé.
    xs = x[used]
    a3 = np.column_stack([np.ones(xs.size), xs, xs**2])
    proj = np.linalg.pinv(a3)                                                   # (3, n_used)
    m_used = means[used].reshape(used.sum(), -1)
    coef = proj @ m_used                                                        # (3, N)
    with np.errstate(divide="ignore", invalid="ignore"):
        kmap = (coef[2] / coef[1] ** 2).reshape(shape)
    # Bruit de κ = q/s² par la méthode delta, avec la covariance de (s, q) :
    # Cov(coef) = P·diag(v_i/K_i)·Pᵀ ; Var κ = Var q/s⁴ + 4q²·Var s/s⁶ − 4q·Cov(q, s)/s⁵.
    # Le terme croisé change de signe avec q : l'oublier fausse la correction
    # dans un sens ou dans l'autre selon le signe de la courbure.
    cov_c = proj @ np.diag(var_t[used] / counts[used]) @ proj.T
    s_px = coef[1][use_px.ravel()]
    q_px = coef[2][use_px.ravel()]
    noise_k = float(np.mean(cov_c[2, 2] / s_px**4 + 4 * q_px**2 * cov_c[1, 1] / s_px**6
                            - 4 * q_px * cov_c[1, 2] / s_px**5))
    kvals = kmap[use_px & np.isfinite(kmap)]
    s2k = float(kvals.var() - noise_k)
    disp = float(np.sqrt(max(s2k, 0.0)))
    u_disp = _u_sigma(s2k, noise_k, kvals.size, np.inf)

    return LinearityResult(
        exposures=x,
        mean_signal=mu,
        u_mean_signal=u_mu,
        saturated_fraction=sat,
        used=used,
        deviation=deviation,
        linearity_error=lin_err,
        slope=float(s),
        curvature=kappa,
        u_curvature=u_kappa,
        curvature_map=np.where(use_px, kmap, np.nan),
        curvature_dispersion=disp,
        u_curvature_dispersion=u_disp,
    )
