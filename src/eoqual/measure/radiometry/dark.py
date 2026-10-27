"""
`dark_frame` — offset, DSNU et bruit temporel à signal nul ;
`dark_current` — courant d'obscurité, sa dispersion, pixels chauds, loi en
température.

`dark_frame` : pile de K images d'obscurité (capteur matriciel) ou bloc
de M lignes d'obscurité (push broom, un détecteur par colonne), même
temps de pose. `dark_current` : piles d'obscurité à plusieurs temps de
pose (et températures). Voir REFERENCE_TECHNIQUE.md, partie IV, chapitres 3 et 4.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from eoqual.measure.radiometry._stack import as_stack, pushbroom_as_stack
from eoqual.measure.radiometry.decomposition import SpatialDecomposition, decompose_spatial


@dataclass(slots=True)
class DarkFrameResult:
    """
    Résultat de `measure_dark_frame` (unités : DN).

    Attributes
    ----------
    layout :
        ``"stack"`` (pile matricielle) ou ``"pushbroom"`` (bloc de lignes).
    n_frames :
        Nombre K de poses (ou de lignes en push broom).
    offset_map :
        Carte d'offset, moyenne temporelle par pixel, forme (H, W) ; en
        push broom (1, W) : la ligne moyenne.
    mean_offset, u_mean_offset :
        Offset moyen (moyenne spatiale de la carte) et son incertitude-type
        due au bruit temporel, √(b/(K·N)).
    temporal_variance_map :
        Variance temporelle par pixel (estimateur en 1/(K − 1)).
    temporal_variance, u_temporal_variance :
        Variance temporelle moyenne b̂ (paramètre b du modèle σ² = a·μ + b)
        et son incertitude-type, b̂·√(2/((K − 1)·N)).
    temporal_noise :
        √b̂, bruit temporel à signal nul σ₀.
    dsnu, u_dsnu :
        DSNU : écart-type spatial de la carte d'offset corrigé du bruit
        temporel résiduel (b̂/K), et son incertitude-type par rapport à la
        carte réalisée.
    decomposition :
        Décomposition ligne / colonne / pixel de la DSNU.
    frame_means :
        Moyenne spatiale de chaque pose, forme (K,).
    drift_per_frame, u_drift :
        Pente de la moyenne spatiale en fonction de l'indice de pose
        (DN/pose) et son incertitude-type.
    stationarity_ratio :
        var(frame_means) / (b̂/N) : proche de 1 pour une pile stationnaire
        sans bruit corrélé ; nettement plus grand en cas de dérive, de
        bruit de ligne ou de bruit commun à toute l'image.
    """

    layout: str
    n_frames: int
    offset_map: NDArray[np.float64]
    mean_offset: float
    u_mean_offset: float
    temporal_variance_map: NDArray[np.float64]
    temporal_variance: float
    u_temporal_variance: float
    temporal_noise: float
    dsnu: float
    u_dsnu: float
    decomposition: SpatialDecomposition
    frame_means: NDArray[np.float64]
    drift_per_frame: float
    u_drift: float
    stationarity_ratio: float


def measure_dark_frame(
    frames: ArrayLike,
    *,
    layout: Literal["stack", "pushbroom"] = "stack",
) -> DarkFrameResult:
    """
    Mesure l'offset, la DSNU et le bruit temporel d'une pile d'obscurité.

    Parameters
    ----------
    frames :
        Pile (K, H, W) d'images d'obscurité (``layout="stack"``), ou bloc
        (M, W) de lignes d'obscurité d'un push broom
        (``layout="pushbroom"``). Au moins deux poses.
    layout :
        Organisation de l'entrée.

    Notes
    -----
    En push broom, la DSNU est la dispersion entre détecteurs ; la
    décomposition ligne / colonne n'a pas de sens (une seule ligne
    moyenne).
    """
    if layout == "stack":
        stack = as_stack(frames, min_frames=2)
    elif layout == "pushbroom":
        stack = pushbroom_as_stack(frames, min_lines=2)
    else:
        raise ValueError(f"layout inconnu : {layout!r}")

    k = stack.shape[0]
    n = stack[0].size
    offset_map = stack.mean(axis=0)
    var_map = stack.var(axis=0, ddof=1)
    b = float(var_map.mean())
    u_b = b * np.sqrt(2.0 / ((k - 1) * n))

    # Bruit spatial : carte moyenne corrigée du résidu temporel b/K.
    v = b / k
    decomposition = decompose_spatial(offset_map, noise_variance=v)
    s2 = decomposition.total_var
    u_dsnu = _u_sigma(s2, v, n, k)

    # Stationnarité : moyennes spatiales des poses.
    frame_means = stack.mean(axis=(1, 2))
    idx = np.arange(k, dtype=np.float64) - (k - 1) / 2.0
    sxx = float(np.sum(idx**2))
    drift = float(np.sum(idx * (frame_means - frame_means.mean())) / sxx)
    u_drift = float(np.sqrt(b / n / sxx))
    ratio = float(frame_means.var(ddof=1) / (b / n))

    return DarkFrameResult(
        layout=layout,
        n_frames=k,
        offset_map=offset_map,
        mean_offset=float(offset_map.mean()),
        u_mean_offset=float(np.sqrt(b / (k * n))),
        temporal_variance_map=var_map,
        temporal_variance=b,
        u_temporal_variance=float(u_b),
        temporal_noise=float(np.sqrt(b)),
        dsnu=decomposition.total,
        u_dsnu=u_dsnu,
        decomposition=decomposition,
        frame_means=frame_means,
        drift_per_frame=drift,
        u_drift=u_drift,
        stationarity_ratio=ratio,
    )


def _u_sigma(s2: float, v: float, n: int, k: float) -> float:
    """
    Incertitude-type de √(var(carte) − v) par rapport à la carte réalisée :
    Var ≈ [4·S·v + 2v²·(1 + 1/(K − 1))]/n (REFERENCE_TECHNIQUE.md, partie IV, §3.2.1).
    """
    var_d = (4.0 * max(s2, 0.0) * v + 2.0 * v**2 * (1.0 + 1.0 / max(k - 1, 1))) / n
    sigma = np.sqrt(max(s2, 0.0))
    return float(np.sqrt(var_d) / (2.0 * sigma)) if sigma > 0 else float(var_d**0.25)


# --------------------------------------------------------------------------
# Courant d'obscurité
# --------------------------------------------------------------------------

BOLTZMANN_EV = 8.617333262e-5
ZERO_CELSIUS = 273.15


@dataclass(slots=True)
class TemperatureFit:
    """
    Ajustement de la loi I(T) = I₀·(T/T₀)³·exp(−(E_g/2k)(1/T − 1/T₀)).

    Attributes
    ----------
    temperatures :
        Températures (°C) des groupes de piles.
    mean_current, u_mean_current :
        Courant moyen des pixels ordinaires par température (DN/s).
    bandgap, u_bandgap :
        Énergie E_g ajustée (eV) et son incertitude-type.
    reference_temperature :
        T₀ (°C) : température médiane des groupes.
    current_at_reference :
        I₀ (DN/s) à T₀.
    doubling_temperature :
        Écart ΔT (K) qui double le courant à T₀ (solution exacte).
    """

    temperatures: NDArray[np.float64]
    mean_current: NDArray[np.float64]
    u_mean_current: NDArray[np.float64]
    bandgap: float
    u_bandgap: float
    reference_temperature: float
    current_at_reference: float
    doubling_temperature: float


@dataclass(slots=True)
class DarkCurrentResult:
    """
    Résultat de `measure_dark_current` (unités : DN et s, sauf mention).

    Attributes
    ----------
    exposure_times :
        Temps de pose utilisés (s), à la température de référence.
    offset_map :
        Ordonnée à l'origine par pixel (offset à t = 0).
    current_map, current_noise :
        Pente par pixel (DN/s) et son incertitude-type (bruit temporel).
    valid :
        Pixels ajustés : au moins deux temps de pose distincts sans
        saturation (un pixel saturé à certaines poses est ajusté sur les
        autres) ; `current_map` vaut NaN ailleurs.
    hot_pixels :
        Pixels chauds : courant > `hot_factor` × médiane et significatif à
        `hot_sigma` incertitudes, ou saturés dans au moins une pose
        d'obscurité.
    mean_current, u_mean_current, median_current :
        Courant moyen des pixels ordinaires (valides, non chauds), son
        incertitude-type, et la médiane.
    dsnu, u_dsnu, dsnu_relative :
        Dispersion du courant des pixels ordinaires, corrigée du bruit de
        la régression (DN/s), son incertitude et rapport au courant moyen.
    poisson_gain, u_poisson_gain :
        Gain K (e⁻/DN) déduit du caractère poissonnien : pente de la
        variance temporelle en fonction du temps de pose = courant / K.
    gain :
        Gain fourni (e⁻/DN), ou `None`.
    mean_current_e :
        Courant moyen en e⁻/s (avec `gain`, sinon avec `poisson_gain`).
    temperature_fit :
        Loi en température si plusieurs températures sont fournies.
    """

    exposure_times: NDArray[np.float64]
    offset_map: NDArray[np.float64]
    current_map: NDArray[np.float64]
    current_noise: NDArray[np.float64]
    valid: NDArray[np.bool_]
    hot_pixels: NDArray[np.bool_]
    mean_current: float
    u_mean_current: float
    median_current: float
    dsnu: float
    u_dsnu: float
    dsnu_relative: float | None
    poisson_gain: float | None
    u_poisson_gain: float | None
    gain: float | None
    mean_current_e: float | None
    temperature_fit: TemperatureFit | None

    @property
    def n_hot(self) -> int:
        """Nombre de pixels chauds."""
        return int(self.hot_pixels.sum())


def _fit_group(stacks, times, saturation_dn, hot_factor, hot_sigma):
    """
    Régression par pixel moyenne(t) = offset + courant·t, pour un groupe de
    même température, sur les seules poses où le pixel ne sature pas
    (moindres carrés pondérés par un masque, vectorisés).
    """
    t = np.asarray(times, dtype=np.float64)
    means = np.stack([s.mean(axis=0) for s in stacks])                    # (n_t, H, W)
    var_maps = np.stack([s.var(axis=0, ddof=1) for s in stacks])          # (n_t, H, W)
    counts = np.array([s.shape[0] for s in stacks], dtype=np.float64)
    if saturation_dn is not None:
        use = np.stack([s.max(axis=0) < saturation_dn for s in stacks]).astype(np.float64)
    else:
        use = np.ones(means.shape)
    tt = t[:, None, None]
    n_use = use.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        t_bar = (use * tt).sum(axis=0) / n_use
        tc = (tt - t_bar) * use
        sxx = (tc * (tt - t_bar)).sum(axis=0)
        m_bar = (use * means).sum(axis=0) / n_use
        slope = (tc * (means - m_bar)).sum(axis=0) / sxx
        intercept = m_bar - slope * t_bar
        # Variance de la pente par pixel : Σ (t_i − t̄)²·v_i/K_i / Sxx².
        slope_var = (tc**2 * var_maps / counts[:, None, None]).sum(axis=0) / sxx**2
    valid = (n_use >= 2) & (sxx > 0)
    saturated = n_use < len(stacks)                                       # sature dans au moins une pose
    med = float(np.median(slope[valid & ~saturated])) if np.any(valid & ~saturated) else float(np.median(slope[valid]))
    with np.errstate(invalid="ignore"):
        criterion = (slope > hot_factor * max(med, 0.0)) & (slope - med > hot_sigma * np.sqrt(slope_var))
    # Un pixel qui sature dans l'obscurité est chaud par définition.
    hot = (valid & criterion) | saturated
    ordinary = valid & ~hot
    slope = np.where(valid, slope, np.nan)
    intercept = np.where(valid, intercept, np.nan)
    slope_var = np.where(valid, slope_var, np.nan)
    return t, means, var_maps, counts, slope, intercept, slope_var, valid, hot, ordinary, med


def measure_dark_current(
    stacks: Sequence[ArrayLike],
    exposure_times: Sequence[float],
    *,
    temperatures: Sequence[float] | None = None,
    gain: float | None = None,
    saturation_dn: float | None = None,
    hot_factor: float = 10.0,
    hot_sigma: float = 5.0,
) -> DarkCurrentResult:
    """
    Mesure le courant d'obscurité par pixel, sa dispersion, les pixels
    chauds et, si plusieurs températures sont fournies, sa loi en
    température.

    Parameters
    ----------
    stacks :
        Piles d'obscurité (K_i ≥ 2, H, W), une par condition.
    exposure_times :
        Temps de pose de chaque pile (s) ; au moins deux valeurs distinctes
        à la température de référence.
    temperatures :
        Température de chaque pile (°C). La carte et la dispersion sont
        mesurées à la température la plus représentée (à égalité, la plus
        basse) ; chaque température doit avoir au moins deux temps de pose
        distincts pour entrer dans la loi en température.
    gain :
        Gain K (e⁻/DN) pour exprimer le courant en e⁻/s ; sinon le gain
        poissonnien est utilisé.
    saturation_dn :
        Code de saturation : une pose où un pixel l'atteint est retirée de
        son ajustement, et le pixel est déclaré chaud.
    hot_factor, hot_sigma :
        Critères des pixels chauds : courant > hot_factor × médiane et
        excès > hot_sigma incertitudes.
    """
    arrs = [as_stack(s, min_frames=2, name="stacks[i]") for s in stacks]
    times = np.asarray(exposure_times, dtype=np.float64)
    if len(arrs) != times.size:
        raise ValueError("une durée de pose par pile")
    if np.any(times < 0):
        raise ValueError("temps de pose négatif")
    shape = arrs[0].shape[1:]
    if any(a.shape[1:] != shape for a in arrs):
        raise ValueError("toutes les piles doivent avoir la même taille d'image")
    temps = np.zeros(times.size) if temperatures is None else np.asarray(temperatures, dtype=np.float64)
    if temps.size != times.size:
        raise ValueError("une température par pile")

    groups = {}
    for i, temp in enumerate(temps):
        groups.setdefault(float(temp), []).append(i)
    usable = {T: idx for T, idx in groups.items() if np.unique(times[idx]).size >= 2}
    if not usable:
        raise ValueError("au moins deux temps de pose distincts à une même température sont nécessaires")
    ref_T = sorted(usable, key=lambda T: (-len(usable[T]), T))[0]

    fits = {}
    for T, idx in usable.items():
        fits[T] = _fit_group([arrs[i] for i in idx], times[idx], saturation_dn, hot_factor, hot_sigma)
    t, means, var_maps, counts, slope, intercept, slope_var, valid, hot, ordinary, med = fits[ref_T]

    n_ord = int(ordinary.sum())
    s_ord = slope[ordinary]
    noise_v = float(slope_var[ordinary].mean())
    mean_cur = float(s_ord.mean())
    u_mean = float(np.sqrt(slope_var[ordinary].sum()) / n_ord)
    s2 = float(s_ord.var() - noise_v)
    dsnu = float(np.sqrt(max(s2, 0.0)))
    u_dsnu = _u_sigma(s2, noise_v, n_ord, np.inf)

    # Caractère poissonnien : variance temporelle moyenne (pixels ordinaires) contre t.
    v_mean = np.array([v[ordinary].mean() for v in var_maps])
    u_v = v_mean * np.sqrt(2.0 / ((counts - 1) * n_ord))
    w = 1.0 / u_v**2
    design = np.column_stack([t, np.ones_like(t)])
    cov = np.linalg.inv(design.T @ (design * w[:, None]))
    v_slope = float((cov @ (design.T @ (w * v_mean)))[0])
    u_v_slope = float(np.sqrt(cov[0, 0]))
    if v_slope > 0:
        k_poisson = mean_cur / v_slope
        u_k = float(k_poisson * np.hypot(u_mean / mean_cur, u_v_slope / v_slope))
    else:
        k_poisson, u_k = None, None
    k_used = gain if gain is not None else k_poisson
    mean_e = None if k_used is None else mean_cur * k_used

    tfit = None
    if len(fits) >= 2:
        Ts = np.array(sorted(fits))
        I = np.array([fits[T][4][fits[T][9]].mean() for T in Ts])
        uI = np.array([np.sqrt(fits[T][6][fits[T][9]].sum()) / fits[T][9].sum() for T in Ts])
        if np.all(I > 0):
            tk = Ts + ZERO_CELSIUS
            T0 = float(np.median(Ts))
            t0k = T0 + ZERO_CELSIUS
            # ln I − 3 ln(T/T₀) = ln I₀ − (E_g/2k)(1/T − 1/T₀) : régression linéaire pondérée.
            y = np.log(I) - 3 * np.log(tk / t0k)
            x = 1 / tk - 1 / t0k
            wy = (I / uI) ** 2
            dsg = np.column_stack([x, np.ones_like(x)])
            cv = np.linalg.inv(dsg.T @ (dsg * wy[:, None]))
            slope_y, c0 = cv @ (dsg.T @ (wy * y))
            eg = float(-2 * BOLTZMANN_EV * slope_y)
            u_eg = float(2 * BOLTZMANN_EV * np.sqrt(cv[0, 0]))
            from scipy.optimize import brentq

            def factor(dt):
                tt = t0k + dt
                return (tt / t0k) ** 3 * np.exp(-eg / (2 * BOLTZMANN_EV) * (1 / tt - 1 / t0k)) - 2.0
            dbl = float(brentq(factor, 1e-3, 200.0)) if eg > 0 else float("nan")
            tfit = TemperatureFit(temperatures=Ts, mean_current=I, u_mean_current=uI, bandgap=eg,
                                  u_bandgap=u_eg, reference_temperature=T0,
                                  current_at_reference=float(np.exp(c0)), doubling_temperature=dbl)

    return DarkCurrentResult(
        exposure_times=t,
        offset_map=intercept,
        current_map=slope,
        current_noise=np.sqrt(slope_var),
        valid=valid,
        hot_pixels=hot,
        mean_current=mean_cur,
        u_mean_current=u_mean,
        median_current=med,
        dsnu=dsnu,
        u_dsnu=u_dsnu,
        dsnu_relative=None if mean_cur <= 0 else dsnu / mean_cur,
        poisson_gain=None if k_poisson is None else float(k_poisson),
        u_poisson_gain=u_k,
        gain=gain,
        mean_current_e=None if mean_e is None else float(mean_e),
        temperature_fit=tfit,
    )
