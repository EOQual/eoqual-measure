"""
`pushbroom_temporal_noise` — bruit temporel d'un push broom, modèle σ²(μ).

Entrée : une ou plusieurs images de scènes peu texturées (désert, neige,
océan), à un ou plusieurs niveaux. Le bruit se sépare de la scène par un
filtre passe-haut séparable : différence d'ordre 1 ou 2 le long de la
trace (retire le niveau et la pente locale, et l'offset de chaque
détecteur) et en travers (retire la texture commune aux détecteurs
voisins). Un bruit blanc garde sa variance au travers du filtre normé ;
une scène, dont le spectre décroît et que l'optique a déjà filtrée, n'en
laisse passer qu'une petite part. Les statistiques sont calculées par
segments de L lignes, dont le signal moyen donne l'abscisse de σ²(μ).
Voir REFERENCE_TECHNIQUE.md, partie IV, chapitre 10.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from eoqual.measure.radiometry.dark import DarkFrameResult

# Noyaux de différence normés (Σk² = 1) : un bruit blanc garde sa variance.
KERNELS: dict[int, NDArray[np.float64]] = {
    0: np.array([1.0]),
    1: np.array([1.0, -1.0]) / np.sqrt(2.0),
    2: np.array([1.0, -2.0, 1.0]) / np.sqrt(6.0),
}


def kernel_autocorrelation(order: int) -> NDArray[np.float64]:
    """Autocorrélation ρ_k (k = 0, 1, …) d'un bruit blanc filtré par le noyau d'ordre `order`."""
    k = KERNELS[order]
    return np.correlate(k, k, mode="full")[len(k) - 1:]


def effective_dof(n: int, order: int) -> float:
    """
    Degrés de liberté effectifs de mean(f²) sur n valeurs filtrées corrélées.

    Var(mean f²) = (2σ⁴/n)·Σ_k ρ_k² (k de −∞ à +∞), d'où ν = n / Σ ρ_k².
    """
    rho = kernel_autocorrelation(order)
    return n / (rho[0] ** 2 + 2.0 * np.sum(rho[1:] ** 2))


def rho1_null(n: int, order: int) -> tuple[float, float]:
    """
    Espérance et écart-type de l'autocorrélation empirique au retard 1 d'un
    bruit blanc filtré (formule de Bartlett, n valeurs).
    """
    if order == 0:
        return 0.0, 1.0 / np.sqrt(n)
    r = kernel_autocorrelation(order)

    def rr(k: int) -> float:
        k = abs(k)
        return float(r[k]) if k < len(r) else 0.0

    rho1 = rr(1)
    # Bartlett : w₁₁ = Σ_{k≥1} (ρ_{k+1} + ρ_{k−1} − 2ρ₁ρ_k)², ρ nulle au-delà du support.
    w11 = sum((rr(k + 1) + rr(k - 1) - 2 * rho1 * rr(k)) ** 2 for k in range(1, len(r) + 2))
    return rho1, float(np.sqrt(w11 / n))


@dataclass(slots=True)
class PushbroomNoiseResult:
    """
    Résultat de `measure_pushbroom_noise` (unités : DN).

    Attributes
    ----------
    segment_length, along_order, across_order :
        Longueur L des segments et ordres des différences le long de la
        trace et en travers.
    dof :
        Degrés de liberté effectifs de la variance d'un segment.
    segment_mean, segment_variance, segment_rho :
        Signal moyen (corrigé de l'obscurité) des détecteurs du groupe,
        variance du bruit filtré et autocorrélation au retard 1, par segment.
    segment_group, segment_image :
        Indice du groupe de détecteurs et de l'image de chaque segment.
    used :
        Segments retenus (signal > 0, et autocorrélation sous le seuil si
        `rho_threshold` est fourni).
    a, b, cov, u_a, u_b :
        Ajustement pondéré σ² = a·μ + b et incertitudes ; si b est imposé,
        u_b = 0 et la covariance ne porte que sur a.
    b_fixed :
        Vrai si b a été imposé (par exemple par `dark_frame`).
    reduced_chi2 :
        χ² réduit de l'ajustement (≈ 1 si le modèle tient).
    rejected_fraction :
        Part des segments rejetés.
    consistency, u_consistency :
        Rapport des variances moyennes obtenues avec l'autre ordre de
        différence le long de la trace (1 ↔ 2) et avec le filtre choisi,
        sur les mêmes segments et détecteurs. Il vaut 1 sur du bruit pur ;
        un écart de plus de quelques u signale une texture résiduelle, donc
        une mesure biaisée.
    """

    segment_length: int
    along_order: int
    across_order: int
    dof: float
    segment_mean: NDArray[np.float64]
    segment_variance: NDArray[np.float64]
    segment_rho: NDArray[np.float64]
    segment_group: NDArray[np.int64]
    segment_image: NDArray[np.int64]
    used: NDArray[np.bool_]
    a: float
    b: float
    cov: NDArray[np.float64]
    u_a: float
    u_b: float
    b_fixed: bool
    reduced_chi2: float
    rejected_fraction: float
    consistency: float
    u_consistency: float

    @property
    def texture_suspected(self) -> bool:
        """
        Vrai si le rapport de cohérence s'écarte de 1 de plus de 5 u : la
        scène laisse passer de la texture dans le filtre et σ² est
        surestimé. Ne pas exploiter le résultat dans ce cas.
        """
        return bool(abs(self.consistency - 1.0) > 5.0 * self.u_consistency)

    def sigma(self, mu: ArrayLike) -> NDArray[np.float64]:
        """Bruit temporel modélisé σ(μ) = √(a·μ + b) (DN)."""
        return np.sqrt(self.a * np.asarray(mu, dtype=np.float64) + self.b)

    def binned(self, n_bins: int = 10) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """
        Moyennes par classes de signal des segments retenus (pour tracé).

        Renvoie (signal moyen, variance moyenne, incertitude-type) par classe,
        classes de même effectif.
        """
        mu = self.segment_mean[self.used]
        var = self.segment_variance[self.used]
        order = np.argsort(mu)
        out: tuple[list, list, list] = ([], [], [])
        for chunk in np.array_split(order, n_bins):
            if chunk.size < 2:
                continue
            out[0].append(mu[chunk].mean())
            out[1].append(var[chunk].mean())
            out[2].append(var[chunk].std(ddof=1) / np.sqrt(chunk.size))
        return tuple(np.asarray(x) for x in out)


def _detector_offset(dark, w: int) -> NDArray[np.float64]:
    if dark is None:
        return np.zeros(w)
    if isinstance(dark, DarkFrameResult):
        off = dark.offset_map.mean(axis=0)
    else:
        off = np.asarray(dark, dtype=np.float64)
        if off.ndim == 0:
            off = np.full(w, float(off))
        elif off.ndim == 2 and off.shape[0] == 1:
            off = off[0]
    if off.shape != (w,):
        raise ValueError(f"dark doit donner un offset par détecteur ({w},), reçu {off.shape}")
    return off


def _filtered_segments(image: NDArray[np.float64], length: int, along: int, across: int):
    """
    Pour chaque segment × groupe de détecteurs : niveau moyen, mean(f²) pour
    le filtre (along, across) et pour le filtre de contrôle (1, 1 ou 1, 0),
    autocorrélation au retard 1.
    """
    m, w = image.shape
    kc = KERNELS[across]
    g = len(kc)
    n_grp = w // g
    grp = image[:, : n_grp * g].reshape(m, n_grp, g)
    n_seg = m // length
    grp = grp[: n_seg * length].reshape(n_seg, length, n_grp, g)
    level = grp.mean(axis=(1, 3))                                  # (n_seg, n_grp)

    def along_filter(x, order):
        ka = KERNELS[order]
        n_out = length - len(ka) + 1
        return sum(ka[i] * x[:, i:i + n_out] for i in range(len(ka)))

    across_filtered = grp @ kc                                     # (n_seg, L, n_grp)
    f = along_filter(across_filtered, along)                       # (n_seg, n_out, n_grp)
    ss = np.sum(f**2, axis=1)
    var = ss / f.shape[1]
    with np.errstate(invalid="ignore", divide="ignore"):
        rho = np.sum(f[:, 1:] * f[:, :-1], axis=1) / ss
    # Filtre de contrôle : mêmes détecteurs et même pondération en travers,
    # autre ordre le long de la trace (1 ↔ 2) : seule change la part de
    # texture transmise, pas le bruit.
    fc = along_filter(across_filtered, 1 if along == 2 else 2)
    var_ctrl = np.mean(fc**2, axis=1)
    idx = np.broadcast_to(np.arange(n_grp), (n_seg, n_grp))
    return level.ravel(), var.ravel(), rho.ravel(), var_ctrl.ravel(), idx.ravel(), f.shape[1]


def _fit(mu, var, dof, b_value=None, n_iter=5):
    """
    Moindres carrés pondérés itératifs, poids tirés du modèle.

    Var(s²) = 2σ⁴/ν : pondérer par la variance mesurée favoriserait les
    segments de faible s² (biais vers le bas) ; on pondère par le modèle.
    """
    floor = 0.05 * float(np.median(var))     # garde-fou : modèle provisoire ≤ 0
    if b_value is None:
        design = np.column_stack([mu, np.ones_like(mu)])
        a, b = np.linalg.lstsq(design, var, rcond=None)[0]
    else:
        a = float(np.sum(mu * (var - b_value)) / np.sum(mu**2))
        b = b_value
    for _ in range(n_iter):
        model = np.maximum(a * mu + b, floor)
        w = dof / (2.0 * model**2)
        if b_value is None:
            design = np.column_stack([mu, np.ones_like(mu)])
            cov = np.linalg.inv(design.T @ (design * w[:, None]))
            a, b = cov @ (design.T @ (w * var))
        else:
            cov = np.array([[1.0 / np.sum(w * mu**2)]])
            a = float(cov[0, 0] * np.sum(w * mu * (var - b_value)))
    model = np.maximum(a * mu + b, floor)
    w = dof / (2.0 * model**2)
    n_par = 2 if b_value is None else 1
    chi2 = float(np.sum(w * (var - model) ** 2) / max(len(mu) - n_par, 1))
    return float(a), float(b), cov, chi2


def measure_pushbroom_noise(
    images: ArrayLike | Sequence[ArrayLike],
    *,
    dark: float | ArrayLike | DarkFrameResult | None = None,
    segment_length: int = 32,
    along_order: int = 2,
    across_order: int = 2,
    rho_threshold: float | None = None,
    b: float | None = None,
) -> PushbroomNoiseResult:
    """
    Mesure le modèle de bruit temporel σ²(μ) = a·μ + b d'un push broom.

    Parameters
    ----------
    images :
        Une image (M, W) ou une séquence d'images du même capteur, de
        scènes peu texturées ; les segments de toutes les images sont
        cumulés (plusieurs niveaux de signal).
    dark :
        Offset par détecteur : scalaire, profil (W,) ou (1, W), ou résultat
        de `measure_dark_frame(..., layout="pushbroom")`. Sert au niveau μ
        seulement (le filtre élimine l'offset de la variance).
    segment_length :
        Longueur L des segments le long de la trace (≥ 8).
    along_order, across_order :
        Ordres des différences le long de la trace (1 ou 2) et en travers
        (0 : détecteur par détecteur, 1 : paires, 2 : triplets).
    rho_threshold :
        Si fourni, rejet d'un segment dont l'autocorrélation au retard 1
        dépasse son espérance sous bruit blanc de plus de `rho_threshold`
        écarts-types. Désactivé par défaut : sur un bruit filtré,
        l'autocorrélation et la variance du segment sont corrélées
        (≈ −0,45), et le rejet biaise σ² vers le haut (+0,5 % pour 3 σ) sans
        presque rien retirer de la texture.
    b :
        Variance à signal nul imposée (DN²), par exemple
        `dark_result.temporal_variance` ; sinon b est ajusté.

    Notes
    -----
    La différence en travers élimine aussi un bruit commun à tous les
    détecteurs d'une ligne (bruit de ligne) : c'est le bruit temporel non
    corrélé qui est mesuré.
    """
    imgs = [np.asarray(images, dtype=np.float64)] if np.ndim(images) == 2 else [np.asarray(x, dtype=np.float64) for x in images]
    if not imgs:
        raise ValueError("au moins une image est nécessaire")
    if segment_length < 8:
        raise ValueError("segment_length doit être >= 8")
    if along_order not in (1, 2) or across_order not in (0, 1, 2):
        raise ValueError("along_order ∈ {1, 2}, across_order ∈ {0, 1, 2}")
    w = imgs[0].shape[1] if imgs[0].ndim == 2 else -1
    for x in imgs:
        if x.ndim != 2 or x.shape[1] != w:
            raise ValueError("toutes les images doivent être de forme (M, W) avec le même W")
        if x.shape[0] < segment_length or w < len(KERNELS[across_order]):
            raise ValueError("image trop petite pour un segment")
    offset = _detector_offset(dark, w)

    parts = [_filtered_segments(x - offset[None, :], segment_length, along_order, across_order) for x in imgs]
    n_out = parts[0][5]
    mu = np.concatenate([p[0] for p in parts])
    var = np.concatenate([p[1] for p in parts])
    rho = np.concatenate([p[2] for p in parts])
    var_ctrl = np.concatenate([p[3] for p in parts])
    grp = np.concatenate([p[4] for p in parts])
    img_idx = np.concatenate([np.full(p[0].size, i) for i, p in enumerate(parts)])

    used = (mu > 0) & np.isfinite(rho)
    if rho_threshold is not None:
        rho0, rho_sd = rho1_null(n_out, along_order)
        used &= rho <= rho0 + rho_threshold * rho_sd
    if used.sum() < 3:
        raise ValueError("moins de trois segments exploitables")

    dof = effective_dof(n_out, along_order)
    a, b_fit, cov, chi2 = _fit(mu[used], var[used], dof, b_value=b)

    # Cohérence : rapport des moyennes (filtre de contrôle / filtre choisi), méthode delta.
    x, y = var_ctrl[used], var[used]
    ratio = float(x.mean() / y.mean())
    u_ratio = float(np.std(x - ratio * y, ddof=1) / (np.sqrt(x.size) * y.mean()))

    return PushbroomNoiseResult(
        segment_length=segment_length,
        along_order=along_order,
        across_order=across_order,
        dof=float(dof),
        segment_mean=mu,
        segment_variance=var,
        segment_rho=rho,
        segment_group=grp.astype(np.int64),
        segment_image=img_idx.astype(np.int64),
        used=used,
        a=a,
        b=b_fit,
        cov=cov,
        u_a=float(np.sqrt(cov[0, 0])),
        u_b=0.0 if b is not None else float(np.sqrt(cov[1, 1])),
        b_fixed=b is not None,
        reduced_chi2=chi2,
        rejected_fraction=float(1.0 - used.mean()),
        consistency=ratio,
        u_consistency=u_ratio,
    )
