"""
`noise_power_spectrum` et `structured_noise` — spectre de puissance du
bruit et bruits structurés (motifs périodiques, bruit de ligne / colonne).

NPS 2D par imagettes (méthode de Welch) : retrait d'un plan, fenêtre de
Hann (ou décomposition périodique + lisse de Moisan), |FFT|², moyenne.
Normalisation : l'intégrale du NPS sur [−½, ½]² égale la variance, si
bien qu'un bruit blanc σ² a un NPS plat de niveau σ² (DN²·pixel²). Voir
REFERENCE_TECHNIQUE.md, partie IV, chapitre 12.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import ndimage, stats


# --------------------------------------------------------------------------
# NPS 2D
# --------------------------------------------------------------------------

@dataclass(slots=True)
class NPSResult:
    """
    Spectre de puissance du bruit (fréquences en cycles/pixel, centrées).

    Attributes
    ----------
    frequency :
        Axe des fréquences (P,), de −½ à ½ − 1/P (ordre `fftshift`).
    nps :
        NPS 2D (P, P), axe 0 = f_y (le long des colonnes, indice de ligne i),
        axe 1 = f_x (indice de colonne j) ; DN²·pixel².
    u_nps :
        Incertitude-type de chaque valeur : dispersion entre imagettes / √n.
        Avec la fenêtre de Hann et 50 % de recouvrement, le nombre de
        moyennes indépendantes mesuré en 2D vaut ≈ n, si bien que u_nps est
        juste (REFERENCE_TECHNIQUE.md, partie IV, §12.2.2).
    variance :
        Intégrale du NPS, mean(nps) : variance du bruit analysé.
    n_patches, patch_size :
        Nombre et taille des imagettes moyennées.
    n_images, n_patch_rows, n_patch_cols :
        Nombre d'images analysées (paires en mode « difference ») et
        nombre de positions distinctes d'imagettes en hauteur et en
        largeur.
    mode, window, detrend :
        Paramètres de l'estimation.
    """

    frequency: NDArray[np.float64]
    nps: NDArray[np.float64]
    u_nps: NDArray[np.float64]
    variance: float
    n_patches: int
    patch_size: int
    n_images: int
    n_patch_rows: int
    n_patch_cols: int
    mode: str
    window: str
    detrend: str

    @property
    def horizontal(self) -> NDArray[np.float64]:
        """Coupe à f_y = 0 en fonction de f_x."""
        return self.nps[self.patch_size // 2]

    @property
    def vertical(self) -> NDArray[np.float64]:
        """Coupe à f_x = 0 en fonction de f_y."""
        return self.nps[:, self.patch_size // 2]

    def radial(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Moyenne radiale : (fréquence radiale, NPS moyen), anneaux de largeur
        1/P, de 0 à ½.
        """
        f = self.frequency
        fr = np.hypot(f[:, None], f[None, :])
        p = self.patch_size
        idx = np.rint(fr * p).astype(int)
        n_bins = p // 2 + 1
        sums = np.bincount(idx.ravel(), weights=self.nps.ravel(), minlength=n_bins)[:n_bins]
        counts = np.bincount(idx.ravel(), minlength=n_bins)[:n_bins]
        return np.arange(n_bins) / p, sums / np.maximum(counts, 1)


def _plane(patch: NDArray[np.float64], design: NDArray[np.float64], pinv: NDArray[np.float64]) -> NDArray[np.float64]:
    coef = pinv @ patch.ravel()
    return (design @ coef).reshape(patch.shape)


def _patches(image: NDArray[np.float64], size: int, step: int, mask: NDArray[np.bool_] | None):
    h, w = image.shape
    for y in range(0, h - size + 1, step):
        for x in range(0, w - size + 1, step):
            if mask is not None and not mask[y:y + size, x:x + size].all():
                continue
            yield y, x, image[y:y + size, x:x + size]


def noise_power_spectrum(
    data: ArrayLike,
    *,
    patch_size: int = 64,
    step: int | None = None,
    detrend: Literal["plane", "mean", "none"] = "plane",
    window: Literal["hann", "none", "perdecomp"] = "hann",
    mode: Literal["image", "difference"] = "image",
    mask: ArrayLike | None = None,
) -> NPSResult:
    """
    NPS 2D d'une image ou d'une pile, par imagettes.

    Parameters
    ----------
    data :
        Image (H, W) ou pile (K, H, W) de zones uniformes (ou produit
        traité : dématriçage, rééchantillonnage, compression…).
    patch_size :
        Taille P des imagettes carrées.
    step :
        Pas entre imagettes (défaut P/2 : recouvrement de 50 %, Welch).
    detrend :
        Retrait par imagette : plan (variations lentes de la scène),
        moyenne, ou rien.
    window :
        ``"hann"`` (fenêtre 2D séparable, NPS normalisé par Σw²),
        ``"none"``, ou ``"perdecomp"`` (composante périodique de Moisan,
        sans fenêtre ; nécessite eoqual-core).
    mode :
        ``"image"`` : NPS de chaque image (bruit temporel et spatial fixe) ;
        ``"difference"`` : NPS de (x_k − x_{k+1})/√2 sur des paires
        disjointes de poses, bruit temporel seul (pile requise).
    mask :
        Masque (H, W) des zones utilisables : seules les imagettes
        entièrement dans le masque sont retenues.
    """
    x = np.asarray(data, dtype=np.float64)
    if x.ndim == 2:
        x = x[None]
    if x.ndim != 3:
        raise ValueError("data doit être une image (H, W) ou une pile (K, H, W)")
    if mode == "difference":
        if x.shape[0] < 2:
            raise ValueError("mode='difference' demande au moins deux poses")
        n_pairs = x.shape[0] // 2
        x = (x[0:2 * n_pairs:2] - x[1:2 * n_pairs:2]) / np.sqrt(2.0)
    elif mode != "image":
        raise ValueError(f"mode inconnu : {mode!r}")
    p = int(patch_size)
    if p < 8 or p > min(x.shape[1:]):
        raise ValueError("patch_size doit être ≥ 8 et ≤ la taille de l'image")
    step = p // 2 if step is None else int(step)
    m = None if mask is None else np.asarray(mask, dtype=bool)
    if m is not None and m.shape != x.shape[1:]:
        raise ValueError("mask doit avoir la forme (H, W) des images")

    if window == "hann":
        w1 = np.hanning(p + 2)[1:-1]                 # fenêtre sans zéros aux bords
        win = np.outer(w1, w1)
    elif window in ("none", "perdecomp"):
        win = np.ones((p, p))
    else:
        raise ValueError(f"window inconnue : {window!r}")
    if window == "perdecomp":
        try:
            from eoqual.core.fourier import perdecomp
        except ImportError as exc:                    # pragma: no cover - dépend de l'installation
            raise ImportError("window='perdecomp' nécessite eoqual-core (extra « eoqual »)") from exc
    norm = float(np.sum(win**2))

    yy, xx = np.mgrid[0:p, 0:p]
    design = np.column_stack([np.ones(p * p), xx.ravel(), yy.ravel()])
    pinv = np.linalg.pinv(design)

    acc = np.zeros((p, p))
    acc2 = np.zeros((p, p))
    n = 0
    ys_used, xs_used = set(), set()
    for img in x:
        for y0, x0, patch in _patches(img, p, step, m):
            ys_used.add(y0)
            xs_used.add(x0)
            if detrend == "plane":
                patch = patch - _plane(patch, design, pinv)
            elif detrend == "mean":
                patch = patch - patch.mean()
            elif detrend != "none":
                raise ValueError(f"detrend inconnu : {detrend!r}")
            if window == "perdecomp":
                patch, _ = perdecomp(patch)
                patch = patch - patch.mean()
            spec = np.abs(np.fft.fft2(patch * win)) ** 2 / norm
            acc += spec
            acc2 += spec**2
            n += 1
    if n == 0:
        raise ValueError("aucune imagette exploitable (masque trop restrictif ?)")
    mean = acc / n
    sd = np.sqrt(np.maximum(acc2 / n - mean**2, 0.0) * n / max(n - 1, 1))
    freq = np.fft.fftshift(np.fft.fftfreq(p))
    return NPSResult(
        frequency=freq,
        nps=np.fft.fftshift(mean),
        u_nps=np.fft.fftshift(sd / np.sqrt(n)),
        variance=float(mean.mean()),
        n_patches=n,
        patch_size=p,
        n_images=int(x.shape[0]),
        n_patch_rows=len(ys_used),
        n_patch_cols=len(xs_used),
        mode=mode,
        window=window,
        detrend=detrend,
    )


# --------------------------------------------------------------------------
# Bruits structurés
# --------------------------------------------------------------------------

@dataclass(slots=True)
class StructuredPeak:
    """
    Pic du NPS (motif périodique), demi-plan f_y > 0 ou (f_y = 0, f_x > 0).

    Attributes
    ----------
    kind :
        ``"temporal"`` (NPS des différences de poses : motif dont la phase
        change d'une pose à l'autre), ``"fixed"`` (NPS de l'image moyenne,
        part temporelle retirée : motif fixe) ou ``"image"`` (image seule,
        nature indéterminée).
    fx, fy :
        Fréquence (cycles/pixel), barycentre de l'excès de puissance.
    frequency, orientation :
        Module (cycles/pixel) et orientation du vecteur d'onde (degrés,
        0 = variation le long des colonnes j, 90 = le long des lignes i).
    variance, rms, amplitude :
        Variance du motif (excès intégré sur le pic et son symétrique,
        DN²), valeur efficace et amplitude crête √2·rms (DN).
    variance_fraction :
        variance / variance totale du NPS où le pic a été trouvé.
    significance :
        Excès au maximum, en écarts-types relatifs du fond (1/√ν).
    """

    kind: str
    fx: float
    fy: float
    frequency: float
    orientation: float
    variance: float
    rms: float
    amplitude: float
    variance_fraction: float
    significance: float


@dataclass(slots=True)
class LineNoise:
    """
    Bruit de ligne (ou de colonne) : offset commun à tous les pixels d'une
    ligne, estimé sur la séquence des moyennes de lignes.

    Attributes
    ----------
    sigma, u_sigma :
        Écart-type corrigé du bruit pixel moyenné (DN) et incertitude-type.
    floor_variance :
        Variance du bruit pixel dans une moyenne de ligne, σ²_pixel/W.
    beta, u_beta :
        Exposant spectral du bruit de ligne, S(f) ∝ f^−β (ajusté après
        retrait du plancher blanc), ou `None` si le bruit de ligne est trop
        faible pour l'ajustement.
    frequency, spectrum :
        Densité spectrale unilatérale de la séquence (Σ S·Δf = variance).
    """

    sigma: float
    u_sigma: float
    floor_variance: float
    beta: float | None
    u_beta: float | None
    frequency: NDArray[np.float64]
    spectrum: NDArray[np.float64]


@dataclass(slots=True)
class PeakSearch:
    """
    Recherche de pics sur un NPS.

    Attributes
    ----------
    kind :
        Nature du NPS (voir `StructuredPeak.kind`).
    nps :
        NPS analysé.
    background :
        Fond lisse (médiane locale, plus la fuite temporelle prévue pour
        ``"fixed"``), même forme que le NPS.
    threshold :
        Seuil sur NPS/fond par fréquence (infini aux fréquences exclues).
    dof :
        Degrés de liberté ν de la loi Gamma ajustée, par fréquence.
    """

    kind: str
    nps: NPSResult
    background: NDArray[np.float64]
    threshold: NDArray[np.float64]
    dof: NDArray[np.float64]


@dataclass(slots=True)
class StructuredNoiseResult:
    """
    Résultat de `find_structured_noise`.

    Attributes
    ----------
    searches :
        Recherches effectuées, par nature (``"temporal"`` et ``"fixed"``
        pour une pile, ``"image"`` pour une image seule).
    peaks :
        Tous les pics détectés, par variance décroissante.
    rows, cols :
        Bruit de ligne et de colonne (`None` si non demandé).
    """

    searches: dict[str, PeakSearch]
    peaks: list[StructuredPeak]
    rows: LineNoise | None = None
    cols: LineNoise | None = None

    def of_kind(self, kind: str) -> list[StructuredPeak]:
        """Pics d'une nature donnée."""
        return [q for q in self.peaks if q.kind == kind]


def _one_sided_density(seq: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    s = seq - seq.mean()
    n = s.size
    power = np.abs(np.fft.rfft(s)) ** 2 / n**2
    power[1:] *= 2.0
    if n % 2 == 0:
        power[-1] /= 2.0
    f = np.fft.rfftfreq(n)
    return f[1:], power[1:] * n


def _line_noise(x: NDArray[np.float64], axis: int) -> LineNoise:
    """
    x : pile (K, H, W) de résidus temporels (moyenne temporelle retirée).
    axis = 2 : moyennes de lignes (sur W) ; axis = 1 : de colonnes (sur H).
    """
    k = x.shape[0]
    means = x.mean(axis=axis)                                   # (K, H) ou (K, W)
    n_avg = x.shape[axis]
    resid = x - np.expand_dims(means, axis)
    # Variance pixel corrigée : retrait de la moyenne (n_avg) et de la moyenne temporelle (K).
    corr = (n_avg / (n_avg - 1)) * (k / (k - 1) if k > 1 else 1.0)
    pix_var = float(resid.var() * corr)
    floor = pix_var / n_avg
    seq = means.ravel()                                         # séquence de lecture continue
    seq_var = float(seq.var() * (k / (k - 1) if k > 1 else 1.0))
    var = seq_var - floor
    n_seq = seq.size
    sigma = float(np.sqrt(max(var, 0.0)))
    var_d = 2.0 * seq_var**2 / n_seq                            # ≈ séquence de valeurs indépendantes
    u_sigma = float(np.sqrt(var_d) / (2 * sigma)) if sigma > 0 else float(var_d**0.25)

    f, spec = _one_sided_density(seq)
    beta = u_beta = None
    excess_floor = 2.0 * floor
    # Ajustement de la pente sur des bandes log de fréquences où le bruit de ligne domine.
    edges = np.geomspace(f[0], f[-1] * 1.0001, 25)
    fb, sb, nb = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (f >= lo) & (f < hi)
        if sel.sum() >= 3:
            fb.append(np.exp(np.mean(np.log(f[sel]))))
            sb.append(spec[sel].mean())
            nb.append(sel.sum())
    fb, sb, nb = map(np.asarray, (fb, sb, nb))
    good = sb > 3.0 * excess_floor
    if good.sum() >= 4:
        y = np.log(sb[good] - excess_floor)
        xlog = np.log(fb[good])
        # Erreur relative d'une moyenne de n valeurs d'un périodogramme ≈ 1/√n (en log).
        wts = nb[good] * ((sb[good] - excess_floor) / sb[good]) ** 2
        coef, cov = np.polyfit(xlog, y, 1, w=np.sqrt(wts), cov="unscaled")
        beta, u_beta = float(-coef[0]), float(np.sqrt(cov[0, 0]))
    return LineNoise(sigma=sigma, u_sigma=u_sigma, floor_variance=floor, beta=beta, u_beta=u_beta,
                     frequency=f, spectrum=spec)


def _gamma_dof(values: NDArray[np.float64]) -> float:
    """
    Degrés de liberté ν d'une loi Gamma(ν) à partir du rapport de ses
    quantiles 25 % / 50 % (invariant d'échelle, robuste aux pics).
    """
    from scipy.optimize import brentq

    v = values[np.isfinite(values) & (values > 0)]
    r = float(np.quantile(v, 0.25) / np.quantile(v, 0.5))

    def f(log_nu):
        nu = np.exp(log_nu)
        return stats.gamma.ppf(0.25, nu) / stats.gamma.ppf(0.5, nu) - r

    lo, hi = np.log(0.3), np.log(1e5)
    if f(lo) >= 0:
        return 0.3
    if f(hi) <= 0:
        return 1e5
    return float(np.exp(brentq(f, lo, hi)))


def _background(spec: NDArray[np.float64], c: int, size: int) -> tuple[NDArray[np.float64], NDArray[np.bool_], list]:
    """
    Fond lisse : médiane locale 2D hors des bandes d'axes ; médiane 1D le
    long de chaque ligne de fréquence des bandes |f_y| ≤ 1/P et |f_x| ≤ 1/P
    (crêtes des bruits de ligne et de colonne, élargies par le lobe
    principal de la fenêtre de Hann). Renvoie aussi le coin central et les
    régions de seuil.
    """
    bg = ndimage.median_filter(spec, size=size, mode="wrap")
    corner = np.zeros_like(spec, dtype=bool)
    corner[c - 1:c + 2, c - 1:c + 2] = True
    regions, in_band = [], np.zeros_like(spec, dtype=bool)
    # Médiane 1D plus longue (2·size + 1) : une ligne de fréquence offre
    # moins de valeurs qu'un voisinage 2D, et un fond bruité alourdit la
    # queue du rapport NPS/fond.
    for d in (-1, 0, 1):
        bg[c + d, :] = ndimage.median_filter(spec[c + d, :], size=2 * size + 1, mode="wrap")
        bg[:, c + d] = ndimage.median_filter(spec[:, c + d], size=2 * size + 1, mode="wrap")
    for d in (-1, 0, 1):
        for axis in (0, 1):
            m = np.zeros_like(spec, dtype=bool)
            if axis == 0:
                m[c + d, :] = True
            else:
                m[:, c + d] = True
            m &= ~corner
            in_band |= m
            regions.append(m)
    regions.insert(0, ~in_band & ~corner)
    bg[corner] = spec[corner]
    return np.maximum(bg, np.finfo(float).tiny), corner, regions


def _search(nps: NPSResult, kind: str, false_alarm: float, size: int,
            extra_background: NDArray[np.float64] | None = None) -> tuple[PeakSearch, list[StructuredPeak]]:
    """Seuils par région (loi Gamma), maxima locaux, intégration des pics."""
    p = nps.patch_size
    c = p // 2
    spec = nps.nps
    bg, corner, regions = _background(spec, c, size)
    if extra_background is not None:
        bg = bg + extra_background
    ratio = spec / bg
    f = nps.frequency
    fgrid = np.hypot(f[:, None], f[None, :])
    low = fgrid < 3.0 / p
    q = false_alarm / (p * p // 2)
    thr = np.full(spec.shape, np.inf)
    nu_map = np.full(spec.shape, np.nan)
    # ν hors axes, estimé sur la plupart des fréquences, borne celui des
    # lignes des bandes d'axes : un motif fixe de ligne ou de colonne ne
    # peut que réduire le nombre de moyennes indépendantes. Sans cette
    # borne, ν estimé sur ~P valeurs fluctue fortement aux grands ν (le
    # rapport des quantiles tend vers 1) et le seuil devient trop confiant.
    #
    # Sur une bande d'axe, un motif fixe le long de l'axe (profil de
    # colonnes pour f_y ≈ 0, de lignes pour f_x ≈ 0) est le même dans
    # toutes les imagettes alignées : le nombre de moyennes indépendantes
    # peut tomber au nombre de positions en travers de l'axe × images,
    # ×0,95 (Welch, Hann, recouvrement de 50 %). L'estimateur par quantiles
    # le surestime sur une seule ligne (le fond médian suit ses
    # fluctuations) : on retient cette borne, prudente.
    nu_off = _gamma_dof(ratio[regions[0] & ~low])
    welch = 0.95
    bound = {0: welch * nps.n_images * nps.n_patch_cols,     # bande f_y ≈ 0 : structure par colonnes
             1: welch * nps.n_images * nps.n_patch_rows}     # bande f_x ≈ 0 : structure par lignes
    for i, region in enumerate(regions):
        if i == 0:
            nu = nu_off
        else:
            axis = (i - 1) % 2                               # ordre de _background : ligne c+d puis colonne c+d
            nu = min(_gamma_dof(ratio[region & ~low]), nu_off, max(bound[axis], 1.0))
        thr[region] = stats.gamma.isf(q, nu) / stats.gamma.median(nu)
        nu_map[region] = nu
    # Basses fréquences (|f| < 3/P) et coin central exclus : le retrait du
    # plan et la pente d'un bruit en 1/f y faussent le fond.
    thr[low | corner] = np.inf

    half = np.zeros_like(spec, dtype=bool)
    half[c + 1:, :] = True
    half[c, c + 1:] = True
    local_max = spec == ndimage.maximum_filter(spec, size=3, mode="wrap")
    cand = (ratio > thr) & half & local_max
    peaks = []
    for iy, ix in zip(*np.nonzero(cand)):
        ys = np.arange(iy - 2, iy + 3) % p
        xs = np.arange(ix - 2, ix + 3) % p
        box = np.ix_(ys, xs)
        excess = np.clip(spec[box] - bg[box], 0, None)
        if excess.sum() <= 0:
            continue
        fy_c = float(np.sum(excess * f[ys][:, None]) / excess.sum())
        fx_c = float(np.sum(excess * f[xs][None, :]) / excess.sum())
        var = 2.0 * float(excess.sum() / p**2)                   # pic et son symétrique
        rms = float(np.sqrt(var))
        peaks.append(StructuredPeak(
            kind=kind, fx=fx_c, fy=fy_c, frequency=float(np.hypot(fx_c, fy_c)),
            orientation=float(np.degrees(np.arctan2(fy_c, fx_c))),
            variance=var, rms=rms, amplitude=float(np.sqrt(2.0) * rms),
            variance_fraction=var / nps.variance,
            significance=float((ratio[iy, ix] - 1.0) * np.sqrt(nu_map[iy, ix])),
        ))
    return PeakSearch(kind=kind, nps=nps, background=bg, threshold=thr, dof=nu_map), peaks


def find_structured_noise(
    data: ArrayLike,
    *,
    patch_size: int = 64,
    false_alarm: float = 1e-3,
    background_size: int = 9,
    min_variance_fraction: float = 0.0,
    line_noise: bool = True,
) -> StructuredNoiseResult:
    """
    Détecte les motifs périodiques et mesure le bruit de ligne et de colonne.

    Parameters
    ----------
    data :
        Pile (K ≥ 2, H, W) ou image (H, W) de zones uniformes (obscurité
        ou champ uniforme).
    patch_size :
        Taille des imagettes du NPS (fenêtre de Hann, retrait d'un plan).
    false_alarm :
        Probabilité de fausse alarme par NPS, pour l'ensemble du demi-plan
        de fréquences (correction de Bonferroni).
    background_size :
        Taille de la médiane locale qui estime le fond lisse.
    min_variance_fraction :
        Seuil d'importance pratique, en plus du test statistique : un pic
        n'est retenu que si sa variance dépasse cette fraction de la
        variance du NPS (0 : test statistique seul).
    line_noise :
        Mesure aussi le bruit de ligne et de colonne. Sur une pile, la
        moyenne temporelle est retirée d'abord : c'est le bruit **temporel**.
        Sur une image seule, il inclut le motif fixe de ligne.

    Notes
    -----
    Sur une pile, deux recherches séparées évitent de mélanger dans un
    même NPS des composantes de statistiques différentes :

    - ``"temporal"`` : NPS des différences de poses (motifs fixes annulés) ;
    - ``"fixed"`` : NPS de l'image moyenne. Un motif temporel y laisse un
      résidu de variance divisée par K, prévu à partir du NPS temporel et
      ajouté au fond.

    Le bruit de ligne forme une crête sur l'axe f_x = 0 (bruit de colonne :
    f_y = 0) ; il est mesuré à part (`rows`, `cols`), pas comme pic.
    """
    x = np.asarray(data, dtype=np.float64)
    if x.ndim == 2:
        x = x[None]
    if x.ndim != 3:
        raise ValueError("data doit être une image (H, W) ou une pile (K, H, W)")
    k = x.shape[0]
    searches: dict[str, PeakSearch] = {}
    peaks: list[StructuredPeak] = []
    if k >= 2:
        nps_t = noise_power_spectrum(x, patch_size=patch_size, mode="difference")
        s_t, p_t = _search(nps_t, "temporal", false_alarm, background_size)
        nps_f = noise_power_spectrum(x.mean(axis=0), patch_size=patch_size)
        leak = np.clip(nps_t.nps - s_t.background, 0, None) / k   # résidu temporel dans la moyenne
        s_f, p_f = _search(nps_f, "fixed", false_alarm, background_size, extra_background=leak)
        # Un motif temporel moyenné sur K poses laisse un résidu dont la
        # variance suit une loi ≈ exponentielle de moyenne var/K : un pic
        # fixe coïncidant avec un pic temporel n'est gardé que s'il dépasse
        # le quantile 1 − p_FA de cette loi, (var/K)·ln(1/p_FA).
        bin_ = 1.5 / nps_t.patch_size
        kept = []
        for q in p_f:
            twin = [r for r in p_t if abs(r.fx - q.fx) < bin_ and abs(r.fy - q.fy) < bin_]
            if twin and q.variance < max(r.variance for r in twin) / k * np.log(1.0 / false_alarm):
                continue
            kept.append(q)
        searches.update(temporal=s_t, fixed=s_f)
        peaks += p_t + kept
    else:
        nps_i = noise_power_spectrum(x[0], patch_size=patch_size)
        s_i, p_i = _search(nps_i, "image", false_alarm, background_size)
        searches["image"] = s_i
        peaks += p_i
    peaks = [q for q in peaks if q.variance_fraction >= min_variance_fraction]
    peaks.sort(key=lambda q: -q.variance)

    rows = cols = None
    if line_noise:
        resid = x - x.mean(axis=0) if k > 1 else x - x.mean()
        rows = _line_noise(resid, axis=2)
        # Colonnes : retrait de la moyenne de chaque pose, sinon la part du
        # bruit de ligne commune à toute la pose passe pour du bruit de colonne.
        cols = _line_noise(resid - resid.mean(axis=(1, 2), keepdims=True), axis=1)
    return StructuredNoiseResult(searches=searches, peaks=peaks, rows=rows, cols=cols)
