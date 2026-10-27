"""
`defective_pixels` — carte des pixels défectueux.

Classes (bits de la carte de codes) :

- 1 ``stuck`` (bloqué) : variance temporelle quasi nulle dans toutes les
  piles (code constant) ; seuil d'une loi du χ² ;
- 2 ``dead`` (mort) : réponse à la lumière < `dead_fraction` × médiane
  locale ;
- 4 ``hot`` (chaud) : repris de `measure_dark_current` (régression sur
  plusieurs temps de pose) ;
- 8 ``noisy`` (bruyant) : variance temporelle d'obscurité au-dessus du
  quantile du χ² de sa variance attendue (pixels télégraphiques…), hors
  pixels chauds. La variance attendue tient compte du bruit d'obscurité
  poissonnien de chaque pixel si `dark_current` et `dark_exposure_time`
  sont fournis ; sinon c'est la médiane des variances (valable pour une
  obscurité courte, où le bruit de lecture domine) ;
- 16 ``response`` (réponse aberrante) : réponse / médiane locale à plus de
  `response_sigma` écarts-types robustes de 1, hors pixels morts.

Les pixels chauds, souvent saturés dans les poses longues (code constant,
réponse apparente nulle), sont exclus des autres classes. La médiane
locale (fenêtre `window`, 5 par défaut) rend le critère de réponse
insensible au vignettage et aux poussières : il ne retient que des
pixels isolés. Voir REFERENCE_TECHNIQUE.md, partie IV, chapitre 7.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import ndimage, stats

from eoqual.measure.radiometry._stack import as_stack
from eoqual.measure.radiometry.dark import DarkCurrentResult

STUCK, DEAD, HOT, NOISY, RESPONSE = 1, 2, 4, 8, 16
CLASSES = {"stuck": STUCK, "dead": DEAD, "hot": HOT, "noisy": NOISY, "response": RESPONSE}


@dataclass(slots=True)
class DefectMap:
    """
    Résultat de `map_defective_pixels`.

    Attributes
    ----------
    codes :
        Carte (H, W) de bits (voir `CLASSES`), 0 pour un pixel sain.
    stuck, dead, hot, noisy, response :
        Masques par classe.
    response_ratio :
        Réponse / médiane locale (1 pour un pixel typique).
    thresholds :
        Seuils utilisés (variance relative du bloqué, du bruyant, réponse
        morte, bornes de réponse).
    """

    codes: NDArray[np.uint8]
    stuck: NDArray[np.bool_]
    dead: NDArray[np.bool_]
    hot: NDArray[np.bool_]
    noisy: NDArray[np.bool_]
    response: NDArray[np.bool_]
    response_ratio: NDArray[np.float64]
    thresholds: dict[str, float]

    @property
    def any(self) -> NDArray[np.bool_]:
        """Masque de tous les pixels défectueux."""
        return self.codes > 0

    def counts(self) -> dict[str, int]:
        """Nombre de pixels par classe."""
        return {name: int(getattr(self, name).sum()) for name in CLASSES}


def map_defective_pixels(
    dark: ArrayLike,
    flats: Sequence[ArrayLike],
    *,
    dark_current: DarkCurrentResult | None = None,
    dark_exposure_time: float | None = None,
    false_alarm: float = 1e-6,
    dead_fraction: float = 0.1,
    response_sigma: float = 6.0,
    window: int = 5,
) -> DefectMap:
    """
    Classe les pixels défectueux.

    Parameters
    ----------
    dark :
        Pile d'obscurité (K ≥ 3, H, W).
    flats :
        Piles de champs uniformes (K ≥ 3) ; la plus lumineuse sert à la
        réponse.
    dark_current :
        Résultat de `measure_dark_current` pour les pixels chauds (sinon la
        classe ``hot`` est vide) et, avec `dark_exposure_time`, pour la
        variance attendue de chaque pixel.
    dark_exposure_time :
        Temps de pose de `dark` (s).
    false_alarm :
        Probabilité de fausse alarme **par pixel** des tests du χ² (bloqué,
        bruyant).
    dead_fraction :
        Réponse relative sous laquelle un pixel est mort.
    response_sigma :
        Écart, en écarts-types robustes (MAD), qui définit une réponse
        aberrante.
    window :
        Taille de la médiane locale de la réponse.
    """
    dk = as_stack(dark, min_frames=3, name="dark")
    fl = [as_stack(f, min_frames=3, name="flats[i]") for f in flats]
    if not fl:
        raise ValueError("au moins une pile de champ uniforme")
    shape = dk.shape[1:]
    if any(f.shape[1:] != shape for f in fl):
        raise ValueError("dark et flats doivent avoir la même taille d'image")

    # Bloqués : variance quasi nulle dans toutes les piles. Sous l'hypothèse
    # d'un pixel sain, (K − 1)·s²/σ² ~ χ²_{K−1} : seuil bas au quantile p_FA,
    # σ² estimé par la médiane des variances de la pile.
    # La médiane d'une variance empirique vaut médiane(χ²_{K−1})/(K−1) fois sa
    # moyenne (0,956 pour K = 16) : on l'en corrige, sinon la référence est trop
    # basse et, aux quantiles extrêmes, les fausses alarmes se multiplient (×3,5).
    def median_scale(k):
        return float(stats.chi2.median(k - 1) / (k - 1))

    stuck = np.ones(shape, dtype=bool)
    stuck_thr = []
    for s in [dk] + fl:
        k = s.shape[0]
        var = s.var(axis=0, ddof=1)
        ref = float(np.median(var)) / median_scale(k)
        thr = stats.chi2.ppf(false_alarm, k - 1) / (k - 1)
        stuck &= var < thr * ref
        stuck_thr.append(thr)

    hot = np.zeros(shape, dtype=bool) if dark_current is None else np.asarray(dark_current.hot_pixels, dtype=bool)
    if hot.shape != shape:
        raise ValueError("dark_current de taille incompatible")
    stuck &= ~hot                       # un pixel chaud saturé sort un code constant

    # Bruyants : variance d'obscurité au-dessus du quantile 1 − p_FA de la
    # variance attendue : b + a·I·t par pixel (bruit d'obscurité poissonnien),
    # ou la médiane des variances à défaut.
    kd = dk.shape[0]
    var_d = dk.var(axis=0, ddof=1)
    noisy_thr = float(stats.chi2.isf(false_alarm, kd - 1) / (kd - 1))
    if dark_current is not None and dark_exposure_time is not None and dark_current.poisson_gain:
        a = 1.0 / dark_current.poisson_gain
        dark_dn = np.nan_to_num(np.clip(dark_current.current_map, 0, None)) * dark_exposure_time
        ordinary = ~hot & ~stuck
        # b tel que R = var / (b + a·d) ait la médiane de χ²_{K−1}/(K−1) :
        # chaque pixel normalisé par sa propre variance attendue.
        from scipy.optimize import brentq

        vv, dd = var_d[ordinary], dark_dn[ordinary]
        target = median_scale(kd)
        def gap(bb):
            return float(np.median(vv / (bb + a * dd))) - target
        hi_b = float(np.median(vv)) * 10 + 1.0
        b = brentq(gap, 1e-9, hi_b) if gap(1e-9) > 0 > gap(hi_b) else float(np.median(vv)) / target
        expected = b + a * dark_dn
    else:
        expected = np.full(shape, float(np.median(var_d)) / median_scale(kd))
    noisy = (var_d > noisy_thr * expected) & ~hot & ~stuck

    # Réponse : flat le plus lumineux − obscurité, rapportée à la médiane locale.
    brightest = max(fl, key=lambda s: float(np.median(s)))
    resp = brightest.mean(axis=0) - dk.mean(axis=0)
    local = ndimage.median_filter(resp, size=window, mode="reflect")
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(local > 0, resp / local, np.nan)
    dead = (ratio < dead_fraction) & ~stuck & ~hot
    core = ratio[~dead & ~stuck & ~hot & np.isfinite(ratio)]
    sd = 1.4826 * float(np.median(np.abs(core - np.median(core))))
    lo, hi = 1 - response_sigma * sd, 1 + response_sigma * sd
    response = ((ratio < lo) | (ratio > hi)) & ~dead & ~stuck & ~hot & np.isfinite(ratio)

    codes = (stuck * STUCK + dead * DEAD + hot * HOT + noisy * NOISY + response * RESPONSE).astype(np.uint8)
    return DefectMap(
        codes=codes, stuck=stuck, dead=dead, hot=hot, noisy=noisy, response=response,
        response_ratio=ratio,
        thresholds={"stuck_variance": float(min(stuck_thr)), "noisy_variance": noisy_thr,
                    "dead_response": dead_fraction, "response_low": lo, "response_high": hi},
    )


# --------------------------------------------------------------------------
# Pixels télégraphiques
# --------------------------------------------------------------------------

@dataclass(slots=True)
class RTSResult:
    """
    Résultat de `find_rts_pixels`.

    Attributes
    ----------
    index :
        Indices aplatis des pixels télégraphiques détectés.
    low, high, amplitude :
        Niveaux bas et haut (DN) et amplitude du saut, par pixel détecté.
    noise :
        Écart-type du bruit autour de chaque niveau (DN).
    p_up, p_down :
        Probabilités de bascule par pose (bas → haut, haut → bas), tirées du
        décodage de Viterbi.
    high_fraction :
        Part des poses à l'état haut.
    delta_bic :
        Gain de BIC du modèle à deux niveaux sur le modèle gaussien.
    states :
        États décodés (K, n_détectés), vrai : niveau haut.
    n_candidates :
        Pixels présélectionnés (variance excessive).
    """

    index: NDArray[np.int64]
    low: NDArray[np.float64]
    high: NDArray[np.float64]
    amplitude: NDArray[np.float64]
    noise: NDArray[np.float64]
    p_up: NDArray[np.float64]
    p_down: NDArray[np.float64]
    high_fraction: NDArray[np.float64]
    delta_bic: NDArray[np.float64]
    states: NDArray[np.bool_]
    n_candidates: int


def _em_two_levels(x: NDArray[np.float64], n_iter: int = 60):
    """
    Mélange de deux gaussiennes de même variance, par pixel (colonnes de x).

    Renvoie μ_bas, μ_haut, σ, poids du haut, log-vraisemblances du mélange
    et d'une gaussienne seule.
    """
    k = x.shape[0]
    mu1 = np.quantile(x, 0.25, axis=0)
    mu2 = np.quantile(x, 0.75, axis=0)
    var = x.var(axis=0) / 4 + 1e-6
    w = np.full(x.shape[1], 0.5)
    for _ in range(n_iter):
        l1 = (1 - w) * np.exp(-0.5 * (x - mu1) ** 2 / var)
        l2 = w * np.exp(-0.5 * (x - mu2) ** 2 / var)
        r = l2 / np.maximum(l1 + l2, 1e-300)
        w = np.clip(r.mean(axis=0), 1e-6, 1 - 1e-6)
        mu1 = ((1 - r) * x).sum(axis=0) / np.maximum((1 - r).sum(axis=0), 1e-12)
        mu2 = (r * x).sum(axis=0) / np.maximum(r.sum(axis=0), 1e-12)
        var = ((1 - r) * (x - mu1) ** 2 + r * (x - mu2) ** 2).mean(axis=0) + 1e-9
    dens = ((1 - w) * np.exp(-0.5 * (x - mu1) ** 2 / var) + w * np.exp(-0.5 * (x - mu2) ** 2 / var)) / np.sqrt(2 * np.pi * var)
    ll2 = np.log(np.maximum(dens, 1e-300)).sum(axis=0)
    v1 = x.var(axis=0) + 1e-12
    ll1 = -0.5 * k * (np.log(2 * np.pi * v1) + 1)
    lo, hi = np.minimum(mu1, mu2), np.maximum(mu1, mu2)
    w_hi = np.where(mu2 >= mu1, w, 1 - w)
    return lo, hi, np.sqrt(var), w_hi, ll2, ll1


def _viterbi_two_states(x, lo, hi, sigma, p_up, p_down, n_iter=5):
    """
    Décodage de Viterbi d'une chaîne à deux états gaussiens, par pixel, puis
    réestimation des probabilités de bascule (apprentissage de Viterbi).
    """
    k, n = x.shape
    for _ in range(n_iter):
        e_lo = -0.5 * ((x - lo) / sigma) ** 2
        e_hi = -0.5 * ((x - hi) / sigma) ** 2
        lt = np.log(np.stack([[1 - p_up, p_up], [p_down, 1 - p_down]]).clip(1e-12))   # (2, 2, n)
        score = np.stack([e_lo[0], e_hi[0]])                                           # (2, n)
        back = np.zeros((k, 2, n), dtype=np.int8)
        for t in range(1, k):
            cand = score[:, None, :] + lt                                              # (from, to, n)
            back[t] = np.argmax(cand, axis=0)
            score = np.max(cand, axis=0) + np.stack([e_lo[t], e_hi[t]])
        states = np.zeros((k, n), dtype=bool)
        s = np.argmax(score, axis=0)
        for t in range(k - 1, -1, -1):
            states[t] = s == 1
            s = back[t, s, np.arange(n)] if t > 0 else s
        low_prev = ~states[:-1]
        n_up = np.sum(low_prev & states[1:], axis=0)
        n_down = np.sum(~low_prev & ~states[1:], axis=0)
        p_up = (n_up + 0.5) / (low_prev.sum(axis=0) + 1.0)
        p_down = (n_down + 0.5) / ((~low_prev).sum(axis=0) + 1.0)
    return states, p_up, p_down


def find_rts_pixels(
    stack: ArrayLike,
    *,
    candidate_false_alarm: float = 1e-3,
    min_delta_bic: float = 10.0,
    exclude: ArrayLike | None = None,
) -> RTSResult:
    """
    Détecte les pixels télégraphiques d'une longue pile temporelle.

    Parameters
    ----------
    stack :
        Pile (K, H, W), K de quelques centaines à quelques milliers de poses
        (obscurité ou champ uniforme stable).
    candidate_false_alarm :
        Présélection : variance > quantile 1 − p du χ² de la variance de
        référence (médiane corrigée de son biais).
    min_delta_bic :
        Gain minimal de BIC du modèle à deux niveaux (10 : preuve forte).
    exclude :
        Pixels à ignorer (chauds, bloqués…).

    Notes
    -----
    Pour chaque candidat : mélange de deux gaussiennes de même variance
    (EM), comparé à une gaussienne seule par
    ΔBIC = 2(ln L₂ − ln L₁) − 2·ln K (deux paramètres de plus), puis
    décodage de Viterbi et réestimation des probabilités de bascule.
    """
    x = as_stack(stack, min_frames=50, name="stack")
    k = x.shape[0]
    flat = x.reshape(k, -1)
    var = flat.var(axis=0, ddof=1)
    excl = np.zeros(flat.shape[1], dtype=bool) if exclude is None else np.asarray(exclude, dtype=bool).ravel()
    ref = float(np.median(var[~excl])) / float(stats.chi2.median(k - 1) / (k - 1))
    thr = float(stats.chi2.isf(candidate_false_alarm, k - 1) / (k - 1))
    cand = np.flatnonzero((var > thr * ref) & ~excl)
    empty = np.zeros(0)
    if cand.size == 0:
        return RTSResult(index=cand.astype(np.int64), low=empty, high=empty, amplitude=empty, noise=empty,
                         p_up=empty, p_down=empty, high_fraction=empty, delta_bic=empty,
                         states=np.zeros((k, 0), bool), n_candidates=0)
    xc = flat[:, cand]
    lo, hi, sig, w_hi, ll2, ll1 = _em_two_levels(xc)
    dbic = 2 * (ll2 - ll1) - 2 * np.log(k)
    keep = dbic > min_delta_bic
    idx = cand[keep]
    lo, hi, sig, w_hi, dbic = lo[keep], hi[keep], sig[keep], w_hi[keep], dbic[keep]
    xk = xc[:, keep]
    # Initialisation des bascules par seuillage au milieu des niveaux.
    s0 = xk > (lo + hi) / 2
    pu0 = (np.sum(~s0[:-1] & s0[1:], axis=0) + 0.5) / ((~s0[:-1]).sum(axis=0) + 1.0)
    pd0 = (np.sum(s0[:-1] & ~s0[1:], axis=0) + 0.5) / (s0[:-1].sum(axis=0) + 1.0)
    states, p_up, p_down = _viterbi_two_states(xk, lo, hi, sig, pu0, pd0) if idx.size else (s0, pu0, pd0)
    return RTSResult(index=idx.astype(np.int64), low=lo, high=hi, amplitude=hi - lo, noise=sig,
                     p_up=p_up, p_down=p_down, high_fraction=states.mean(axis=0), delta_bic=dbic,
                     states=states, n_candidates=int(cand.size))
