"""
`variogram_noise` — variance du bruit et SNR d'une scène par la pépite du
variogramme.

Le variogramme γ(h) = ½·E[(Z(x + h) − Z(x))²] d'une image Z = S + N, avec
un bruit blanc N de variance σ² indépendant de la scène S, vaut
γ_Z(h) = γ_S(h) + σ² pour h ≠ 0 : le bruit ajoute un saut σ² à l'origine,
la « pépite », que l'on obtient en extrapolant à h = 0 les premières
valeurs du variogramme (Curran et Dungan, 1989).

Chaîne : découpage en tuiles, retrait d'un plan par tuile, sélection des
tuiles les plus lisses sur un critère peu sensible au bruit (gradient de
l'image lissée), variogramme le long des lignes et des colonnes,
extrapolation à h = 0, moyenne des tuiles. La dispersion entre tuiles
donne l'incertitude ; la répartition des tuiles en luminance donne un
modèle σ² = a + b·μ exploitable par `snr_at_levels`.

Voir docs/technique/bruit-variogramme/index.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import ndimage

__all__ = ["VariogramResult", "VariogramNoiseResult", "variogram", "variogram_noise"]

Model = Literal["linear", "quadratic"]

#: Coefficients d'extrapolation à h = 0 sur γ(1), γ(2), γ(3).
#: linéaire : droite par h = 1, 2 → 2γ(1) − γ(2) ;
#: quadratique : parabole par h = 1, 2, 3 → 3γ(1) − 3γ(2) + γ(3).
_EXTRAPOLATION = {"linear": np.array([2.0, -1.0, 0.0]), "quadratic": np.array([3.0, -3.0, 1.0])}


# --------------------------------------------------------------------------
# Tuiles
# --------------------------------------------------------------------------

def _tiles(image: NDArray[np.float64], tile: int) -> tuple[NDArray[np.float64], NDArray[np.intp]]:
    """Tuiles (n, tile, tile) sans recouvrement et position (ligne, colonne) de leur coin."""
    m, w = image.shape[0] // tile, image.shape[1] // tile
    if m == 0 or w == 0:
        raise ValueError(f"image {image.shape} plus petite qu'une tuile de {tile} pixels")
    t = image[: m * tile, : w * tile].reshape(m, tile, w, tile).swapaxes(1, 2).reshape(-1, tile, tile)
    ii, jj = np.mgrid[0:m, 0:w]
    return t, np.column_stack([ii.ravel() * tile, jj.ravel() * tile])


def _detrend(tiles: NDArray[np.float64]) -> NDArray[np.float64]:
    """Retire de chaque tuile le plan ajusté par moindres carrés."""
    n, t, _ = tiles.shape
    yy, xx = np.mgrid[0:t, 0:t]
    design = np.column_stack([xx.ravel(), yy.ravel(), np.ones(t * t)])
    coef = tiles.reshape(n, -1) @ np.linalg.pinv(design).T
    return tiles - (coef @ design.T).reshape(n, t, t)


def _detrend_gamma(tile: int, max_lag: int, axis: int) -> NDArray[np.float64]:
    """γ(h) d'un bruit blanc de variance 1 après retrait du plan, h = 1 … max_lag.

    Le résidu r = (I − H)·n d'un bruit blanc a la covariance σ²(I − H), H étant
    la projection sur le plan (H_ab = d_aᵀ (DᵀD)⁻¹ d_b, d_a = (x_a, y_a, 1)) :
    γ(h) = σ²·[1 − ½(H_aa + H_bb) + H_ab], moyenné sur les paires (a, b)
    distantes de h. Le variogramme et la pépite sont divisés par ce facteur,
    calculé exactement. L'effet croît avec h et décroît avec la taille de
    tuile : 1,0002 à h = 1 mais 0,975 à h = 8 pour T = 16.
    """
    yy, xx = np.mgrid[0:tile, 0:tile]
    d = np.stack([xx, yy, np.ones_like(xx)], axis=-1).astype(np.float64)   # (T, T, 3)
    m = np.linalg.inv(d.reshape(-1, 3).T @ d.reshape(-1, 3))
    hd = np.einsum("...i,ij,...j->...", d, m, d)                             # H_aa
    out = np.empty(max_lag)
    for lag in range(1, max_lag + 1):
        if axis == 2:
            da, db, ha, hb = d[:, :-lag], d[:, lag:], hd[:, :-lag], hd[:, lag:]
        else:
            da, db, ha, hb = d[:-lag, :], d[lag:, :], hd[:-lag, :], hd[lag:, :]
        hab = np.einsum("...i,ij,...j->...", da, m, db)
        out[lag - 1] = np.mean(1.0 - 0.5 * (ha + hb) + hab)
    return out


def _smoothness(tiles: NDArray[np.float64], sigma: float) -> NDArray[np.float64]:
    """Énergie de gradient de chaque tuile lissée (σ pixels) : critère de sélection.

    Le lissage atténue fortement le bruit pixel à pixel : le critère dépend
    surtout de la scène, ce qui évite de retenir les tuiles où le bruit est
    par hasard faible (biais de sélection).
    """
    s = ndimage.gaussian_filter(tiles, sigma=(0, sigma, sigma))
    gy, gx = np.gradient(s, axis=(1, 2))
    return np.mean(gx**2 + gy**2, axis=(1, 2))


def _gamma(tiles: NDArray[np.float64], max_lag: int, axis: int) -> NDArray[np.float64]:
    """γ(h) de chaque tuile le long des colonnes (axis=2, « lignes ») ou des lignes (axis=1)."""
    out = np.empty((tiles.shape[0], max_lag))
    for h in range(1, max_lag + 1):
        if axis == 2:
            d = tiles[:, :, h:] - tiles[:, :, :-h]
        else:
            d = tiles[:, h:, :] - tiles[:, :-h, :]
        out[:, h - 1] = 0.5 * np.mean(d**2, axis=(1, 2))
    return out


def _select(image: ArrayLike, tile: int, keep_fraction: float, saturation: float | None,
            selection_sigma: float):
    img = np.asarray(image, dtype=np.float64)
    if img.ndim != 2:
        raise ValueError(f"image 2D attendue, reçu ndim={img.ndim}")
    if not 0.0 < keep_fraction <= 1.0:
        raise ValueError("keep_fraction doit être dans ]0, 1]")
    if tile < 16:
        raise ValueError("tile doit valoir au moins 16 pixels")
    raw, corners = _tiles(img, tile)
    valid = np.ones(raw.shape[0], dtype=bool)
    if saturation is not None:
        valid &= np.all(raw < saturation, axis=(1, 2))
    if not np.any(valid):
        raise ValueError("aucune tuile valide (toutes contiennent des pixels saturés)")
    raw, corners = raw[valid], corners[valid]
    flat = _detrend(raw)
    crit = _smoothness(flat, selection_sigma)
    n_keep = max(1, int(round(keep_fraction * raw.shape[0])))
    idx = np.sort(np.argsort(crit, kind="stable")[:n_keep])
    return raw[idx], flat[idx], corners[idx], int(valid.sum())


# --------------------------------------------------------------------------
# Variogramme
# --------------------------------------------------------------------------

@dataclass(slots=True)
class VariogramResult:
    """
    Variogramme moyen des tuiles retenues.

    Attributes
    ----------
    lags :
        Distances h = 1 … max_lag (pixels).
    gamma_rows, gamma_cols :
        γ(h) le long des lignes (paires sur une même ligne, décalées de h
        colonnes) et le long des colonnes (DN²).
    u_gamma_rows, u_gamma_cols :
        Incertitude-type : dispersion entre tuiles / √n.
    n_tiles :
        Nombre de tuiles retenues.
    """

    lags: NDArray[np.int_]
    gamma_rows: NDArray[np.float64]
    gamma_cols: NDArray[np.float64]
    u_gamma_rows: NDArray[np.float64]
    u_gamma_cols: NDArray[np.float64]
    n_tiles: int

    @property
    def gamma(self) -> NDArray[np.float64]:
        """Moyenne des deux directions."""
        return 0.5 * (self.gamma_rows + self.gamma_cols)


def variogram(
    image: ArrayLike,
    *,
    max_lag: int = 8,
    tile: int = 16,
    keep_fraction: float = 0.5,
    saturation: float | None = None,
    selection_sigma: float = 3.0,
) -> VariogramResult:
    """
    Variogramme directionnel d'une image, moyenné sur les tuiles les plus lisses.

    Parameters
    ----------
    image :
        Image 2D en DN, brute de préférence (voir les limites, chapitre 14).
    max_lag :
        Distance maximale (pixels), au plus tile/2.

    Notes
    -----
    La sélection des tuiles lisses (``keep_fraction`` < 1) retient un peu
    plus souvent les tuiles où le bruit basse fréquence est faible : sur un
    bruit blanc, la courbe est creusée de −0,04 % à h = 1 et de −1,7 % à
    h = 8 (T = 16, ``keep_fraction`` = 0,5). Pour tracer la forme exacte du
    variogramme d'un bruit, utiliser ``keep_fraction=1``.
    tile, keep_fraction, saturation, selection_sigma :
        Voir :func:`variogram_noise`.
    """
    if not 1 <= max_lag <= tile // 2:
        raise ValueError("max_lag doit être compris entre 1 et tile/2")
    _, flat, _, _ = _select(image, tile, keep_fraction, saturation, selection_sigma)
    n = flat.shape[0]
    gr = _gamma(flat, max_lag, 2) / _detrend_gamma(tile, max_lag, 2)
    gc = _gamma(flat, max_lag, 1) / _detrend_gamma(tile, max_lag, 1)
    u = (lambda g: g.std(axis=0, ddof=1) / np.sqrt(n) if n > 1 else np.full(max_lag, np.nan))
    return VariogramResult(np.arange(1, max_lag + 1), gr.mean(axis=0), gc.mean(axis=0), u(gr), u(gc), n)


# --------------------------------------------------------------------------
# Pépite et SNR
# --------------------------------------------------------------------------

@dataclass(slots=True)
class VariogramNoiseResult:
    """
    Variance du bruit par la pépite du variogramme.

    Attributes
    ----------
    variance, u_variance :
        Pépite σ² (moyenne des deux directions et des tuiles) et son
        incertitude-type (DN²).
    variance_rows, u_variance_rows, variance_cols, u_variance_cols :
        Pépite le long des lignes et des colonnes.
    anisotropy :
        (pépite lignes − pépite colonnes) / incertitude combinée. |·| > 5
        signale un bruit structuré (rayures, bruit de ligne ou de colonne) :
        la pépite de la direction la plus basse est alors celle du bruit
        blanc, l'autre y ajoute le bruit structuré.
    gamma1 :
        γ(1) moyen (DN²).
    nugget_fraction :
        F = σ² / γ(1) : part du bruit dans l'écart entre pixels voisins.
        Proche de 1 : la scène varie peu d'un pixel au suivant, la pépite est
        fiable. Faible ou négative : scène trop texturée à l'échelle du pixel,
        ou bruit corrélé (image traitée) — voir `reliable`.
    reliable :
        F ≥ `min_nugget_fraction`, pépite significative (> 3 u) et pas
        d'anisotropie (|anisotropy| ≤ 5).
    mean, snr, u_snr :
        Luminance moyenne des tuiles retenues (DN), SNR = mean/σ et son
        incertitude-type.
    a, b, cov :
        Modèle σ² = a + b·μ ajusté sur les tuiles (moindres carrés, covariance
        tirée des résidus), utilisable par `snr_at_levels`. NaN si moins de
        trois tuiles ou si leurs luminances ne varient pas.
    tile_variance, tile_mean, tile_corner :
        Pépite, luminance moyenne et coin (ligne, colonne) de chaque tuile
        retenue.
    model, tile, n_tiles, n_valid :
        Modèle d'extrapolation, taille des tuiles, nombre de tuiles retenues
        et de tuiles valides (avant sélection).
    """

    variance: float
    u_variance: float
    variance_rows: float
    u_variance_rows: float
    variance_cols: float
    u_variance_cols: float
    anisotropy: float
    gamma1: float
    nugget_fraction: float
    reliable: bool
    mean: float
    snr: float
    u_snr: float
    a: float
    b: float
    cov: NDArray[np.float64]
    tile_variance: NDArray[np.float64]
    tile_mean: NDArray[np.float64]
    tile_corner: NDArray[np.intp]
    model: str
    tile: int
    n_tiles: int
    n_valid: int
    params: dict = field(default_factory=dict)

    @property
    def sigma(self) -> float:
        """Écart-type du bruit (NaN si la pépite est négative)."""
        return float(np.sqrt(self.variance)) if self.variance > 0 else float("nan")


def _mean_u(x: NDArray[np.float64]) -> tuple[float, float]:
    n = x.size
    return float(x.mean()), float(x.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")


def _noise_model(mu: NDArray[np.float64], var: NDArray[np.float64]) -> tuple[float, float, NDArray[np.float64]]:
    """Moindres carrés var = a + b·mu ; covariance s²·(XᵀX)⁻¹."""
    nan = (float("nan"), float("nan"), np.full((2, 2), np.nan))
    if mu.size < 3 or np.ptp(mu) <= 1e-9 * max(1.0, abs(mu.mean())):
        return nan
    x = np.column_stack([np.ones_like(mu), mu])
    coef, *_ = np.linalg.lstsq(x, var, rcond=None)
    resid = var - x @ coef
    s2 = float(resid @ resid) / (mu.size - 2)
    return float(coef[0]), float(coef[1]), s2 * np.linalg.inv(x.T @ x)


def variogram_noise(
    image: ArrayLike,
    *,
    tile: int = 16,
    keep_fraction: float = 0.5,
    model: Model = "linear",
    saturation: float | None = None,
    selection_sigma: float = 3.0,
    min_nugget_fraction: float = 0.8,
) -> VariogramNoiseResult:
    """
    Variance du bruit et SNR d'une image par la pépite du variogramme.

    Parameters
    ----------
    image :
        Image 2D en DN. Le bruit doit être blanc : image brute, avant
        dématriçage, rééchantillonnage, compression ou débruitage.
    tile :
        Côté des tuiles (pixels), 16 par défaut : de petites tuiles évitent
        mieux les bords d'objets (chapitre 14, tableau 2). Chaque tuile donne
        une estimation ; elles doivent être assez nombreuses (≥ 30) pour
        l'incertitude. Le biais du retrait du plan est corrigé exactement.
    keep_fraction :
        Fraction des tuiles retenues, les plus lisses d'après l'énergie de
        gradient de la tuile lissée (écarte les bords d'objets).
    model :
        ``"linear"`` (défaut) : 2γ(1) − γ(2) ; ``"quadratic"`` :
        3γ(1) − 3γ(2) + γ(3), pour une scène lisse (courbe en h² à
        l'origine), au prix d'une variance plus forte.
    saturation :
        Si fourni, les tuiles contenant un pixel ≥ ce niveau sont exclues
        (l'écrêtage réduit la variance).
    selection_sigma :
        Écart-type (pixels) du lissage du critère de sélection. Assez fort
        pour que le bruit n'influe presque plus sur le choix des tuiles :
        biais de sélection mesuré à +0,1 % sur bruit pur (chapitre 14).
    min_nugget_fraction :
        Seuil de F = σ²/γ(1) au-dessous duquel l'estimation est déclarée non
        fiable.
    """
    if model not in _EXTRAPOLATION:
        raise ValueError(f"model doit valoir 'linear' ou 'quadratic' (reçu {model!r})")
    raw, flat, corners, n_valid = _select(image, tile, keep_fraction, saturation, selection_sigma)
    coefs = _EXTRAPOLATION[model]
    # γ corrigé du retrait du plan, distance par distance
    gr = _gamma(flat, 3, 2) / _detrend_gamma(tile, 3, 2)
    gc = _gamma(flat, 3, 1) / _detrend_gamma(tile, 3, 1)
    nug_r, nug_c = gr @ coefs, gc @ coefs
    nug = 0.5 * (nug_r + nug_c)
    tile_mean = raw.mean(axis=(1, 2))

    var, u_var = _mean_u(nug)
    var_r, u_r = _mean_u(nug_r)
    var_c, u_c = _mean_u(nug_c)
    g1 = float(0.5 * (gr[:, 0] + gc[:, 0]).mean())
    fraction = var / g1 if g1 > 0 else float("nan")
    anis = (var_r - var_c) / np.hypot(u_r, u_c) if np.hypot(u_r, u_c) > 0 else float("nan")
    reliable = bool(np.isfinite(fraction) and fraction >= min_nugget_fraction and var > 3.0 * u_var
                    and np.isfinite(anis) and abs(anis) <= 5.0)

    mean = float(tile_mean.mean())
    if var > 0:
        snr = mean / np.sqrt(var)
        u_snr = snr * u_var / (2.0 * var)
    else:
        snr = u_snr = float("nan")
    a, b, cov = _noise_model(tile_mean, nug)
    return VariogramNoiseResult(
        variance=var, u_variance=u_var, variance_rows=var_r, u_variance_rows=u_r,
        variance_cols=var_c, u_variance_cols=u_c, anisotropy=float(anis), gamma1=g1,
        nugget_fraction=float(fraction), reliable=reliable, mean=mean, snr=float(snr),
        u_snr=float(u_snr), a=a, b=b, cov=cov, tile_variance=nug, tile_mean=tile_mean,
        tile_corner=corners, model=model, tile=tile, n_tiles=int(nug.size), n_valid=n_valid,
        params=dict(keep_fraction=keep_fraction, saturation=saturation,
                    selection_sigma=selection_sigma, min_nugget_fraction=min_nugget_fraction),
    )
