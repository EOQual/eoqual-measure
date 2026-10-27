"""
`cnr_target` — contraste sur bruit (CNR) d'une mire à deux plages.

CNR = (μ_clair − μ_sombre) / √(σ²_clair + σ²_sombre), où μ et σ² sont la
moyenne et la variance des pixels de chaque plage d'**une** image : le
bruit compte le bruit temporel et le bruit spatial fixe dans la plage.
Distinct du SNR radiométrique μ/σ(μ) (`snr_model`). Les plages sont
fournies (masques) ou détectées par le seuil d'Otsu, puis érodées pour
écarter la transition floue entre les deux. Voir REFERENCE_TECHNIQUE.md, partie IV, chapitre 13.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import ndimage


@dataclass(slots=True)
class CNRResult:
    """
    Résultat de `measure_cnr`.

    Attributes
    ----------
    cnr, u_cnr :
        Contraste sur bruit (moyenne sur les images d'une pile) et son
        incertitude-type.
    cnr_per_image :
        CNR de chaque image.
    mean_bright, mean_dark, var_bright, var_dark :
        Moyennes et variances des plages (moyennes sur les images).
    n_bright, n_dark :
        Nombre de pixels de chaque plage.
    bright_mask, dark_mask :
        Plages utilisées.
    threshold :
        Seuil d'Otsu si les plages ont été détectées, sinon `None`.
    """

    cnr: float
    u_cnr: float
    cnr_per_image: NDArray[np.float64]
    mean_bright: float
    mean_dark: float
    var_bright: float
    var_dark: float
    n_bright: int
    n_dark: int
    bright_mask: NDArray[np.bool_]
    dark_mask: NDArray[np.bool_]
    threshold: float | None


def otsu_threshold(values: ArrayLike, bins: int = 256) -> float:
    """Seuil d'Otsu : maximise la variance inter-classes de l'histogramme."""
    v = np.asarray(values, dtype=np.float64).ravel()
    hist, edges = np.histogram(v, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2
    w0 = np.cumsum(hist)
    w1 = w0[-1] - w0
    m0 = np.cumsum(hist * centers)
    mu0 = m0 / np.maximum(w0, 1)
    mu1 = (m0[-1] - m0) / np.maximum(w1, 1)
    between = w0 * w1 * (mu0 - mu1) ** 2
    return float(centers[np.argmax(between[:-1])])


def _cnr_and_u(mb, md, vb, vd, nb, nd):
    """CNR et incertitude (méthode delta : u(μ) = σ/√n, u(σ²) = σ²√(2/(n − 1)))."""
    s = np.sqrt(vb + vd)
    c = (mb - md) / s
    var_c = (vb / nb + vd / nd) / s**2 + (c / (2 * s**2)) ** 2 * (2 * vb**2 / (nb - 1) + 2 * vd**2 / (nd - 1))
    return c, np.sqrt(var_c)


def measure_cnr(
    data: ArrayLike,
    *,
    bright_mask: ArrayLike | None = None,
    dark_mask: ArrayLike | None = None,
    margin: int = 3,
    smoothing: float = 2.0,
) -> CNRResult:
    """
    Mesure le CNR d'une mire à deux plages.

    Parameters
    ----------
    data :
        Image (H, W) ou pile (K, H, W) d'une mire à deux plages.
    bright_mask, dark_mask :
        Plages claire et sombre ; si absentes, détectées par seuil d'Otsu
        sur la moyenne lissée de la pile.
    margin :
        Érosion (pixels) des plages détectées, pour écarter la transition.
    smoothing :
        Écart-type (pixels) du lissage avant seuillage (détection seulement).

    Notes
    -----
    Sur une pile, le CNR est calculé image par image puis moyenné ; son
    incertitude est la plus grande de l'incertitude analytique moyenne /√K
    et de la dispersion entre images /√K.
    """
    x = np.asarray(data, dtype=np.float64)
    if x.ndim == 2:
        x = x[None]
    if x.ndim != 3:
        raise ValueError("data doit être une image (H, W) ou une pile (K, H, W)")
    shape = x.shape[1:]
    thr = None
    if bright_mask is None or dark_mask is None:
        ref = ndimage.gaussian_filter(x.mean(axis=0), smoothing)
        thr = otsu_threshold(ref)
        b = ref > thr
        d = ~b
        if margin > 0:
            b = ndimage.binary_erosion(b, iterations=margin, border_value=0)
            d = ndimage.binary_erosion(d, iterations=margin, border_value=0)
    else:
        b = np.asarray(bright_mask, dtype=bool)
        d = np.asarray(dark_mask, dtype=bool)
        if b.shape != shape or d.shape != shape:
            raise ValueError("masques de forme (H, W) attendus")
    if np.any(b & d):
        raise ValueError("les plages se recouvrent")
    nb, nd = int(b.sum()), int(d.sum())
    if nb < 10 or nd < 10:
        raise ValueError("plages trop petites (moins de 10 pixels)")
    mb = x[:, b].mean(axis=1)
    md = x[:, d].mean(axis=1)
    vb = x[:, b].var(axis=1, ddof=1)
    vd = x[:, d].var(axis=1, ddof=1)
    c, u = _cnr_and_u(mb, md, vb, vd, nb, nd)
    k = x.shape[0]
    u_mean = float(np.mean(u) / np.sqrt(k))
    if k > 1:
        u_mean = max(u_mean, float(np.std(c, ddof=1) / np.sqrt(k)))
    return CNRResult(
        cnr=float(c.mean()), u_cnr=u_mean, cnr_per_image=c,
        mean_bright=float(mb.mean()), mean_dark=float(md.mean()),
        var_bright=float(vb.mean()), var_dark=float(vd.mean()),
        n_bright=nb, n_dark=nd, bright_mask=b, dark_mask=d, threshold=thr,
    )
