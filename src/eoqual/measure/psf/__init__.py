"""
Mesure de la qualité d'une PSF observée sur une source ponctuelle (étoile,
collimateur) : rapport de Strehl et énergie encerclée.

Exemple
-------
>>> from eoqual.measure.psf import measure_psf
>>> res = measure_psf(star, sampling=4.0)    # 4 pixels par λ/D
>>> res.strehl, res.r50, res.r80, res.ee_first_dark_ring

Chaîne : estimation et retrait du fond (couronne au bord de l'imagette),
centroïde, recentrage sous-pixel par translation de Fourier (le pic
mesuré ne doit pas dépendre de la position du centre dans le pixel),
normalisation à la somme 1, énergie encerclée ; puis, si l'échantillonnage
optique ``sampling`` est connu, comparaison du pic à celui d'une tache
d'Airy de même grille (rapport de Strehl).

Briques numériques : :mod:`eoqual.core.psf` (Airy, Strehl, énergie
encerclée) et :mod:`eoqual.core.fourier` (translation sous-pixel).
Validation : :class:`eoqual.measure.synthetic.SyntheticPSF` (aberrations
de Zernike connues, Strehl exact). Voir ``REFERENCE_TECHNIQUE.md``,
partie III.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from eoqual.core import psf as core_psf
from eoqual.core.fourier import translate

__all__ = ["PSFMeasurement", "measure_psf"]

#: Rayon du premier anneau noir de la tache d'Airy, en unités de λ/D.
FIRST_DARK_RING = 1.2197


@dataclass(slots=True)
class PSFMeasurement:
    """
    Résultat de :func:`measure_psf`. Distances en pixels de l'image.

    Attributes
    ----------
    strehl :
        Rapport de Strehl (NaN si ``sampling`` n'est pas fourni).
    radii, encircled_energy :
        Courbe d'énergie encerclée EE(r), normalisée à l'énergie contenue
        dans l'imagette.
    r50, r80 :
        Rayons contenant 50 % et 80 % de l'énergie.
    ee_first_dark_ring :
        EE au premier anneau noir (1.22 λ/D) ; 0.838 pour une pupille
        circulaire parfaite. NaN si ``sampling`` n'est pas fourni.
    centroid :
        Centroïde ``(ligne, colonne)`` de la PSF dans l'image d'entrée.
    background :
        Fond retiré, par pixel.
    psf :
        PSF recentrée, fond retiré, de somme 1.
    reference :
        Tache d'Airy de référence sur la même grille (``None`` sans
        ``sampling``).
    details :
        Paramètres et diagnostics (rapport signal sur fond, avertissements).
    """

    strehl: float
    radii: NDArray[np.float64]
    encircled_energy: NDArray[np.float64]
    r50: float
    r80: float
    ee_first_dark_ring: float
    centroid: tuple[float, float]
    background: float
    psf: NDArray[np.float64]
    reference: NDArray[np.float64] | None
    details: dict[str, Any] = field(default_factory=dict)


def _border_background(image: NDArray[np.float64], width: int) -> float:
    mask = np.ones(image.shape, dtype=bool)
    mask[width:-width, width:-width] = False
    return float(np.median(image[mask]))


def _centroid(image: NDArray[np.float64], radius: float) -> tuple[float, float]:
    """Centroïde des valeurs positives dans un disque de rayon ``radius`` autour du maximum, itéré."""
    y, x = np.indices(image.shape)
    cy, cx = np.unravel_index(int(np.argmax(image)), image.shape)
    cy, cx = float(cy), float(cx)
    weights = np.clip(image, 0.0, None)
    for _ in range(5):
        disk = (y - cy) ** 2 + (x - cx) ** 2 <= radius**2
        w = weights * disk
        total = w.sum()
        if total <= 0:
            break
        cy, cx = float((w * y).sum() / total), float((w * x).sum() / total)
    return cy, cx


def _radius_at(radii: NDArray[np.float64], ee: NDArray[np.float64], level: float) -> float:
    above = np.flatnonzero(ee >= level)
    if above.size == 0:
        return float("nan")
    i = above[0]
    if i == 0:
        return float(radii[0])
    return float(np.interp(level, [ee[i - 1], ee[i]], [radii[i - 1], radii[i]]))


def measure_psf(
    image: NDArray,
    sampling: float | None = None,
    background: float | str | None = "border",
    border_width: int | None = None,
    centroid_radius: float | None = None,
    recenter: bool = True,
) -> PSFMeasurement:
    """
    Mesure le rapport de Strehl et l'énergie encerclée d'une PSF.

    Parameters
    ----------
    image :
        Imagette 2D centrée approximativement sur une source ponctuelle
        isolée. Elle doit contenir l'essentiel de l'énergie (au moins une
        dizaine de λ/D de côté) : l'énergie hors champ est perdue pour la
        normalisation.
    sampling :
        Pixels par λ/D (2 = Nyquist). Nécessaire au Strehl et à l'EE au
        premier anneau noir. Sous 2, le pic est sous-échantillonné et le
        Strehl n'a plus de sens : un avertissement est émis.
    background :
        ``"border"`` (médiane d'une couronne au bord de l'imagette, défaut),
        une valeur numérique, ou ``None`` (aucun retrait).
    border_width :
        Largeur de la couronne de fond (défaut : 1/10 du petit côté, au
        moins 2 pixels).
    centroid_radius :
        Rayon de calcul du centroïde (défaut : 3 λ/D si ``sampling`` est
        donné, sinon 5 pixels).
    recenter :
        Recentre la PSF sur le centre géométrique ``(n - 1) / 2`` par
        translation de Fourier, comme la référence d'Airy.

    Returns
    -------
    PSFMeasurement
    """
    psf = np.asarray(image, dtype=np.float64)
    if psf.ndim != 2:
        raise ValueError(f"Imagette 2D attendue, reçu ndim={psf.ndim}.")
    details: dict[str, Any] = {"sampling": sampling, "recenter": recenter}

    width = border_width or max(2, min(psf.shape) // 10)
    if background == "border":
        bg = _border_background(psf, width)
    elif background is None:
        bg = 0.0
    else:
        bg = float(background)
    psf = psf - bg

    radius = centroid_radius or (3.0 * sampling if sampling else 5.0)
    cy, cx = _centroid(psf, radius)
    centre = ((psf.shape[0] - 1) / 2.0, (psf.shape[1] - 1) / 2.0)
    if recenter:
        psf = translate(psf, centre[0] - cy, centre[1] - cx)

    total = psf.sum()
    if total <= 0:
        raise ValueError("Énergie nulle ou négative après retrait du fond.")
    psf = psf / total
    details["peak_to_background"] = float(psf.max() * total / max(abs(bg), 1e-12)) if bg else float("inf")

    radii, ee = core_psf.encircled_energy(psf)
    r50, r80 = _radius_at(radii, ee, 0.5), _radius_at(radii, ee, 0.8)

    strehl, ee_ring, reference = float("nan"), float("nan"), None
    if sampling is not None:
        if sampling < 2.0:
            details["warning"] = "sous-échantillonné (sampling < 2) : Strehl non significatif"
            warnings.warn(details["warning"], stacklevel=2)
        reference = core_psf.airy_psf(psf.shape, wavelength=1.0, diameter=1.0, pixel_scale=1.0 / sampling)
        # Même traitement que la PSF mesurée : les ailes au bord de l'imagette
        # sont comptées comme « fond » dans les deux cas, puis normalisation à
        # l'énergie restante. Sans cela, le retrait de fond (qui ôte aussi les
        # ailes de la PSF) biaise le Strehl d'environ +0.5 à +1 %.
        if background == "border":
            reference = reference - _border_background(reference, width)
        reference = reference / reference.sum()
        strehl = core_psf.strehl_ratio(psf, reference)
        _, ee_at = core_psf.encircled_energy(psf, radii=np.array([FIRST_DARK_RING * sampling]))
        ee_ring = float(ee_at[0])

    return PSFMeasurement(
        strehl=strehl,
        radii=radii,
        encircled_energy=ee,
        r50=r50,
        r80=r80,
        ee_first_dark_ring=ee_ring,
        centroid=(cy, cx),
        background=bg,
        psf=psf,
        reference=reference,
        details=details,
    )


# EOF
