"""
PSF synthétiques à aberrations connues, pour valider la mesure de Strehl et
d'énergie encerclée.

La PSF est calculée par diffraction de Fraunhofer : module au carré de la
transformée de Fourier d'une pupille circulaire portant une phase
d'aberration, développée sur quelques polynômes de Zernike normalisés
(écart quadratique moyen unité sur la pupille, convention de Noll). Le
rapport de Strehl vrai est obtenu par la même simulation sans aberration,
ce qui évite toute ambiguïté de discrétisation.

Références : V. N. Mahajan, *Optical Imaging and Aberrations*, SPIE Press
(polynômes de Zernike normalisés, approximation de Maréchal) ; R. J. Noll,
"Zernike polynomials and atmospheric turbulence", JOSA 66(3), 1976.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

__all__ = ["ZERNIKE_TERMS", "SyntheticPSF"]


def _zernike(name: str, rho: NDArray[np.float64], theta: NDArray[np.float64]) -> NDArray[np.float64]:
    if name == "defocus":
        return np.sqrt(3) * (2 * rho**2 - 1)
    if name == "astigmatism":
        return np.sqrt(6) * rho**2 * np.cos(2 * theta)
    if name == "coma":
        return np.sqrt(8) * (3 * rho**3 - 2 * rho) * np.cos(theta)
    if name == "spherical":
        return np.sqrt(5) * (6 * rho**4 - 6 * rho**2 + 1)
    raise ValueError(f"Terme de Zernike inconnu : {name!r} (disponibles : {', '.join(ZERNIKE_TERMS)}).")


#: Termes d'aberration disponibles (Zernike de Noll, rms unité sur la pupille).
ZERNIKE_TERMS = ("defocus", "astigmatism", "coma", "spherical")


@dataclass
class SyntheticPSF:
    """
    PSF d'une pupille circulaire aberrée, échantillonnée et bruitée.

    Parameters
    ----------
    shape :
        Taille de l'image ``(ny, nx)``.
    sampling :
        Nombre de pixels par ``λ/D`` (2 = échantillonnage de Nyquist).
    aberrations_rms_waves :
        Coefficients de Zernike en vagues rms, ex. ``{"defocus": 0.05}``.
        L'erreur de front d'onde rms totale est la racine de la somme des
        carrés (termes orthogonaux).
    offset :
        Décalage ``(dy, dx)`` du centre de la PSF par rapport au centre de
        l'image, en pixels (sous-pixel autorisé).
    photons :
        Nombre total de photons (bruit de Poisson) ; ``None`` = pas de
        bruit de photon.
    background :
        Fond uniforme ajouté, en photons par pixel.
    read_noise :
        Écart-type du bruit de lecture gaussien, en photons.
    pupil_pixels :
        Diamètre de la pupille dans la grille de calcul (précision de la
        discrétisation).
    seed :
        Graine du bruit.
    """

    shape: tuple[int, int] = (129, 129)
    sampling: float = 4.0
    aberrations_rms_waves: dict[str, float] = field(default_factory=dict)
    offset: tuple[float, float] = (0.0, 0.0)
    photons: float | None = None
    background: float = 0.0
    read_noise: float = 0.0
    pupil_pixels: int = 128
    seed: int | None = 0

    def _pupil(self, with_aberrations: bool) -> NDArray[np.complex128]:
        n = self.pupil_pixels
        c = (n - 1) / 2.0
        y, x = np.indices((n, n))
        rho = np.hypot(x - c, y - c) / (n / 2.0)
        theta = np.arctan2(y - c, x - c)
        inside = rho <= 1.0
        phase = np.zeros((n, n))
        if with_aberrations:
            for name, coeff in self.aberrations_rms_waves.items():
                phase += 2 * np.pi * coeff * _zernike(name, rho, theta)
        return np.where(inside, np.exp(1j * phase), 0.0)

    def _intensity(self, with_aberrations: bool) -> NDArray[np.float64]:
        """PSF sans bruit, de somme 1 sur la grille de calcul, recadrée sur ``shape``."""
        n_fft = int(round(self.sampling * self.pupil_pixels))
        if n_fft < max(self.shape):
            raise ValueError(
                f"Grille de calcul ({n_fft} px = sampling x pupil_pixels) plus petite que "
                f"l'image demandée {self.shape} : augmenter pupil_pixels."
            )
        pupil = np.zeros((n_fft, n_fft), dtype=np.complex128)
        p = self._pupil(with_aberrations)
        pupil[: p.shape[0], : p.shape[1]] = p
        # Décalage sous-pixel (dy, dx) de la PSF : rampe de phase linéaire
        # sur la pupille (théorème du décalage).
        dy, dx = self.offset
        if dy or dx:
            k = np.arange(n_fft)
            pupil = pupil * np.exp(2j * np.pi * (k[:, None] * dy + k[None, :] * dx) / n_fft)
        # Centre de la PSF non décalée : pixel n_fft // 2.
        field_ = np.fft.fftshift(np.fft.fft2(pupil))
        intensity = np.abs(field_) ** 2
        total = intensity.sum()
        cy, cx = n_fft // 2, n_fft // 2
        hy, hx = self.shape[0] // 2, self.shape[1] // 2
        crop = intensity[cy - hy: cy - hy + self.shape[0], cx - hx: cx - hx + self.shape[1]]
        return crop / total

    @property
    def wavefront_rms_waves(self) -> float:
        """Erreur de front d'onde rms totale (vagues)."""
        return float(np.sqrt(sum(c**2 for c in self.aberrations_rms_waves.values())))

    @property
    def strehl_true(self) -> float:
        """Rapport de Strehl exact de la PSF simulée (pic aberré / pic sans aberration)."""
        if self.offset != (0.0, 0.0):
            centred = SyntheticPSF(**{**self.__dict__, "offset": (0.0, 0.0)})
            return centred.strehl_true
        return float(self._intensity(True).max() / self._intensity(False).max())

    @property
    def strehl_marechal(self) -> float:
        """Approximation de Maréchal ``exp(-(2π σ)²)``, valable pour σ ≲ 0.1 vague."""
        return float(np.exp(-((2 * np.pi * self.wavefront_rms_waves) ** 2)))

    def generate(self) -> NDArray[np.float64]:
        """Image de la PSF (photons si ``photons`` est donné, sinon somme 1), fond et bruits inclus."""
        rng = np.random.default_rng(self.seed)
        psf = self._intensity(True)
        if self.photons is not None:
            psf = rng.poisson(psf * self.photons + self.background).astype(np.float64)
        else:
            psf = psf + self.background
        if self.read_noise > 0:
            psf = psf + rng.normal(0.0, self.read_noise, psf.shape)
        return psf


# EOF
