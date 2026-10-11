"""
API fonctionnelle de la mesure de bruit : un appel = un résultat typé.

Fine couche au-dessus de :class:`~eoqual.measure.noise.variance.NoiseVariance`
(qui reste disponible telle quelle) : elle remplace le tuple positionnel
``(variance, moyenne[, échantillons, luminances])`` par un
:class:`NoiseEstimate` nommé, que les mesures construites sur le bruit
(courbe de transfert photonique, bruit d'égalisation...) peuvent consommer
sans dépendre de l'ordre des champs.

Exemple
-------
>>> from eoqual.measure.noise import estimate_noise
>>> est = estimate_noise(image, "ponomarenko", num_bins=8)
>>> est.variance, est.mean          # une valeur par classe de luminance
>>> est.sigma                       # écart-type(s) correspondant(s)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from eoqual.measure.noise.variance import NoiseVariance

__all__ = ["NoiseEstimate", "estimate_noise", "list_noise_methods"]


@dataclass(slots=True)
class NoiseEstimate:
    """
    Résultat d'une estimation de variance de bruit.

    Attributes
    ----------
    method :
        Nom de l'estimateur (voir :func:`list_noise_methods`).
    variance :
        Variance du bruit (unités de l'image au carré). Scalaire, ou une
        valeur par classe de luminance (``ponomarenko`` avec
        ``num_bins > 1``).
    mean :
        Luminance moyenne des zones retenues, même forme que ``variance``.
    samples :
        Mesures élémentaires ayant servi à l'estimation (patchs, blocs,
        segments, tuiles ou tirages bootstrap selon la méthode) — base
        d'une barre d'erreur intra-image. Une array par classe pour
        ``ponomarenko``.
    luminances :
        Luminance associée à chaque élément de ``samples`` (même ordre).
    params :
        Paramètres transmis à l'estimateur.
    """

    method: str
    variance: float | list[float]
    mean: float | list[float]
    samples: Any
    luminances: Any
    params: dict[str, Any]

    @property
    def sigma(self) -> float | NDArray[np.float64]:
        """Écart-type du bruit (racine de ``variance``)."""
        if np.isscalar(self.variance):
            return float(np.sqrt(self.variance))
        return np.sqrt(np.asarray(self.variance, dtype=np.float64))


def list_noise_methods() -> dict[str, dict[str, str]]:
    """Méthodes d'estimation disponibles et description de leurs paramètres."""
    return dict(NoiseVariance._REQUIRED_PARAMS)


def estimate_noise(image: NDArray, method: str = "ponomarenko", **params: Any) -> NoiseEstimate:
    """
    Estime la variance du bruit d'une image 2D.

    Parameters
    ----------
    image :
        Image 2D (H, W). Les zones utilisées sont détectées
        automatiquement : aucune zone homogène n'a besoin d'être fournie.
    method :
        Estimateur (voir :func:`list_noise_methods`).
    **params :
        Paramètres de l'estimateur (voir :func:`list_noise_methods`).

    Returns
    -------
    NoiseEstimate
    """
    image = np.asarray(image)
    if image.ndim != 2:
        raise ValueError(f"Image 2D attendue, reçu ndim={image.ndim}.")
    variance, mean, samples, luminances = NoiseVariance().estimate_noise(
        image, method, params=params, return_samples=True
    )
    return NoiseEstimate(
        method=method,
        variance=variance,
        mean=mean,
        samples=samples,
        luminances=luminances,
        params=params,
    )


# EOF
