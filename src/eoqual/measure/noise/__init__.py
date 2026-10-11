"""
Mesure du bruit radiométrique d'une image 2D sur zones homogènes détectées
automatiquement (patchs plats, ondelettes, Ponomarenko, superpixels, PCA).

Exemple
-------
>>> from eoqual.measure.noise import estimate_noise, list_noise_methods
>>> est = estimate_noise(image, "ponomarenko", num_bins=8)
>>> est.variance, est.mean, est.sigma

Voir ``REFERENCE_TECHNIQUE.md`` (section Bruit) pour le choix de méthode
et les biais connus.
"""
from eoqual.measure.noise.estimate import (  # noqa: F401
    NoiseEstimate,
    estimate_noise,
    list_noise_methods,
)
from eoqual.measure.noise.variance import NoiseVariance  # noqa: F401

__all__ = ["NoiseEstimate", "NoiseVariance", "estimate_noise", "list_noise_methods"]
