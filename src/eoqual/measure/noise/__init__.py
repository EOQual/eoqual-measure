"""
Mesure du bruit radiométrique d'une image 2D sur zones homogènes détectées
automatiquement (patchs plats, ondelettes, Ponomarenko, superpixels, PCA),
spectre de puissance du bruit et bruits structurés.

Exemple
-------
>>> from eoqual.measure.noise import estimate_noise, list_noise_methods
>>> est = estimate_noise(image, "ponomarenko", num_bins=8)
>>> est.variance, est.mean, est.sigma
>>> from eoqual.measure.noise import noise_power_spectrum, find_structured_noise
>>> nps = noise_power_spectrum(stack, mode="difference")

Voir ``REFERENCE_TECHNIQUE.md`` (partie II pour l'estimation de la
variance, partie IV, chapitre 12, pour le spectre et les bruits
structurés).
"""
from eoqual.measure.noise.estimate import (  # noqa: F401
    NoiseEstimate,
    estimate_noise,
    list_noise_methods,
)
from eoqual.measure.noise.spectrum import (  # noqa: F401
    LineNoise,
    NPSResult,
    PeakSearch,
    StructuredNoiseResult,
    StructuredPeak,
    find_structured_noise,
    noise_power_spectrum,
)
from eoqual.measure.noise.variance import NoiseVariance  # noqa: F401

__all__ = [
    "LineNoise",
    "NPSResult",
    "NoiseEstimate",
    "NoiseVariance",
    "PeakSearch",
    "StructuredNoiseResult",
    "StructuredPeak",
    "estimate_noise",
    "find_structured_noise",
    "list_noise_methods",
    "noise_power_spectrum",
]
