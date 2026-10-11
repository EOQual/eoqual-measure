"""
Mesure de FTM (MTF) sur cible contrôlée.

Une seule méthode pour l'instant : le bord incliné
(:mod:`eoqual.measure.mtf.slanted_edge`). Le sous-paquet est prévu pour
accueillir d'autres cibles (mire sinusoïdale, impulsion, couteau) sous la
même forme de résultat (:class:`MTFResult`).

Exemple
-------
>>> from eoqual.measure.mtf import measure_mtf
>>> res = measure_mtf(image)                    # ESF/LSF, PCHIP
>>> res.value_at(0.5), res.frequency_at(0.5)    # MTF@Nyquist, MTF50
"""
from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from eoqual.measure.mtf.slanted_edge import (
    EsfLsfMTFEstimator,
    MTFResult,
    SlantedEdgePreprocessor,
    SpatialMTFEstimator,
    SpectralCorrelationMTFEstimator,
    SpectralMTFEstimator,
)

__all__ = ["MTFResult", "SLANTED_EDGE_ESTIMATORS", "measure_mtf"]

#: Estimateurs bord incliné disponibles via :func:`measure_mtf`.
SLANTED_EDGE_ESTIMATORS = {
    "esf_lsf": EsfLsfMTFEstimator,
    "spatial": SpatialMTFEstimator,
    "spectral": SpectralMTFEstimator,
    "spectral_correlation": SpectralCorrelationMTFEstimator,
}


def measure_mtf(
    image: NDArray,
    method: str = "esf_lsf",
    oversample_factor: int = 4,
    canonicalization_percentile: float = 10.0,
    **estimator_params: Any,
) -> MTFResult:
    """
    Mesure la MTF d'une zone contenant un unique bord incliné.

    Parameters
    ----------
    image :
        Zone 2D (niveaux de gris) contenant un seul bord, d'orientation
        quelconque (la canonicalisation transpose/retourne au besoin).
    method :
        Clé de :data:`SLANTED_EDGE_ESTIMATORS`.
    oversample_factor :
        1, 2 ou 4.
    canonicalization_percentile :
        Pourcentage de colonnes servant à estimer les niveaux noir/blanc.
    **estimator_params :
        Transmis au constructeur de l'estimateur (ex. ``pixel_pitch=0.5``
        pour un plan Bayer, ``esf_model="tikhonov"`` pour ``esf_lsf``).

    Returns
    -------
    MTFResult
    """
    if method not in SLANTED_EDGE_ESTIMATORS:
        raise ValueError(
            f"Méthode inconnue '{method}' — disponibles : {sorted(SLANTED_EDGE_ESTIMATORS)}."
        )
    frame = SlantedEdgePreprocessor(percentile=canonicalization_percentile).preprocess(
        np.asarray(image, dtype=np.float64)
    )
    estimator = SLANTED_EDGE_ESTIMATORS[method](**estimator_params)
    return estimator.estimate(frame, oversample_factor=oversample_factor)
