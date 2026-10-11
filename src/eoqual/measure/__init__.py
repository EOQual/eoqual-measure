"""
eoqual.measure — Mesures physiques contrôlées de la qualité image d'un
système d'observation de la Terre : FTM sur bord incliné, variance du
bruit, Strehl et énergie encerclée, et à terme courbe de transfert
photonique, bruit d'égalisation, contraste sur bruit sur mire.

Public API
----------
>>> from eoqual.measure import measure_mtf, estimate_noise, measure_psf, list_measures
>>> list_measures(include_planned=True)
>>> mtf = measure_mtf(edge_zone)                # MTFResult
>>> noise = estimate_noise(image, "ponomarenko", num_bins=8)
>>> psf = measure_psf(star, sampling=4.0)       # Strehl, énergie encerclée

Les API détaillées vivent dans les sous-paquets :
:mod:`eoqual.measure.mtf.slanted_edge` (comparateurs, estimateurs,
bootstrap, aberration chromatique), :mod:`eoqual.measure.noise`,
:mod:`eoqual.measure.synthetic` (mires à vérité terrain connue). Les
briques numériques partagées avec les autres modules EOQual viennent
d'``eoqual-core``.

Différence avec ``eoqual-metrics`` : ici la cible ou le protocole est
contrôlé et le résultat est une grandeur physique ; ``eoqual-metrics``
calcule des indicateurs de qualité, dont certains sans référence sur
scène quelconque (ex. MTF@Nyquist sur arêtes naturelles).
"""

__version__ = "0.1.0"
__title__ = "eoqual.measure"
__description__ = "Controlled physical image quality measurements for Earth observation imaging systems"
__url__ = "https://github.com/EOQual/eoqual-measure"
__uri__ = __url__
__doc__ = __description__ + " <" + __uri__ + ">"
__author__ = "Olivier Amram"
__email__ = "olivier.amram@free.fr"
__license__ = "MIT"
__copyright__ = "Copyright (c) 2026 Olivier Amram"

from .config import MEASURES_CONFIGS  # noqa: F401
from .mtf import MTFResult, measure_mtf  # noqa: F401
from .noise import NoiseEstimate, estimate_noise  # noqa: F401
from .psf import PSFMeasurement, measure_psf  # noqa: F401
from .runner import list_measures  # noqa: F401

__all__ = [
    "MEASURES_CONFIGS",
    "MTFResult",
    "NoiseEstimate",
    "PSFMeasurement",
    "estimate_noise",
    "list_measures",
    "measure_mtf",
    "measure_psf",
]
