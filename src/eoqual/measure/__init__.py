"""
eoqual.measure — Mesures physiques contrôlées de la qualité image d'un
système d'observation de la Terre : FTM sur bord incliné, variance du
bruit, Strehl et énergie encerclée, et mesures radiométriques
(obscurité, courant d'obscurité, bruit temporel, courbe de transfert
photonique, PRNU, linéarité, pixels défectueux et télégraphiques,
convertisseur, bruit d'un push broom, bruit spatial et d'égalisation,
spectre du bruit et bruits structurés, contraste sur bruit sur mire).

**En cours de développement** : les mesures radiométriques sont validées
sur données simulées mais pas encore qualifiées sur données réelles ;
l'API peut évoluer et des anomalies sont possibles.

Public API
----------
>>> from eoqual.measure import measure_mtf, estimate_noise, measure_psf, list_measures
>>> list_measures(include_planned=True)
>>> mtf = measure_mtf(edge_zone)                # MTFResult
>>> noise = estimate_noise(image, "ponomarenko", num_bins=8)
>>> psf = measure_psf(star, sampling=4.0)       # Strehl, énergie encerclée
>>> from eoqual.measure.radiometry import measure_dark_frame, measure_ptc
>>> dark = measure_dark_frame(dark_stack)        # offset, DSNU, b
>>> ptc = measure_ptc(flat_pairs, dark)          # gain K, bruit de lecture

Les API détaillées vivent dans les sous-paquets :
:mod:`eoqual.measure.mtf.slanted_edge` (comparateurs, estimateurs,
bootstrap, aberration chromatique), :mod:`eoqual.measure.noise`,
:mod:`eoqual.measure.radiometry` (mesures radiométriques),
:mod:`eoqual.measure.contrast` (contraste sur bruit),
:mod:`eoqual.measure.synthetic` (mires et capteurs simulés à vérité
terrain connue). Les
briques numériques partagées avec les autres modules EOQual viennent
d'``eoqual-core``.

Différence avec ``eoqual-metrics`` : ici la cible ou le protocole est
contrôlé et le résultat est une grandeur physique ; ``eoqual-metrics``
calcule des indicateurs de qualité, dont certains sans référence sur
scène quelconque (ex. MTF@Nyquist sur arêtes naturelles).
"""

__version__ = "0.2.0"
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
