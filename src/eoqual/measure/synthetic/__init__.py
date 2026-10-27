"""
Scènes, mires et capteurs synthétiques à vérité terrain connue, pour
valider les mesures sans donnée réelle :

- bord incliné à FTM analytique (`SyntheticSlantedEdge`), PSF aberrée à
  Strehl exact (`SyntheticPSF`), image propre à bruit injecté de variance
  connue (`add_noise`) ;
- simulateur de capteur radiométrique (`MatrixSensor`, `PushbroomSensor`,
  `WhiskbroomSensor`) : bruit de photons, de lecture et d'obscurité,
  DSNU/PRNU, non-linéarité, défauts, bruits structurés, convertisseur ;
  scènes naturelles et traitements à fonction de transfert connue.
  Voir REFERENCE_TECHNIQUE.md, partie V.
"""
from eoqual.measure.synthetic.noise import add_noise, generate_clean_image  # noqa: F401
from eoqual.measure.synthetic.patterns import (  # noqa: F401
    StructuredMap,
    pattern_rng,
    power_law_noise,
    structured_map,
)
from eoqual.measure.synthetic.processing import apply_kernel, gaussian_kernel, kernel_power_transfer  # noqa: F401
from eoqual.measure.synthetic.psf import ZERNIKE_TERMS, SyntheticPSF  # noqa: F401
from eoqual.measure.synthetic.scene import Scene, natural_scene  # noqa: F401
from eoqual.measure.synthetic.sensor import (  # noqa: F401
    Acquisition,
    MatrixSensor,
    PushbroomAcquisition,
    PushbroomSensor,
    SensorParams,
    SensorTruth,
    WhiskbroomAcquisition,
    WhiskbroomSensor,
)
from eoqual.measure.synthetic.slanted_edge import SyntheticSlantedEdge  # noqa: F401
from eoqual.measure.synthetic.temporal_effects import PeriodicPattern  # noqa: F401

__all__ = [
    "Acquisition",
    "MatrixSensor",
    "PeriodicPattern",
    "PushbroomAcquisition",
    "PushbroomSensor",
    "Scene",
    "SensorParams",
    "SensorTruth",
    "StructuredMap",
    "SyntheticPSF",
    "SyntheticSlantedEdge",
    "WhiskbroomAcquisition",
    "WhiskbroomSensor",
    "ZERNIKE_TERMS",
    "add_noise",
    "apply_kernel",
    "gaussian_kernel",
    "generate_clean_image",
    "kernel_power_transfer",
    "natural_scene",
    "pattern_rng",
    "power_law_noise",
    "structured_map",
]
