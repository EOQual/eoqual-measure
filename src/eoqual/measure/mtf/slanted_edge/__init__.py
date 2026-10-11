"""
MTF par la méthode du bord incliné (slanted-edge, ISO 12233) sur mire ou
bord choisi : canonicalisation, orientation (3 algorithmes), MTF (4
estimateurs comparables), bootstrap, aberration chromatique latérale
(Bayer), études de sensibilité.

Architecture « template method + comparateur » : chaque étape a une
classe de base et plusieurs implémentations interchangeables, exécutables
côte à côte sur la même donnée (``EdgeOrientationComparator``,
``MTFComparator``). Voir ``REFERENCE_TECHNIQUE.md`` (section MTF).

Les modules ``visualization`` (plotly) et ``sensitivity`` (pandas) ne sont
pas importés ici : ils relèvent des extras ``plot``/``study``.
"""
from eoqual.measure.mtf.slanted_edge.bootstrap import BootstrapMTFResult, bootstrap_mtf  # noqa: F401
from eoqual.measure.mtf.slanted_edge.chromatic import (  # noqa: F401
    ChromaticShiftResult,
    compute_chromatic_shifts,
)
from eoqual.measure.mtf.slanted_edge.estimators import (  # noqa: F401
    ESF_FIT_MODELS,
    EsfLsfMTFEstimator,
    GeometricESFBuilder,
    MTFComparator,
    MTFResult,
    SlantedEdgeMTFEstimator,
    SpatialMTFEstimator,
    SpectralCorrelationMTFEstimator,
    SpectralMTFEstimator,
    compute_fwhm,
    compute_rer,
)
from eoqual.measure.mtf.slanted_edge.orientation import (  # noqa: F401
    EdgeOrientationComparator,
    EdgeOrientationDetector,
    FourierOrientationDetector,
    GradientOrientationDetector,
    OrientationResult,
    PCAOrientationDetector,
)
from eoqual.measure.mtf.slanted_edge.preprocess import (  # noqa: F401
    PreprocessingResult,
    SlantedEdgePreprocessor,
)
