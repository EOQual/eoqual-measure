"""
Scènes et mires synthétiques à vérité terrain connue, pour valider les
mesures sans donnée réelle : bord incliné à FTM analytique, image propre
à bruit injecté de variance connue.

Inclut la PSF aberrée à Strehl exact (Strehl et énergie encerclée) ;
destiné à accueillir les générateurs des mesures à venir (champ uniforme
pour la courbe de transfert photonique, mire à contraste connu).
"""
from eoqual.measure.synthetic.noise import add_noise, generate_clean_image  # noqa: F401
from eoqual.measure.synthetic.psf import ZERNIKE_TERMS, SyntheticPSF  # noqa: F401
from eoqual.measure.synthetic.slanted_edge import SyntheticSlantedEdge  # noqa: F401

__all__ = ["SyntheticPSF", "SyntheticSlantedEdge", "ZERNIKE_TERMS", "add_noise", "generate_clean_image"]
