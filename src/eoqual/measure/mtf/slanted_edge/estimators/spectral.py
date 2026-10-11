#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Estimation de MTF par la méthode « spectrale ».

On travaille directement dans le domaine fréquentiel : la phase linéaire
due à l'inclinaison est retirée ligne par ligne, puis on résout un
problème linéaire (moindres carrés) pour extraire la MTF à chaque
fréquence, en combinant plusieurs répliques spectrales lorsque
`oversample_factor > 1`.

Voir `SpectralMTFEstimator` pour le détail.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from eoqual.measure.mtf.slanted_edge.fourier_utils import reference_spectrum
from eoqual.measure.mtf.slanted_edge.estimators.base import SlantedEdgeMTFEstimator


class SpectralMTFEstimator(SlantedEdgeMTFEstimator):
    """
    Estime la MTF en résolvant, fréquence par fréquence, un système
    linéaire dans le domaine spectral.

    Principe : la FFT de chaque ligne du bord incliné est la même MTF,
    modulée par une rampe de phase due au décalage `dx` propre à cette
    ligne (théorème du décalage). On retire d'abord cette rampe
    (`mes`), puis, pour chaque fréquence cible, on exprime la valeur
    mesurée sur toutes les lignes comme une combinaison linéaire de la
    MTF aux répliques spectrales pertinentes (1, 2 ou 4 selon
    `oversample_factor`) et on résout ce système par moindres carrés
    (`np.linalg.lstsq`).

    Avantage : ne suppose pas d'interpolation spatiale, exploite
    directement la redondance entre toutes les lignes pour chaque
    fréquence. Inconvénient : plus coûteux (une résolution par
    fréquence) et suppose lui aussi `dx` linéaire sur toute la hauteur
    de `rect`.
    """

    name = "spectral"

    def _compute_mtf(
        self,
        rect: NDArray[np.float64],
        dx: float,
        theta_rad: float,
        oversample_factor: int,
    ) -> tuple[NDArray[np.float64], NDArray[np.complex128], dict]:
        n_rows, n_cols = rect.shape

        # Facteur de phase linéaire dû au décalage ligne-à-ligne (à supprimer).
        phase_factor = np.exp(complex(0, -2.0 * np.pi * dx))

        # FFT ligne-par-ligne (normalisée).
        tf_rect = np.fft.fft(rect, axis=1) / float(n_cols)

        # Suppression de la rampe de phase.
        x_grid, y_grid = np.meshgrid(np.arange(n_cols), np.arange(n_rows))
        mes = tf_rect * np.exp(
            x_grid * y_grid * dx / n_cols * complex(0, 2.0 * np.pi)
        )

        # Allocation du tableau de solution (MTF).
        sp = np.zeros((oversample_factor, n_cols // 2), dtype=complex)

        # Spectre de référence (taille adaptée au sur-échantillonnage).
        ref_spec = reference_spectrum(n_cols * oversample_factor // 4)

        # Boucle sur chaque fréquence impaire. Bornée à `ref_spec.size`
        # plutôt qu'à `n_cols // 2` : pour oversample_factor=1,
        # `ref_spec` ne contient que `n_cols // 4` éléments (les
        # positions au-delà ne sont de toute façon jamais lues via
        # `ftm_vec[:n_needed]`) ; pour 2 et 4 les deux tailles
        # coïncident déjà, ce qui laisse leur comportement inchangé.
        for c in range(min(n_cols // 2, ref_spec.size)):
            if oversample_factor == 1:
                A = (np.ones(n_rows) * ref_spec[c]).reshape(-1, 1)
            elif oversample_factor == 2:
                A = np.vstack(
                    (
                        np.ones(n_rows) * ref_spec[c],
                        ref_spec[n_cols // 2 - 1 - c]
                        * np.power(phase_factor, np.arange(n_rows)),
                    )
                ).conj().T
            elif oversample_factor == 4:
                A = np.vstack(
                    (
                        np.ones(n_rows) * ref_spec[c],
                        ref_spec[n_cols // 2 + c]
                        * np.power(phase_factor, -np.arange(n_rows)),
                        ref_spec[n_cols - 1 - c]
                        * np.power(phase_factor, -2 * np.arange(n_rows)),
                        ref_spec[n_cols // 2 - 1 - c]
                        * np.power(phase_factor, np.arange(n_rows)),
                    )
                ).conj().T
            else:
                raise ValueError(
                    f"oversample_factor={oversample_factor} non supporté (1,2,4)."
                )
            sp[:, c], *_ = np.linalg.lstsq(A, mes[:, 2 * c + 1], rcond=None)

        mtf_vec = sp.ravel()
        n_needed = n_cols * oversample_factor // 4
        mtf = mtf_vec[:n_needed]
        frequency = (1 + 2 * np.arange(n_needed)) / n_cols

        return frequency, mtf, {}


# EOF
