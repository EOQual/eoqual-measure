#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Estimateur d'orientation de bord incliné basé sur le gradient.

Voir `GradientOrientationDetector` pour l'algorithme et l'hypothèse que
sa métrique de qualité teste.
"""
from __future__ import annotations

import numpy as np

from eoqual.measure.mtf.slanted_edge.orientation.base import CanonicalEstimate, CanonicalFrame, EdgeOrientationDetector


class GradientOrientationDetector(EdgeOrientationDetector):
    """
    Estime l'orientation du bord à partir de la position sous-pixel du
    maximum local du gradient, ligne par ligne.

    Hypothèse testée : le profil d'intensité de chaque ligne présente un
    unique maximum local bien marqué de |d(intensité)/d(colonne)| au
    niveau du bord. Métrique de qualité ("peak_sharpness") : la
    parabole ajustée sur le maximum et ses deux voisins a une courbure
    grande devant la hauteur du pic lorsque le maximum est net et sans
    ambiguïté, et petite lorsque le profil est plat ou bruité --
    exactement ce qu'exige l'hypothèse.
    """

    name = "gradient"

    def _estimate_canonical(self, frame: CanonicalFrame) -> CanonicalEstimate:
        image = frame.image
        n_rows, n_cols = image.shape
        if n_cols < 3:
            raise ValueError("Image is too narrow to locate an edge.")

        # Dérivée ligne par ligne (différences centrées le long des colonnes).
        gradient = np.gradient(image, axis=1)

        row_indices = np.arange(n_rows)
        # Exclut la première/dernière colonne pour que chaque pic ait deux voisins.
        peak_columns = 1 + np.argmax(
            np.abs(gradient[:, 1:-1]), axis=1
        )

        left = gradient[row_indices, peak_columns - 1]
        center = gradient[row_indices, peak_columns]
        right = gradient[row_indices, peak_columns + 1]

        denominator = left - 2.0 * center + right
        valid = np.abs(denominator) > 1e-9

        delta = np.zeros(n_rows)
        delta[valid] = 0.5 * (left[valid] - right[valid]) / denominator[valid]
        delta = np.clip(delta, -1.0, 1.0)

        sub_pixel_columns = peak_columns + delta

        sharpness = np.zeros(n_rows)
        sharpness[valid] = np.abs(denominator[valid]) / (
            np.abs(center[valid]) + 1e-12
        )

        if not np.any(valid):
            raise ValueError("No well-defined gradient peak on any row.")

        rows = row_indices[valid].astype(np.float64)
        columns = sub_pixel_columns[valid]

        angle_deg, slope, r_squared = self._fit_angle_from_row_positions(
            rows, columns
        )

        return CanonicalEstimate(
            angle_deg=angle_deg,
            rows=rows,
            columns=columns,
            quality_metric_name="peak_sharpness",
            quality_metric_value=float(np.median(sharpness[valid])),
            details={"slope": slope, "r_squared": r_squared},
        )


# EOF
