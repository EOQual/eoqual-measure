#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Estimateur d'orientation de bord incliné basé sur la PCA.

Voir `PCAOrientationDetector` pour l'algorithme et l'hypothèse que sa
métrique de qualité teste.
"""
from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray

from eoqual.measure.mtf.slanted_edge.orientation.base import CanonicalEstimate, CanonicalFrame, EdgeOrientationDetector


class PCAOrientationDetector(EdgeOrientationDetector):
    """
    Estime l'orientation d'un bord incliné à l'aide d'une PCA sur les
    coordonnées des pixels de fort gradient.

    Hypothèse testée : les pixels de fort gradient sont alignés sur une
    droite fine (le bord). Métrique de qualité : rapport des deux
    valeurs propres de la matrice de covariance, calculé sur les pixels
    candidats bruts issus du seuillage de gradient -- proche de 1 pour
    un nuage de points isotrope, sans droite, grand pour une droite
    fine et bien définie. Comme elle est calculée avant tout rejet
    d'aberrants, une valeur faible est un signal honnête et direct que
    les pixels candidats eux-mêmes sont bruités ou multimodaux (par
    exemple un fond bruité faisant passer de nombreux pixels parasites
    au-dessus du seuil de gradient), et non simplement que l'ajustement
    final a fini par paraître propre.

    La position sous-pixel du bord par ligne est obtenue par
    *projection* : pour chaque ligne, le centroïde pondéré par la
    magnitude des pixels sélectionnés sur cette ligne. Dans une image
    bruitée, le seul seuillage de gradient laisse couramment passer
    plus de pixels de bruit que de véritables pixels de bord (un simple
    seuil n'a aucune notion de "proche du bord"), et ces pixels de
    bruit ne sont pas dispersés uniformément : plusieurs peuvent tomber
    sur la même ligne, biaisant de façon cohérente la moyenne de cette
    ligne loin du vrai bord -- une dérive systématique et lisse plutôt
    qu'une dispersion aléatoire, qu'une seule passe de rejet d'aberrants
    (par exemple basée sur l'écart absolu médian) ne suffit pas à
    corriger une fois que les aberrants sont majoritaires. On utilise
    plutôt RANSAC : de nombreuses droites candidates sont ajustées à
    partir de paires de points aléatoires, et celle qui compte le plus
    de points à l'intérieur d'une faible tolérance est retenue. Seuls
    ces inliers alimentent le centroïde par ligne, ce qui reste fidèle
    à l'hypothèse propre de l'estimateur (des points proches d'une
    droite) même quand la majorité des pixels seuillés n'y sont pas.

    L'angle rapporté est ajusté sur ce même ensemble d'inliers, et non
    sur le nuage de candidats brut : l'ajuster sur les points bruts non
    filtrés le rendrait incohérent avec un tracé de positions visible-
    ment propre, filtré par RANSAC -- des positions d'apparence correcte
    avec un angle biaisé, plutôt que les deux étant également fiables.
    Seule la métrique de qualité est délibérément calculée sur le nuage
    brut, puisque son rôle est de diagnostiquer à quel point les
    candidats étaient bruités au départ.

    Parameters
    ----------
    gradient_percentile :
        Seuil de percentile utilisé pour ne conserver que les pixels de
        fort gradient. Une valeur plus élevée conserve moins de pixels
        et se concentre sur le bord.
    ransac_tolerance :
        Distance perpendiculaire maximale, en pixels, à une droite
        candidate pour qu'un point soit considéré comme un inlier.
    ransac_iterations :
        Nombre de droites candidates, tirées de paires de points
        aléatoires, à essayer.
    """

    name = "pca"

    def __init__(
        self,
        gradient_percentile: float = 90.0,
        ransac_tolerance: float = 2.0,
        ransac_iterations: int = 300,
        canonicalization_percentile: float = 10.0,
    ) -> None:
        super().__init__(canonicalization_percentile)
        if not 0.0 < gradient_percentile < 100.0:
            raise ValueError(
                "'gradient_percentile' must belong to (0, 100)."
            )
        self.gradient_percentile = gradient_percentile
        self.ransac_tolerance = ransac_tolerance
        self.ransac_iterations = ransac_iterations

    def _estimate_canonical(self, frame: CanonicalFrame) -> CanonicalEstimate:
        image = frame.image

        gradient_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        gradient_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.hypot(gradient_x, gradient_y)

        threshold = self._robust_threshold(magnitude, self.gradient_percentile)
        row_idx, col_idx = np.nonzero(magnitude >= threshold)

        if row_idx.size < 10:
            raise ValueError("Not enough gradient points detected.")

        weights = magnitude[row_idx, col_idx]
        points = np.column_stack(
            (col_idx.astype(np.float64), row_idx.astype(np.float64))
        )

        # Première passe : ajuste le nuage de candidats entier. C'est sur
        # CETTE passe qu'est calculée la métrique de qualité -- voir le
        # docstring de la classe -- afin qu'elle reste un diagnostic
        # honnête des candidats bruts, pas du résultat nettoyé.
        _, eigenvalue_ratio, explained_variance, _, _ = self._fit_pca_line(
            points
        )

        # Seconde passe : rejette via RANSAC les points qui ne sont pas
        # proches d'une droite commune avant de construire le centroïde
        # par ligne, afin que les pixels de bruit (potentiellement
        # majoritaires parmi les candidats) ne puissent pas le biaiser.
        inlier_mask = self._ransac_inliers(points)

        if np.count_nonzero(inlier_mask) < 10:
            inlier_mask = np.ones_like(inlier_mask, dtype=bool)  # repli

        row_in = row_idx[inlier_mask]
        col_in = col_idx[inlier_mask]
        weights_in = weights[inlier_mask]

        # L'angle rapporté doit être ajusté sur les MÊMES points inliers
        # que les positions par ligne ci-dessous -- l'ajuster plutôt sur
        # le nuage brut et bruité (comme le faisait une première version
        # de ce détecteur) donne un angle incohérent avec un tracé de
        # positions visiblement propre.
        angle_deg, _, _, _, _ = self._fit_pca_line(
            np.column_stack((col_in, row_in)).astype(np.float64)
        )

        rows = np.unique(row_in).astype(np.float64)
        columns = np.empty_like(rows)
        for i, row in enumerate(rows.astype(int)):
            row_mask = row_in == row
            columns[i] = np.average(
                col_in[row_mask], weights=weights_in[row_mask]
            )

        return CanonicalEstimate(
            angle_deg=angle_deg,
            rows=rows,
            columns=columns,
            quality_metric_name="eigenvalue_ratio",
            quality_metric_value=eigenvalue_ratio,
            details={
                "n_points": int(row_idx.size),
                "n_inliers": int(np.count_nonzero(inlier_mask)),
                "inlier_fraction": float(np.mean(inlier_mask)),
                "explained_variance": explained_variance,
            },
        )

    def _fit_pca_line(
        self,
        points: NDArray[np.float64],
    ) -> tuple[float, float, float, NDArray[np.float64], NDArray[np.float64]]:
        """
        Ajuste une droite passant par `points` (colonnes 0/1 = x/y) via
        une PCA.

        Returns
        -------
        tuple
            `(angle_deg, eigenvalue_ratio, explained_variance,
            principal_direction, centroid)`.
        """
        centroid = np.mean(points, axis=0)
        centered_points = points - centroid
        covariance = np.cov(centered_points, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)

        direction = eigenvectors[:, np.argmax(eigenvalues)]
        dx, dy = direction
        angle_deg = self._normalize_angle_deg(np.degrees(np.arctan2(dy, dx)))

        lambda_max = float(np.max(eigenvalues))
        lambda_min = float(np.min(eigenvalues))
        eigenvalue_ratio = lambda_max / (lambda_min + 1e-12)
        explained_variance = lambda_max / (lambda_max + lambda_min + 1e-12)

        return angle_deg, eigenvalue_ratio, explained_variance, direction, centroid

    def _ransac_inliers(
        self,
        points: NDArray[np.float64],
    ) -> NDArray[np.bool_]:
        """
        Trouve, parmi de nombreuses droites candidates tirées de paires
        aléatoires, celle qui compte le plus de points à l'intérieur de
        `self.ransac_tolerance`, et renvoie son masque d'inliers.
        """
        rng = np.random.default_rng(0)
        n_points = points.shape[0]

        sample_a = points[rng.integers(0, n_points, self.ransac_iterations)]
        sample_b = points[rng.integers(0, n_points, self.ransac_iterations)]

        direction = sample_b - sample_a
        length = np.hypot(direction[:, 0], direction[:, 1])
        degenerate = length < 1e-6
        length[degenerate] = 1.0  # évite une division par zéro, écarté ensuite
        direction /= length[:, None]

        # Distance perpendiculaire de chaque point à chaque droite
        # candidate, vectorisée : normal_k . (point_i - a_k) pour chaque (i, k).
        normals = np.column_stack((-direction[:, 1], direction[:, 0]))
        offsets = points[:, None, :] - sample_a[None, :, :]
        distances = np.abs(np.einsum("ikj,kj->ik", offsets, normals))

        inlier_counts = np.sum(distances <= self.ransac_tolerance, axis=0)
        inlier_counts[degenerate] = -1

        best = int(np.argmax(inlier_counts))
        return distances[:, best] <= self.ransac_tolerance


# EOF
