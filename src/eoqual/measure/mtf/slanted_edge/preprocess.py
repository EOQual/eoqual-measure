#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prétraite une image de mire à bord incliné avant la localisation du bord.

Ce module ne réalise que la canonicalisation *générique* géométrique et
photométrique, partagée par tous les algorithmes d'estimation d'angle
(gradient, PCA, Fourier, ...) : ramener le bord à une orientation de
référence commune afin que les algorithmes en aval puissent le balayer
ligne par ligne de façon cohérente.

preprocess() : orchestre le pipeline ;
_compute_column_means() : extrait le profil 1D de luminance ;
_estimate_black_white_levels() : estime les niveaux de référence ;
_is_dark_on_left() : décide de l'orientation photométrique ;
_is_horizontal_edge() : décide de l'orientation géométrique.

Les représentations spécifiques à un algorithme -- comme le signal
symétrique demi-onde utilisé par l'estimateur à phase de Fourier --
n'ont pas leur place ici et sont construites par l'estimateur qui en a
besoin (voir `orientation/base.py`).
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray


@dataclass(slots=True)
class PreprocessingResult:
    """
    Résultat du prétraitement d'une image de mire à bord incliné.

    Attributes
    ----------
    image :
        Image prétraitée avec un bord vertical, région sombre à gauche
        et région claire à droite.
    black_level :
        Niveau de référence noir estimé.
    white_level :
        Niveau de référence blanc estimé.
    was_transposed :
        Indique si l'image d'entrée a été transposée.
    was_flipped :
        Indique si l'image d'entrée a été retournée horizontalement.
    """

    image: NDArray[np.float64]
    black_level: float
    white_level: float
    was_transposed: bool
    was_flipped: bool


class SlantedEdgePreprocessor:
    """
    Prétraite une image de mire à bord incliné avant la localisation du bord.

    Le prétraitement effectue les opérations suivantes :

    1. Détecter si le bord est majoritairement horizontal ou vertical.
    2. Transposer l'image si nécessaire.
    3. S'assurer que la région sombre est à gauche.
    4. Estimer des niveaux de référence noir et blanc robustes.
    """

    def __init__(self, percentile: float = 10.0) -> None:
        """
        Parameters
        ----------
        percentile :
            Pourcentage de colonnes utilisées pour estimer les niveaux
            de référence noir et blanc.
        """
        if not 0.0 < percentile <= 100.0:
            raise ValueError("'percentile' must belong to (0, 100].")

        self.percentile = percentile

    def preprocess(
        self,
        image: NDArray[np.float64],
    ) -> PreprocessingResult:
        """
        Prétraite une image de mire à bord incliné.

        Parameters
        ----------
        image :
            Image d'entrée en niveaux de gris.

        Returns
        -------
        PreprocessingResult
            Résultat complet du prétraitement.
        """
        processed_image = image.copy()

        was_transposed = False
        was_flipped = False

        if self._is_horizontal_edge(processed_image):
            processed_image = processed_image.T
            was_transposed = True

        column_means = self._compute_column_means(processed_image)

        if not self._is_dark_on_left(column_means):
            processed_image = np.fliplr(processed_image)
            column_means = column_means[::-1]
            was_flipped = True

        black_level, white_level = self._estimate_black_white_levels(
            column_means
        )

        return PreprocessingResult(
            image=processed_image,
            black_level=black_level,
            white_level=white_level,
            was_transposed=was_transposed,
            was_flipped=was_flipped,
        )

    def _compute_column_means(
        self,
        image: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """
        Calcule l'intensité moyenne de chaque colonne de l'image.

        Parameters
        ----------
        image :
            Image d'entrée en niveaux de gris.

        Returns
        -------
        NDArray[np.float64]
            Intensité moyenne de chaque colonne de l'image.
        """
        return np.mean(image, axis=0)

    def _estimate_black_white_levels(
        self,
        column_means: NDArray[np.float64],
    ) -> tuple[float, float]:
        """
        Estime les niveaux de référence noir et blanc.

        Parameters
        ----------
        column_means :
            Intensité moyenne de chaque colonne de l'image.

        Returns
        -------
        tuple[float, float]
            Niveaux noir et blanc estimés.
        """
        sorted_means = np.sort(column_means)

        number_of_columns = max(
            1,
            int(np.ceil(sorted_means.size * self.percentile / 100.0)),
        )

        black_level = float(np.mean(sorted_means[:number_of_columns]))
        white_level = float(np.mean(sorted_means[-number_of_columns:]))

        if black_level >= white_level:
            raise ValueError(
                "Unable to estimate a valid image dynamic range."
            )

        return black_level, white_level

    def _is_dark_on_left(
        self,
        column_means: NDArray[np.float64],
    ) -> bool:
        """
        Détermine si la région sombre se trouve à gauche.

        Parameters
        ----------
        column_means :
            Intensité moyenne de chaque colonne de l'image.

        Returns
        -------
        bool
            True si le côté gauche est plus sombre que le côté droit.
        """
        number_of_columns = max(
            1,
            int(np.ceil(column_means.size * self.percentile / 100.0)),
        )

        left_level = float(
            np.mean(column_means[:number_of_columns])
        )

        right_level = float(
            np.mean(column_means[-number_of_columns:])
        )

        return left_level < right_level

    @staticmethod
    def _is_horizontal_edge(
        image: NDArray[np.float64],
    ) -> bool:
        """
        Détermine si le bord est majoritairement horizontal.

        Parameters
        ----------
        image :
            Image d'entrée en niveaux de gris.

        Returns
        -------
        bool
            True si le bord est majoritairement horizontal.
        """
        gradient_x = cv2.Sobel(image, cv2.CV_64F, 1, 0)
        gradient_y = cv2.Sobel(image, cv2.CV_64F, 0, 1)

        horizontal_gradient_energy = np.sum(np.abs(gradient_x))
        vertical_gradient_energy = np.sum(np.abs(gradient_y))

        return vertical_gradient_energy > horizontal_gradient_energy


@dataclass(slots=True)
class PCAOrientationResult:
    """
    Résultat de l'analyse d'orientation du bord par PCA.

    Attributes
    ----------
    angle_deg :
        Angle estimé du bord, en degrés.
    is_horizontal :
        True si le bord est majoritairement horizontal.
    gradient_points :
        Coordonnées des pixels utilisés pour la PCA.
    """

    angle_deg: float
    is_horizontal: bool
    gradient_points: NDArray[np.float64]


class PCAEdgeOrientationDetector:
    """
    Estime l'orientation d'un bord incliné à partir des pixels de fort
    gradient et d'une PCA.

    Parameters
    ----------
    gradient_percentile :
        Seuil de percentile utilisé pour ne conserver que les pixels de
        fort gradient. Une valeur plus élevée conserve moins de pixels
        et se concentre sur le bord.
    """

    def __init__(
        self,
        gradient_percentile: float = 90.0,
    ) -> None:

        if not 0 < gradient_percentile < 100:
            raise ValueError(
                "'gradient_percentile' must belong to (0, 100)."
            )

        self.gradient_percentile = gradient_percentile

    def analyse(
        self,
        image: NDArray[np.float64],
    ) -> PCAOrientationResult:
        """
        Estime l'orientation d'un bord incliné.

        Parameters
        ----------
        image :
            Image d'entrée en niveaux de gris.

        Returns
        -------
        PCAOrientationResult
            Informations sur l'orientation du bord.
        """

        gradient_x = cv2.Sobel(
            image,
            cv2.CV_64F,
            1,
            0,
            ksize=3,
        )

        gradient_y = cv2.Sobel(
            image,
            cv2.CV_64F,
            0,
            1,
            ksize=3,
        )

        gradient_magnitude = np.hypot(
            gradient_x,
            gradient_y,
        )

        threshold = np.percentile(
            gradient_magnitude,
            self.gradient_percentile,
        )

        y, x = np.nonzero(
            gradient_magnitude >= threshold
        )

        if len(x) < 10:
            raise ValueError(
                "Not enough gradient points detected."
            )

        points = np.column_stack(
            (
                x.astype(np.float64),
                y.astype(np.float64),
            )
        )

        # Centrage
        centered_points = (
            points - np.mean(points, axis=0)
        )

        covariance = np.cov(
            centered_points,
            rowvar=False,
        )

        eigenvalues, eigenvectors = np.linalg.eigh(
            covariance
        )

        # Plus grande valeur propre = direction du bord
        principal_direction = eigenvectors[
            :, np.argmax(eigenvalues)
        ]

        dx, dy = principal_direction

        angle_rad = np.arctan2(
            dy,
            dx,
        )

        angle_deg = np.degrees(angle_rad)

        # Ramener dans [-90°, 90°]
        if angle_deg > 90:
            angle_deg -= 180

        if angle_deg < -90:
            angle_deg += 180

        is_horizontal = abs(angle_deg) < 45

        return PCAOrientationResult(
            angle_deg=float(angle_deg),
            is_horizontal=is_horizontal,
            gradient_points=points,
        )


#EOF
