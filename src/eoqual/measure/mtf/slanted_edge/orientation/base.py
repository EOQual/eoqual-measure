#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mécanique générique d'estimation de l'orientation d'un bord incliné.

Ce module contient uniquement ce qui est *partagé* par tous les
algorithmes d'estimation, et rien de spécifique à l'un d'eux :

- `FrameCanonicalizer` : ramène une image à une orientation canonique
  (bord balayé ligne par ligne, région sombre à gauche) et convertit les
  résultats vers le repère d'origine de l'appelant. Générique, agnostique
  de l'algorithme -- chaque estimateur utilise exactement la même
  canonicalisation.
- `OrientationResult` : le type de résultat commun renvoyé par chaque
  estimateur.
- `EdgeOrientationDetector` : une classe de base en *template method*.
  Elle prend en charge la canonicalisation et la conversion vers le
  repère d'origine ; les sous-classes n'implémentent que
  `_estimate_canonical()`.
- `EdgeOrientationComparator` : exécute une collection de détecteurs sur
  la même image et collecte leurs `OrientationResult`.

Les algorithmes concrets eux-mêmes vivent chacun dans leur propre
module :

- `orientation/gradient.py` : `GradientOrientationDetector`.
- `orientation/pca.py` : `PCAOrientationDetector`.
- `orientation/fourier.py` : `FourierOrientationDetector`.

Le résultat de chaque estimateur porte des positions sous-pixel du bord
par ligne en plus de son angle, ainsi qu'une *unique* métrique de
qualité qui n'est pas arbitraire -- elle teste l'hypothèse spécifique
sur laquelle repose l'algorithme (voir le docstring de chaque module).
Construire un tableau ou un graphique de comparaison à partir de
plusieurs `OrientationResult` est laissé à l'appelant (voir `cli.py`).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from eoqual.measure.mtf.slanted_edge.preprocess import PreprocessingResult, SlantedEdgePreprocessor


# --------------------------------------------------------------------------
# Canonicalisation générique (partagée par tous les estimateurs)
# --------------------------------------------------------------------------

# `SlantedEdgePreprocessor` implémente déjà exactement la canonicalisation
# générique, agnostique de l'algorithme, dont chaque estimateur a besoin
# (transposer pour que le bord soit balayé ligne par ligne, retourner pour
# que la région sombre soit à gauche, estimer des niveaux noir/blanc
# robustes). On la réutilise telle quelle et on ne fait qu'aliaser son
# type de résultat pour la lisibilité de ce module.
CanonicalFrame = PreprocessingResult


class FrameCanonicalizer:
    """
    Fine enveloppe autour de `SlantedEdgePreprocessor`, plus les deux
    conversions de repère dont chaque estimateur a besoin pour ramener
    son résultat en repère canonique vers l'image d'origine de
    l'appelant.
    """

    def __init__(self, percentile: float = 10.0) -> None:
        self._preprocessor = SlantedEdgePreprocessor(percentile=percentile)

    def canonicalize(self, image: NDArray[np.float64]) -> CanonicalFrame:
        """Ramène `image` à l'orientation canonique. Voir le docstring du module."""
        return self._preprocessor.preprocess(image)

    @staticmethod
    def angle_to_original_frame(
        angle_deg_canonical: float,
        frame: CanonicalFrame,
    ) -> float:
        """Défait le flip/transpose du repère canonique sur un angle."""
        angle_deg = angle_deg_canonical
        if frame.was_flipped:
            angle_deg = -angle_deg
        if frame.was_transposed:
            angle_deg = 90.0 - angle_deg
        return EdgeOrientationDetector._normalize_angle_deg(angle_deg)

    @staticmethod
    def positions_to_original_frame(
        rows_canonical: NDArray[np.float64],
        columns_canonical: NDArray[np.float64],
        frame: CanonicalFrame,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Défait le flip/transpose du repère canonique sur des positions par ligne."""
        rows = rows_canonical.astype(np.float64).copy()
        columns = columns_canonical.astype(np.float64).copy()

        if frame.was_flipped:
            n_cols = frame.image.shape[1]
            columns = (n_cols - 1) - columns

        if frame.was_transposed:
            rows, columns = columns, rows

        return rows, columns


# --------------------------------------------------------------------------
# Types de résultat communs
# --------------------------------------------------------------------------


@dataclass(slots=True)
class OrientationResult:
    """
    Résultat d'une estimation d'orientation de bord, exprimé dans le
    repère d'image d'origine de l'appelant.

    Attributes
    ----------
    algorithm :
        Nom de l'algorithme ayant produit ce résultat.
    angle_deg :
        Angle de bord estimé en degrés, replié dans [-90, 90] et mesuré
        depuis l'axe des colonnes (`atan2(delta_row, delta_column)`).
    is_horizontal :
        True si le bord est majoritairement horizontal
        (`abs(angle_deg) < 45`).
    was_transposed, was_flipped :
        Indique si l'étape de canonicalisation a dû transposer et/ou
        retourner l'image pour atteindre son orientation de travail
        interne.
    edge_rows, edge_columns :
        Localisation sous-pixel du bord, une paire de valeurs par ligne
        balayée, dans les coordonnées ligne/colonne de l'image
        ORIGINALE -- directement comparables entre algorithmes et
        directement superposables à l'image d'origine.
    quality_metric_name, quality_metric_value :
        L'unique métrique de qualité qui teste l'hypothèse centrale de
        cet algorithme (voir le docstring du module de chaque
        estimateur). Plus grand est toujours meilleur.
    details :
        Toute information supplémentaire spécifique à l'algorithme
        (métriques secondaires, points intermédiaires, messages
        d'erreur, ...).
    """

    algorithm: str
    angle_deg: float
    is_horizontal: bool
    was_transposed: bool
    was_flipped: bool
    edge_rows: NDArray[np.float64]
    edge_columns: NDArray[np.float64]
    quality_metric_name: str
    quality_metric_value: float
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class CanonicalEstimate:
    """
    Ce qu'un estimateur concret calcule, exprimé dans le repère
    CANONIQUE. Transformé en `OrientationResult` par
    `EdgeOrientationDetector`.
    """

    angle_deg: float
    rows: NDArray[np.float64]
    columns: NDArray[np.float64]
    quality_metric_name: str
    quality_metric_value: float
    details: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------
# Interface générique de stratégie (template method)
# --------------------------------------------------------------------------


class EdgeOrientationDetector(ABC):
    """
    Interface commune pour les stratégies d'estimation d'orientation
    d'un bord incliné.

    Les sous-classes n'implémentent que `_estimate_canonical()` ; la
    canonicalisation et la conversion vers le repère d'origine sont
    prises en charge une fois pour toutes, ici. Les estimateurs concrets
    vivent chacun dans leur propre module (`gradient_orientation`,
    `pca_orientation`, `fourier_orientation`), chacun important cette
    classe.
    """

    #: Nom court et unique, utilisé comme clé par `EdgeOrientationComparator`.
    name: str = "base"

    def __init__(self, canonicalization_percentile: float = 10.0) -> None:
        self._canonicalizer = FrameCanonicalizer(
            percentile=canonicalization_percentile
        )

    def analyse(self, image: NDArray[np.float64]) -> OrientationResult:
        """
        Estime l'orientation d'un bord incliné.

        Parameters
        ----------
        image :
            Image d'entrée en niveaux de gris contenant un unique bord
            incliné.

        Returns
        -------
        OrientationResult
            Angle estimé, positions sous-pixel par ligne et métrique de
            qualité, tous exprimés dans le repère propre d'`image`.
        """
        frame = self._canonicalizer.canonicalize(image)
        estimate = self._estimate_canonical(frame)

        angle_deg = self._canonicalizer.angle_to_original_frame(
            estimate.angle_deg, frame
        )
        rows, columns = self._canonicalizer.positions_to_original_frame(
            estimate.rows, estimate.columns, frame
        )

        return OrientationResult(
            algorithm=self.name,
            angle_deg=angle_deg,
            is_horizontal=self._is_horizontal(angle_deg),
            was_transposed=frame.was_transposed,
            was_flipped=frame.was_flipped,
            edge_rows=rows,
            edge_columns=columns,
            quality_metric_name=estimate.quality_metric_name,
            quality_metric_value=estimate.quality_metric_value,
            details=estimate.details,
        )

    @abstractmethod
    def _estimate_canonical(self, frame: CanonicalFrame) -> CanonicalEstimate:
        """Effectue l'estimation proprement dite dans le repère canonique."""
        raise NotImplementedError

    # -- utilitaires partagés, disponibles pour chaque sous-classe -------

    @staticmethod
    def _normalize_angle_deg(angle_deg: float) -> float:
        """Replie un angle exprimé en degrés dans [-90, 90]."""
        while angle_deg > 90.0:
            angle_deg -= 180.0
        while angle_deg < -90.0:
            angle_deg += 180.0
        return angle_deg

    @staticmethod
    def _is_horizontal(angle_deg: float) -> bool:
        """Détermine si un angle correspond à un bord horizontal."""
        return abs(angle_deg) < 45.0

    @staticmethod
    def _robust_threshold(
        magnitude: NDArray[np.float64],
        percentile: float,
    ) -> float:
        """
        Calcule un seuil de magnitude de gradient qui conserve les
        `100 - percentile` pourcents les plus élevés parmi les pixels
        réellement non nuls.

        Un simple `np.percentile(magnitude, percentile)` dégénère à 0
        dès que plus de `100 - percentile` pourcents de l'image sont
        exactement plats (cas fréquent pour un bord fin, non flouté),
        ce qui sélectionnerait silencieusement l'image entière au lieu
        du seul bord. Restreindre le percentile aux valeurs non nulles
        évite ce piège.
        """
        non_zero = magnitude[magnitude > 0]
        if non_zero.size == 0:
            raise ValueError("Gradient magnitude is zero everywhere.")
        return float(np.percentile(non_zero, percentile))

    @staticmethod
    def _fit_angle_from_row_positions(
        rows: NDArray[np.float64],
        columns: NDArray[np.float64],
    ) -> tuple[float, float, float]:
        """
        Ajuste `columns = slope * rows + intercept` et convertit la
        pente en angle de bord (voir le docstring du module pour la
        convention).

        Returns
        -------
        tuple[float, float, float]
            `(angle_deg, slope, r_squared)`.
        """
        if rows.size < 2:
            raise ValueError("Not enough valid rows to fit a slope.")

        slope, intercept = np.polyfit(rows, columns, 1)
        fitted = slope * rows + intercept
        residuals = columns - fitted
        ss_res = float(np.sum(residuals**2))
        ss_tot = float(np.sum((columns - np.mean(columns)) ** 2))
        r_squared = 1.0 - ss_res / (ss_tot + 1e-12)

        angle_deg = EdgeOrientationDetector._normalize_angle_deg(
            np.degrees(np.arctan2(1.0, slope))
        )
        return angle_deg, float(slope), r_squared


# --------------------------------------------------------------------------
# Comparateur
# --------------------------------------------------------------------------


class EdgeOrientationComparator:
    """
    Exécute plusieurs stratégies `EdgeOrientationDetector` sur la même
    image.

    Parameters
    ----------
    detectors :
        Mapping nom -> détecteur à exécuter. Par défaut, une instance de
        chaque détecteur intégré (gradient, PCA, Fourier), importée
        paresseusement depuis leurs propres modules pour éviter un
        import circulaire au chargement du module (chacun d'eux importe
        ce module).
    """

    def __init__(
        self,
        detectors: dict[str, EdgeOrientationDetector] | None = None,
    ) -> None:
        if detectors is None:
            from eoqual.measure.mtf.slanted_edge.orientation.fourier import FourierOrientationDetector
            from eoqual.measure.mtf.slanted_edge.orientation.gradient import GradientOrientationDetector
            from eoqual.measure.mtf.slanted_edge.orientation.pca import PCAOrientationDetector

            detectors = {
                "gradient": GradientOrientationDetector(),
                "pca": PCAOrientationDetector(),
                "fourier": FourierOrientationDetector(),
            }
        self.detectors = detectors

    def compare(
        self,
        image: NDArray[np.float64],
    ) -> dict[str, OrientationResult]:
        """
        Exécute chaque détecteur enregistré sur `image`.

        Parameters
        ----------
        image :
            Image d'entrée en niveaux de gris contenant un unique bord
            incliné.

        Returns
        -------
        dict[str, OrientationResult]
            Un résultat par nom de détecteur. Un détecteur qui lève une
            exception est rapporté avec un angle NaN et le message
            d'erreur dans `details["error"]`, plutôt que d'interrompre
            la comparaison.
        """
        results: dict[str, OrientationResult] = {}

        for name, detector in self.detectors.items():
            try:
                results[name] = detector.analyse(image)
            except Exception as exc:  # noqa: BLE001 - rapportée, non levée
                results[name] = OrientationResult(
                    algorithm=name,
                    angle_deg=float("nan"),
                    is_horizontal=False,
                    was_transposed=False,
                    was_flipped=False,
                    edge_rows=np.array([]),
                    edge_columns=np.array([]),
                    quality_metric_name="",
                    quality_metric_value=float("nan"),
                    details={"error": str(exc)},
                )

        return results


# EOF
