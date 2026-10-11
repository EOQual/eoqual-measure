#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Estimation générique de la MTF (Modulation Transfer Function) à partir
d'un bord incliné (méthode « bord de plage » / slanted-edge).

Même architecture que `orientation/base.py` pour l'estimation d'angle :

- `MTFResult` : le type de résultat commun renvoyé par chaque estimateur.
- `SlantedEdgeMTFEstimator` : une classe de base en *template method*.
- `MTFComparator` : exécute plusieurs estimateurs sur la même zone et
  collecte leurs `MTFResult`, pour les comparer.

Les approches concrètes vivent chacune dans leur propre module :

- `estimators/spatial.py` : `SpatialMTFEstimator` (dé-interlacement + FFT).
- `estimators/spectral.py` : `SpectralMTFEstimator` (moindres carrés multi-
  répliques dans le domaine fréquentiel, avec spectre de référence).
- `estimators/esf_lsf.py` : `EsfLsfMTFEstimator` (détection de bord paramétrique
  ligne par ligne, ESF/LSF, MTF par FFT de la LSF).
- `estimators/spectral_correlation.py` : `SpectralCorrelationMTFEstimator`
  (corrélation de phase à une composante, fréquence par fréquence, sans
  spectre de référence ni mirroring).

Entrée attendue par tous les estimateurs
-----------------------------------------
`frame` est un `CanonicalFrame` (voir `preprocess.SlantedEdgePreprocessor`) :
l'image déjà canonicalisée (bord scanné ligne par ligne, région sombre à
gauche), telle quelle -- PAS encore mise sous forme symétrique demi-onde.

Ce choix (frame canonique plutôt que représentation déjà symétrisée) est
délibéré : les approches spatiale et spectrale ont besoin de la symétrie
demi-onde (voir `FourierOrientationDetector._build_symmetric_
representation`) pour leur FFT, mais ce n'est pas vrai de toutes les
approches -- `EsfLsfMTFEstimator`, par exemple, détecte le bord
directement sur l'image canonique, sans mirroring. Chaque estimateur
construit donc lui-même, dans `_compute_mtf()` ou en surchargeant
`estimate()`, la représentation dont il a besoin à partir de `frame`.
`SlantedEdgeMTFEstimator.estimate()` fournit `rect` (symétrisé) tout
fait aux sous-classes qui en ont besoin, pour éviter de dupliquer cette
construction dans chacune d'elles.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from eoqual.core.edge import frequency_at

from eoqual.measure.mtf.slanted_edge.fourier_utils import estimate_slope_multiharmonic
from eoqual.measure.mtf.slanted_edge.orientation.fourier import FourierOrientationDetector
from eoqual.measure.mtf.slanted_edge.preprocess import PreprocessingResult

#: Alias de lisibilité, comme dans `orientation/base.py`.
CanonicalFrame = PreprocessingResult


# --------------------------------------------------------------------------
# Type de résultat commun
# --------------------------------------------------------------------------


@dataclass(slots=True)
class MTFResult:
    """
    Résultat d'une estimation de MTF à partir d'un bord incliné.

    Attributes
    ----------
    method :
        Nom de la méthode ayant produit ce résultat.
    mtf :
        Fonction de transfert de modulation à chaque fréquence de
        `frequency`. Complexe pour les méthodes qui préservent la phase
        (spatiale, spectrale) ; réelle (module déjà pris) pour celles
        qui ne calculent que le module (ESF/LSF). `np.abs(mtf)` est
        toujours la MTF au sens usuel.
    frequency :
        Fréquences normalisées (`f / f_e`, `f_e` = fréquence
        d'échantillonnage, 0.5 = Nyquist), NON corrigées de
        l'inclinaison.
    frequency_corrected :
        Mêmes fréquences, corrigées par `cos(theta)` : la MTF est
        mesurée le long du bord incliné, cette correction la ramène à
        la direction perpendiculaire au bord (la direction physiquement
        pertinente pour une MTF de capteur).
    dx :
        Déplacement moyen ligne-à-ligne du bord (pixels).
    theta_rad :
        Angle d'inclinaison du bord (radians), `theta = arctan(dx)`.
    oversample_factor :
        Facteur de sur-échantillonnage utilisé (1, 2 ou 4). Sans effet
        pour les méthodes qui n'en ont pas besoin (valeur conservée
        pour la cohérence de l'interface).
    details :
        Toute information supplémentaire spécifique à la méthode.
    """

    method: str
    mtf: NDArray[np.complex128]
    frequency: NDArray[np.float64]
    frequency_corrected: NDArray[np.float64]
    dx: float
    theta_rad: float
    oversample_factor: int
    details: dict[str, Any] = field(default_factory=dict)

    def value_at(
        self,
        relative_frequency: float,
        angle_correction: bool = True,
    ) -> float:
        """
        Interpole la valeur de MTF (module) à une fréquence normalisée
        donnée.

        Parameters
        ----------
        relative_frequency :
            Valeur comprise entre 0 (continu) et 0.5 (Nyquist).
        angle_correction :
            Si `True`, utilise `frequency_corrected` (par `cos(theta)`).

        Returns
        -------
        float
            Valeur de MTF (module) interpolée.
        """
        freq = self.frequency_corrected if angle_correction else self.frequency
        return float(np.interp(relative_frequency, freq, np.abs(self.mtf)))

    def frequency_at(
        self,
        level: float = 0.5,
        angle_correction: bool = True,
    ) -> float:
        """
        Fréquence normalisée à laquelle la MTF (module) descend pour la
        première fois sous `level` -- MTF50 pour `level=0.5`.

        Interpolation linéaire entre les deux échantillons qui encadrent
        le premier passage sous `level`. NaN si la MTF ne descend jamais
        sous `level` sur la plage mesurée.
        """
        freq = self.frequency_corrected if angle_correction else self.frequency
        return frequency_at(freq, self.mtf, level)


# --------------------------------------------------------------------------
# Interface générique de stratégie (template method)
# --------------------------------------------------------------------------


class SlantedEdgeMTFEstimator(ABC):
    """
    Interface commune pour les stratégies d'estimation de MTF par bord
    incliné.

    La plupart des sous-classes n'implémentent que `_compute_mtf()` :
    `estimate()` construit pour elles la représentation symétrisée
    (`rect`, voir docstring du module) à partir de `frame`, estime la
    pente du bord (partagée par ces approches, voir
    `fourier_utils.estimate_slope_multiharmonic`) et applique la
    correction de fréquence par `cos(theta)`.

    Une sous-classe dont l'approche ne repose pas sur cette
    représentation symétrisée (ex. `EsfLsfMTFEstimator`, qui détecte le
    bord directement sur l'image canonique) peut simplement surcharger
    `estimate()` en entier ; `_compute_mtf()` reste alors inutilisée
    (voir sa docstring dans ce cas).

    Parameters
    ----------
    low_freq_ratio :
        Portion des harmoniques impaires les plus basses utilisées pour
        l'estimation de la pente du bord (transmis à
        `estimate_slope_multiharmonic`).
    block_gap :
        Distance (en lignes) entre les deux lignes comparées pour
        l'estimation de la pente du bord.
    pixel_pitch :
        Facteur de conversion de la fréquence "brute" (cycles par
        échantillon du tableau fourni) vers la fréquence physique
        rapportée (`frequency`/`frequency_corrected`) : `frequency =
        frequency_brute * pixel_pitch`. Défaut 1.0 (aucune conversion :
        un échantillon du tableau = un pixel physique du capteur).
        Pour un plan Bayer sous-échantillonné (un échantillon du plan =
        `1/sampling_factor` pixels physiques), utiliser `pixel_pitch =
        sampling_factor` (ex. 0.5 pour RGGB) -- voir `eoqual.core.bayer`. Ce
        paramètre est le même pour les quatre méthodes (spatiale,
        spectrale, ESF/LSF, corrélation spectrale), ce qui les rend
        directement comparables sur un même plan, quelle que soit sa
        résolution native.
    """

    #: Nom court et unique, utilisé comme clé par `MTFComparator`.
    name: str = "base"

    def __init__(
        self,
        low_freq_ratio: float = 0.2,
        block_gap: int = 2,
        pixel_pitch: float = 1.0,
    ) -> None:
        self.low_freq_ratio = low_freq_ratio
        self.block_gap = block_gap
        self.pixel_pitch = pixel_pitch

    def estimate(
        self,
        frame: CanonicalFrame,
        oversample_factor: int = 4,
    ) -> MTFResult:
        """
        Estime la MTF à partir d'une zone contenant un bord incliné.

        Parameters
        ----------
        frame :
            Image canonicalisée (voir docstring du module).
        oversample_factor :
            1, 2 ou 4. Un facteur plus grand génère des points
            supplémentaires entre `f_e/4` et `f_e/2`.

        Returns
        -------
        MTFResult
            MTF estimée, avec l'angle et le déplacement ayant servi à
            la calculer.
        """
        if oversample_factor not in (1, 2, 4):
            raise ValueError(
                f"oversample_factor={oversample_factor} non supporté (1, 2, 4)."
            )

        rect = self._build_symmetric_representation(frame)

        dx, theta_rad = estimate_slope_multiharmonic(
            rect, self.low_freq_ratio, self.block_gap
        )

        frequency, mtf, details = self._compute_mtf(
            rect, dx, theta_rad, oversample_factor
        )
        frequency = frequency * self.pixel_pitch
        frequency_corrected = frequency / np.cos(theta_rad)

        return MTFResult(
            method=self.name,
            mtf=mtf,
            frequency=frequency,
            frequency_corrected=frequency_corrected,
            dx=dx,
            theta_rad=theta_rad,
            oversample_factor=oversample_factor,
            details=details,
        )

    @staticmethod
    def _build_symmetric_representation(
        frame: CanonicalFrame,
    ) -> NDArray[np.float64]:
        """
        Construit la représentation symétrique demi-onde à partir de
        `frame` -- délègue à `FourierOrientationDetector._build_
        symmetric_representation` plutôt que de dupliquer cette
        construction ici (import au niveau module : `orientation/fourier.py`
        ne dépend pas de ce module, aucun cycle).
        """
        return FourierOrientationDetector._build_symmetric_representation(
            frame.image, frame.black_level, frame.white_level
        )

    @abstractmethod
    def _compute_mtf(
        self,
        rect: NDArray[np.float64],
        dx: float,
        theta_rad: float,
        oversample_factor: int,
    ) -> tuple[NDArray[np.float64], NDArray[np.complex128], dict[str, Any]]:
        """
        Calcule la MTF proprement dite, une fois la pente du bord connue.

        Returns
        -------
        tuple
            `(frequency, mtf, details)` -- fréquences normalisées NON
            corrigées, MTF à ces fréquences, et informations
            complémentaires propres à la méthode.
        """
        raise NotImplementedError


# --------------------------------------------------------------------------
# Comparateur
# --------------------------------------------------------------------------


class MTFComparator:
    """
    Exécute plusieurs stratégies `SlantedEdgeMTFEstimator` sur la même
    zone.

    Parameters
    ----------
    estimators :
        Mapping nom -> estimateur à exécuter. Par défaut, une instance
        de chaque estimateur intégré (spatial, spectral, esf_lsf,
        spectral_correlation),
        importée paresseusement pour éviter un import circulaire au
        chargement du module.
    """

    def __init__(
        self,
        estimators: dict[str, SlantedEdgeMTFEstimator] | None = None,
    ) -> None:
        if estimators is None:
            from eoqual.measure.mtf.slanted_edge.estimators.esf_lsf import EsfLsfMTFEstimator
            from eoqual.measure.mtf.slanted_edge.estimators.spatial import SpatialMTFEstimator
            from eoqual.measure.mtf.slanted_edge.estimators.spectral_correlation import SpectralCorrelationMTFEstimator
            from eoqual.measure.mtf.slanted_edge.estimators.spectral import SpectralMTFEstimator

            estimators = {
                "spatial": SpatialMTFEstimator(),
                "spectral": SpectralMTFEstimator(),
                "esf_lsf": EsfLsfMTFEstimator(),
                "spectral_correlation": SpectralCorrelationMTFEstimator(),
            }
        self.estimators = estimators

    def compare(
        self,
        frame: CanonicalFrame,
        oversample_factor: int = 4,
    ) -> dict[str, MTFResult]:
        """
        Exécute chaque estimateur enregistré sur `frame`.

        Parameters
        ----------
        frame :
            Image canonicalisée (voir docstring du module).
        oversample_factor :
            1, 2 ou 4.

        Returns
        -------
        dict[str, MTFResult]
            Un résultat par nom d'estimateur. Un estimateur qui lève
            une exception est rapporté avec `details["error"]` renseigné
            plutôt que d'interrompre la comparaison.
        """
        results: dict[str, MTFResult] = {}

        for name, estimator in self.estimators.items():
            try:
                results[name] = estimator.estimate(frame, oversample_factor)
            except Exception as exc:  # noqa: BLE001 - rapportée, non levée
                results[name] = MTFResult(
                    method=name,
                    mtf=np.array([]),
                    frequency=np.array([]),
                    frequency_corrected=np.array([]),
                    dx=float("nan"),
                    theta_rad=float("nan"),
                    oversample_factor=oversample_factor,
                    details={"error": str(exc)},
                )

        return results


# EOF
