#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Estimateur d'orientation de bord incliné basé sur la phase de Fourier.

Voir `FourierOrientationDetector` pour la raison pour laquelle une
représentation symétrique demi-onde doit être construite avant de
prendre la FFT, et pour l'hypothèse que sa métrique de qualité teste.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from eoqual.measure.mtf.slanted_edge.orientation.base import CanonicalEstimate, CanonicalFrame, EdgeOrientationDetector
from eoqual.measure.mtf.slanted_edge.fourier_utils import estimate_slope_multiharmonic


class FourierOrientationDetector(EdgeOrientationDetector):
    """
    Estime l'orientation d'un bord incliné à partir de la phase de
    l'harmonique fondamentale de Fourier.

    Pourquoi une représentation symétrique demi-onde est nécessaire
    ------------------------------------------------------------------
    Une ligne de bord brute est un *échelon*, pas un signal périodique :
    la FFT le traite néanmoins comme périodique, ce qui crée
    silencieusement une seconde discontinuité artificielle là où la
    ligne se rebouclerait. Ce signal périodisé n'a aucune raison d'avoir
    des harmoniques paires nulles, donc sa phase ne peut pas être
    utilisée pour localiser le bord.

    Un vrai créneau à 50 % de rapport cyclique, en revanche, vérifie la
    symétrie demi-onde `f(x + T/2) = -f(x)` : sa seconde demi-période
    est exactement l'opposée de la première. En développant ce signal
    en série de Fourier, on montre que cette symétrie impose l'annula-
    tion de tout coefficient d'ordre pair (`a_2m = b_2m = 0`) ; seules
    les harmoniques impaires subsistent. C'est la propriété dont
    dépendent silencieusement les algorithmes du type `computeSlope()`
    lorsqu'ils ne regardent que `spectrum[1::2]`.

    Ainsi, avant de prendre la FFT, chaque ligne est transformée en un
    tel créneau par mirroring : `symmetric_row = concatenate(row,
    row[::-1])`. Pour une ligne valant `black_level` jusqu'au bord puis
    `white_level` après, cette construction produit une impulsion exacte
    à 50 % de rapport cyclique, de largeur égale à la longueur de la
    ligne, centrée dans une période deux fois plus longue -- ce qui
    possède bien la symétrie demi-onde requise, si bien que ses
    harmoniques paires sont négligeables (toute énergie résiduelle sur
    ces harmoniques est un signe direct et mesurable que la ligne
    s'écarte d'un échelon propre). Cette construction est entièrement
    spécifique à cet estimateur, ce pour quoi elle est construite ici
    plutôt que dans l'étape de canonicalisation générique.

    Hypothèse testée : la ligne symétrique construite de cette manière
    se comporte comme un signal symétrique demi-onde. Métrique de
    qualité : rapport de l'énergie des harmoniques impaires sur celle
    des harmoniques paires -- grand quand l'hypothèse est vérifiée,
    petit lorsqu'elle est violée (asymétrie du flou, saturation, bruit,
    niveaux noir/blanc estimés sur de mauvaises colonnes, ...).

    Localisation sous-pixel : décaler un signal de `d` multiplie son
    k-ième coefficient de Fourier par `exp(-2j*pi*k*d/T)` (théorème du
    décalage). Pour k=1 et une période `T = 2 * n_cols`, cette phase
    encode directement la colonne `d` du bord (modulo `n_cols`) -- voir
    `_locate_edge_from_phase()`.

    Deux façons d'estimer l'angle
    ------------------------------
    `_locate_edge_from_phase()` n'utilise que l'harmonique fondamentale
    (k=1) pour obtenir une position absolue par ligne, ensuite ajustée
    par régression linéaire sur toutes les lignes (`_fit_angle_from_row_
    positions`). C'est ce calcul qui alimente `rows`/`columns`, car ce
    sont ces positions par ligne que `cli.py` superpose au tracé
    comparatif des trois algorithmes -- elles sont donc toujours
    calculées, quel que soit `angle_source`.

    `_estimate_multiharmonic_angle()` implémente une seconde stratégie,
    différente : plutôt qu'une position absolue par ligne, elle mesure
    la différence de phase entre paires de lignes voisines (séparées de
    `block_gap`), moyennée sur plusieurs harmoniques impaires basses
    (fraction `low_freq_ratio`) et sur toutes les paires de lignes. En
    exploitant plusieurs harmoniques et de nombreuses paires au lieu
    d'un seul harmonique et d'une régression globale, cette approche est
    statistiquement plus robuste au bruit -- au prix de ne fournir
    qu'un angle global, sans position par ligne.

    Le paramètre `angle_source` choisit laquelle des deux alimente
    `angle_deg` (le résultat "officiel"). L'autre reste disponible dans
    `details`, avec son propre préfixe (`*_fundamental` /
    `*_multiharmonic`), pour comparaison ou diagnostic.

    Parameters
    ----------
    canonicalization_percentile :
        Percentile de colonnes utilisé pour estimer les niveaux de
        référence noir/blanc (transmis à `FrameCanonicalizer`).
    angle_source :
        `"fundamental"` (par défaut) pour utiliser l'angle issu de la
        régression sur les positions par ligne (k=1) ; `"multiharmonic"`
        pour utiliser l'estimation multi-harmoniques par différence de
        phase entre lignes voisines.
    low_freq_ratio :
        Fraction des harmoniques impaires les plus basses utilisées par
        l'estimation multi-harmoniques (les plus basses fréquences sont
        les moins bruitées).
    block_gap :
        Écart, en lignes, entre les deux lignes comparées par
        l'estimation multi-harmoniques.
    """

    name = "fourier"

    def __init__(
        self,
        canonicalization_percentile: float = 10.0,
        angle_source: str = "fundamental",
        low_freq_ratio: float = 0.2,
        block_gap: int = 2,
    ) -> None:
        super().__init__(canonicalization_percentile)
        if angle_source not in ("fundamental", "multiharmonic"):
            raise ValueError(
                "'angle_source' must be 'fundamental' or 'multiharmonic'."
            )
        if block_gap < 1:
            raise ValueError("'block_gap' must be >= 1.")
        if not 0.0 < low_freq_ratio <= 1.0:
            raise ValueError("'low_freq_ratio' must belong to (0, 1].")

        self.angle_source = angle_source
        self.low_freq_ratio = low_freq_ratio
        self.block_gap = block_gap

    def _estimate_canonical(self, frame: CanonicalFrame) -> CanonicalEstimate:
        image = frame.image
        n_rows, n_cols = image.shape

        symmetric_image = self._build_symmetric_representation(
            image, frame.black_level, frame.white_level
        )

        spectrum = np.fft.rfft(symmetric_image, axis=1)
        magnitude = np.abs(spectrum)

        # Rapport d'énergie harmoniques impaires/paires (DC exclue) : le
        # test direct et mesurable de l'hypothèse de symétrie demi-onde.
        odd_energy = float(np.sum(magnitude[:, 1::2] ** 2))
        even_energy = float(np.sum(magnitude[:, 2::2] ** 2))
        odd_even_ratio = odd_energy / (even_energy + 1e-12)

        # Toujours calculées : ce sont elles qui alimentent le tracé
        # comparatif entre algorithmes (voir docstring de la classe).
        columns = self._locate_edge_from_phase(spectrum[:, 1], n_cols)
        rows = np.arange(n_rows, dtype=np.float64)
        angle_fundamental, slope_fundamental, r_squared_fundamental = (
            self._fit_angle_from_row_positions(rows, columns)
        )

        # Toujours calculée aussi, pour rester disponible en comparaison
        # même quand elle n'est pas la source retenue pour angle_deg.
        dx_multiharmonic, angle_multiharmonic = (
            self._estimate_multiharmonic_angle(symmetric_image)
        )

        if self.angle_source == "fundamental":
            angle_deg = angle_fundamental
        else:
            angle_deg = angle_multiharmonic

        return CanonicalEstimate(
            angle_deg=angle_deg,
            rows=rows,
            columns=columns,
            quality_metric_name="odd_even_ratio",
            quality_metric_value=odd_even_ratio,
            details={
                "angle_source": self.angle_source,
                "angle_deg_fundamental": angle_fundamental,
                "slope_fundamental": slope_fundamental,
                "r_squared_fundamental": r_squared_fundamental,
                "angle_deg_multiharmonic": angle_multiharmonic,
                "dx_multiharmonic": dx_multiharmonic,
            },
        )

    @staticmethod
    def _build_symmetric_representation(
        image: NDArray[np.float64],
        black_level: float,
        white_level: float,
    ) -> NDArray[np.float64]:
        """
        Construit, pour chaque ligne, un créneau symétrique demi-onde à
        50 % de rapport cyclique à partir de la transition de bord
        (éventuellement douce). Voir le docstring de la classe pour la
        raison pour laquelle cette construction précise est nécessaire.
        """
        dynamic_range = white_level - black_level
        distance_to_white = white_level - image
        distance_to_black = image - black_level
        return (
            np.concatenate((distance_to_white, distance_to_black), axis=1)
            / dynamic_range
        )

    @staticmethod
    def _locate_edge_from_phase(
        fundamental: NDArray[np.complex128],
        n_cols: int,
    ) -> NDArray[np.float64]:
        """
        Récupère la colonne sous-pixel du bord de chaque ligne à partir
        de la phase de l'harmonique k=1 de sa représentation symétrique
        demi-onde (période `T = 2 * n_cols`), puis déroule (unwrap) à
        travers les lignes afin qu'un bord doucement incliné produise
        une séquence de positions lisse plutôt qu'une séquence qui se
        rebouclerait à chaque franchissement de bord de ligne.
        """
        phase = np.angle(fundamental)

        # La représentation symétrique a une période T = 2 * n_cols (elle
        # est deux fois plus large que la ligne d'origine).
        # exp(-2j*pi*d/T) avec un décalage de phase de référence de
        # T/4 = n_cols/2, calibré par rapport à une position d'échelon
        # synthétique exactement connue, donne :
        # d = -phase/(2*pi)*T + T/4, replié dans [0, n_cols).
        period = 2.0 * n_cols
        position = -phase / (2.0 * np.pi) * period + n_cols / 2.0
        position = np.mod(position, n_cols)

        # Déroule (unwrap) à travers les lignes (période n_cols) en
        # supposant que le bord ne se déplace pas de plus d'une demi-
        # largeur de ligne d'une ligne à l'autre.
        position_rad = position * (2.0 * np.pi / n_cols)
        unwrapped_rad = np.unwrap(position_rad)
        return unwrapped_rad * (n_cols / (2.0 * np.pi))

    def _estimate_multiharmonic_angle(
        self,
        symmetric_image: NDArray[np.float64],
    ) -> tuple[float, float]:
        """
        Estime le déplacement moyen ligne-à-ligne (`dx`) et l'angle du
        bord à partir de la différence de phase entre paires de lignes
        séparées de `self.block_gap`, moyennée sur les
        `self.low_freq_ratio` harmoniques impaires les plus basses.

        Contrairement à `_locate_edge_from_phase()` (qui n'utilise que
        l'harmonique fondamentale pour une position absolue par ligne),
        cette méthode moyenne l'information de nombreux harmoniques et
        de nombreuses paires de lignes, ce qui la rend statistiquement
        plus robuste au bruit -- au prix de ne donner qu'un angle
        global, sans position par ligne. Le calcul de `dx` lui-même est
        partagé avec l'estimation de MTF (voir `fourier_utils.py`), qui
        a besoin de la même quantité.

        Parameters
        ----------
        symmetric_image :
            Représentation symétrique demi-onde (voir
            `_build_symmetric_representation`), de largeur
            `T = 2 * n_cols`.

        Returns
        -------
        tuple[float, float]
            `(dx, angle_deg)` où `dx` est le déplacement moyen en
            colonnes par ligne, et `angle_deg` sa conversion dans la
            convention d'angle du module (voir docstring de la classe).
        """
        dx, _theta_rad_reference_convention = estimate_slope_multiharmonic(
            symmetric_image, self.low_freq_ratio, self.block_gap
        )
        angle_deg = self._normalize_angle_deg(
            np.degrees(np.arctan2(1.0, dx))
        )
        return dx, angle_deg


# EOF
