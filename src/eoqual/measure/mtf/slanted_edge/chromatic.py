#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Aberration chromatique latérale : compare la position du bord mesurée
sur plusieurs plans Bayer (R, G, B, ...) pour en déduire leur décalage
relatif, en pixels pleine résolution.

Principe : chaque plan Bayer est sous-échantillonné à une position
différente de la grille pleine résolution du capteur
(`eoqual.core.bayer.bayer_offsets("RGGB")`). La position sous-pixel du bord, déjà
mesurée indépendamment sur chaque plan par n'importe quel détecteur
d'orientation (`edge_rows`/`edge_columns` d'un `OrientationResult`),
est ramenée dans ce repère commun ; l'écart entre plans à une même ligne
de référence est le décalage chromatique latéral.

Ce module ne fait aucun nouveau calcul d'orientation : il réutilise les
`OrientationResult` déjà produits par `EdgeOrientationComparator` pour
chaque plan (voir `cli.py`).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from eoqual.core.bayer import bayer_offsets
from eoqual.measure.mtf.slanted_edge.orientation.base import OrientationResult


@dataclass(slots=True)
class ChromaticShiftResult:
    """
    Résultat d'une mesure d'aberration chromatique latérale.

    Attributes
    ----------
    reference_plane :
        Plan pris comme référence (décalage nul par définition).
    shifts_px :
        Décalage de chaque plan par rapport à la référence, en pixels
        PLEINE RÉSOLUTION, le long de l'axe perpendiculaire au bord --
        positif ou négatif selon le sens. Ne contient pas la référence
        elle-même (toujours 0 par construction).
    reference_row_full :
        Ligne (repère pleine résolution) à laquelle le décalage a été
        évalué -- les droites de bord de chaque plan étant linéaires,
        ce choix n'affecte le résultat qu'à l'ordre 2 près (léger écart
        d'angle résiduel entre plans), mais est documenté pour
        traçabilité.
    """

    reference_plane: str
    shifts_px: dict[str, float]
    reference_row_full: float


def _fit_line_full_resolution(
    result: OrientationResult,
    row_offset: float,
    col_offset: float,
) -> tuple[float, float]:
    """
    Ajuste la droite de bord (`edge_rows`/`edge_columns`, dans le repère
    du plan sous-échantillonné) et la convertit en repère pleine
    résolution : `col_plein = pente * ligne_pleine + ordonnée_pleine`.

    La pente est invariante par ce changement de repère (les deux axes
    sont mis à l'échelle par le même facteur 2) ; seule l'ordonnée à
    l'origine se décale, selon `row_offset`/`col_offset` (voir
    `eoqual.core.bayer.bayer_offsets("RGGB")`).
    """
    if result.edge_rows.size < 2:
        raise ValueError(
            f"Not enough edge positions to fit a line for '{result.algorithm}'."
        )
    slope, intercept = np.polyfit(result.edge_rows, result.edge_columns, 1)
    full_intercept = 2.0 * intercept - slope * row_offset + col_offset
    return float(slope), float(full_intercept)


def compute_chromatic_shifts(
    orientation_results: dict[str, OrientationResult],
    reference_plane: str = "G",
    reference_row_full: float | None = None,
    plane_offsets: dict[str, tuple[float, float]] | None = None,
) -> ChromaticShiftResult:
    """
    Calcule le décalage chromatique latéral entre plans Bayer, à partir
    de résultats d'orientation déjà calculés indépendamment sur chacun.

    Parameters
    ----------
    orientation_results :
        Un `OrientationResult` par plan (ex. `{"R": ..., "G": ...,
        "B": ...}`), typiquement produit en appelant
        `EdgeOrientationComparator` séparément sur chaque plan et en
        choisissant un même algorithme (le plus fiable/cohérent, ex.
        `"fourier"`) pour chacun -- mélanger des algorithmes différents
        entre plans introduirait un biais propre à chaque algorithme
        dans la mesure de décalage, pas seulement l'aberration
        chromatique elle-même.
    reference_plane :
        Plan de référence (décalage nul par définition). Doit être une
        clé de `orientation_results`.
    reference_row_full :
        Ligne pleine résolution à laquelle évaluer le décalage. Par
        défaut, le milieu de la plage de lignes couverte par le plan de
        référence (converti en repère plein).
    plane_offsets :
        Décalages `(ligne, colonne)` de chaque plan dans la grille
        pleine résolution -- par défaut `eoqual.core.bayer.bayer_offsets("RGGB")`.

    Returns
    -------
    ChromaticShiftResult
    """
    if reference_plane not in orientation_results:
        raise ValueError(
            f"'{reference_plane}' not in orientation_results "
            f"(available: {sorted(orientation_results)})."
        )
    offsets = plane_offsets or bayer_offsets("RGGB")
    missing = set(orientation_results) - set(offsets)
    if missing:
        raise ValueError(
            f"No known plane offset for {sorted(missing)} -- pass "
            "'plane_offsets' explicitly for custom plane names."
        )

    lines = {
        plane: _fit_line_full_resolution(result, *offsets[plane])
        for plane, result in orientation_results.items()
    }

    ref_slope, ref_intercept = lines[reference_plane]
    if reference_row_full is None:
        row_off, _ = offsets[reference_plane]
        ref_result = orientation_results[reference_plane]
        reference_row_full = float(2.0 * np.mean(ref_result.edge_rows) + row_off)

    ref_col = ref_slope * reference_row_full + ref_intercept

    shifts_px = {}
    for plane, (slope, intercept) in lines.items():
        if plane == reference_plane:
            continue
        col_at_ref = slope * reference_row_full + intercept
        shifts_px[plane] = float(col_at_ref - ref_col)

    return ChromaticShiftResult(
        reference_plane=reference_plane,
        shifts_px=shifts_px,
        reference_row_full=reference_row_full,
    )


def print_chromatic_shifts(result: ChromaticShiftResult) -> None:
    """Affiche un résumé texte d'un `ChromaticShiftResult`."""
    print(f"Aberration chromatique latérale (référence : {result.reference_plane}, "
          f"ligne pleine résolution {result.reference_row_full:.1f}) :")
    for plane, shift in result.shifts_px.items():
        print(f"  {plane:4s} : {shift:+.4f} px")


# EOF
