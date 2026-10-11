#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Harnais d'étude de sensibilité : exécute tous les estimateurs
d'orientation et de MTF sur une grille de paramètres (angle, bruit, ...)
générée par une mire synthétique fournie par l'appelant, avec plusieurs
répétitions Monte-Carlo par point de grille, et agrège les résultats.

Couvre trois études (voir §12.1 de la partie I de
`REFERENCE_TECHNIQUE.md`) : effet de l'angle,
effet du bruit, comparaison des algorithmes (temps de calcul, erreur
systématique).

Interface attendue pour le générateur de mire
-----------------------------------------------
Toute fonction respectant ce contrat convient :

```python
def mire_generator(
    angle_deg: float,
    noise_param: float,
    seed: int,
) -> tuple[NDArray[np.float64], Callable[[NDArray[np.float64]], NDArray[np.float64]]]:
    ...
    return image, mtf_theoretical  # mtf_theoretical : f_over_fe -> MTF
```

Le générateur fournit lui-même sa fonction de MTF théorique (`f_over_fe
-> MTF`), plutôt que ce module ne suppose une PSF gaussienne fixe --
elle peut donc être gaussienne, lorentzienne, pillbox, ou toute autre
forme, du moment que le générateur sait l'évaluer analytiquement.
`angle_deg` et `noise_param` sont les deux facteurs balayés par
`SensitivityGrid` ; tout autre paramètre du générateur (taille d'image,
type de PSF, largeur de PSF, type de bruit, ...) reste fixe pour toute
l'étude et se règle par fermeture (`functools.partial` ou lambda) --
voir `examples/mtf_sensitivity_study.py` pour un exemple complet avec
`SyntheticSlantedEdge`.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from eoqual.measure.mtf.slanted_edge.orientation.base import EdgeOrientationComparator
from eoqual.measure.mtf.slanted_edge.estimators.base import MTFComparator
from eoqual.measure.mtf.slanted_edge.preprocess import SlantedEdgePreprocessor

TheoreticalMTF = Callable[[NDArray[np.float64]], NDArray[np.float64]]
MireGenerator = Callable[[float, float, int], tuple[NDArray[np.float64], TheoreticalMTF]]


@dataclass
class SensitivityGrid:
    """
    Grille de paramètres à balayer.

    Attributes
    ----------
    angles_deg :
        Angles de bord à tester (degrés, dans la convention du
        générateur -- voir `true_angle_to_convention`).
    noise_levels :
        Valeurs du paramètre de bruit à tester (unité propre au
        générateur : écart-type, niveau de Poisson, ...).
    n_repetitions :
        Nombre de tirages (graines différentes) par point de grille
        (angle, bruit) -- pour séparer biais systématique et
        sensibilité au bruit.
    mtf_frequency_band :
        Bande de fréquence `(f_min, f_max)`, en `f/fe`, sur laquelle
        calculer le RMSE de la MTF -- exclure les hautes fréquences où
        le signal théorique est proche de 0 et où seul le bruit de
        mesure domine (voir §9.3 de la partie I de REFERENCE_TECHNIQUE.md).
    seed_offset :
        Décalage appliqué aux graines (pour rendre plusieurs études
        reproductibles indépendamment).
    true_angle_to_convention :
        Convertit l'angle tel que paramétré par `mire_generator` vers la
        convention des estimateurs (`atan2(delta_ligne, delta_colonne)`,
        repliée dans [-90, 90] -- voir §3.2 de la documentation
        technique). Par défaut, identité. Pour `SyntheticSlantedEdge`
        : `orientation="vertical"` ->
        `lambda a: fold(a - 90)` ; `orientation="horizontal"` ->
        `lambda a: fold(-a)` (`fold` replie dans [-90, 90] -- voir
        `fold_angle_deg` dans ce module). Ces deux formules sont
        vérifiées empiriquement dans `examples/mtf_sensitivity_study.py`.
    """

    angles_deg: list[float]
    noise_levels: list[float]
    n_repetitions: int = 10
    mtf_frequency_band: tuple[float, float] = (0.02, 0.3)
    seed_offset: int = 0
    true_angle_to_convention: Callable[[float], float] = lambda a: a


def fold_angle_deg(angle_deg: float) -> float:
    """Replie un angle dans [-90, 90] -- même convention que le pipeline."""
    while angle_deg > 90.0:
        angle_deg -= 180.0
    while angle_deg < -90.0:
        angle_deg += 180.0
    return angle_deg


@dataclass
class TrialRecord:
    """Une ligne de résultat : un essai (angle, bruit, répétition), une méthode."""

    family: str  # "orientation" ou "mtf"
    method: str
    angle_deg_true: float
    noise_level: float
    repetition: int
    seed: int
    value_error: float  # erreur d'angle (deg) ou RMSE de MTF, selon `family`
    compute_time_s: float
    failed: bool
    error_message: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


class SensitivityStudy:
    """
    Exécute tous les estimateurs d'orientation et de MTF sur une grille
    de mires synthétiques générées à la volée, et collecte les erreurs
    (vs vérité terrain), temps de calcul, et échecs de chaque méthode.

    Parameters
    ----------
    mire_generator :
        Fonction respectant le contrat documenté en tête de module.
    orientation_comparator, mtf_comparator :
        Comparateurs à utiliser (par défaut : jeu complet de chaque
        module). Passer un sous-ensemble réduit les temps de calcul --
        utile pour un premier balayage grossier, en particulier avec
        `esf_model="tikhonov"` (coûteux : plusieurs résolutions de
        système linéaire par image, voir sa docstring dans
        `estimators/esf_lsf.py`).
    """

    def __init__(
        self,
        mire_generator: MireGenerator,
        orientation_comparator: EdgeOrientationComparator | None = None,
        mtf_comparator: MTFComparator | None = None,
    ) -> None:
        self.mire_generator = mire_generator
        self.orientation_comparator = orientation_comparator or EdgeOrientationComparator()
        self.mtf_comparator = mtf_comparator or MTFComparator()

    def run(self, grid: SensitivityGrid) -> pd.DataFrame:
        """
        Exécute la grille complète (angle x bruit x répétitions) et
        renvoie un DataFrame long format (une ligne par essai/méthode).
        """
        records: list[TrialRecord] = []
        seed = grid.seed_offset

        total = len(grid.angles_deg) * len(grid.noise_levels) * grid.n_repetitions
        done = 0

        for angle_deg in grid.angles_deg:
            for noise_level in grid.noise_levels:
                for repetition in range(grid.n_repetitions):
                    image, theoretical_mtf = self.mire_generator(angle_deg, noise_level, seed)
                    records.extend(
                        self._run_single_trial(
                            image, theoretical_mtf, angle_deg, noise_level,
                            repetition, seed, grid,
                        )
                    )
                    seed += 1
                    done += 1
                    if done % max(1, total // 20) == 0:
                        print(f"  [{done}/{total}] essais terminés")

        return pd.DataFrame([r.__dict__ for r in records])

    def _run_single_trial(
        self,
        image: NDArray[np.float64],
        theoretical_mtf: TheoreticalMTF,
        angle_deg_true: float,
        noise_level: float,
        repetition: int,
        seed: int,
        grid: SensitivityGrid,
    ) -> list[TrialRecord]:
        records: list[TrialRecord] = []

        # -- Orientation ---------------------------------------------------
        angle_deg_true_convention = grid.true_angle_to_convention(angle_deg_true)
        for name, detector in self.orientation_comparator.detectors.items():
            t0 = time.perf_counter()
            try:
                result = detector.analyse(image)
                elapsed = time.perf_counter() - t0
                error = float(result.angle_deg - angle_deg_true_convention)
                records.append(TrialRecord(
                    family="orientation", method=name,
                    angle_deg_true=angle_deg_true, noise_level=noise_level,
                    repetition=repetition, seed=seed,
                    value_error=error, compute_time_s=elapsed, failed=False,
                    extra={"quality_metric": result.quality_metric_value},
                ))
            except Exception as exc:  # noqa: BLE001
                elapsed = time.perf_counter() - t0
                records.append(TrialRecord(
                    family="orientation", method=name,
                    angle_deg_true=angle_deg_true, noise_level=noise_level,
                    repetition=repetition, seed=seed,
                    value_error=float("nan"), compute_time_s=elapsed, failed=True,
                    error_message=str(exc),
                ))

        # -- MTF -------------------------------------------------------
        frame = SlantedEdgePreprocessor().preprocess(image)
        for name, estimator in self.mtf_comparator.estimators.items():
            t0 = time.perf_counter()
            try:
                result = estimator.estimate(frame)
                elapsed = time.perf_counter() - t0
                f_min, f_max = grid.mtf_frequency_band
                mask = (result.frequency_corrected >= f_min) & (result.frequency_corrected <= f_max)
                freq_band = result.frequency_corrected[mask]
                mtf_band = np.abs(result.mtf)[mask]
                theoretical = theoretical_mtf(freq_band)
                rmse = float(np.sqrt(np.mean((mtf_band - theoretical) ** 2))) if freq_band.size else float("nan")
                records.append(TrialRecord(
                    family="mtf", method=name,
                    angle_deg_true=angle_deg_true, noise_level=noise_level,
                    repetition=repetition, seed=seed,
                    value_error=rmse, compute_time_s=elapsed, failed=False,
                    extra={"n_freq_points": int(freq_band.size)},
                ))
            except Exception as exc:  # noqa: BLE001
                elapsed = time.perf_counter() - t0
                records.append(TrialRecord(
                    family="mtf", method=name,
                    angle_deg_true=angle_deg_true, noise_level=noise_level,
                    repetition=repetition, seed=seed,
                    value_error=float("nan"), compute_time_s=elapsed, failed=True,
                    error_message=str(exc),
                ))

        return records

    @staticmethod
    def summarize(df: pd.DataFrame) -> pd.DataFrame:
        """
        Agrège les essais par (family, method, angle, bruit) : biais
        systématique (moyenne de l'erreur), sensibilité (écart-type),
        temps moyen, taux d'échec.
        """
        grouped = df.groupby(["family", "method", "angle_deg_true", "noise_level"])
        summary = grouped.agg(
            bias=("value_error", lambda s: np.nanmean(s)),
            std=("value_error", lambda s: np.nanstd(s)),
            mean_time_s=("compute_time_s", "mean"),
            failure_rate=("failed", "mean"),
            n=("value_error", "size"),
        ).reset_index()
        return summary


def plot_sensitivity(
    summary: pd.DataFrame,
    x: str,
    family: str,
    fixed: dict[str, Any],
    output_path: str,
    title: str | None = None,
) -> Any:
    """
    Trace le biais (± écart-type en enveloppe) en fonction de `x`
    (`"angle_deg_true"` ou `"noise_level"`), une courbe par méthode, les
    autres facteurs de la grille fixés (`fixed`).

    Parameters
    ----------
    summary :
        Sortie de `SensitivityStudy.summarize`.
    x :
        `"angle_deg_true"` (sensibilité à l'angle) ou `"noise_level"`
        (sensibilité au bruit).
    family :
        `"orientation"` ou `"mtf"`.
    fixed :
        Valeurs des autres colonnes de la grille à fixer pour ce tracé,
        ex. `{"noise_level": 0.005}` si `x="angle_deg_true"`.
    output_path :
        Chemin du fichier HTML Plotly à écrire.
    """
    import plotly.graph_objects as go

    subset = summary[summary["family"] == family]
    for col, val in fixed.items():
        subset = subset[np.isclose(subset[col], val)]
    subset = subset.sort_values(x)

    fig = go.Figure()
    for method, group in subset.groupby("method"):
        fig.add_trace(go.Scatter(
            x=group[x], y=group["bias"], mode="lines+markers", name=method,
            error_y=dict(type="data", array=group["std"], visible=True),
        ))

    fig.update_layout(
        title=title or f"Sensibilité ({family}) -- {x}, {fixed}",
        xaxis_title=x,
        yaxis_title="Biais (± écart-type)",
        template="simple_white",
        width=800,
        height=500,
    )
    fig.write_html(output_path)
    print(f"Sensitivity plot saved to: {output_path}")
    return fig


def plot_heatmap(
    summary: pd.DataFrame,
    family: str,
    method: str,
    output_path: str,
    value: str = "bias",
    title: str | None = None,
) -> Any:
    """
    Carte de chaleur (angle x bruit) d'une méthode donnée, pour repérer
    d'un coup d'oeil les régimes où elle décroche.

    Parameters
    ----------
    value :
        `"bias"`, `"std"`, ou `"mean_time_s"`.
    """
    import plotly.graph_objects as go

    subset = summary[(summary["family"] == family) & (summary["method"] == method)]
    pivot = subset.pivot(index="noise_level", columns="angle_deg_true", values=value)

    fig = go.Figure(go.Heatmap(
        x=pivot.columns, y=pivot.index, z=pivot.values,
        colorscale="RdBu_r" if value == "bias" else "Viridis",
        colorbar=dict(title=value),
    ))
    fig.update_layout(
        title=title or f"{method} ({family}) -- {value}",
        xaxis_title="angle_deg_true",
        yaxis_title="noise_level",
        template="simple_white",
        width=700,
        height=500,
    )
    fig.write_html(output_path)
    print(f"Heatmap saved to: {output_path}")
    return fig


# EOF
