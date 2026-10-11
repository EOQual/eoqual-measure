#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualisation (tableaux texte + figures Plotly) du pipeline bord incliné :
comparaison des algorithmes d'orientation, et comparaison des courbes de
MTF.

Toute la présentation est regroupée ici dans `SlantedEdgeVisualizer` :
`cli.py` ne fait qu'orchestrer : charger l'image, lancer les
estimateurs, appeler ce module pour afficher les résultats.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import plotly.graph_objects as go

from eoqual.measure.mtf.slanted_edge.orientation.base import OrientationResult
from eoqual.measure.mtf.slanted_edge.estimators.base import MTFResult


class SlantedEdgeVisualizer:
    """
    Regroupe toute la visualisation du pipeline bord incliné.

    Toutes les méthodes sont statiques : cette classe ne porte pas
    d'état, elle sert seulement à organiser la visualisation en un seul
    endroit.
    """

    #: Couleur attribuée à chaque algorithme d'orientation, pour rester
    #: cohérent entre les figures.
    ORIENTATION_COLORS = {"gradient": "red", "pca": "royalblue", "fourier": "green"}

    #: Couleur attribuée à chaque méthode de MTF.
    MTF_COLORS = {
        "spatial": "darkorange",
        "spectral": "seagreen",
        "esf_lsf_numeric": "mediumpurple",
        "esf_lsf_sigmoid": "crimson",
        "esf_lsf_pchip": "goldenrod",
        "spectral_correlation": "teal",
    }

    # -- Image ----------------------------------------------------------

    @staticmethod
    def plot_image(
        image: np.ndarray,
        output_path: Path,
        title: str = "Mire",
    ) -> go.Figure:
        """
        Affiche une image 2D (heatmap en niveaux de gris), typiquement
        la mire complète avant extraction de zone, écrite en HTML
        plutôt qu'affichée interactivement.

        Parameters
        ----------
        image :
            Matrice d'intensité à afficher.
        output_path :
            Chemin du fichier HTML à écrire.
        title :
            Titre de la figure.

        Returns
        -------
        plotly.graph_objects.Figure
            La figure générée (déjà écrite sur disque).
        """
        fig = go.Figure(
            go.Heatmap(z=image[::-1, :], colorscale="gray", showscale=False)
        )
        fig.update_layout(
            title=title,
            xaxis_showgrid=False,
            yaxis_showgrid=False,
            xaxis_visible=False,
            yaxis_visible=False,
        )
        fig.update_yaxes(scaleanchor="x", scaleratio=1)

        fig.write_html(output_path)
        print(f"Image plot saved to: {output_path}")
        return fig

    # -- Orientation ------------------------------------------------------

    @staticmethod
    def print_orientation_summary(results: dict[str, OrientationResult]) -> None:
        """Affiche un tableau de comparaison : angle, orientation, métrique de qualité."""
        header = (
            f"{'algorithm':10s} {'angle_deg':>10s} {'orientation':>12s} "
            f"{'transp.':>8s} {'flipped':>8s}   quality metric"
        )
        print(header)
        print("-" * len(header))
        for name, result in results.items():
            if "error" in result.details:
                print(f"{name:10s}  ERROR: {result.details['error']}")
                continue
            orientation = "horizontal" if result.is_horizontal else "vertical"
            print(
                f"{name:10s} {result.angle_deg:10.3f} {orientation:>12s} "
                f"{str(result.was_transposed):>8s} {str(result.was_flipped):>8s}   "
                f"{result.quality_metric_name}={result.quality_metric_value:.4g}"
            )

    @staticmethod
    def plot_orientation_comparison(
        image: np.ndarray,
        results: dict[str, OrientationResult],
        output_path: Path,
    ) -> go.Figure:
        """
        Superpose les positions sous-pixel du bord (par ligne/colonne) de
        chaque algorithme sur l'image d'origine (figure Plotly
        interactive), afin que les trois méthodes puissent être
        comparées directement sur les mêmes données.

        Parameters
        ----------
        image :
            Image d'origine (niveaux de gris).
        results :
            Résultats d'orientation, par nom d'algorithme.
        output_path :
            Chemin du fichier HTML à écrire.

        Returns
        -------
        plotly.graph_objects.Figure
            La figure générée (déjà écrite sur disque).
        """
        n_rows, n_cols = image.shape

        fig = go.Figure()

        # Image d'origine sous forme de heatmap en niveaux de gris,
        # ligne 0 en haut comme avec imshow.
        fig.add_trace(
            go.Heatmap(
                z=image[::-1, :],
                colorscale="gray",
                showscale=False,
                hoverinfo="skip",
            )
        )

        for name, result in results.items():
            if "error" in result.details:
                continue
            color = SlantedEdgeVisualizer.ORIENTATION_COLORS.get(name)
            fig.add_trace(
                go.Scatter(
                    x=result.edge_columns,
                    # Inverse les lignes pour correspondre à l'axe y
                    # inversé de la heatmap.
                    y=(n_rows - 1) - result.edge_rows,
                    mode="markers",
                    marker=dict(size=3, color=color),
                    name=f"{name} ({result.angle_deg:.2f} deg)",
                )
            )

        fig.update_layout(
            title="Sub-pixel edge position per algorithm",
            xaxis=dict(range=[0, n_cols], constrain="domain", title="column"),
            yaxis=dict(
                range=[0, n_rows], scaleanchor="x", scaleratio=1, title="row"
            ),
            legend=dict(x=1.02, y=1.0),
            width=800,
            height=800,
        )

        fig.write_html(output_path)
        print(f"Orientation comparison plot saved to: {output_path}")
        return fig

    # -- MTF ----------------------------------------------------------------

    @staticmethod
    def print_mtf_summary(
        results: dict[str, MTFResult],
        sample_frequencies: tuple[float, ...] = (0.1, 0.25, 0.5),
        angle_correction: bool = True,
    ) -> None:
        """
        Affiche un tableau de comparaison : angle, dx, et MTF interpolée
        à quelques fréquences normalisées repères.

        Parameters
        ----------
        results :
            Résultats de MTF, par nom de méthode.
        sample_frequencies :
            Fréquences normalisées (f / f_e) auxquelles échantillonner
            la MTF pour le tableau.
        angle_correction :
            Si `True`, interpole sur `frequency_corrected` (par
            `cos(theta)`) plutôt que sur `frequency` brute.
        """
        freq_headers = "".join(f"MTF@{f:<5.2f}  " for f in sample_frequencies)
        header = f"{'method':16s} {'angle_deg':>10s} {'dx':>8s}   {freq_headers}"
        print(header)
        print("-" * len(header))
        for name, result in results.items():
            if "error" in result.details:
                print(f"{name:16s}  ERROR: {result.details['error']}")
                continue
            angle_deg = np.degrees(result.theta_rad)
            values = "".join(
                f"{result.value_at(f, angle_correction):<9.4f}  "
                for f in sample_frequencies
            )
            print(f"{name:16s} {angle_deg:10.3f} {result.dx:8.4f}   {values}")

    @staticmethod
    def plot_mtf_comparison(
        results: dict[str, MTFResult],
        output_path: Path,
        angle_correction: bool = True,
        title: str = "MTF -- bord incliné",
    ) -> go.Figure:
        """
        Trace les courbes de MTF (module) de chaque méthode, superposées
        (figure Plotly interactive).

        Parameters
        ----------
        results :
            Résultats de MTF, par nom de méthode.
        output_path :
            Chemin du fichier HTML à écrire.
        angle_correction :
            Si `True`, trace en fonction de `frequency_corrected` (par
            `cos(theta)`) plutôt que de `frequency` brute.
        title :
            Titre de la figure.

        Returns
        -------
        plotly.graph_objects.Figure
            La figure générée (déjà écrite sur disque).
        """
        fig = go.Figure()

        for name, result in results.items():
            if "error" in result.details:
                continue
            color = SlantedEdgeVisualizer.MTF_COLORS.get(name)
            frequency = (
                result.frequency_corrected if angle_correction else result.frequency
            )
            fig.add_trace(
                go.Scatter(
                    x=frequency,
                    y=np.abs(result.mtf),
                    mode="lines",
                    name=name,
                    line=dict(color=color),
                )
            )

        fig.update_layout(
            title=title,
            xaxis=dict(title="Fréquence normalisée (f / fₑ)", range=[0, 0.5]),
            yaxis=dict(title="MTF (module)", range=[0, 1]),
            legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
            template="simple_white",
            width=800,
            height=550,
        )

        # Repères verticaux fₑ/4 et fₑ/2 (Nyquist), communs à toutes les courbes.
        for fraction, label in [(0.25, "fₑ/4"), (0.5, "fₑ/2")]:
            fig.add_vline(
                x=fraction,
                line_dash="dot",
                line_color="gray",
                line_width=1,
                annotation_text=label,
                annotation_position="top",
                annotation_font=dict(color="gray", size=11),
            )

        fig.write_html(output_path)
        print(f"MTF comparison plot saved to: {output_path}")
        return fig

    # -- ESF / LSF ------------------------------------------------------

    @staticmethod
    def plot_esf_lsf(
        result: MTFResult,
        output_path: Path,
        title: str | None = None,
    ) -> go.Figure:
        """
        Trace l'ESF et la LSF d'un résultat de MTF, côte à côte.

        Réservé aux méthodes qui exposent `x_esf`/`esf`/`x_lsf`/`lsf`
        dans `result.details` (actuellement `esf_lsf_*` uniquement --
        `spatial`/`spectral`/`spectral_correlation` ne construisent pas
        d'ESF explicite, voir docstring de `estimators/base.py`).

        Parameters
        ----------
        result :
            Résultat produit par un `EsfLsfMTFEstimator` (ou tout
            estimateur exposant les mêmes clés dans `details`).
        output_path :
            Chemin du fichier HTML à écrire.
        title :
            Titre de la figure.

        Raises
        ------
        ValueError
            Si `result.details` ne contient pas d'ESF/LSF (méthode
            incompatible).
        """
        required = {"x_esf", "esf", "x_lsf", "lsf"}
        missing = required - result.details.keys()
        if missing:
            raise ValueError(
                f"'{result.method}' ne fournit pas {sorted(missing)} dans "
                "details -- seuls les estimateurs esf_lsf_* exposent l'ESF/LSF."
            )

        from plotly.subplots import make_subplots

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("ESF (normalisée)", "LSF"),
        )

        fig.add_trace(
            go.Scatter(
                x=result.details["x_esf"], y=result.details["esf"],
                mode="markers", marker=dict(size=3, color="royalblue"),
                name="ESF",
            ),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=result.details["x_lsf"], y=result.details["lsf"],
                mode="lines", line=dict(color="crimson"),
                name="LSF",
            ),
            row=1, col=2,
        )

        fwhm = result.details.get("fwhm")
        if fwhm:
            fig.add_annotation(
                xref="x2 domain", yref="y2 domain", x=0.98, y=0.95,
                text=f"FWHM = {fwhm:.3f} px", showarrow=False,
                font=dict(size=11), align="right",
            )

        fig.update_xaxes(title_text="Distance au bord (pixels)", row=1, col=1)
        fig.update_xaxes(title_text="Distance au bord (pixels)", row=1, col=2)
        fig.update_yaxes(title_text="Intensité normalisée", row=1, col=1)
        fig.update_yaxes(title_text="LSF", row=1, col=2)

        fig.update_layout(
            title=title or f"ESF / LSF -- {result.method}",
            template="simple_white",
            showlegend=False,
            width=950,
            height=450,
        )

        fig.write_html(output_path)
        print(f"ESF/LSF plot saved to: {output_path}")
        return fig


# EOF
