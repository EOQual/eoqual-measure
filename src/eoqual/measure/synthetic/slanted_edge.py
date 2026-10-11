#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Génération d'une mire synthétique de type "bord incliné" (slanted-edge) avec
une fonction de transfert de modulation (FTM / MTF) contrôlée et connue
analytiquement.

La classe `SyntheticSlantedEdge` permet de :
    1. Générer un bord idéal (échelon) incliné d'un angle donné.
    2. Le convoluer avec une PSF (point spread function) 2D isotrope dont
       la FTM théorique est connue en forme close (gaussienne, lorentzienne,
       ou porte/pillbox).
    3. Évaluer cette FTM théorique en tout point, en particulier à fe/4 et
       fe/2 (Nyquist), où fe est la fréquence d'échantillonnage (1 cycle/pixel
       par défaut).
    4. Sauvegarder la courbe de FTM théorique dans un fichier CSV dont l'axe
       des fréquences est exprimé en fraction de fe (f/fe).
"""
from __future__ import annotations

from dataclasses import dataclass, field
import csv
from pathlib import Path
from typing import Callable, Literal

import numpy as np
from numpy.typing import NDArray
from scipy.signal import fftconvolve

# plotly (``plot``) et imageio (``save_image``) sont importés à l'appel :
# ils relèvent de l'extra ``plot`` et ne sont pas requis pour générer une
# mire en mémoire.

# CONSTANTES
PSFType = Literal["gaussian", "lorentzian", "pillbox"]
EdgeOrientation = Literal["vertical", "horizontal"]
NoiseType = Literal["none", "gaussian", "poisson_gaussian"]


@dataclass
class SyntheticSlantedEdge:
    """
    Générateur de mire synthétique "bord incliné" avec FTM théorique connue.

    La FTM théorique est celle d'une PSF 2D isotrope choisie parmi
    ``"gaussian"``, ``"lorentzian"`` ou ``"pillbox"``. Elle est évaluée le
    long d'une direction radiale (les trois PSF proposées étant à symétrie
    circulaire, la FTM 1D ne dépend pas de la direction).

    Parameters
    ----------
    nx : int, default=256
        Largeur de l'image générée, en pixels.
    ny : int, default=256
        Hauteur de l'image générée, en pixels.
    angle_deg : float, default=5.0
        Angle d'inclinaison du bord par rapport à sa direction de référence
        (voir ``orientation``), en degrés. C'est ce petit angle qui permet
        la méthode du bord incliné (suréchantillonnage de l'ESF).
    orientation : {"vertical", "horizontal"}, default="vertical"
        Direction de référence du bord avant inclinaison.

        - ``"vertical"`` : bord de référence vertical (transition selon x),
          incliné de ``angle_deg`` par rapport à la verticale.
        - ``"horizontal"`` : bord de référence horizontal (transition selon
          y), incliné de ``angle_deg`` par rapport à l'horizontale.
    psf_type : {"gaussian", "lorentzian", "pillbox"}, default="gaussian"
        Type de PSF appliquée à l'image, dont dépend la forme de la FTM
        théorique.
    sigma_psf : float, default=1.0
        Paramètre d'échelle de la PSF, en pixels.

        - ``"gaussian"`` : écart-type σ de la gaussienne.
        - ``"lorentzian"`` : paramètre d'échelle γ (largeur à mi-hauteur = 2γ).
        - ``"pillbox"`` : rayon R du disque uniforme.
    fe : float, default=1.0
        Fréquence d'échantillonnage, en cycles/pixel (avec la convention
        usuelle d'une image discrète, un pixel = un échantillon, fe = 1.0
        cycle/pixel). Ce paramètre sert uniquement à convertir en interne
        la fréquence réduite ``f/fe`` — seule échelle utilisée en entrée
        et en sortie de :meth:`mtf_theoretical`, du CSV et des graphiques
        — vers une fréquence absolue pour les formules physiques des PSF.
    noise_type : {"none", "gaussian", "poisson_gaussian"}, default="gaussian"
        Modèle de bruit appliqué à l'image finale (après convolution par
        la PSF).

        - ``"none"`` : aucun bruit, image parfaitement déterministe.
        - ``"gaussian"`` : bruit additif gaussien blanc, indépendant du
          signal (typiquement un bruit de lecture), d'écart-type
          ``noise_level``.
        - ``"poisson_gaussian"`` : modèle mixte réaliste d'un capteur —
          bruit de photon (Poisson, dépendant du signal, piloté par
          ``poisson_scale``) additionné d'un bruit de lecture gaussien
          d'écart-type ``noise_level``.
    noise_level : float, default=0.001
        Écart-type (en unités d'intensité normalisée [0, 1]) du bruit
        gaussien additif — utilisé seul (``noise_type="gaussian"``) ou
        comme bruit de lecture en complément du bruit de photon
        (``noise_type="poisson_gaussian"``). Sans effet si
        ``noise_type="none"``.
    poisson_scale : float, default=1000.0
        Nombre de photons équivalent à une intensité normalisée de 1.0,
        utilisé uniquement pour ``noise_type="poisson_gaussian"``. Plus
        cette valeur est grande, plus le bruit de photon relatif est
        faible (meilleur rapport signal/bruit).
    seed : int | None, default=None
        Graine du générateur aléatoire pour la reproductibilité du bruit.

    Attributes
    ----------
    image : NDArray[np.float32] | None
        Dernière image générée par :meth:`generate_image` (``None`` avant
        le premier appel).
    psf : NDArray[np.float64] | None
        Dernière PSF construite par :meth:`build_psf`.

    Examples
    --------
    >>> edge = SyntheticSlantedEdge(nx=128, ny=128, angle_deg=4.0,
    ...                              orientation="horizontal",
    ...                              psf_type="gaussian", sigma_psf=1.2,
    ...                              noise_type="poisson_gaussian")
    >>> image = edge.generate_image()
    >>> mtf_quarter, mtf_half = edge.mtf_at_key_frequencies()
    >>> edge.save_mtf_csv("mtf_curve.csv")
    """

    nx: int = 256
    ny: int = 256
    angle_deg: float = 5.0
    orientation: EdgeOrientation = "vertical"
    psf_type: PSFType = "gaussian"
    sigma_psf: float = 1.0
    fe: float = 1.0
    noise_type: NoiseType = "gaussian"
    noise_level: float = 0.001
    poisson_scale: float = 1000.0
    seed: int | None = None

    image: NDArray[np.float32] | None = field(default=None, init=False, repr=False)
    psf: NDArray[np.float64] | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.nx <= 0 or self.ny <= 0:
            raise ValueError("nx et ny doivent être strictement positifs.")
        if self.sigma_psf <= 0:
            raise ValueError("sigma_psf doit être strictement positif.")
        if self.fe <= 0:
            raise ValueError("fe doit être strictement positif.")
        if self.orientation not in ("vertical", "horizontal"):
            raise ValueError(f"orientation inconnue : {self.orientation!r}")
        if self.noise_type not in ("none", "gaussian", "poisson_gaussian"):
            raise ValueError(f"noise_type inconnu : {self.noise_type!r}")
        if self.noise_level < 0:
            raise ValueError("noise_level doit être positif ou nul.")
        if self.poisson_scale <= 0:
            raise ValueError("poisson_scale doit être strictement positif.")
        self._rng = np.random.default_rng(self.seed)

    # ------------------------------------------------------------------
    # Génération du bord idéal
    # ------------------------------------------------------------------
    def build_ideal_edge(self) -> NDArray[np.float32]:
        """
        Construit le bord idéal (fonction échelon) incliné, avant flou.

        Selon ``self.orientation``, le bord de référence (avant inclinaison
        de ``angle_deg``) est soit vertical (transition selon x), soit
        horizontal (transition selon y).

        Returns
        -------
        edge_ideal : NDArray[np.float32]
            Image binaire (0/1) de forme ``(ny, nx)`` représentant le bord
            idéal, non convoluée.
        """
        angle = np.deg2rad(self.angle_deg)
        y_idx, x_idx = np.meshgrid(
            np.arange(self.ny, dtype=float),
            np.arange(self.nx, dtype=float),
            indexing="ij",
        )

        # Milieu de l'image
#TODO: vérifier si le milieu est bien centré sur un pixel ou entre pixels.
# Pour une image de taille nx=ny=256, le centre exact est à (127.5, 127.5) 
# si on considère les pixels comme des intervalles [0,1), [1,2), ..., [255,256).
# Donc le centre est à (nx-1)/2 et (ny-1)/2.
#        cx = self.nx / 2.0
#        cy = self.ny / 2.0
        cx = (self.nx - 1)/2
        cy = (self.ny - 1)/2

        if self.orientation == "vertical":
            # Bord de référence vertical : transition principale selon x.
            distance = (x_idx - cx) * np.cos(angle) + (y_idx - cy) * np.sin(angle)
        else:
            # Bord de référence horizontal : transition principale selon y.
            distance = (y_idx - cy) * np.cos(angle) + (x_idx - cx) * np.sin(angle)

        return (distance > 0).astype(np.float32)

    # ------------------------------------------------------------------
    # Construction de la PSF
    # ------------------------------------------------------------------
    def build_psf(self) -> NDArray[np.float64]:
        """
        Construit la PSF 2D isotrope, normalisée en énergie (somme = 1).

        Returns
        -------
        psf : NDArray[np.float64]
            Noyau de convolution carré, de taille impaire, adapté au type
            de PSF sélectionné (``psf_type``).

        Raises
        ------
        ValueError
            Si ``psf_type`` n'est pas reconnu.
        """
        if self.psf_type == "gaussian":
            radius = int(np.ceil(4 * self.sigma_psf))  # 4σ -> ~99.99% énergie
        elif self.psf_type == "lorentzian":
            radius = int(np.ceil(10 * self.sigma_psf))  # queue plus lourde
        elif self.psf_type == "pillbox":
            radius = int(np.ceil(self.sigma_psf)) + 1
        else:
            raise ValueError(f"psf_type inconnu : {self.psf_type!r}")

        coords = np.arange(-radius, radius + 1, dtype=float)
        X, Y = np.meshgrid(coords, coords)
        R2 = X**2 + Y**2

        if self.psf_type == "gaussian":
            psf = np.exp(-R2 / (2 * self.sigma_psf**2))
        elif self.psf_type == "lorentzian":
            psf = 1.0 / (1.0 + R2 / self.sigma_psf**2)
        else:  # pillbox
            psf = (np.sqrt(R2) <= self.sigma_psf).astype(float)

        psf_sum = psf.sum()
        if psf_sum <= 0:
            raise RuntimeError("La PSF construite est nulle : vérifier sigma_psf.")
        psf /= psf_sum

        self.psf = psf
        return psf

    # ------------------------------------------------------------------
    # Génération de l'image finale (bord + flou + bruit)
    # ------------------------------------------------------------------
    def generate_image(self) -> NDArray[np.float32]:
        """
        Génère l'image de mire complète : bord idéal convolué par la PSF,
        puis bruité selon ``self.noise_type``.

        Returns
        -------
        image : NDArray[np.float32]
            Image finale de forme ``(ny, nx)``, valeurs bornées dans [0, 1].
        """
        edge_ideal = self.build_ideal_edge()
        psf = self.build_psf()

        image = fftconvolve(edge_ideal, psf, mode="same")
        image = self._apply_noise(image)

        image = np.clip(image, 0.0, 1.0).astype(np.float32)
        self.image = image
        return image

    def _apply_noise(self, image: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Applique le modèle de bruit sélectionné (``self.noise_type``) à une
        image (non encore bornée/castée).

        Parameters
        ----------
        image : NDArray[np.float64]
            Image d'entrée, typiquement dans [0, 1], avant ajout de bruit.

        Returns
        -------
        noisy_image : NDArray[np.float64]
            Image bruitée, potentiellement hors de [0, 1] (le clipping
            final est fait par l'appelant).

        Notes
        -----
        - ``"none"`` : l'image est retournée inchangée.
        - ``"gaussian"`` : bruit additif blanc indépendant du signal,
          ``I_bruitée = I + N(0, noise_level²)``.
        - ``"poisson_gaussian"`` : modèle mixte capteur,
          ``I_bruitée = Poisson(I · poisson_scale) / poisson_scale + N(0, noise_level²)``,
          combinant bruit de photon (dépendant du signal) et bruit de
          lecture (indépendant du signal).
        """
        if self.noise_type == "none":
            return image

        if self.noise_type == "gaussian":
            return image + self._rng.normal(0.0, self.noise_level, image.shape)

        # poisson_gaussian
        photon_counts = np.clip(image, 0.0, None) * self.poisson_scale
        shot_noisy = self._rng.poisson(photon_counts).astype(np.float64) / self.poisson_scale
        return shot_noisy + self._rng.normal(0.0, self.noise_level, image.shape)

    # ------------------------------------------------------------------
    # Normalisation et export image (float32, via imageio)
    # ------------------------------------------------------------------
    def normalize_image(self, image: NDArray | None = None) -> NDArray[np.float32]:
        """
        Normalise une image par étirement min-max vers l'intervalle [0, 1].

        Parameters
        ----------
        image : NDArray | None, default=None
            Image à normaliser. Si ``None``, utilise ``self.image`` (et le
            génère au préalable via :meth:`generate_image` s'il n'existe
            pas encore).

        Returns
        -------
        normalized : NDArray[np.float32]
            Image normalisée en float32, telle que
            ``normalized.min() == 0`` et ``normalized.max() == 1``
            (image constante retournée telle quelle, en évitant une
            division par zéro).
        """
        if image is None:
            image = self.image if self.image is not None else self.generate_image()

        image = np.asarray(image, dtype=np.float64)
        img_min, img_max = image.min(), image.max()
        span = img_max - img_min

        if span <= 0:
            # Image constante : rien à étirer, on la retourne telle quelle.
            normalized = image
        else:
            normalized = (image - img_min) / span

        return normalized.astype(np.float32)

    def save_image(
        self,
        path: str | Path,
        normalize: bool = True,
        image: NDArray | None = None,
    ) -> Path:
        """
        Sauvegarde l'image de la mire en float32 à l'aide d'imageio.

        Parameters
        ----------
        path : str | Path
            Chemin du fichier de sortie. Un format supportant nativement
            le float32 (typiquement ``.tiff``/``.tif``) est recommandé :
            les formats entiers (PNG, JPEG...) ne peuvent pas représenter
            du float32 sans perte de précision.
        normalize : bool, default=True
            Si ``True``, applique :meth:`normalize_image` (étirement
            min-max vers [0, 1]) avant la sauvegarde. Si ``False``,
            l'image est sauvegardée telle quelle (simplement castée en
            float32).
        image : NDArray | None, default=None
            Image à sauvegarder. Si ``None``, utilise ``self.image`` (et
            le génère au préalable via :meth:`generate_image` s'il
            n'existe pas encore).

        Returns
        -------
        path : Path
            Chemin du fichier effectivement écrit.
        """
        if image is None:
            image = self.image if self.image is not None else self.generate_image()

        if normalize:
            image_to_save = self.normalize_image(image)
        else:
            image_to_save = np.asarray(image, dtype=np.float32)

        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if out_path.suffix.lower() not in (".tif", ".tiff"):
            print(
                f"Attention : le format '{out_path.suffix}' ne supporte "
                "généralement pas le float32 nativement ; préférez "
                "'.tiff' pour préserver la précision."
            )

        import imageio.v3 as iio

        iio.imwrite(out_path, image_to_save)
        return out_path

    # ------------------------------------------------------------------
    # FTM théorique
    # ------------------------------------------------------------------
    def mtf_theoretical(self, f_over_fe: NDArray[np.float64] | float) -> NDArray[np.float64]:
        """
        Évalue la FTM théorique 1D (le long d'une direction radiale) pour
        la PSF isotrope sélectionnée.

        Parameters
        ----------
        f_over_fe : NDArray[np.float64] | float
            Fréquence(s) spatiale(s) réduite(s), exprimée(s) en fraction de
            la fréquence d'échantillonnage fe (sans unité). La fréquence de
            Nyquist correspond à ``f_over_fe = 0.5``.

        Returns
        -------
        mtf : NDArray[np.float64]
            Valeur(s) de la FTM théorique, sans unité, dans [0, 1].

        Notes
        -----
        Les formules ci-dessous sont exprimées à partir de la fréquence
        absolue ``f = f_over_fe * fe`` (en cycles/pixel), puis ramenées à
        la fréquence réduite ``f_over_fe`` (PSF normalisée à énergie unité,
        isotrope 2D) :

        - Gaussienne : ``MTF(f_over_fe) = exp(-2 π² σ² (f_over_fe · fe)²)``
        - Lorentzienne (PSF ∝ 1 / (1 + r²/γ²)) :
          ``MTF(f_over_fe) = exp(-2π γ (f_over_fe · fe))``
        - Porte circulaire (pillbox) de rayon R :
          ``MTF(f_over_fe) = 2 J1(2π R (f_over_fe · fe)) / (2π R (f_over_fe · fe))``
          (fonction de Bessel J1), avec la convention ``MTF(0) = 1``.
        """
        f_reduced = np.asarray(f_over_fe, dtype=float)
        f_absolute = f_reduced * self.fe  # conversion en cycles/pixel pour les formules

        if self.psf_type == "gaussian":
            mtf = np.exp(-2 * (np.pi**2) * (self.sigma_psf**2) * (f_absolute**2))
        elif self.psf_type == "lorentzian":
            mtf = np.exp(-2 * np.pi * self.sigma_psf * np.abs(f_absolute))
        else:  # pillbox
            from scipy.special import j1

            arg = 2 * np.pi * self.sigma_psf * f_absolute
            with np.errstate(divide="ignore", invalid="ignore"):
                mtf = np.where(arg == 0, 1.0, 2 * j1(arg) / arg)

        return mtf

    def mtf_at_key_frequencies(self) -> tuple[float, float]:
        """
        Calcule la FTM théorique aux fréquences clés réduites f/fe = 1/4 et
        f/fe = 1/2 (Nyquist).

        Returns
        -------
        mtf_quarter : float
            Valeur de la FTM à f/fe = 0.25.
        mtf_half : float
            Valeur de la FTM à f/fe = 0.5 (fréquence de Nyquist).
        """
        mtf_quarter = float(self.mtf_theoretical(np.array([0.25]))[0])
        mtf_half = float(self.mtf_theoretical(np.array([0.5]))[0])
        return mtf_quarter, mtf_half

    # ------------------------------------------------------------------
    # Export CSV
    # ------------------------------------------------------------------
    def save_mtf_csv(
        self,
        path: str | Path,
        n_points: int = 501,
        f_over_fe_max: float = 0.5,
    ) -> Path:
        """
        Sauvegarde la courbe de FTM théorique dans un fichier CSV.

        L'axe des abscisses est exprimé en fréquence réduite ``f / fe``
        (sans unité), de 0 à ``f_over_fe_max`` (0.5 = fréquence de Nyquist
        par défaut).

        Parameters
        ----------
        path : str | Path
            Chemin du fichier CSV de sortie.
        n_points : int, default=501
            Nombre de points échantillonnant la courbe.
        f_over_fe_max : float, default=0.5
            Valeur maximale de f/fe couverte par la courbe (0.5 correspond
            à la fréquence de Nyquist).

        Returns
        -------
        path : Path
            Chemin du fichier effectivement écrit.
        """
        f_over_fe = np.linspace(0.0, f_over_fe_max, n_points)
        mtf = self.mtf_theoretical(f_over_fe)

        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with out_path.open("w", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["f_over_fe", "mtf"])
            for row_f_over_fe, row_mtf in zip(f_over_fe, mtf):
                writer.writerow([f"{row_f_over_fe:.6f}", f"{row_mtf:.6f}"])

        return out_path

    # ------------------------------------------------------------------
    # Visualisation Plotly
    # ------------------------------------------------------------------
    def plot(
        self,
        n_points: int = 500,
        html_path: str | Path | None = None,
        show: bool = False,
    ) -> go.Figure:
        """
        Construit une figure Plotly à deux panneaux : l'image de la mire
        générée et la courbe de FTM théorique (avec les points fe/4 et
        fe/2 mis en évidence).

        Parameters
        ----------
        n_points : int, default=500
            Nombre de points utilisés pour tracer la courbe de FTM, entre
            0 et fe/2.
        html_path : str | Path | None, default=None
            Si fourni, chemin du fichier HTML autonome où exporter la
            figure interactive.
        show : bool, default=False
            Si ``True``, ouvre la figure (``fig.show()``) en plus de
            l'éventuelle sauvegarde.

        Returns
        -------
        fig : plotly.graph_objects.Figure
            Figure Plotly composée de deux sous-graphiques : l'image
            (heatmap en niveaux de gris) et la courbe FTM(f/fe).
        """
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        if self.image is None:
            self.generate_image()

        mtf_quarter, mtf_half = self.mtf_at_key_frequencies()

        f_over_fe = np.linspace(0.0, 0.5, n_points)
        mtf_curve = self.mtf_theoretical(f_over_fe)

        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=(
                f"Mire synthétique (PSF={self.psf_type}, échelle={self.sigma_psf} px)",
                "FTM théorique",
            ),
        )

        # Panneau 1 : image de la mire
        fig.add_trace(
            go.Heatmap(
                z=self.image,
                colorscale="gray",
                showscale=False,
                hovertemplate="x=%{x}<br>y=%{y}<br>valeur=%{z:.3f}<extra></extra>",
            ),
            row=1,
            col=1,
        )

        # Panneau 2 : courbe FTM
        fig.add_trace(
            go.Scatter(
                x=f_over_fe,
                y=mtf_curve,
                mode="lines",
                name="FTM théorique",
                line=dict(color="royalblue", width=2),
                hovertemplate="f/fe=%{x:.3f}<br>FTM=%{y:.4f}<extra></extra>",
            ),
            row=1,
            col=2,
        )
        fig.add_trace(
            go.Scatter(
                x=[0.25, 0.5],
                y=[mtf_quarter, mtf_half],
                mode="markers+text",
                name="Points clés",
                marker=dict(color="crimson", size=10, symbol=["circle", "square"]),
                text=[f"fe/4 = {mtf_quarter:.3f}", f"fe/2 = {mtf_half:.3f}"],
                textposition="top center",
                hovertemplate="f/fe=%{x:.3f}<br>FTM=%{y:.4f}<extra></extra>",
            ),
            row=1,
            col=2,
        )

        fig.update_yaxes(scaleanchor="x", scaleratio=1, autorange="reversed", row=1, col=1)
        fig.update_xaxes(title_text="x (pixels)", row=1, col=1)
        fig.update_yaxes(title_text="y (pixels)", row=1, col=1)
        fig.update_xaxes(title_text="Fréquence réduite f / fe", range=[0, 0.5], row=1, col=2)
        fig.update_yaxes(title_text="FTM", range=[0, 1.05], row=1, col=2)

        fig.update_layout(
            title=f"Mire bord incliné — orientation={self.orientation}, "
                  f"angle={self.angle_deg}°, PSF={self.psf_type}",
            width=1000,
            height=480,
            template="plotly_white",
        )

        if html_path is not None:
            out_path = Path(html_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            fig.write_html(str(out_path))

        if show:
            fig.show()

        return fig

    # ------------------------------------------------------------------
    # Pipeline complet
    # ------------------------------------------------------------------
    def generate(
        self,
        csv_path: str | Path | None = None,
        html_path: str | Path | None = None,
        image_path: str | Path | None = None,
        normalize_image: bool = True,
    ) -> tuple[NDArray[np.float32], Callable[[NDArray[np.float64] | float], NDArray[np.float64]]]:
        """
        Pipeline complet : génère l'image, affiche un résumé, exporte
        éventuellement la courbe de FTM en CSV, la visualisation Plotly en
        HTML et l'image en float32 (via imageio).

        Parameters
        ----------
        csv_path : str | Path | None, default=None
            Si fourni, chemin du fichier CSV où sauvegarder la courbe de
            FTM théorique (voir :meth:`save_mtf_csv`).
        html_path : str | Path | None, default=None
            Si fourni, chemin du fichier HTML où sauvegarder la figure
            Plotly interactive (voir :meth:`plot`).
        image_path : str | Path | None, default=None
            Si fourni, chemin du fichier image où sauvegarder la mire en
            float32 (voir :meth:`save_image`). Privilégier une extension
            ``.tiff``.
        normalize_image : bool, default=True
            Transmis à :meth:`save_image` : normalise l'image (min-max
            vers [0, 1]) avant sauvegarde si ``image_path`` est fourni.

        Returns
        -------
        image : NDArray[np.float32]
            Image de mire générée.
        mtf_theoretical : Callable
            Fonction ``f -> MTF(f)`` (voir :meth:`mtf_theoretical`).
        """
        image = self.generate_image()
        mtf_quarter, mtf_half = self.mtf_at_key_frequencies()

        print(f"Mire générée : {self.nx}x{self.ny} px, orientation={self.orientation}, "
              f"angle={self.angle_deg}°, PSF={self.psf_type} (échelle={self.sigma_psf} px)")
        print(f"Bruit : {self.noise_type}"
              + (f" (noise_level={self.noise_level})" if self.noise_type != "none" else "")
              + (f", poisson_scale={self.poisson_scale}" if self.noise_type == "poisson_gaussian" else ""))
        print(f"FTM théorique à f/fe = 0.25 : {mtf_quarter:.4f}")
        print(f"FTM théorique à f/fe = 0.50 (Nyquist) : {mtf_half:.4f}")
        print(f"Plage image : [{image.min():.3f}, {image.max():.3f}]")

        if csv_path is not None:
            saved_path = self.save_mtf_csv(csv_path)
            print(f"Courbe FTM sauvegardée dans : {saved_path}")

        if html_path is not None:
            self.plot(html_path=html_path)
            print(f"Visualisation Plotly sauvegardée dans : {html_path}")

        if image_path is not None:
            saved_image_path = self.save_image(image_path, normalize=normalize_image)
            print(f"Image (float32) sauvegardée dans : {saved_image_path}")

        return image, self.mtf_theoretical


# EOF
