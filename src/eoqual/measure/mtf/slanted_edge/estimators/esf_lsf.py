#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Estimation de MTF par construction ESF/LSF -- détection de bord
paramétrique ligne par ligne, construction d'une ESF sur-échantillonnée
par projection géométrique sur la normale au bord, puis LSF/MTF selon un
modèle de fit choisi indépendamment.

Voir `GeometricESFBuilder` et `EsfLsfMTFEstimator` pour le détail. Les
fonctions modèles pures (`sigmoid`, `gaussian`, ...) vivent dans
`estimators/esf_models.py`, sur le même principe que `fourier_utils.py`.

Architecture (mise à jour)
---------------------------
La construction de l'ESF (`GeometricESFBuilder`) est maintenant
découplée du modèle utilisé pour en tirer la LSF : elle est calculée
**une seule fois**, puis n'importe lequel des modèles de fit
(`fit_esf_numeric`, `fit_esf_sigmoid`, `fit_esf_pchip`) peut y être
appliqué sans refaire la détection de bord ni la projection. Avant cette
mise à jour, `EsfLsfMTFEstimator(mode="sigmoid")` et
`EsfLsfMTFEstimator(mode="gaussian")` relançaient chacune tout le
pipeline depuis la détection de bord -- ce n'est plus le cas : seul
`esf_model` change désormais le calcul de la LSF ; `mode` ne concerne
plus que la détection de bord elle-même (localisation, pas modélisation
de la forme de l'ESF).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from eoqual.core import edge as core_edge

from eoqual.measure.mtf.slanted_edge.estimators.esf_models import gaussian, sigmoid, sigmoid_derivative
from eoqual.measure.mtf.slanted_edge.estimators.base import CanonicalFrame, MTFResult, SlantedEdgeMTFEstimator


# --------------------------------------------------------------------------
# Détecteur de bord sous-pixel, ligne par ligne
# --------------------------------------------------------------------------


class ParametricEdgeDetector:
    """
    Détecteur de bord sous-pixel par ajustement paramétrique local.

    Localise grossièrement le bord au maximum du gradient, puis affine
    la position en ajustant, sur une petite fenêtre autour de ce
    maximum, soit une sigmoïde sur l'intensité elle-même (`mode=
    "sigmoid"`), soit une gaussienne sur le gradient (`mode="gaussian"`).

    Ce choix ne concerne que la LOCALISATION du bord (utilisée pour
    aligner/projeter les lignes dans `GeometricESFBuilder`), pas la
    modélisation de la forme de l'ESF agrégée -- voir `esf_model` sur
    `EsfLsfMTFEstimator` pour cette seconde question, découplée.

    Parameters
    ----------
    mode :
        `"sigmoid"` pour un fit sur l'intensité, `"gaussian"` pour un
        fit sur le gradient.
    """

    def __init__(self, mode: str = "sigmoid") -> None:
        if mode.lower() not in ("sigmoid", "gaussian"):
            raise ValueError("'mode' must be 'sigmoid' or 'gaussian'.")
        self.mode = mode.lower()

    def detect(self, profile: NDArray[np.float64], window: int = 5) -> float:
        """
        Détecte la position sous-pixel du bord dans un profil 1D.

        Parameters
        ----------
        profile :
            Profil d'intensité 1D (une ligne de l'image).
        window :
            Demi-largeur de la fenêtre de fit autour du pic de gradient.

        Returns
        -------
        float
            Position sous-pixel du bord. NaN si le fit échoue vraiment
            (profil trop plat pour contenir un bord exploitable).
        """
        # Import différé : scipy.optimize n'est nécessaire qu'ici.
        from scipy.optimize import curve_fit

        x = np.arange(profile.size)

        # Marge adaptative : au moins 2 px, au plus 10 % de la taille du profil.
        margin = max(2, profile.size // 10)
        if profile.size < 2 * margin + 2:
            return float(profile.size / 2)

        grad = np.gradient(profile)
        # On ignore les bords pour trouver le centre du bord réel.
        inner_grad = grad[margin:-margin]
        x0 = int(np.argmax(np.abs(inner_grad))) + margin

        # Seuil relatif à l'amplitude du profil.
        amplitude = profile.max() - profile.min()
        if amplitude < 1e-3 or np.abs(grad[x0]) < 0.01 * amplitude:
            return float("nan")

        # Fenêtre de fit.
        start, end = max(0, x0 - window), min(profile.size, x0 + window)
        x_fit = x[start:end]

        min_pts = 4
        if len(x_fit) < min_pts:
            return float(x0)

        try:
            if self.mode == "sigmoid":
                y_fit = profile[start:end]
                amplitude_guess = (y_fit[-1] - y_fit[0]) / 2.0
                offset_guess = np.mean(y_fit)
                popt, _ = curve_fit(
                    sigmoid,
                    x_fit,
                    y_fit,
                    p0=[offset_guess, amplitude_guess, float(x0), 1.0],
                    maxfev=2000,
                )
                return float(popt[2])  # x0

            y_fit = grad[start:end]  # mode == "gaussian" : fit sur la dérivée
            popt, _ = curve_fit(
                gaussian,
                x_fit,
                y_fit,
                p0=[np.max(y_fit), float(x0), 1.0, 0.0],
                maxfev=5000,
                bounds=(
                    [-np.inf, start, 0.1, -np.inf],
                    [np.inf, end, 20.0, np.inf],
                ),
            )
            return float(popt[1])  # x0

        except Exception:
            # En cas d'échec du fit, on reste sur l'estimation pixélique.
            return float(x0)


# --------------------------------------------------------------------------
# ESF par projection géométrique sur la normale au bord
# --------------------------------------------------------------------------


@dataclass(slots=True)
class ESFData:
    """
    Résultat de `GeometricESFBuilder.build()` : tout ce dont a besoin un
    modèle de fit (`fit_esf_*`) pour calculer une LSF, sans jamais avoir
    à refaire la détection de bord ni la projection géométrique.

    Attributes
    ----------
    x_esf, esf :
        Grille de l'ESF (position le long de la normale au bord, en
        pixels ; intensité normalisée dans [0, 1]).
    bin_count :
        Nombre d'échantillons ayant contribué à chaque bin de l'ESF --
        une mesure directe de la précision de chaque point, utilisée
        comme poids par `fit_esf_tikhonov` (voir sa docstring).
    edge_positions, rows :
        Positions de bord détectées par ligne (repère colonne d'origine)
        et indices de ligne correspondants, après filtrage des aberrants.
    dx, theta_rad :
        Pente (colonnes par ligne) et angle du bord, estimés par
        régression linéaire sur `edge_positions`.
    details :
        Diagnostics de construction (nombre de lignes valides avant/
        après filtrage, occupation des bins, ...).
    """

    x_esf: NDArray[np.float64]
    esf: NDArray[np.float64]
    bin_count: NDArray[np.float64]
    edge_positions: NDArray[np.float64]
    rows: NDArray[np.float64]
    dx: float
    theta_rad: float
    details: dict[str, Any] = field(default_factory=dict)


class GeometricESFBuilder:
    """
    Construit l'ESF par projection géométrique de chaque pixel sur la
    NORMALE au bord (méthode ISO 12233), plutôt que par un simple
    décalage colonne par colonne.

    Différence avec l'ancien `LineESFBuilder`
    --------------------------------------------
    `LineESFBuilder` décalait chaque ligne uniquement selon sa position
    de bord en colonne (`row, x - x0`), ce qui revient à négliger la
    composante verticale du décalage dû à l'inclinaison -- une bonne
    approximation pour un angle faible, mais géométriquement incorrecte
    au-delà. Ici, chaque pixel `(x, y)` autour du bord de sa ligne est
    projeté sur la direction perpendiculaire au bord :
    `d = (x - x0)*cos(theta) - (y - y_centre)*sin(theta)`, où `x0` est
    la position de la droite de bord AJUSTÉE (régression), évaluée à la
    ligne de référence -- une référence FIXE, pas la position détectée
    (bruitée) de chaque ligne individuellement.

    Piège évité : utiliser la position de bord PAR LIGNE directement
    dans cette formule de projection (`dx_pix = j - x_edge_ligne`),
    variante fréquente de la projection de [Kohm2004]. Or la distance signée
    point-droite correcte, développée analytiquement, ne se simplifie
    à cette forme que si `x_edge` est une référence fixe -- utiliser la
    position par ligne (qui suit déjà la pente) réinjecte quasiment
    l'indice de ligne brut dans la distance projetée. Vérifié
    numériquement : sur un cas de test, la plage de `d` obtenue passe de
    ±39 (avec la position par ligne, pour une fenêtre de ±20 px)
    à ±21 (formule corrigée, référence fixe) -- la première valeur,
    presque le double de la fenêtre demandée, trahissait le problème.
    La position par ligne (bruitée) ne sert plus qu'à choisir la
    fenêtre de colonnes à considérer pour chaque ligne, pas dans la
    formule de distance elle-même.

    Étapes : détection de bord par ligne (`ParametricEdgeDetector`) ->
    filtrage des aberrants -> régression linéaire pour l'angle -> pour
    chaque ligne, projection des pixels dans une fenêtre autour du bord
    -> moyenne par bin sur une grille régulière -> rejet des bins
    insuffisamment peuplés -> lissage léger optionnel (Savitzky-Golay)
    -> normalisation dans [0, 1].

    Largeur de bin : `"omni"` vs `"iso_cos"`
    -------------------------------------------
    Deux conventions de largeur de bin coexistent dans la littérature
    [[Wu2022]](REFERENCE_TECHNIQUE.md, partie I, §10) (`Wu et
    al., 2022`, qui bâtit sur le jeu de données de comparaison de
    [[ViallefontRobinet2018]](REFERENCE_TECHNIQUE.md, partie I, §10)) :
    - `"omni"` (défaut, comportement historique de cette classe) :
      largeur de bin FIXE `1/oversample`, indépendante de l'angle.
    - `"iso_cos"` : largeur de bin `cos(theta)/oversample`, qui RÉTRÉCIT
      avec l'angle -- Wu et al. montrent numériquement que cette
      convention donne une MTF plus lisse et mieux convergée aux
      moyennes/hautes fréquences que `"omni"`, au prix de bins plus
      nombreux (donc plus de risque de bins vides) pour un même
      `oversample`.

    Suréchantillonnage adaptatif (`oversample="auto"`)
    ------------------------------------------------------
    Toujours d'après Wu et al., le suréchantillonnage optimal (nombre de
    lignes effectivement distinctes après projection) dépend de l'angle :
    `OSR_optimal = 1/tan(theta)`. Avec `oversample="auto"`, ce calcul est
    fait automatiquement à partir de l'angle mesuré (régression sur les
    positions de bord), borné à `[2, 10]` (au-delà de 10, Wu et al.
    notent que le coût de calcul augmente sans gain de précision
    qualitatif ; en-deçà de 2, la précision se dégrade nettement).

    Parameters
    ----------
    edge_detector :
        Détecteur utilisé pour localiser le bord sur chaque ligne.
    fit_window :
        Demi-largeur de la fenêtre de fit du détecteur de bord.
    half_width_px :
        Demi-largeur (en pixels) de la bande autour du bord conservée
        dans l'ESF.
    oversample :
        Nombre d'échantillons d'ESF par pixel capteur, ou `"auto"` pour
        le calculer à partir de l'angle mesuré (voir ci-dessus).
    binning_mode :
        `"omni"` (défaut, largeur de bin fixe) ou `"iso_cos"` (largeur
        de bin réduite par `cos(theta)`, voir ci-dessus).
    outlier_threshold :
        Une ligne dont la position de bord détectée s'écarte de la
        médiane de plus de `outlier_threshold` pixels est exclue.
    min_samples_per_bin :
        Un bin de l'ESF avec moins de `min_samples_per_bin` échantillons
        est rejeté plutôt que conservé avec une moyenne peu fiable.
    smoothing :
        Applique un lissage Savitzky-Golay léger après le moyennage par
        bin (fenêtre adaptative).
        Désactivé par défaut : vérifié numériquement que ce lissage,
        avec cette fenêtre adaptative, dégrade nettement
        la MTF aux hautes fréquences sur un cas de test propre (MTF à
        f=0.25 passant de 0.32 à 0.066) -- un compromis qui peut valoir
        le coup sur une image réellement bruitée, mais pas en général.
    """

    def __init__(
        self,
        edge_detector: ParametricEdgeDetector,
        fit_window: int = 5,
        half_width_px: float = 20.0,
        oversample: int | str = 5,
        binning_mode: str = "omni",
        outlier_threshold: float = 5.0,
        min_samples_per_bin: int = 5,
        smoothing: bool = False,
    ) -> None:
        if binning_mode not in ("omni", "iso_cos"):
            raise ValueError("'binning_mode' must be 'omni' or 'iso_cos'.")
        if oversample != "auto" and (not isinstance(oversample, int) or oversample < 1):
            raise ValueError("'oversample' must be a positive int, or 'auto'.")

        self.edge_detector = edge_detector
        self.fit_window = fit_window
        self.half_width_px = half_width_px
        self.oversample = oversample
        self.binning_mode = binning_mode
        self.outlier_threshold = outlier_threshold
        self.min_samples_per_bin = min_samples_per_bin
        self.smoothing = smoothing

    def build(self, image: NDArray[np.float64]) -> ESFData:
        """Construit l'ESF à partir d'une image canonicalisée (bord ~vertical)."""
        n_rows, n_cols = image.shape

        edges = np.array(
            [self.edge_detector.detect(image[y], self.fit_window) for y in range(n_rows)]
        )

        valid_mask = np.isfinite(edges)
        n_valid_before = int(np.count_nonzero(valid_mask))
        if n_valid_before > 3:
            median_pos = np.median(edges[valid_mask])
            outlier_mask = np.abs(edges - median_pos) > self.outlier_threshold
            edges[outlier_mask] = np.nan

        valid = np.isfinite(edges)
        n_valid_after = int(np.count_nonzero(valid))
        if n_valid_after < 3:
            raise ValueError("Not enough valid edge rows to build an ESF.")

        rows = np.arange(n_rows, dtype=np.float64)[valid]
        edge_positions = edges[valid]

        # Régression linéaire : pente (colonnes/ligne) et angle du bord.
        dx, intercept = np.polyfit(rows, edge_positions, 1)
        theta_rad = float(np.arctan(dx))
        sin_t, cos_t = np.sin(theta_rad), np.cos(theta_rad)
        row_center = float(np.mean(rows))
        # Position de la droite de bord ajustée, à la ligne de référence
        # row_center -- LA référence FIXE (pas une position par ligne)
        # utilisée par la formule de projection ci-dessous. Voir la
        # docstring de la classe : utiliser la position par ligne (bruitée)
        # directement dans cette formule ne se simplifie pas correctement et réinjecte quasiment l'indice
        # de ligne brut dans la distance projetée -- vérifié numériquement
        # (la plage de "d" obtenue passe de ±39 à ±21, pour une fenêtre de
        # ±20 px, une fois corrigé).
        x0 = intercept + dx * row_center

        # Projection géométrique de chaque pixel, dans une fenêtre
        # autour du bord de sa ligne, sur la normale au bord. La
        # position de bord PAR LIGNE (bruitée) ne sert ici qu'à choisir
        # la fenêtre de colonnes à considérer -- la distance projetée
        # elle-même utilise la référence fixe `x0` (droite ajustée),
        # pas la position par ligne.
        d_all: list[NDArray[np.float64]] = []
        val_all: list[NDArray[np.float64]] = []
        for row, x_edge in zip(rows, edge_positions):
            j0 = max(0, int(np.floor(x_edge - self.half_width_px)))
            j1 = min(n_cols, int(np.ceil(x_edge + self.half_width_px)) + 1)
            if j1 <= j0:
                continue
            j_coords = np.arange(j0, j1, dtype=np.float64)
            d = (j_coords - x0) * cos_t - (row - row_center) * sin_t
            d_all.append(d)
            val_all.append(image[int(row), j0:j1])

        if not d_all:
            raise ValueError("No pixels collected for the ESF (ROI too narrow?).")

        d_all = np.concatenate(d_all)
        val_all = np.concatenate(val_all)

        # Suréchantillonnage adaptatif (Wu et al., 2022) : OSR optimal
        # = 1/tan(theta), borné à [2, 10] -- voir docstring de la classe.
        if self.oversample == "auto":
            osr = 1.0 / max(abs(np.tan(theta_rad)), 1e-6)
            oversample = int(np.clip(round(osr), 2, 10))
        else:
            oversample = self.oversample

        # Moyenne par bin sur une grille régulière. Deux conventions de
        # largeur (voir docstring de la classe) :
        # - "omni" : largeur fixe 1/oversample, indépendante de l'angle.
        # - "iso_cos" : largeur cos(theta)/oversample, plus étroite
        #   quand l'angle augmente -- meilleure convergence MTF d'après
        #   Wu et al. (2022).
        if self.binning_mode == "iso_cos":
            bin_width = max(cos_t, 1e-6) / oversample
        else:
            bin_width = 1.0 / oversample
        d_min = float(d_all.min())
        n_bins = int(np.ceil((d_all.max() - d_min) / bin_width)) + 1
        x_grid = d_min + bin_width * np.arange(n_bins)

        bin_idx = np.floor((d_all - d_min) / bin_width).astype(int)
        in_range = (bin_idx >= 0) & (bin_idx < n_bins)

        bin_sum = np.bincount(bin_idx[in_range], weights=val_all[in_range], minlength=n_bins)
        bin_count = np.bincount(bin_idx[in_range], minlength=n_bins)

        with np.errstate(divide="ignore", invalid="ignore"):
            esf = bin_sum / bin_count

        populated = bin_count >= self.min_samples_per_bin
        x_esf = x_grid[populated]
        esf = esf[populated]
        bin_count = bin_count[populated]

        if x_esf.size < 8:
            raise ValueError(
                "Not enough well-populated ESF bins "
                "(try a smaller min_samples_per_bin or a taller ROI)."
            )

        esf_range = np.nanmax(esf) - np.nanmin(esf)
        if esf_range < 1e-12:
            raise ValueError("ESF has no dynamic range (flat profile).")
        esf = (esf - np.nanmin(esf)) / esf_range

        if self.smoothing:
            from scipy.signal import savgol_filter

            window = min(31, (len(esf) // 5) | 1)  # impair, borné
            if window >= 7:
                esf = savgol_filter(esf, window_length=window, polyorder=3)
                esf_range = esf.max() - esf.min()
                if esf_range > 1e-12:
                    esf = (esf - esf.min()) / esf_range

        return ESFData(
            x_esf=x_esf,
            esf=esf,
            bin_count=bin_count.astype(np.float64),
            edge_positions=edge_positions,
            rows=rows,
            dx=float(dx),
            theta_rad=theta_rad,
            details={
                "n_valid_rows_before_filtering": n_valid_before,
                "n_valid_rows_after_filtering": n_valid_after,
                "n_rows": n_rows,
                "oversample_used": oversample,
                "binning_mode": self.binning_mode,
                "min_bin_count": int(bin_count.min()),
                "max_bin_count": int(bin_count.max()),
            },
        )


# --------------------------------------------------------------------------
# Modèles de fit ESF -> LSF, découplés de la construction de l'ESF
# --------------------------------------------------------------------------


def fit_esf_numeric(
    x_esf: NDArray[np.float64],
    esf: NDArray[np.float64],
    lsf_derivative_method: str = "gradient",
    lsf_filter_type: str = "tukey",
    lsf_window_length: int = 15,
    lsf_polyorder: int = 3,
    lsf_tukey_alpha: float = 0.15,
    **_ignored: Any,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Calcule la LSF par dérivation numérique directe de l'ESF (pas de
    modèle ajusté) -- comportement historique de cette classe, avant
    l'ajout des modèles `"sigmoid"`/`"pchip"`. Voir `MTFComputer.
    compute_lsf` pour le détail des options de dérivation/filtrage.
    """
    dx = float(x_esf[1] - x_esf[0])
    lsf = MTFComputer.compute_lsf(
        esf,
        dx,
        derivative_method=lsf_derivative_method,
        filter_type=lsf_filter_type,
        window_length=lsf_window_length,
        polyorder=lsf_polyorder,
        tukey_alpha=lsf_tukey_alpha,
    )
    return x_esf, lsf


def fit_esf_sigmoid(
    x_esf: NDArray[np.float64],
    esf: NDArray[np.float64],
    lsf_points: int = 1000,
    **_ignored: Any,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Ajuste une sigmoïde (`sigmoid`) à l'ESF agrégée et retourne sa
    dérivée analytique (`sigmoid_derivative`) comme LSF, sur une grille
    régulière de `lsf_points` points.

    Contrairement à `fit_esf_numeric`, la LSF est ici une fonction
    analytique lisse -- pas de dérivation numérique ni de fenêtrage
    séparé nécessaire (les paramètres `lsf_*` de `fit_esf_numeric` ne
    s'appliquent pas ici).

    Mise en garde : biais de forme (pas un bug, une hypothèse de modèle)
    ---------------------------------------------------------------------
    La dérivée d'une sigmoïde (`sech²`) décroît, dans le domaine
    fréquentiel, plus LENTEMENT qu'une PSF gaussienne aux hautes
    fréquences (queues spectrales plus lourdes). Si le vrai bord a une
    PSF plus proche d'une gaussienne que d'une sigmoïde (cas courant),
    ce modèle SURESTIME systématiquement la MTF aux hautes fréquences --
    et ce, même sans aucun bruit sur l'image, puisqu'il s'agit d'un
    désaccord de forme, pas d'un artefact de bruit. Vérifié
    numériquement sur un bord synthétique à PSF gaussienne connue
    (sigma=1.2 px, sans bruit) : MTF=0.153 à f=0.3 c/px contre 0.077
    théorique (~2x). `fit_esf_pchip`, qui n'impose aucune forme
    particulière, ne montre pas ce biais (0.098 dans le même test) --
    c'est pourquoi c'est le modèle par défaut de `EsfLsfMTFEstimator`,
    pas celui-ci. À réserver aux cas où l'on a une raison de penser que
    le bord suit effectivement une transition en tanh.
    """
    from scipy.optimize import curve_fit

    y_range = float(np.max(esf) - np.min(esf))
    p0 = [
        float(np.min(esf)),
        y_range if esf[-1] > esf[0] else -y_range,
        float(np.median(x_esf)),
        max(0.5, (x_esf.max() - x_esf.min()) / 6.0),
    ]
    try:
        popt, _ = curve_fit(sigmoid, x_esf, esf, p0=p0, maxfev=8000)
    except Exception:
        popt = p0  # repli sur l'estimation initiale si l'ajustement échoue

    x_lsf = np.linspace(x_esf.min(), x_esf.max(), lsf_points)
    lsf = sigmoid_derivative(x_lsf, *popt)
    return x_lsf, lsf


def fit_esf_pchip(
    x_esf: NDArray[np.float64],
    esf: NDArray[np.float64],
    lsf_points: int = 1000,
    **_ignored: Any,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Interpole l'ESF agrégée par une spline PCHIP (Piecewise Cubic
    Hermite Interpolating Polynomial) -- monotone par construction, sans
    hypothèse de forme symétrique (contrairement à `fit_esf_sigmoid`) --
    et en dérive la LSF, recentrée au sous-pixel par rampe de phase FFT.

    Le recentrage est nécessaire ici car, contrairement à un
    modèle ajusté (dont le centre sort directement du fit), une
    interpolation n'est pas garantie centrée sur le vrai milieu du bord.
    """
    from scipy.interpolate import PchipInterpolator

    pchip = PchipInterpolator(x_esf, esf)
    x_lsf = np.linspace(x_esf.min(), x_esf.max(), lsf_points)
    dx = float(x_lsf[1] - x_lsf[0])
    lsf = pchip.derivative()(x_lsf)

    # Recentrage sous-pixel (barycentre puis rampe de phase FFT).
    lsf_positive = np.clip(lsf, 0, None)
    positive_sum = np.sum(lsf_positive)
    if positive_sum < 1e-12:
        return x_lsf, lsf  # LSF dégénérée : pas de recentrage possible

    center_x = float(np.sum(x_lsf * lsf_positive) / positive_sum)
    shift_samples = center_x / dx

    window = np.hanning(lsf.size)
    lsf_windowed = lsf * window
    spectrum = np.fft.fft(lsf_windowed)
    freqs = np.fft.fftfreq(lsf.size)
    phase_ramp = np.exp(2j * np.pi * freqs * shift_samples)
    lsf_centered = np.real(np.fft.ifft(spectrum * phase_ramp))

    x_lsf_centered = x_lsf - center_x
    return x_lsf_centered, lsf_centered


def fit_esf_tikhonov(
    x_esf: NDArray[np.float64],
    esf: NDArray[np.float64],
    bin_count: NDArray[np.float64] | None = None,
    tikhonov_lambda: float | str = "auto",
    tikhonov_lambda_grid: NDArray[np.float64] | None = None,
    **_ignored: Any,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    r"""
    Régularise l'ESF par une pénalisation de Tikhonov d'ordre 2 (lissage
    par courbure), pondérée par la précision de chaque bin, puis en tire
    la LSF par dérivation numérique de l'ESF régularisée -- inspiré de
    Xie et al. (2018) [[Xie2018]](REFERENCE_TECHNIQUE.md, partie I, §10),
    qui posent le problème comme l'estimation d'une ESF mal conditionnée
    par le bruit et le résolvent par régularisation de Tikhonov + poids
    de précision variables, via le principe variationnel.

    **Avertissement de fidélité** : l'article de Xie et al. n'étant pas
    accessible (accès payant), ce qui suit est une formulation de
    Tikhonov standard et bien établie (lissage par pénalisation de la
    dérivée seconde -- équivalent à un "smoothing spline" / Whittaker-
    Henderson), construite pour répondre au même problème (ESF mal
    posée face au bruit) avec les mêmes ingrédients qu'ils décrivent
    (régularisation de Tikhonov, poids de précision, choix de λ par
    courbe en L) -- ce n'est PAS un portage de leurs équations exactes,
    que je n'ai pas pu consulter.

    Formulation
    -----------
    On cherche l'ESF régularisée `f` minimisant :

    ```
    J(f) = sum_i w_i * (esf_i - f_i)^2  +  lambda * sum_i (f_{i+1} - 2*f_i + f_{i-1})^2
    ```

    Le premier terme est l'attache aux données, pondérée par `w_i`
    (poids de précision -- voir ci-dessous) ; le second est la
    pénalisation de courbure (Tikhonov d'ordre 2), qui pousse `f` vers
    une forme lisse. `lambda` arbitre le compromis fidélité/lissage.
    Solution explicite (système linéaire, `L` = opérateur de dérivée
    seconde discret) :

    ```
    f = (W + lambda * L^T @ L)^{-1} @ (W @ esf)
    ```

    La LSF est ensuite `np.gradient(f, x_esf)` -- une dérivation
    numérique simple, mais appliquée à `f` déjà lissée, donc bien moins
    sensible au bruit qu'une dérivation directe de l'ESF brute
    (`esf_model="numeric"`). Comme cette dérivation reste numérique, le
    biais de `np.gradient` (voir `MTFCorrections.derivative`) s'y
    applique toujours, contrairement à `"sigmoid"`/`"pchip"`.

    Poids de précision
    -------------------
    `w_i = bin_count_i` (nombre d'échantillons ayant contribué à ce bin
    de l'ESF, voir `GeometricESFBuilder`) : un bin construit à partir de
    plus de pixels est une estimation plus précise de sa moyenne, donc
    doit peser davantage dans l'attache aux données -- c'est la
    correspondance directe des "poids de précision variables" mentionnés
    par Xie et al. Sans `bin_count` (poids uniformes), la régularisation
    reste valide mais ignore cette information.

    Choix de λ (`tikhonov_lambda="auto"`)
    ----------------------------------------
    Recherche par courbe en L (L-curve, Hansen) : `f` est calculé pour
    une grille de λ (`tikhonov_lambda_grid`, log-espacée par défaut), et
    la valeur retenue est celle qui maximise la courbure du tracé
    `log(||résidu pondéré||)` vs `log(||courbure de f||)` -- le
    "coin" classique de la courbe en L, qui équilibre fidélité aux
    données et lissage sans réglage manuel. `log_res`/`log_rough` sont
    lissés (Savitzky-Golay) avant de calculer cette courbure : dériver
    deux fois de suite (`np.gradient` appliqué deux fois) une courbe
    échantillonnée sur seulement 40 (grille par défaut initiale) à 80
    valeurs de λ amplifie fortement la moindre irrégularité
    d'échantillonnage, au point de déplacer le maximum de courbure
    détecté -- voir §9.13 de la partie I de REFERENCE_TECHNIQUE.md pour le cas
    concret ayant motivé ce lissage (deux angles synthétiques très
    proches donnant des λ dans un rapport ~24×).

    **Limite connue, non résolue.** Ce lissage réduit mais n'élimine pas
    une fragilité plus profonde : quand la courbe en L est lisse, sans
    coin net (cas rencontré en pratique, pas rare), la détection par
    courbure peut sélectionner un λ insuffisant, donnant une LSF
    visiblement dentelée et une FWHM sous-estimée. Vérifié
    numériquement : sur des bords synthétiques à angles proches (-1° à
    -20°, PSF gaussienne identique), la FWHM récupérée en mode "auto"
    varie entre ~0.5 et ~2.9 px pour une valeur théorique de 2.826 px --
    y compris avec λ FIXE (essayé 50.0), certains angles restent mal
    régularisés. Une piste de correction plus robuste (critère de
    Morozov, validation croisée généralisée) n'a pas été implémentée
    (voir §9.13). **En pratique** : si un résultat semble implausible,
    inspecter `details["fwhm"]` et essayer plusieurs valeurs fixes de
    `tikhonov_lambda` plutôt que de faire confiance à "auto" sans
    vérification.

    Parameters
    ----------
    x_esf, esf :
        Grille de l'ESF (voir `GeometricESFBuilder`).
    bin_count :
        Nombre d'échantillons par bin (`ESFData.bin_count`), utilisé
        comme poids de précision. Poids uniformes si `None`.
    tikhonov_lambda :
        `"auto"` (sélection par courbe en L) ou une valeur numérique
        fixée directement.
    tikhonov_lambda_grid :
        Grille de λ testée si `tikhonov_lambda="auto"` (par défaut :
        80 valeurs log-espacées entre 1e-6 et 1e3).

    Returns
    -------
    tuple[NDArray[np.float64], NDArray[np.float64]]
        `(x_esf, lsf)` -- sur la même grille que l'ESF d'entrée (pas de
        ré-échantillonnage, contrairement à `"sigmoid"`/`"pchip"`).
    """
    n = esf.size
    if n < 8:
        raise ValueError("Not enough ESF points for Tikhonov regularization.")

    weights = np.ones(n) if bin_count is None else np.clip(np.asarray(bin_count, dtype=float), 1.0, None)

    # Opérateur de dérivée seconde discret L, (n-2, n).
    L = np.zeros((n - 2, n))
    idx = np.arange(n - 2)
    L[idx, idx] = 1.0
    L[idx, idx + 1] = -2.0
    L[idx, idx + 2] = 1.0
    LtL = L.T @ L
    W = np.diag(weights)
    Wy = weights * esf
    sqrt_w = np.sqrt(weights)

    def solve(lam: float) -> NDArray[np.float64]:
        A = W + lam * LtL
        return np.linalg.solve(A, Wy)

    if tikhonov_lambda == "auto":
        if tikhonov_lambda_grid is None:
            # Densifiée (80 plutôt que 40 points) : réduit l'erreur de
            # discrétisation de la dérivation numérique ci-dessous, en
            # plus du lissage (voir juste après).
            tikhonov_lambda_grid = np.logspace(-6, 3, 80)

        residual_norms = np.empty(len(tikhonov_lambda_grid))
        roughness_norms = np.empty(len(tikhonov_lambda_grid))
        candidates = []
        for i, lam in enumerate(tikhonov_lambda_grid):
            f = solve(lam)
            candidates.append(f)
            residual_norms[i] = np.linalg.norm(sqrt_w * (esf - f)) + 1e-300
            roughness_norms[i] = np.linalg.norm(L @ f) + 1e-300

        # Courbure de la courbe en L (log-log). La dérivée SECONDE d'un
        # signal brut (`np.gradient` appliqué deux fois de suite) est
        # une opération bruyante -- exactement la même classe de
        # problème que celle déjà rencontrée et corrigée ailleurs dans
        # ce pipeline (biais de dérivation numérique, sur-lissage ESF,
        # voir §9 de la partie I de REFERENCE_TECHNIQUE.md). Observé concrètement
        # sur ce point précis : sur deux cas de test très proches (bord
        # synthétique à -4° et -5°, PSF identique), la courbure brute
        # sélectionnait des λ dans un rapport ~24x (14 contre 345),
        # donnant une LSF nettement sous-régularisée (visiblement
        # dentelée) dans le premier cas. Lisser `log_res`/`log_rough`
        # avant de dériver (au lieu de dériver le signal brut deux fois
        # de suite) stabilise cette sélection.
        from scipy.signal import savgol_filter

        log_res = np.log(residual_norms)
        log_rough = np.log(roughness_norms)
        window = min(11, len(log_res) - (1 - len(log_res) % 2))  # impair, borné
        if window >= 5:
            log_res = savgol_filter(log_res, window, 3)
            log_rough = savgol_filter(log_rough, window, 3)
        d1_res, d1_rough = np.gradient(log_res), np.gradient(log_rough)
        d2_res, d2_rough = np.gradient(d1_res), np.gradient(d1_rough)
        denom = (d1_res**2 + d1_rough**2) ** 1.5 + 1e-300
        curvature = (d1_res * d2_rough - d1_rough * d2_res) / denom

        search_slice = slice(4, -4) if len(curvature) > 8 else slice(None)
        best_local = int(np.argmax(np.abs(curvature[search_slice])))
        best_idx = best_local + (4 if len(curvature) > 8 else 0)
        f_reg = candidates[best_idx]
        lambda_used = float(tikhonov_lambda_grid[best_idx])
    else:
        lambda_used = float(tikhonov_lambda)
        f_reg = solve(lambda_used)

    lsf = np.gradient(f_reg, x_esf)
    return x_esf, lsf


#: Modèles de fit ESF -> LSF disponibles, par nom.
ESF_FIT_MODELS = {
    "numeric": fit_esf_numeric,
    "sigmoid": fit_esf_sigmoid,
    "pchip": fit_esf_pchip,
    "tikhonov": fit_esf_tikhonov,
}


# FWHM et RER : implémentations partagées (eoqual-core), conservées ici
# sous leurs noms historiques.
compute_fwhm = core_edge.fwhm
compute_rer = core_edge.rer


# --------------------------------------------------------------------------
# LSF (numérique) + MTF, et corrections
# --------------------------------------------------------------------------


class MTFComputer:
    """
    LSF (dérivée numérique de l'ESF) puis MTF (FFT de la LSF) — délègue à
    :func:`eoqual.core.edge.esf_to_lsf` et :func:`eoqual.core.edge.lsf_to_mtf`
    (voir leurs docstrings pour les options et la normalisation).
    """

    compute_lsf = staticmethod(core_edge.esf_to_lsf)
    compute_mtf = staticmethod(core_edge.lsf_to_mtf)


class MTFCorrections:
    """Corrections physiques d'une MTF brute — délègue à :mod:`eoqual.core.edge`."""

    derivative = staticmethod(core_edge.derivative_correction)
    pixel_aperture = staticmethod(core_edge.pixel_aperture_correction)


# --------------------------------------------------------------------------
# Estimateur
# --------------------------------------------------------------------------


class EsfLsfMTFEstimator(SlantedEdgeMTFEstimator):
    """
    Estime la MTF par détection de bord ligne par ligne, construction
    d'ESF par projection géométrique, puis LSF/MTF selon un modèle de
    fit choisi indépendamment (`esf_model`).

    Différences avec les approches spatiale et spectrale
    -------------------------------------------------------
    Les approches spatiale (`estimators/spatial.py`) et spectrale
    (`estimators/spectral.py`) exploitent toutes deux la représentation
    symétrique demi-onde et l'estimation de pente par différence de
    phase multi-harmoniques (`fourier_utils.estimate_slope_
    multiharmonic`) -- c'est le socle partagé par `SlantedEdgeMTFEstimator.
    estimate()`. Cette approche-ci n'utilise ni l'une ni l'autre : elle
    détecte sa propre position de bord sur *chaque ligne*
    (`ParametricEdgeDetector`), et en déduit `dx`/`theta` par régression
    linéaire -- une méthode d'estimation de pente entièrement
    différente, indépendante de la construction symétrique. Elle
    surcharge donc `estimate()` en entier plutôt que de n'implémenter
    que `_compute_mtf()` (qui reste inutilisée -- voir plus bas).

    Deux choix indépendants, découplés
    -------------------------------------
    - `mode` (`ParametricEdgeDetector`) : comment LOCALISER le bord sur
      chaque ligne (`"sigmoid"` ou `"gaussian"`), pour la construction de
      l'ESF (`GeometricESFBuilder`, voir sa docstring pour le détail de
      la projection géométrique sur la normale au bord).
    - `esf_model` : comment MODÉLISER l'ESF agrégée pour en tirer la LSF
      (`ESF_FIT_MODELS`) : `"numeric"` (dérivation numérique directe,
      comportement historique de cette classe -- seul mode où
      `lsf_derivative_method`/`lsf_filter_type`/... s'appliquent),
      `"sigmoid"` (fit d'une sigmoïde, dérivée analytique), ou `"pchip"`
      (interpolation spline monotone, sans hypothèse de forme, recentrée
      au sous-pixel par rampe de phase FFT).

    Comme l'ESF n'est construite qu'une fois (`GeometricESFBuilder.
    build()`) et que seul le modèle de fit change ensuite, comparer
    `esf_model="numeric"` / `"sigmoid"` / `"pchip"` sur la même image ne
    refait jamais la détection de bord ni la projection géométrique.

    Convention de fréquence : avec `pixel_pitch=1.0` (défaut), les
    fréquences produites restent en cycles/pixel, 0.5 = Nyquist -- la
    même convention `f / f_e` que les approches spatiale et spectrale,
    ce qui les rend directement comparables sur le même graphique.

    Parameters
    ----------
    mode :
        `"sigmoid"` ou `"gaussian"`, transmis à `ParametricEdgeDetector`
        (localisation du bord par ligne).
    esf_model :
        `"numeric"`, `"sigmoid"` ou `"pchip"` (modèle de fit ESF -> LSF,
        défaut `"pchip"` -- voir la mise en garde dans `fit_esf_sigmoid`
        sur le biais de forme de `"sigmoid"` aux hautes fréquences).
    pixel_pitch :
        Taille du pixel (défaut : 1.0, c'est-à-dire pas de conversion
        physique -- voir "Convention de fréquence" ci-dessus).
    fit_window :
        Demi-largeur de la fenêtre de fit du détecteur de bord.
    esf_half_width, esf_oversample :
        Demi-largeur (pixels) et suréchantillonnage (échantillons par
        pixel, ou `"auto"`) de la grille d'ESF -- transmis à
        `GeometricESFBuilder`.
    esf_binning_mode :
        `"omni"` (défaut) ou `"iso_cos"` -- voir `GeometricESFBuilder`.
    outlier_threshold :
        Une ligne dont la position de bord détectée s'écarte de la
        médiane de plus de `outlier_threshold` pixels est exclue avant
        la construction de l'ESF (bord mal détecté sur cette ligne).
    esf_min_samples_per_bin :
        Un bin de l'ESF avec moins d'échantillons que cela est rejeté.
    esf_smoothing :
        Lissage Savitzky-Golay léger de l'ESF après moyennage par bin.
    lsf_points :
        Nombre de points de la grille de LSF pour `esf_model="sigmoid"`
        ou `"pchip"` (sans effet pour `"numeric"`, qui reste sur la
        grille de l'ESF).
    lsf_derivative_method, lsf_filter_type, lsf_window_length,
    lsf_polyorder, lsf_tukey_alpha :
        Options de calcul de la LSF pour `esf_model="numeric"`
        uniquement -- voir `MTFComputer.compute_lsf`.
    tikhonov_lambda, tikhonov_lambda_grid :
        Options pour `esf_model="tikhonov"` uniquement -- voir
        `fit_esf_tikhonov`.
    """

    def __init__(
        self,
        mode: str = "sigmoid",
        esf_model: str = "pchip",
        pixel_pitch: float = 1.0,
        fit_window: int = 5,
        esf_half_width: float = 20.0,
        esf_oversample: int | str = 5,
        esf_binning_mode: str = "omni",
        outlier_threshold: float = 5.0,
        esf_min_samples_per_bin: int = 5,
        esf_smoothing: bool = False,
        lsf_points: int = 1000,
        lsf_derivative_method: str = "gradient",
        lsf_filter_type: str = "tukey",
        lsf_window_length: int = 15,
        lsf_polyorder: int = 3,
        lsf_tukey_alpha: float = 0.15,
        tikhonov_lambda: float | str = "auto",
        tikhonov_lambda_grid: NDArray[np.float64] | None = None,
    ) -> None:
        super().__init__(pixel_pitch=pixel_pitch)  # low_freq_ratio/block_gap hérités mais inutilisés ici
        if esf_model not in ESF_FIT_MODELS:
            raise ValueError(
                f"'esf_model' must be one of {sorted(ESF_FIT_MODELS)}, got {esf_model!r}."
            )
        self.name = f"esf_lsf_{esf_model}"
        self.mode = mode
        self.esf_model = esf_model
        self.lsf_points = lsf_points
        self.lsf_derivative_method = lsf_derivative_method
        self.lsf_filter_type = lsf_filter_type
        self.lsf_window_length = lsf_window_length
        self.lsf_polyorder = lsf_polyorder
        self.lsf_tukey_alpha = lsf_tukey_alpha
        self.tikhonov_lambda = tikhonov_lambda
        self.tikhonov_lambda_grid = tikhonov_lambda_grid

        self._builder = GeometricESFBuilder(
            edge_detector=ParametricEdgeDetector(mode=mode),
            fit_window=fit_window,
            half_width_px=esf_half_width,
            oversample=esf_oversample,
            binning_mode=esf_binning_mode,
            outlier_threshold=outlier_threshold,
            min_samples_per_bin=esf_min_samples_per_bin,
            smoothing=esf_smoothing,
        )

    def estimate(
        self,
        frame: CanonicalFrame,
        oversample_factor: int = 4,
    ) -> MTFResult:
        """
        Estime la MTF. Surcharge complète : n'utilise ni la
        représentation symétrique ni `estimate_slope_multiharmonic` du
        socle commun -- voir la docstring de la classe.

        `oversample_factor` est accepté pour la cohérence de l'interface
        avec `MTFComparator`, mais sans effet ici.
        """
        esf_data = self._builder.build(frame.image)

        fit_model = ESF_FIT_MODELS[self.esf_model]
        x_lsf, lsf = fit_model(
            esf_data.x_esf,
            esf_data.esf,
            bin_count=esf_data.bin_count,
            lsf_points=self.lsf_points,
            lsf_derivative_method=self.lsf_derivative_method,
            lsf_filter_type=self.lsf_filter_type,
            lsf_window_length=self.lsf_window_length,
            lsf_polyorder=self.lsf_polyorder,
            lsf_tukey_alpha=self.lsf_tukey_alpha,
            tikhonov_lambda=self.tikhonov_lambda,
            tikhonov_lambda_grid=self.tikhonov_lambda_grid,
        )
        lsf_dx = float(x_lsf[1] - x_lsf[0]) if x_lsf.size > 1 else 1.0

        fwhm = compute_fwhm(x_lsf, lsf)
        # RER calculé sur l'ESF elle-même (pas x_lsf, qui peut être
        # recentrée par le modèle -- voir compute_rer) : reste valable
        # quel que soit esf_model.
        rer = compute_rer(esf_data.x_esf, esf_data.esf)

        freqs, mtf = MTFComputer.compute_mtf(lsf, lsf_dx)
        # Correction du biais de dérivation numérique : uniquement
        # pertinente si la LSF vient effectivement d'une dérivation
        # numérique -- "numeric" ET "tikhonov" (qui dérive aussi par
        # np.gradient, seulement sur une ESF déjà régularisée). Pour
        # "sigmoid"/"pchip", la LSF est une dérivée analytique (ou
        # spline) sans ce biais ; l'appliquer quand même revient à
        # diviser par sinc(2f) -- qui tend vers 0 près de Nyquist -- et
        # gonfle artificiellement la MTF aux hautes fréquences, de façon
        # lisse et donc trompeuse (pas de bruit visible pour signaler le
        # problème). Vérifié numériquement sur un bord synthétique
        # bruité (sigma=1.2 px) : sans ce correctif, `esf_model=
        # "sigmoid"` donnait 0.153 à f=0.3 contre 0.077 théorique (~2x)
        # même SANS bruit ajouté.
        if self.esf_model in ("numeric", "tikhonov"):
            mtf = MTFCorrections.derivative(freqs, mtf, lsf_dx)
        mtf = MTFCorrections.pixel_aperture(freqs, mtf, self.pixel_pitch)
        frequency = freqs * self.pixel_pitch  # cycles / pixel_pitch-unit

        frequency_corrected = frequency / np.cos(esf_data.theta_rad)

        details: dict[str, Any] = {
            "mode": self.mode,
            "esf_model": self.esf_model,
            "fwhm": fwhm,
            "rer": rer,
            "x_esf": esf_data.x_esf,
            "esf": esf_data.esf,
            "x_lsf": x_lsf,
            "lsf": lsf,
            "edge_positions": esf_data.edge_positions,
            **esf_data.details,
        }

        return MTFResult(
            method=self.name,
            mtf=mtf,
            frequency=frequency,
            frequency_corrected=frequency_corrected,
            dx=esf_data.dx,
            theta_rad=esf_data.theta_rad,
            oversample_factor=oversample_factor,
            details=details,
        )

    def _compute_mtf(
        self,
        rect: NDArray[np.float64],
        dx: float,
        theta_rad: float,
        oversample_factor: int,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], dict[str, Any]]:
        """
        Non utilisée : `estimate()` est entièrement surchargée ci-dessus
        (voir docstring de la classe pour la raison). Cette méthode
        n'existe que pour satisfaire `SlantedEdgeMTFEstimator`, qui la
        déclare abstraite.
        """
        raise NotImplementedError(
            "EsfLsfMTFEstimator overrides estimate() directly; "
            "_compute_mtf() is unused."
        )


# EOF
