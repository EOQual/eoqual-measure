#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
Intercomparaison des méthodes d'estimation de la variance du bruit.

Deux usages bien séparés, pour éviter toute confusion entre bruit
préexistant et bruit de test :

1. **Validation sur image synthétique** (défaut, sans `--images`) : une
   image "propre" est générée en mémoire et un bruit de variance connue y
   est injecté à la volée (pas de fichier écrit sur disque, inutile pour ce
   cas puisque l'image de départ est garantie sans bruit). Les résultats
   sont comparés à cette référence connue.
2. **Mesure sur images réelles** (`--images ...`) : par défaut, **aucun
   bruit n'est ajouté** — le script mesure directement le bruit déjà présent
   dans vos images. Injecter un bruit supplémentaire sur des images réelles
   n'a de sens que si vous en avez explicitement besoin (ex. tester la
   sensibilité des méthodes à un niveau de bruit connu ajouté par-dessus un
   bruit capteur inconnu) ; il faut alors le demander explicitement via
   `--sigma2` ou `--a`/`--b`, et un avertissement rappelle que la variance
   mesurée inclura le bruit préexistant en plus du bruit injecté.

Utilisation
-----------
    # Validation sur image synthétique (bruit connu, injecté à la volée)
    python examples/noise_intercomparison.py --sigma2 25.0 --a 0.08 --b 5.0

    # Mesure directe sur vos images (aucun bruit ajouté)
    python examples/noise_intercomparison.py --images photo1.tif photo2.tif

    # Vos images + bruit de test explicite par-dessus (avertissement affiché)
    python examples/noise_intercomparison.py --images photo1.tif --sigma2 25.0

Images multiples et barres d'erreur
-------------------------------------
Les images fournies sont supposées issues du même système d'acquisition
(même capteur, même statistique de bruit) : les mesures élémentaires
(patchs, blocs, segments ou tuiles) de toutes les images sont **cumulées**
avant calcul de la variance finale et de sa barre d'erreur. Le nombre de
zones homogènes détectées (cumulé) est rapporté pour chaque méthode.

Non-linéarité aux extrêmes de luminance
------------------------------------------
La relation bruit-luminance σ²(I) = a·I + b n'est en général linéaire que
sur une plage intermédiaire de luminance : en basse luminance, la
quantification du capteur devient non négligeable ; en haute luminance,
l'écrêtage/saturation comprime artificiellement la variance mesurée. Deux
options permettent d'en tenir compte :
- `--trim_low_pct` / `--trim_high_pct` excluent les bins de luminance les
  plus extrêmes du calcul de la régression affine (mais les affichent quand
  même sur le graphique, pour visualiser l'écart), afin que ces effets ne
  biaisent pas l'estimation de la pente/ordonnée sur la plage utile.
- `--fit_degree N` (N=2 quadratique, N=3 cubique, etc.) superpose une
  régression polynomiale de degré N à la droite affine pour visualiser la
  courbure éventuelle. Non affichée par défaut (N=1, affine seule).
- Pour la validation synthétique, `--bit_depth` (ex. 8 ou 16) quantifie et
  écrête l'image bruitée générée, pour reproduire ces effets non linéaires
  de façon contrôlée plutôt que de les supposer.

Toutes les méthodes sont superposées sur la courbe bruit-luminance
------------------------------------------------------------------
Pas seulement `ponomarenko` : `flat_patches`, `adaptive_segmentation`,
`wavelets` et `skimage_estimate_sigma` exposent aussi la luminance de
chacun de leurs échantillons élémentaires (patchs/segments/tuiles), ce qui
permet de les regrouper dans les mêmes bins de luminance que `ponomarenko`
et de les afficher sur le même graphique. Seule `pca_estimate` n'a pas de
courbe par luminance (son estimateur repose sur une covariance globale, pas
sur des variances individuelles par patch) et apparaît comme un point
unique. La régression (affine + degré supérieur optionnel) n'est calculée
et tracée que sur la méthode de référence (`ponomarenko` par défaut) ; les
autres méthodes servent de comparaison visuelle.
"""

import argparse
import os
import time
from typing import Dict, List, Optional, Tuple

import imageio.v3 as iio
import numpy as np
import plotly.graph_objects as go
from numpy.typing import NDArray
from plotly.subplots import make_subplots

from eoqual.core.bayer import extract_bayer_planes
from eoqual.measure.noise.variance import NoiseVariance
from eoqual.measure.synthetic.noise import add_noise, generate_clean_image



def load_real_images(paths: List[str]) -> List[NDArray[np.float32]]:
    """
    Charge une ou plusieurs images réelles en niveaux de gris via `imageio`.

    Parameters
    ----------
    paths : List[str]
        Chemins vers les fichiers image à charger.

    Returns
    -------
    List[NDArray[np.float32]]
        Images 2D en niveaux de gris, type float32.
    """

    images: List[NDArray[np.float32]] = []
    for path in paths:
        raw = np.asarray(iio.imread(path))
        if raw.ndim == 3:
            weights = np.array([0.2989, 0.5870, 0.1140])
            raw = np.tensordot(raw[..., :3], weights, axes=([-1], [0]))
        images.append(raw.astype(np.float32))
    return images


def load_bayer_images(paths: List[str]) -> List[NDArray[np.float32]]:
    """
    Charge une ou plusieurs images Bayer **mosaïquées** (un seul plan, motif
    de filtres couleur non démosaïqué) via `imageio`, sans appliquer de
    conversion en niveaux de gris.

    Contrairement à `load_real_images`, aucune pondération RGB n'est
    appliquée : chaque pixel du plan chargé correspond à une seule
    composante couleur (R, G ou B) selon sa position dans le motif de
    Bayer, et doit rester tel quel pour pouvoir être séparé par bande via
    `split_bayer_channels`.

    Parameters
    ----------
    paths : List[str]
        Chemins vers les fichiers image Bayer (mosaïque brute) à charger.

    Returns
    -------
    List[NDArray[np.float32]]
        Images 2D mosaïquées, type float32.
    """

    images: List[NDArray[np.float32]] = []
    for path in paths:
        raw = np.asarray(iio.imread(path))
        if raw.ndim == 3:
            # Un fichier "Bayer" à 3 canaux est probablement déjà démosaïqué
            # (ou stocké en RGB par le lecteur) : on ne peut pas retrouver le
            # motif d'origine. On prévient et on se rabat sur le premier canal.
            print(
                f"ATTENTION : '{path}' contient {raw.shape[-1]} canaux, "
                "alors qu'une image Bayer mosaïquée est attendue (1 seul "
                "plan). Utilisation du premier canal uniquement -- le "
                "résultat peut ne pas correspondre au motif Bayer réel."
            )
            raw = raw[..., 0]
        images.append(raw.astype(np.float32))
    return images


def split_bayer_channels(
    image: NDArray[np.float32], pattern: str = 'RGGB'
) -> Dict[str, NDArray[np.float32]]:
    """
    Sépare une image Bayer mosaïquée en trois bandes R, G, B en sous-
    échantillonnant chaque composante à sa position dans le motif 2x2.

    Chaque bande résultante a une résolution moitié (en ligne et en
    colonne) de l'image d'origine. Les deux pixels verts du motif (G1, G2)
    sont combinés par moyenne simple en une seule bande G. Délègue à
    ``eoqual.core.bayer.extract_bayer_planes`` (dimension impaire : la
    dernière ligne ou colonne est ignorée).

    Parameters
    ----------
    image : NDArray[np.float32]
        Image 2D mosaïquée (un seul plan, non démosaïqué).
    pattern : str
        Motif de Bayer : 'RGGB', 'BGGR', 'GRBG' ou 'GBRG' (défaut : 'RGGB').

    Returns
    -------
    Dict[str, NDArray[np.float32]]
        Dictionnaire {'R': ..., 'G': ..., 'B': ...}, chaque bande étant un
        tableau 2D de résolution (H/2, W/2).
    """

    if image.ndim != 2:
        raise ValueError(
            f"split_bayer_channels attend une image 2D mosaïquée, reçu ndim={image.ndim}."
        )
    planes = extract_bayer_planes(image, pattern.upper())  # ValueError si motif inconnu
    return {band: planes[band].astype(np.float32) for band in ('R', 'G', 'B')}


def load_bayer_band_images(
    paths: List[str], pattern: str = 'RGGB'
) -> Dict[str, List[NDArray[np.float32]]]:
    """
    Charge des images Bayer mosaïquées et les sépare en trois listes
    d'images par bande (R, G, B), prêtes à être passées telles quelles aux
    fonctions d'analyse existantes (`run_comparison`, `run_noise_vs_luminance`),
    exactement comme des images en niveaux de gris.

    Parameters
    ----------
    paths : List[str]
        Chemins vers les fichiers image Bayer (mosaïque brute).
    pattern : str
        Motif de Bayer ('RGGB', 'BGGR', 'GRBG' ou 'GBRG').

    Returns
    -------
    Dict[str, List[NDArray[np.float32]]]
        Dictionnaire {'R': [images...], 'G': [images...], 'B': [images...]}.
    """

    mosaics = load_bayer_images(paths)
    bands: Dict[str, List[NDArray[np.float32]]] = {'R': [], 'G': [], 'B': []}
    for mosaic in mosaics:
        split = split_bayer_channels(mosaic, pattern)
        for name in bands:
            bands[name].append(split[name])
    return bands


def compute_snr(
    band_images: List[NDArray[np.float32]], noise_variance: float
) -> Tuple[float, float]:
    """
    Calcule le rapport signal/bruit (SNR) d'une bande à partir de son
    intensité moyenne (signal) et d'une variance de bruit déjà estimée.

    SNR = intensité_moyenne / écart-type_du_bruit, exprimé en linéaire et
    en décibels (20*log10(SNR)).

    Parameters
    ----------
    band_images : List[NDArray[np.float32]]
        Images (toutes de la même bande) sur lesquelles calculer le signal moyen.
    noise_variance : float
        Variance du bruit estimée pour cette bande (ex. par `ponomarenko`).

    Returns
    -------
    Tuple[float, float]
        (SNR linéaire, SNR en dB). (nan, nan) si la variance est non finie,
        nulle ou négative.
    """

    if not np.isfinite(noise_variance) or noise_variance <= 0:
        return float('nan'), float('nan')

    mean_signal = float(np.mean(np.concatenate([img.ravel() for img in band_images])))
    noise_std = float(np.sqrt(noise_variance))
    snr_linear = mean_signal / noise_std if noise_std > 0 else float('nan')
    snr_db = 20.0 * np.log10(snr_linear) if snr_linear > 0 else float('nan')
    return snr_linear, snr_db


def run_comparison(
    noisy_images: List[NDArray[np.float32]],
    methods: List[str] = None,
    method_params: Dict[str, dict] = None,
) -> Dict[str, Dict[str, object]]:
    """
    Applique toutes les méthodes d'estimation de bruit demandées à chaque
    image fournie, puis cumule les mesures élémentaires (patchs, blocs,
    segments ou tuiles) de toutes les images avant de calculer la variance
    finale et sa barre d'erreur pour chaque méthode.

    Parameters
    ----------
    noisy_images : List[NDArray[np.float32]]
        Images (même système d'acquisition) sur lesquelles appliquer les
        estimateurs -- bruitées ou non selon le mode d'utilisation.
    methods : List[str] ou None
        Liste des méthodes à comparer. Si None, toutes les méthodes
        disponibles dans `NoiseVariance` sont utilisées.
    method_params : Dict[str, dict] ou None
        Dictionnaire optionnel {méthode: paramètres} pour surcharger les
        valeurs par défaut de certaines méthodes.

    Returns
    -------
    Dict[str, Dict[str, object]]
        Pour chaque méthode : variance moyenne, écart-type, nombre de zones
        homogènes détectées (cumulé), temps de calcul total, erreurs éventuelles.
    """

    if methods is None:
        methods = list(NoiseVariance(noisy_images[0]).get_available_methods().keys())
    if method_params is None:
        method_params = {}

    results: Dict[str, Dict[str, object]] = {}

    for method in methods:
        params = method_params.get(method)
        all_samples: List[NDArray[np.float64]] = []
        total_time = 0.0
        errors: List[str] = []

        for image in noisy_images:
            nv = NoiseVariance(image)
            t0 = time.perf_counter()
            try:
                _variance, _mean, samples, _luminance = nv.estimate_noise(
                    image, method, params, return_samples=True
                )
                total_time += time.perf_counter() - t0

                if isinstance(samples, list):
                    for s in samples:
                        if len(s) > 0:
                            all_samples.append(np.asarray(s))
                else:
                    samples = np.asarray(samples)
                    if len(samples) > 0:
                        all_samples.append(samples)
            except Exception as exc:  # noqa: BLE001 - on veut comparer même les échecs
                total_time += time.perf_counter() - t0
                errors.append(str(exc))

        pooled = np.concatenate(all_samples) if all_samples else np.array([], dtype=np.float64)

        results[method] = {
            'variance': float(np.mean(pooled)) if len(pooled) else float('nan'),
            'std': float(np.std(pooled, ddof=1)) if len(pooled) > 1 else 0.0,
            'n_zones': int(len(pooled)),
            'time_s': total_time,
            'errors': errors,
        }

    return results


def _common_luminance_bins(
    noisy_images: List[NDArray[np.float32]], num_bins: int
) -> NDArray[np.float64]:
    """
    Détermine des bornes de bin de luminance communes à toutes les méthodes,
    à partir de la distribution des valeurs de pixels des images fournies
    (percentiles 1-99 pour éviter que quelques valeurs extrêmes n'étirent
    excessivement la plage), afin de pouvoir superposer les courbes
    bruit-luminance de plusieurs méthodes sur le même axe.

    Parameters
    ----------
    noisy_images : List[NDArray[np.float32]]
        Images sur lesquelles calculer la plage de luminance.
    num_bins : int
        Nombre de bins souhaité.

    Returns
    -------
    NDArray[np.float64]
        Bornes des bins, de longueur `num_bins + 1`.
    """

    all_pixels = np.concatenate([img.ravel().astype(np.float64) for img in noisy_images])
    lo, hi = np.percentile(all_pixels, [1.0, 99.0])
    if hi <= lo:
        lo, hi = float(np.min(all_pixels)), float(np.max(all_pixels) + 1.0)
    return np.linspace(lo, hi, num_bins + 1)


def _bin_variance_samples(
    var_samples: NDArray[np.float64],
    luminance_samples: NDArray[np.float64],
    edges: NDArray[np.float64],
) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.int64]]:
    """
    Regroupe des couples (variance, luminance) individuels dans des bins de
    luminance communs (`edges`), et calcule la moyenne/écart-type/effectif
    de la variance dans chaque bin non vide.

    Parameters
    ----------
    var_samples : NDArray[np.float64]
        Estimations individuelles de variance (une par patch/bloc/segment/tuile).
    luminance_samples : NDArray[np.float64]
        Luminance moyenne associée à chaque échantillon (même ordre).
    edges : NDArray[np.float64]
        Bornes de bin communes (voir `_common_luminance_bins`).

    Returns
    -------
    Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.int64]]
        Luminance moyenne, variance moyenne, écart-type, effectif -- un
        quadruplet de valeurs par bin non vide.
    """

    if len(var_samples) == 0:
        empty = np.array([], dtype=np.float64)
        return empty, empty, empty, np.array([], dtype=np.int64)

    num_bins = len(edges) - 1
    bin_idx = np.clip(np.digitize(luminance_samples, edges[1:-1]), 0, num_bins - 1)

    means_out, vars_out, stds_out, n_out = [], [], [], []
    for b in range(num_bins):
        mask = bin_idx == b
        if not np.any(mask):
            continue
        lum_b = luminance_samples[mask]
        var_b = var_samples[mask]
        means_out.append(float(np.mean(lum_b)))
        vars_out.append(float(np.mean(var_b)))
        stds_out.append(float(np.std(var_b, ddof=1)) if mask.sum() > 1 else 0.0)
        n_out.append(int(mask.sum()))

    return (
        np.array(means_out, dtype=np.float64),
        np.array(vars_out, dtype=np.float64),
        np.array(stds_out, dtype=np.float64),
        np.array(n_out, dtype=np.int64),
    )


def run_noise_vs_luminance(
    noisy_images: List[NDArray[np.float32]],
    num_bins: int = 8,
    methods: List[str] = None,
    method_params: Dict[str, dict] = None,
    reference_method: str = 'ponomarenko',
    trim_low_pct: float = 0.0,
    trim_high_pct: float = 0.0,
    fit_degree: int = 1,
) -> Dict[str, object]:
    """
    Estime la relation bruit-luminance σ²(I) pour **toutes les méthodes**
    (pas seulement `ponomarenko`), en cumulant sur toutes les images les
    échantillons individuels (patchs/blocs/segments/tuiles) de chaque
    méthode, puis en les regroupant dans des bins de luminance **communs**
    à toutes les méthodes pour permettre une superposition directe.

    Une régression polynomiale de degré `fit_degree` (1 = affine, 2 =
    quadratique, 3 = cubique, etc.) est ajustée sur la courbe de la méthode
    `reference_method` uniquement (par défaut `ponomarenko`, la plus adaptée
    à ce type d'analyse car elle stratifie nativement par luminance) — les
    autres méthodes servent de comparaison visuelle (leurs points doivent
    suivre la même tendance si l'estimation est fiable), sans qu'une
    régression ne soit tracée pour chacune (par lisibilité).

    Pour tenir compte d'un éventuel écart à la linéarité aux luminances
    extrêmes (quantification en bas, écrêtage/saturation en haut), les bins
    dont la luminance se situe dans les `trim_low_pct` % les plus bas ou
    `trim_high_pct` % les plus hauts de la plage observée sont exclus du
    calcul de la régression de `reference_method` (mais restent affichés).

    Parameters
    ----------
    noisy_images : List[NDArray[np.float32]]
        Images sur lesquelles estimer la relation bruit-luminance.
    num_bins : int
        Nombre de bins de luminance communs à toutes les méthodes.
    methods : List[str] ou None
        Méthodes à superposer. Si None, toutes les méthodes disponibles.
    method_params : Dict[str, dict] ou None
        Paramètres optionnels par méthode.
    reference_method : str
        Méthode sur laquelle la régression (affine/quadratique/cubique...)
        est calculée et tracée (défaut : `'ponomarenko'`).
    trim_low_pct : float
        Pourcentage de la plage de luminance à exclure en bas pour la
        régression de `reference_method` (défaut : 0).
    trim_high_pct : float
        Pourcentage de la plage de luminance à exclure en haut (défaut : 0).
    fit_degree : int
        Degré du polynôme ajusté sur `reference_method` (défaut : 1, affine).
        `fit_degree=2` ajoute une régression quadratique, `fit_degree=3` une
        cubique, etc., en plus de la droite affine de référence.

    Returns
    -------
    Dict[str, object]
        Dictionnaire `{'curves': {méthode: {'means','variances','stds','n_zones'}},
        'reference_method', 'used_in_fit', 'fit_coeffs_1' (a, b de la droite
        affine), 'fit_coeffs_extra' (coefficients du degré `fit_degree` si
        > 1, sinon None)}`.
    """

    if methods is None:
        methods = list(NoiseVariance(noisy_images[0]).get_available_methods().keys())
    if method_params is None:
        method_params = {}

    edges = _common_luminance_bins(noisy_images, num_bins)

    curves: Dict[str, Dict[str, NDArray]] = {}

    for method in methods:
        params = method_params.get(method)
        all_var: List[NDArray[np.float64]] = []
        all_lum: List[NDArray[np.float64]] = []

        for image in noisy_images:
            nv = NoiseVariance(image)
            try:
                _var, _mean, var_samples, lum_samples = nv.estimate_noise(
                    image, method, params, return_samples=True
                )
            except Exception:  # noqa: BLE001 - méthode indisponible sur cette image, on l'ignore ici
                continue

            if isinstance(var_samples, list):
                # ponomarenko : liste d'arrays (une par bin interne) -> on aplatit
                for vs, ls in zip(var_samples, lum_samples):
                    if len(vs) > 0:
                        all_var.append(np.asarray(vs))
                        all_lum.append(np.asarray(ls))
            else:
                var_samples = np.asarray(var_samples)
                lum_samples = np.asarray(lum_samples)
                if len(var_samples) > 0:
                    all_var.append(var_samples)
                    all_lum.append(lum_samples)

        var_pool = np.concatenate(all_var) if all_var else np.array([], dtype=np.float64)
        lum_pool = np.concatenate(all_lum) if all_lum else np.array([], dtype=np.float64)

        means_b, vars_b, stds_b, n_b = _bin_variance_samples(var_pool, lum_pool, edges)
        curves[method] = {'means': means_b, 'variances': vars_b, 'stds': stds_b, 'n_zones': n_b}

    ref = curves.get(reference_method, {'means': np.array([]), 'variances': np.array([]), 'stds': np.array([])})
    means_arr, variances_arr, stds_arr = ref['means'], ref['variances'], ref['stds']

    used_in_fit = np.ones(len(means_arr), dtype=bool)
    if len(means_arr) > 0 and (trim_low_pct > 0 or trim_high_pct > 0):
        lo, hi = np.min(means_arr), np.max(means_arr)
        span = hi - lo if hi > lo else 1.0
        low_cut = lo + span * (trim_low_pct / 100.0)
        high_cut = hi - span * (trim_high_pct / 100.0)
        used_in_fit = (means_arr >= low_cut) & (means_arr <= high_cut)
        if used_in_fit.sum() < 2:
            used_in_fit = np.ones(len(means_arr), dtype=bool)  # garde-fou

    if used_in_fit.sum() >= 2:
        weights = 1.0 / np.maximum(stds_arr[used_in_fit], 1e-6) ** 2
        fit_a, fit_b = np.polyfit(means_arr[used_in_fit], variances_arr[used_in_fit], deg=1, w=weights)
    else:
        fit_a, fit_b = float('nan'), float('nan')

    fit_coeffs_extra = None
    if fit_degree > 1 and len(means_arr) >= fit_degree + 1:
        weights_all = 1.0 / np.maximum(stds_arr, 1e-6) ** 2
        fit_coeffs_extra = np.polyfit(means_arr, variances_arr, deg=fit_degree, w=weights_all)

    return {
        'curves': curves,
        'reference_method': reference_method,
        'used_in_fit': used_in_fit,
        'fit_a': float(fit_a),
        'fit_b': float(fit_b),
        'fit_degree': fit_degree,
        'fit_coeffs_extra': fit_coeffs_extra,
    }


def plot_comparison(
    results: Dict[str, Dict[str, object]],
    true_variance: Optional[float],
    luminance_curve: Dict[str, object],
    true_a: Optional[float],
    true_b: Optional[float],
    n_images: int,
    output_html: str = "comparaison_bruit.html",
    band_label: Optional[str] = None,
    snr: Optional[Tuple[float, float]] = None,
) -> None:
    """
    Construit un tableau de bord `plotly` : comparaison des variances
    estimées avec barres d'erreur, temps de calcul, et courbe bruit versus
    luminance avec **toutes les méthodes superposées** (barres d'erreur par
    bin), régression affine et polynôme de degré supérieur optionnel
    calculés sur la méthode de référence.

    Parameters
    ----------
    results : Dict[str, Dict[str, object]]
        Résultats produits par `run_comparison`.
    true_variance : float ou None
        Variance de référence (bruit homoscédastique injecté), si connue
        (None en mesure directe sur images réelles sans injection).
    luminance_curve : Dict[str, object]
        Résultat de `run_noise_vs_luminance` (toutes méthodes).
    true_a : float ou None
        Pente réelle du bruit hétéroscédastique injecté, si connue.
    true_b : float ou None
        Ordonnée à l'origine réelle du bruit hétéroscédastique injecté, si connue.
    n_images : int
        Nombre d'images utilisées.
    output_html : str
        Chemin du fichier HTML de sortie.
    band_label : str ou None
        Si fourni (ex. 'R', 'G', 'B'), identifie la bande couleur analysée
        et l'ajoute au titre du graphique et aux impressions console (mode
        Bayer par bande).
    snr : Tuple[float, float] ou None
        (SNR linéaire, SNR en dB) déjà calculé pour cette bande, affiché en
        annotation sur le graphique et dans le résumé console si fourni.

    Returns
    -------
    None
    """

    band_suffix = f" -- canal {band_label}" if band_label else ""

    methods = list(results.keys())
    means = [results[m]['variance'] for m in methods]
    stds = [results[m]['std'] for m in methods]
    times = [results[m]['time_s'] for m in methods]
    n_zones = [results[m]['n_zones'] for m in methods]
    errors_pct = [
        100.0 * (v - true_variance) / true_variance
        if (true_variance is not None and np.isfinite(v)) else np.nan
        for v in means
    ]

    fig = make_subplots(
        rows=2, cols=2,
        specs=[[{}, {}], [{"colspan": 2}, None]],
        subplot_titles=(
            f"Variance estimée ± écart-type (n={n_images} image(s), zones cumulées){band_suffix}",
            "Temps de calcul total par méthode",
            f"Bruit vs luminance -- toutes méthodes superposées{band_suffix}",
        ),
        row_heights=[0.45, 0.55],
    )

    fig.add_trace(
        go.Bar(
            x=methods, y=means, name="Variance estimée",
            error_y=dict(type='data', array=stds, visible=True),
            text=[f"n={n}" for n in n_zones],
            textposition='outside',
            marker_color="steelblue",
        ),
        row=1, col=1,
    )
    if true_variance is not None:
        fig.add_hline(
            y=true_variance, line_dash="dash", line_color="firebrick",
            annotation_text=f"Référence = {true_variance:.2f}",
            row=1, col=1,
        )

    fig.add_trace(
        go.Bar(x=methods, y=times, name="Temps (s)", marker_color="darkorange"),
        row=1, col=2,
    )

    # --- Courbe bruit-luminance : toutes les méthodes superposées ---
    curves = luminance_curve['curves']
    reference_method = luminance_curve['reference_method']
    used_in_fit = luminance_curve['used_in_fit']
    fit_a, fit_b = luminance_curve['fit_a'], luminance_curve['fit_b']
    fit_degree = luminance_curve['fit_degree']
    fit_coeffs_extra = luminance_curve['fit_coeffs_extra']

    palette = {
        'flat_patches': '#1f77b4', 'wavelets': '#ff7f0e', 'skimage_estimate_sigma': '#2ca02c',
        'ponomarenko': '#d62728', 'adaptive_segmentation': '#9467bd', 'pca_estimate': '#8c564b',
    }

    all_lum_for_range: List[float] = []
    for method in methods:
        c = curves.get(method)
        if c is None or len(c['means']) == 0:
            continue
        all_lum_for_range.extend(c['means'].tolist())

        is_ref = (method == reference_method)
        fig.add_trace(
            go.Scatter(
                x=c['means'], y=c['variances'], mode='markers',
                name=f"{method}" + (" (référence, régression)" if is_ref else ""),
                error_y=dict(type='data', array=c['stds'], visible=True),
                text=[f"n={n}" for n in c['n_zones']],
                marker=dict(
                    size=11 if is_ref else 8,
                    color=palette.get(method, 'gray'),
                    symbol='circle' if is_ref else 'diamond',
                    line=dict(width=2, color='black') if is_ref else None,
                ),
            ),
            row=2, col=1,
        )

    if all_lum_for_range:
        x_line = np.linspace(np.min(all_lum_for_range), np.max(all_lum_for_range), 100)

        # Points de la méthode de référence exclus de la régression (marqués)
        ref_curve = curves.get(reference_method)
        if ref_curve is not None and len(ref_curve['means']) > 0 and (~used_in_fit).any():
            fig.add_trace(
                go.Scatter(
                    x=ref_curve['means'][~used_in_fit], y=ref_curve['variances'][~used_in_fit], mode='markers',
                    name=f"{reference_method} -- bins exclus de la régression",
                    marker=dict(size=13, color='lightgray', line=dict(width=2, color='darkgray'), symbol='x'),
                ),
                row=2, col=1,
            )

        fig.add_trace(
            go.Scatter(
                x=x_line, y=fit_a * x_line + fit_b, mode='lines',
                name=f"Régression affine sur {reference_method} (a={fit_a:.3g}, b={fit_b:.3g})",
                line=dict(color="black", dash="dash"),
            ),
            row=2, col=1,
        )
        if fit_coeffs_extra is not None:
            y_extra = np.polyval(fit_coeffs_extra, x_line)
            fig.add_trace(
                go.Scatter(
                    x=x_line, y=y_extra, mode='lines',
                    name=f"Régression degré {fit_degree} sur {reference_method}",
                    line=dict(color="black", dash="dot"),
                ),
                row=2, col=1,
            )
        if true_a is not None and true_b is not None:
            fig.add_trace(
                go.Scatter(
                    x=x_line, y=true_a * x_line + true_b, mode='lines',
                    name=f"Modèle réel injecté (a={true_a:.3g}, b={true_b:.3g})",
                    line=dict(color="firebrick"),
                ),
                row=2, col=1,
            )

    title_text = "Intercomparaison des méthodes d'estimation de la variance du bruit" + band_suffix
    if snr is not None and np.isfinite(snr[0]):
        title_text += f" (SNR ≈ {snr[0]:.2f} = {snr[1]:.1f} dB)"

    fig.update_layout(
        title_text=title_text,
        showlegend=True,
        template="plotly_white",
        height=850,
    )
    fig.update_yaxes(title_text="Variance", row=1, col=1)
    fig.update_yaxes(title_text="Secondes", row=1, col=2)
    fig.update_xaxes(title_text="Luminance moyenne du bin", row=2, col=1)
    fig.update_yaxes(title_text="Variance du bruit", row=2, col=1)

    fig.write_html(output_html)
    print(f"Graphique sauvegardé dans : {output_html}")

    if snr is not None:
        if np.isfinite(snr[0]):
            print(f"SNR{band_suffix} : {snr[0]:.3f} ({snr[1]:.2f} dB)")
        else:
            print(f"SNR{band_suffix} : indisponible (variance de référence invalide)")

    print(f"\nRésumé{band_suffix} - ({n_images} image(s), zones cumulées) :")
    print(f"{'Méthode':25s} {'Variance':>10s} {'± écart-type':>14s} {'Zones détectées':>16s} {'Erreur %':>10s} {'Temps (s)':>10s}")
    for m, err in zip(methods, errors_pct):
        v = results[m]['variance']
        s = results[m]['std']
        n = results[m]['n_zones']
        t = results[m]['time_s']
        err_str = f"{err:+.1f}" if np.isfinite(err) else "N/A"
        n_fail = len(results[m]['errors'])
        note = f" [{n_fail} échec(s)]" if n_fail else ""
        print(f"{m:25s} {v:10.4f} {s:14.4f} {n:16d} {err_str:>10s} {t:10.4f}{note}")

    ref_curve = curves.get(reference_method)
    if ref_curve is not None and len(ref_curve['means']) > 0:
        print(f"\nRésumé{band_suffix} - relation bruit vs luminance, méthode de référence '{reference_method}' :")
        print(f"{'Luminance':>12s} {'Variance':>10s} {'± écart-type':>14s} {'Zones détectées':>16s} {'Utilisé (fit)':>14s}")
        for m, v, s, n, u in zip(
            ref_curve['means'], ref_curve['variances'], ref_curve['stds'], ref_curve['n_zones'], used_in_fit
        ):
            print(f"{m:12.2f} {v:10.4f} {s:14.4f} {n:16d} {'oui' if u else 'non':>14s}")
        if true_a is not None:
            print(f"\nModèle réel injecté        : a={true_a:.4g}, b={true_b:.4g}")
        print(f"Modèle estimé (régression affine, plage retenue) : a={fit_a:.4g}, b={fit_b:.4g}")
        if fit_coeffs_extra is not None:
            terms = " + ".join(f"{c:.4g}*I^{fit_degree - i}" for i, c in enumerate(fit_coeffs_extra))
            print(f"Régression degré {fit_degree} (tous bins) : {terms}")
        print(
            "\nAutres méthodes affichées à titre de comparaison visuelle (pas de "
            "régression tracée pour elles, par lisibilité) -- pca_estimate "
            "n'a pas de courbe par luminance (voir limite documentée)."
        )


def run_bayer_analysis(
    paths: List[str],
    pattern: str = 'RGGB',
    num_bins: int = 8,
    trim_low_pct: float = 0.0,
    trim_high_pct: float = 0.0,
    fit_degree: int = 1,
    reference_method: str = 'ponomarenko',
    output_prefix: str = "comparaison_bruit",
) -> Dict[str, Dict[str, object]]:
    """
    Pipeline complet de mesure de bruit **par bande couleur** pour des
    images Bayer mosaïquées (RGGB par défaut) : sépare chaque image en
    plans R, G, B (résolution moitié, G1/G2 moyennés), puis reproduit pour
    chaque bande le même pipeline que pour une image en niveaux de gris
    (`run_comparison` + `run_noise_vs_luminance` + `plot_comparison`), et
    calcule en plus un SNR (signal/bruit) par bande à partir de la
    variance donnée par `reference_method`.

    Un fichier HTML séparé est produit par bande (`{output_prefix}_R.html`,
    `_G.html`, `_B.html`), aucun bruit n'étant ajouté (mesure directe sur
    images réelles, comme le mode `--images` sans injection).

    Parameters
    ----------
    paths : List[str]
        Chemins vers les fichiers image Bayer (mosaïque brute, un seul plan).
    pattern : str
        Motif de Bayer ('RGGB', 'BGGR', 'GRBG' ou 'GBRG').
    num_bins : int
        Nombre de bins de luminance pour la courbe bruit-luminance.
    trim_low_pct, trim_high_pct : float
        Pourcentages de la plage de luminance exclus de la régression.
    fit_degree : int
        Degré du polynôme ajusté sur la méthode de référence, en plus de la droite affine.
    reference_method : str
        Méthode utilisée pour la régression bruit-luminance et pour le
        calcul du SNR (défaut : 'ponomarenko').
    output_prefix : str
        Préfixe des fichiers HTML de sortie (sans extension).

    Returns
    -------
    Dict[str, Dict[str, object]]
        Pour chaque bande ('R', 'G', 'B') : `results` (sortie de
        `run_comparison`), `luminance_curve` (sortie de
        `run_noise_vs_luminance`), `snr_linear`, `snr_db`.
    """

    bands = load_bayer_band_images(paths, pattern)
    print(
        f"Mode Bayer ({pattern}) : {len(paths)} image(s) mosaïquée(s) séparées "
        f"en bandes R/G/B (résolution moitié) -- mesure directe, aucun bruit ajouté."
    )

    summary: Dict[str, Dict[str, object]] = {}

    for band_name in ('R', 'G', 'B'):
        band_images = bands[band_name]
        print(f"\n=== Analyse du canal {band_name} ===")

        results = run_comparison(band_images)
        luminance_curve = run_noise_vs_luminance(
            band_images, num_bins=num_bins,
            trim_low_pct=trim_low_pct, trim_high_pct=trim_high_pct,
            fit_degree=fit_degree, reference_method=reference_method,
        )

        ref_variance = results.get(reference_method, {}).get('variance', float('nan'))
        snr_linear, snr_db = compute_snr(band_images, ref_variance)

        plot_comparison(
            results, true_variance=None, luminance_curve=luminance_curve,
            true_a=None, true_b=None, n_images=len(band_images),
            output_html=f"{output_prefix}_{band_name}.html",
            band_label=band_name, snr=(snr_linear, snr_db),
        )

        summary[band_name] = {
            'results': results,
            'luminance_curve': luminance_curve,
            'snr_linear': snr_linear,
            'snr_db': snr_db,
        }

    print("\nRésumé SNR par bande (méthode de référence : "
          f"'{reference_method}') :")
    print(f"{'Bande':>6s} {'SNR (linéaire)':>16s} {'SNR (dB)':>10s}")
    for band_name in ('R', 'G', 'B'):
        s_lin = summary[band_name]['snr_linear']
        s_db = summary[band_name]['snr_db']
        lin_str = f"{s_lin:.3f}" if np.isfinite(s_lin) else "N/A"
        db_str = f"{s_db:.2f}" if np.isfinite(s_db) else "N/A"
        print(f"{band_name:>6s} {lin_str:>16s} {db_str:>10s}")

    return summary


def main() -> None:
    """
    Point d'entrée : mode validation (image synthétique, bruit injecté à la
    volée) si aucune image n'est fournie ; mode mesure directe (aucune
    injection) si des images réelles sont fournies, sauf demande explicite
    de bruit supplémentaire (avec avertissement).

    Returns
    -------
    None
    """

    parser = argparse.ArgumentParser(
        description="Évalue les méthodes d'estimation de variance du bruit."
    )
    parser.add_argument(
        "--images", type=str, nargs='+', default=None,
        help="Chemin(s) vers une ou plusieurs images réelles du même système "
             "d'acquisition. Si fourni, AUCUN bruit n'est ajouté par défaut "
             "(mesure directe) -- passez --sigma2 ou --a/--b explicitement "
             "pour en injecter un par-dessus (avertissement affiché). Si "
             "absent, une image synthétique propre est générée et bruitée.",
    )
    parser.add_argument(
        "--sigma2", type=float, default=None,
        help="Variance homoscédastique à injecter. Sur image synthétique "
             "(--images absent), défaut 25.0 si ni --sigma2 ni --a/--b ne "
             "sont donnés. Sur images réelles (--images fourni), aucune "
             "injection sauf si précisé explicitement ici.",
    )
    parser.add_argument(
        "--a", type=float, default=None,
        help="Pente 'a' du bruit hétéroscédastique à injecter (voir --sigma2 "
             "pour la logique par défaut / explicite).",
    )
    parser.add_argument(
        "--b", type=float, default=None,
        help="Ordonnée à l'origine 'b' du bruit hétéroscédastique à injecter.",
    )
    parser.add_argument(
        "--bit_depth", type=int, default=None,
        help="Si fourni (ex. 8, 16), quantifie/écrête l'image bruitée sur "
             "[0, 2^bit_depth - 1] pour simuler la saturation/quantification "
             "d'un capteur réel. Ignoré si aucun bruit n'est injecté.",
    )
    parser.add_argument(
        "--bayer", action="store_true",
        help="Traite les images fournies via --images comme des mosaïques "
             "Bayer non démosaïquées (un seul plan) : chaque image est "
             "séparée en trois bandes R, G, B (résolution moitié, G1/G2 "
             "moyennés) et le pipeline complet (comparaison des méthodes + "
             "courbe bruit-luminance + SNR) est exécuté séparément pour "
             "chaque bande. Nécessite --images ; aucun bruit n'est ajouté.",
    )
    parser.add_argument(
        "--bayer_pattern", type=str, default="RGGB",
        choices=["RGGB", "BGGR", "GRBG", "GBRG"],
        help="Motif de la mosaïque Bayer (défaut : RGGB). Ignoré si --bayer "
             "n'est pas fourni.",
    )
    parser.add_argument(
        "--num_bins", type=int, default=8,
        help="Nombre de bins de luminance pour la méthode ponomarenko (défaut : 8).",
    )
    parser.add_argument(
        "--trim_low_pct", type=float, default=0.0,
        help="Pourcentage de la plage de luminance à exclure en bas de la "
             "régression affine (défaut : 0, aucune exclusion).",
    )
    parser.add_argument(
        "--trim_high_pct", type=float, default=0.0,
        help="Pourcentage de la plage de luminance à exclure en haut de la "
             "régression affine (défaut : 0, aucune exclusion).",
    )
    parser.add_argument(
        "--fit_degree", type=int, default=1,
        help="Degré du polynôme ajusté sur la méthode de référence (ponomarenko) "
             "pour la courbe bruit-luminance : 1 = affine (défaut), 2 = quadratique, "
             "3 = cubique, etc. La droite affine (degré 1) est toujours tracée en "
             "plus si un degré supérieur est demandé.",
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="Graine aléatoire de base pour le bruit injecté (si injection).",
    )
    parser.add_argument(
        "--output", type=str, default="comparaison_bruit.html",
        help="Chemin du fichier HTML de sortie pour le graphique plotly.",
    )
    args = parser.parse_args()

    user_gave_noise_params = (
        args.sigma2 is not None or args.a is not None or args.b is not None
    )

    if args.bayer:
        if not args.images:
            parser.error("--bayer nécessite --images (mesure directe sur images Bayer réelles).")
        if user_gave_noise_params:
            print(
                "ATTENTION : --sigma2/--a/--b sont ignorés en mode --bayer "
                "(injection de bruit non supportée par bande pour le moment) ; "
                "mesure directe du bruit déjà présent dans chaque bande."
            )
        output_prefix = args.output[:-5] if args.output.endswith(".html") else args.output
        run_bayer_analysis(
            args.images, pattern=args.bayer_pattern, num_bins=args.num_bins,
            trim_low_pct=args.trim_low_pct, trim_high_pct=args.trim_high_pct,
            fit_degree=args.fit_degree, output_prefix=output_prefix,
        )
        return

    if args.images:
        # --- Mode images réelles ---
        images = load_real_images(args.images)

        if user_gave_noise_params:
            a = args.a if args.a is not None else 0.0
            b = args.b if args.b is not None else (args.sigma2 if args.sigma2 is not None else 0.0)
            true_a, true_b = a, b
            true_variance = b if a == 0.0 else None
            print(
                "ATTENTION : vous injectez un bruit supplémentaire (a="
                f"{a}, b={b}) sur des images réelles, qui contiennent très "
                "probablement déjà du bruit capteur inconnu. La variance "
                "mesurée ci-dessous inclut ce bruit préexistant EN PLUS du "
                "bruit injecté -- ne la comparez pas telle quelle à a/b si "
                "vous cherchez une validation de précision absolue."
            )
            noisy_images = [
                add_noise(img, a=a, b=b, seed=args.seed + i, bit_depth=args.bit_depth)
                for i, img in enumerate(images)
            ]
        else:
            # Aucun bruit demandé : mesure directe, rien n'est ajouté.
            true_a, true_b, true_variance = None, None, None
            noisy_images = images
            print(
                f"Mesure directe sur {len(images)} image(s) réelle(s) fournie(s) "
                "-- aucun bruit n'a été ajouté."
            )

        results = run_comparison(noisy_images)
        luminance_curve = run_noise_vs_luminance(
            noisy_images, num_bins=args.num_bins,
            trim_low_pct=args.trim_low_pct, trim_high_pct=args.trim_high_pct,
            fit_degree=args.fit_degree,
        )
        plot_comparison(
            results, true_variance=true_variance, luminance_curve=luminance_curve,
            true_a=true_a, true_b=true_b, n_images=len(noisy_images), output_html=args.output,
        )

    else:
        # --- Mode validation synthétique : injection à la volée, sans écriture disque ---
        clean_image = generate_clean_image()
        a = args.a if args.a is not None else 0.08
        b = args.b if args.b is not None else 5.0
        sigma2 = args.sigma2 if args.sigma2 is not None else 25.0
        true_a, true_b = a, b

        # Protocole hétéroscédastique (courbe bruit-luminance)
        noisy_image_hetero = add_noise(clean_image, a=a, b=b, seed=args.seed, bit_depth=args.bit_depth)
        # Protocole homoscédastique dédié (panneau de comparaison globale, variance constante)
        noisy_image_homo = add_noise(clean_image, a=0.0, b=sigma2, seed=args.seed, bit_depth=args.bit_depth)

        results = run_comparison([noisy_image_homo])
        luminance_curve = run_noise_vs_luminance(
            [noisy_image_hetero], num_bins=args.num_bins,
            trim_low_pct=args.trim_low_pct, trim_high_pct=args.trim_high_pct,
            fit_degree=args.fit_degree,
        )
        plot_comparison(
            results, true_variance=sigma2, luminance_curve=luminance_curve,
            true_a=true_a, true_b=true_b, n_images=1, output_html=args.output,
        )


if __name__ == '__main__':
    main()

# EOF
