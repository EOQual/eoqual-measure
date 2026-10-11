#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
Mesure de la variance du bruit dans une image 2D.

Ce module rassemble plusieurs estimateurs de la variance du bruit dans des
images (notamment de télédétection) : approche par patchs plats, ondelettes,
``skimage.restoration``, Ponomarenko (bruit dépendant du signal), segmentation
adaptative (superpixels) et estimation robuste par PCA sur patchs.
"""
__all__ = ['NoiseVariance']  # Liste des objets exportés par le module

# IMPORTS STANDARD
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

# IMPORTS TIERS
import numpy as np
import pywt
import skimage.restoration
import skimage.segmentation
from numpy.lib.stride_tricks import sliding_window_view
from numpy.typing import NDArray
from scipy.fft import dctn

# IMPORTS LOCAUX
from eoqual.core.detrend import fit_plane_residual
from eoqual.core.tiling import grid_slices


class NoiseVariance:
    """
    Ensemble de méthodes pour mesurer la variance du bruit dans une image 2D.
    """

    # Liste des méthodes disponibles
    NoiseMeasurementMethods = Literal[
        'flat_patches',
        'wavelets',
        'skimage_estimate_sigma',
        'ponomarenko',
        'adaptive_segmentation',
        'pca_estimate',
    ]

    # Définition des paramètres requis pour chaque méthode de mesure
    # Utiliser 'str' comme clé pour que l'annotation soit compatible avec Dict[str, Any]
    _REQUIRED_PARAMS: Dict[str, Dict[str, Union[str, float]]] = {
        'flat_patches': {
            'patch_size': "int (taille des imagettes carrées, défaut : 8)",
            'percentile': "float (pourcentage des patches les plus plats à utiliser, défaut : 10%)",
            'detrend': "bool (soustraire un plan par moindres carrés avant calcul de variance, défaut : True)",
            'flatness_criterion': "str ('gradient' ou 'variance', critère de sélection des patches plats, défaut : 'gradient')",
        },
        'wavelets': {
            'n_tiles_per_side': "int (nombre de tuiles par côté pour estimer une dispersion intra-image, défaut : 3)",
        },
        'skimage_estimate_sigma': {
            'n_tiles_per_side': "int (nombre de tuiles par côté pour estimer une dispersion intra-image, défaut : 3)",
        },
        'ponomarenko': {
            'num_bins': "int (nombre de bins de luminance, défaut : 1)",
            'block_size': "int (taille des imagettes carrées, défaut : 8)",
            'overlap': "int (pas en pixels entre le coin de deux imagettes successives, défaut : 8, "
                       "= block_size équivaut à un pavage sans recouvrement)",
            'percentil': "float (fraction des imagettes les moins texturées gardées par bin, défaut : 0.05)",
        },
        'adaptive_segmentation': {
            'n_segments': "int (nombre de superpixels visés par SLIC, défaut : 200 ; ignoré si target_zone_size est fourni)",
            'target_zone_size': "int ou None (taille de zone visée en pixels ; si fourni, calcule n_segments = aire_image / target_zone_size, défaut : None)",
            'compactness': "float (compacité SLIC, plus petit = épouse mieux les contours radiométriques, défaut : 0.1)",
            'detrend': "bool (soustraire un plan par moindres carrés par segment, défaut : True)",
            'min_segment_size': "int (nombre de pixels minimum pour garder un segment, défaut : 30)",
            'homogeneity_percentile': "float (pourcentage des segments les plus homogènes conservés, défaut : 20%)",
        },
        'pca_estimate': {
            'patch_size': "int (taille des patchs carrés utilisés pour la covariance, défaut : 7)",
            'stride': "int (pas d'échantillonnage des patchs, défaut : 2)",
            'percentile': "float (pourcentage des patchs les moins texturés conservés pour la covariance, défaut : 20%)",
            'detrend': "bool (soustraire un plan par moindres carrés à chaque patch avant covariance, défaut : True)",
            'calibrate_bias': "bool (corriger le biais d'échantillonnage fini par auto-étalonnage Monte-Carlo, défaut : True)",
            'calibration_seed': "int (graine du tirage aléatoire d'étalonnage, défaut : 0)",
            'n_bootstrap': "int (nombre de ré-échantillonnages bootstrap intra-image pour la dispersion, défaut : 20)",
            'bootstrap_seed': "int (graine du tirage aléatoire du bootstrap, défaut : 1)",
        },
    }

    def __init__(self, image: Optional[NDArray] = None):
        # `image` est conservé pour compatibilité ascendante : l'image est
        # passée à `estimate_noise`, pas au constructeur.

        # Mappage des fonctions de bruit
        self._estimate_noise_functions: Dict[
            str, Callable[[NDArray[np.float32], Optional[dict]], Tuple[Any, Any, Any, Any]]
        ] = {
            'flat_patches': self._estimate_noise_var_flat_patches,
            'wavelets': self._estimate_noise_var_wavelets,
            'skimage_estimate_sigma': self._estimate_noise_var_skimage,
            'ponomarenko': self._estimate_noise_ponomarenko,
            'adaptive_segmentation': self._estimate_noise_var_segmentation,
            'pca_estimate': self._estimate_noise_var_pca,
        }

## ----------------------------------------------------------------------
# Méthode d'Information
# -----------------------------------------------------------------------

    def get_available_methods(self) -> Dict[str, Any]:
        """
        Retourne la liste des méthodes de mesure disponibles
        et leur paramétrisation requise/recommandée.

        Returns
        -------
        Dict[str, Any]
            Dictionnaire des méthodes disponibles et de leurs paramètres.
        """
        return self._REQUIRED_PARAMS

## ----------------------------------------------------------------------
# Outils internes communs
# -----------------------------------------------------------------------

    @staticmethod
    def _fit_plane_residual(values: NDArray[np.float32], coords: NDArray[np.float64]) -> NDArray[np.float32]:
        """
        Ajuste un plan (régression linéaire) par moindres carrés à un jeu de
        valeurs et retourne le résidu (valeurs moins plan ajusté).

        Utile pour retirer une tendance radiométrique basse fréquence
        (relief, éclairement, BRDF) d'une zone supposée homogène avant de
        calculer la variance du bruit, afin de ne pas confondre signal et
        bruit.

        Parameters
        ----------
        values : NDArray[np.float32]
            Valeurs radiométriques 1D (patch ou segment aplati).
        coords : NDArray[np.float64]
            Coordonnées (x, y) associées à chaque valeur, de forme (N, 2).

        Returns
        -------
        NDArray[np.float32]
            Résidu après soustraction du plan ajusté, même forme que `values`.
        """
        return fit_plane_residual(values, coords)

    @staticmethod
    def _tile_samples(
        image: NDArray[np.float32],
        estimator_fn: Callable[[NDArray[np.float32]], float],
        n_tiles_per_side: int = 3,
        min_tile_size: int = 32,
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Découpe l'image en une grille de tuiles non chevauchantes et applique
        un estimateur scalaire à chacune, pour obtenir une distribution de
        mesures répétées au sein d'une même image (utile aux méthodes
        globales comme `wavelets`/`skimage_estimate_sigma`, qui ne
        produisent naturellement qu'une seule valeur pour toute l'image).

        Parameters
        ----------
        image : NDArray[np.float32]
            L'image 2D d'entrée.
        estimator_fn : Callable[[NDArray[np.float32]], float]
            Fonction appliquée à chaque tuile, retournant un scalaire (ex.
            variance du bruit estimée sur la tuile).
        n_tiles_per_side : int
            Nombre de tuiles par côté (grille n_tiles_per_side x n_tiles_per_side).
        min_tile_size : int
            Taille minimale (px) d'une tuile en dessous de laquelle le
            découpage est abandonné (retourne des tableaux vides).

        Returns
        -------
        Tuple[NDArray[np.float64], NDArray[np.float64]]
            Tableau des estimations par tuile, et tableau de la luminance
            moyenne de chaque tuile (même ordre) — permet de construire une
            courbe bruit-luminance même pour les méthodes qui ne produisent
            naturellement qu'une valeur globale.
        """
        h, w = image.shape
        th, tw = h // n_tiles_per_side, w // n_tiles_per_side
        if th < min_tile_size or tw < min_tile_size:
            return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

        tiles = [image[rows, cols] for rows, cols in grid_slices(image.shape, n_tiles_per_side)]
        samples = [estimator_fn(tile) for tile in tiles]
        luminances = [float(np.nanmean(tile)) for tile in tiles]
        return np.array(samples, dtype=np.float64), np.array(luminances, dtype=np.float64)

    @staticmethod
    def _split_by_luminance(means: NDArray[np.float64], num_bins: int) -> List[NDArray[np.intp]]:
        """
        Répartit des blocs en `num_bins` classes de luminance d'effectifs
        égaux (quantiles de la luminance moyenne des blocs).

        Des classes à effectif égal plutôt qu'à largeur égale garantissent
        à chaque classe le même nombre de blocs candidats, donc une
        précision comparable d'une classe à l'autre -- choix habituel des
        estimateurs de courbe bruit-luminance par blocs (voir Colom &
        Buades, IPOL 2013, pour la variante de Ponomarenko).

        Returns
        -------
        List[NDArray[np.intp]]
            Indices des blocs de chaque classe, par luminance croissante.
        """
        if num_bins < 1:
            raise ValueError("'num_bins' must be >= 1.")
        order = np.argsort(means, kind='stable')
        return np.array_split(order, num_bins)

## ----------------------------------------------------------------------
# Méthodes d'estimation du bruit internes
# -----------------------------------------------------------------------

    def _estimate_noise_var_flat_patches(
        self, image: NDArray[np.float32], params: Optional[dict]
    ) -> Tuple[float, float, NDArray[np.float64]]:
        """
        Estime la variance du bruit en analysant les patches plats de l'image.

        Le critère de sélection des patches "plats" (défaut : norme du
        gradient local) est distinct de la grandeur utilisée pour calculer la
        variance (résidu après soustraction d'un plan ajusté). Cette
        séparation limite le biais négatif que produirait un tri direct sur
        la variance brute (les patches sélectionnés seraient alors ceux dont
        le bruit s'est annulé par hasard, biaisant l'estimation vers le bas).

        Parameters
        ----------
        image : NDArray[np.float32]
            L'image 2D d'entrée (h, w).
        params : dict ou None
            Dictionnaire des paramètres (voir `_REQUIRED_PARAMS['flat_patches']`).

        Returns
        -------
        Tuple[float, float, NDArray[np.float64], NDArray[np.float64]]
            Variance du bruit, moyenne du signal, tableau des variances
            résiduelles individuelles des patches retenus (barre d'erreur),
            et tableau de la luminance moyenne de chacun de ces patches
            (même ordre) — permet de tracer une courbe bruit-luminance pour
            cette méthode, comme pour `ponomarenko`.
        """

        if params is None:
            params = {}

        patch_size = params.get('patch_size', 8)
        percentile = params.get('percentile', 10.0)
        detrend = params.get('detrend', True)
        flatness_criterion = params.get('flatness_criterion', 'gradient')

        h, w = image.shape

        variances: List[float] = []
        flatness_scores: List[float] = []
        means: List[float] = []

        yy, xx = np.mgrid[0:patch_size, 0:patch_size]
        coords = np.column_stack([xx.ravel().astype(np.float64), yy.ravel().astype(np.float64)])

        for y in range(0, h - patch_size + 1, patch_size):
            for x in range(0, w - patch_size + 1, patch_size):
                patch = image[y:y + patch_size, x:x + patch_size]

                # Critère de platitude : calculé sur le patch brut, indépendamment
                # de la grandeur utilisée ensuite pour la variance du bruit.
                if flatness_criterion == 'gradient':
                    gy, gx = np.gradient(patch)
                    score = np.nanmean(gx ** 2 + gy ** 2)
                else:
                    score = np.nanvar(patch)
                flatness_scores.append(score)

                # Calcul de la variance sur le résidu (avec ou sans détrend).
                if detrend:
                    residual = self._fit_plane_residual(patch.ravel().astype(np.float64), coords)
                else:
                    residual = patch.ravel() - np.nanmean(patch)

                variances.append(np.nanvar(residual))
                means.append(np.nanmean(patch))

        variances_arr = np.array(variances)
        flatness_arr = np.array(flatness_scores)
        means_arr = np.array(means)

        # Sélection des patches les plus plats selon le critère de flatness
        flat_count = max(1, int(len(variances_arr) * (percentile / 100.0)))
        idx_flat = np.argsort(flatness_arr)[:flat_count]

        image_variance = float(np.mean(variances_arr[idx_flat]))
        image_mean = float(np.mean(means_arr[idx_flat]))
        var_samples = variances_arr[idx_flat]
        luminance_samples = means_arr[idx_flat]
        return (image_variance, image_mean, var_samples, luminance_samples)

    def _estimate_noise_var_wavelets(
        self, image: NDArray[np.float32], params: Optional[dict] = None
    ) -> Tuple[float, float, NDArray[np.float64], NDArray[np.float64]]:
        """
        Estime la variance du bruit via la méthode de Donoho-Johnstone (ondelettes).

        Cette méthode ne produit naturellement qu'une seule valeur pour
        toute l'image (pas de sélection de patches). Pour obtenir une
        dispersion représentative (barre d'erreur) et une courbe
        bruit-luminance comparable à celle de `ponomarenko`, le même
        estimateur est répété sur une grille de tuiles non chevauchantes au
        sein de la même image (voir `_tile_samples`), plutôt que sur des
        images différentes qui ne partagent pas la même scène/le même bruit.

        Parameters
        ----------
        image : NDArray[np.float32]
            L'image 2D d'entrée (h, w).
        params : dict ou None
            Dictionnaire des paramètres (voir `_REQUIRED_PARAMS['wavelets']`).

        Returns
        -------
        Tuple[float, float, NDArray[np.float64], NDArray[np.float64]]
            Variance du bruit, moyenne du signal, tableau des estimations
            obtenues par tuile, et luminance moyenne de chaque tuile.
        """

        if params is None:
            params = {}
        n_tiles_per_side = params.get('n_tiles_per_side', 3)

        def _single(img: NDArray[np.float32]) -> float:
            coeffs = pywt.wavedec2(img.astype(np.float32), 'db1', level=1)
            detail_coeffs = coeffs[-1][-1]
            median_abs_dev = np.median(np.abs(detail_coeffs))
            sigma = median_abs_dev / 0.6745
            return float(sigma ** 2)

        image_variance = _single(image)
        image_mean = float(np.nanmean(image))
        var_samples, luminance_samples = self._tile_samples(image, _single, n_tiles_per_side=n_tiles_per_side)

        return (image_variance, image_mean, var_samples, luminance_samples)

    def _estimate_noise_var_skimage(
        self, image: NDArray[np.float32], params: Optional[dict] = None
    ) -> Tuple[float, float, NDArray[np.float64], NDArray[np.float64]]:
        """
        Estime la variance du bruit gaussien à partir de la méthode
        `estimate_sigma` de `skimage.restoration` (méthode Donoho-Johnstone).

        Comme pour `wavelets`, la dispersion (barre d'erreur) est obtenue en
        répétant l'estimateur sur une grille de tuiles de la même image
        plutôt qu'en comparant des images différentes.

        Parameters
        ----------
        image : NDArray[np.float32]
            L'image 2D d'entrée (h, w).
        params : dict ou None
            Dictionnaire des paramètres (voir `_REQUIRED_PARAMS['skimage_estimate_sigma']`).

        Returns
        -------
        Tuple[float, float, NDArray[np.float64], NDArray[np.float64]]
            Variance du bruit, moyenne du signal, tableau des estimations
            obtenues par tuile, et luminance moyenne de chaque tuile.
        """

        if params is None:
            params = {}
        n_tiles_per_side = params.get('n_tiles_per_side', 3)

        def _single(img: NDArray[np.float32]) -> float:
            sigma = skimage.restoration.estimate_sigma(img)
            return float(sigma ** 2)

        image_variance = _single(image)
        image_mean = float(np.nanmean(image))
        var_samples, luminance_samples = self._tile_samples(image, _single, n_tiles_per_side=n_tiles_per_side)

        return (image_variance, image_mean, var_samples, luminance_samples)

    def _estimate_noise_ponomarenko(
        self, image: NDArray[np.float32], params: Optional[dict]
    ) -> Tuple[List[float], List[float], List[NDArray[np.float64]]]:
        """
        Estime la variance du bruit par la méthode de Ponomarenko, en
        analysant l'énergie hautes fréquences (DCT) de blocs peu texturés,
        éventuellement par classe ("bin") de luminance pour caractériser un
        bruit dépendant du signal (Poisson + lecture).

        Structure de la méthode, d'après la littérature :

        - Ponomarenko et al. (2007) : DCT 2D de blocs w x w ; le bruit
          blanc se répartit uniformément sur tous les coefficients, alors
          que le signal d'une image naturelle se concentre dans les basses
          fréquences. L'énergie basse fréquence d'un bloc (hors continu)
          mesure donc sa texture, et les coefficients hautes fréquences
          des blocs les moins texturés ne portent presque que du bruit.
        - Colom & Buades (2013) : partition du plan DCT par la diagonale
          ``i + j`` (basses fréquences ``0 < i + j <= T``, hautes
          fréquences ``i + j > T``, ici ``T = w``), sélection d'une
          fraction ``p`` des blocs de plus faible énergie basse fréquence,
          et répétition par classe de luminance d'effectifs égaux pour
          obtenir une courbe de bruit dépendant du signal.

        Variante propre à cette implémentation : l'agrégation (moyenne
        par bloc, puis médiane sur les blocs retenus), justifiée
        ci-dessous.

        Références
        ----------
        N. Ponomarenko, V. Lukin, M. Zriakhov, A. Kaarna, J. Astola, "An
        automatic approach to lossy compression of AVIRIS images", Proc.
        IEEE IGARSS, 2007.
        M. Colom, A. Buades, "Analysis and Extension of the Ponomarenko et
        al. Method, Estimating a Noise Curve from a Single Image", Image
        Processing On Line, 3, pp. 173-197, 2013.

        Pour chaque bin, une estimation de variance est calculée
        individuellement sur **chacun des K blocs retenus** (les plus plats
        au sens de l'énergie basses fréquences), avant agrégation (médiane)
        en une valeur unique par bin. Ceci fournit naturellement une
        distribution d'estimations à luminance comparable — plusieurs
        blocs différents de la même image, à un niveau de luminance donné —
        dont la dispersion (écart-type) sert de barre d'erreur pour ce bin,
        conformément à l'idée que la variabilité pertinente est celle entre
        zones homogènes de même luminance au sein de l'image, pas celle
        entre images/scènes différentes.

        Parameters
        ----------
        image : NDArray[np.float32]
            L'image 2D d'entrée (h, w).
        params : dict ou None
            Dictionnaire des paramètres (voir `_REQUIRED_PARAMS['ponomarenko']`).

        Returns
        -------
        Tuple[List[float], List[float], List[NDArray[np.float64]], List[NDArray[np.float64]]]
            Liste des variances du bruit et des moyennes de signal (une
            valeur par bin de luminance), liste des tableaux d'estimations
            individuelles par bloc (une array par bin), et liste des
            tableaux de luminance moyenne de chacun de ces blocs (même
            ordre, même longueur — un couple (variance, luminance) par bloc).
        """

        if params is None:
            params = {}

        num_bins = params.get('num_bins', 1)  # 1 = une seule variance pour toute l'image
        block_size = params.get('block_size', 8)
        step = params.get('overlap', 8)  # pas entre blocs (= block_size : pavage)
        percentil = params.get('percentil', .05)

        # Blocs (fenêtre glissante de pas `step`) et leur DCT-II 2D orthonormée.
        windows = sliding_window_view(image.astype(np.float64), (block_size, block_size))
        blocks = windows[::step, ::step].reshape(-1, block_size, block_size)
        coeffs = dctn(blocks, axes=(1, 2), norm='ortho')
        energy = coeffs ** 2

        # Partition du plan fréquentiel DCT : basses fréquences (hors DC) pour
        # le critère de texture, hautes fréquences pour la variance du bruit.
        ii, jj = np.indices((block_size, block_size))
        low_mask = (ii + jj <= block_size) & ((ii + jj) > 0)
        high_mask = (ii + jj) > block_size

        texture = energy[:, low_mask].mean(axis=1)            # V_low par bloc
        per_block_sigma2_all = energy[:, high_mask].mean(axis=1)
        block_means = blocks.mean(axis=(1, 2))

        image_variance: List[float] = []
        image_mean: List[float] = []
        per_bin_var_samples: List[NDArray[np.float64]] = []
        per_bin_luminance_samples: List[NDArray[np.float64]] = []

        for bin_indices in self._split_by_luminance(block_means, num_bins):
            if bin_indices.size == 0:
                image_variance.append(np.nan)
                image_mean.append(np.nan)
                per_bin_var_samples.append(np.array([], dtype=np.float64))
                per_bin_luminance_samples.append(np.array([], dtype=np.float64))
                continue

            # Les K blocs les moins texturés du bin (K >= 1 : un bin peu
            # peuplé ne doit pas provoquer de division par zéro).
            k = max(1, int(percentil * bin_indices.size))
            selected = bin_indices[np.argsort(texture[bin_indices], kind='stable')[:k]]

            # Estimation par bloc = MOYENNE de l'énergie haute fréquence (la
            # médiane des coefficients au carré d'un seul bloc serait biaisée
            # vers le bas : loi du chi² à 1 ddl, médiane ~0.45x la moyenne).
            # La médiane robuste n'agrège ensuite que les K estimations.
            per_block_sigma2 = per_block_sigma2_all[selected]
            image_variance.append(float(np.median(per_block_sigma2)))
            image_mean.append(float(np.mean(block_means[selected])))
            per_bin_var_samples.append(per_block_sigma2)
            per_bin_luminance_samples.append(block_means[selected])

        return (image_variance, image_mean, per_bin_var_samples, per_bin_luminance_samples)

    def _estimate_noise_var_segmentation(
        self, image: NDArray[np.float32], params: Optional[dict]
    ) -> Tuple[float, float, NDArray[np.float64]]:
        """
        Estime la variance du bruit par segmentation adaptative en superpixels
        (SLIC). Contrairement à un pavage régulier, la segmentation adaptative
        épouse les contours radiométriques réels de la scène : elle limite le
        mélange de plusieurs objets/matériaux dans une même zone d'analyse, ce
        qui est particulièrement utile en résolution métrique où une grille
        fixe de type 8x8 px peut recouvrir plusieurs structures au sol.

        Un plan est ajusté et soustrait dans chaque segment (option
        `detrend`) afin de ne pas confondre un gradient radiométrique
        résiduel (relief, éclairement, BRDF) avec du bruit. Seuls les
        segments les plus homogènes (variance résiduelle la plus faible) sont
        conservés pour le calcul final.

        Parameters
        ----------
        image : NDArray[np.float32]
            L'image 2D d'entrée (h, w).
        params : dict ou None
            Dictionnaire des paramètres (voir `_REQUIRED_PARAMS['adaptive_segmentation']`).

        Returns
        -------
        Tuple[float, float, NDArray[np.float64], NDArray[np.float64]]
            Variance du bruit, moyenne du signal, tableau des variances
            résiduelles individuelles des segments retenus (barre d'erreur),
            et tableau de la luminance moyenne de chacun de ces segments
            (même ordre).
        """

        if params is None:
            params = {}

        n_segments = params.get('n_segments', 200)
        target_zone_size = params.get('target_zone_size', None)
        compactness = params.get('compactness', 0.1)
        detrend = params.get('detrend', True)
        min_segment_size = params.get('min_segment_size', 30)
        homogeneity_percentile = params.get('homogeneity_percentile', 20.0)

        if target_zone_size is not None:
            # Taille de zone visée (en pixels) -> nombre de superpixels équivalent.
            n_segments = max(1, int((image.shape[0] * image.shape[1]) / target_zone_size))

        image = image.astype(np.float64)
        img_min, img_max = np.nanmin(image), np.nanmax(image)
        if img_max > img_min:
            img_norm = (image - img_min) / (img_max - img_min)
        else:
            img_norm = np.zeros_like(image)

        segments = skimage.segmentation.slic(
            img_norm,
            n_segments=n_segments,
            compactness=compactness,
            channel_axis=None,
            start_label=0,
        )

        variances: List[float] = []
        means: List[float] = []

        for label in np.unique(segments):
            mask = segments == label
            if np.sum(mask) < min_segment_size:
                continue

            ys, xs = np.where(mask)
            region = image[mask]

            if detrend:
                coords = np.column_stack([xs.astype(np.float64), ys.astype(np.float64)])
                residual = self._fit_plane_residual(region, coords)
            else:
                residual = region - np.nanmean(region)

            variances.append(float(np.nanvar(residual)))
            means.append(float(np.nanmean(region)))

        if len(variances) == 0:
            return (float('nan'), float('nan'), np.array([], dtype=np.float64), np.array([], dtype=np.float64))

        variances_arr = np.array(variances)
        means_arr = np.array(means)

        # Conserve uniquement les segments les plus homogènes (variance résiduelle
        # la plus faible), afin d'écarter les segments encore contaminés par du
        # signal (bord d'objet mal capturé par SLIC, texture fine, etc.).
        n_keep = max(1, int(len(variances_arr) * (homogeneity_percentile / 100.0)))
        idx_keep = np.argsort(variances_arr)[:n_keep]

        image_variance = float(np.mean(variances_arr[idx_keep]))
        image_mean = float(np.mean(means_arr[idx_keep]))
        var_samples = variances_arr[idx_keep]
        luminance_samples = means_arr[idx_keep]
        return (image_variance, image_mean, var_samples, luminance_samples)

    def _estimate_noise_var_pca(
        self, image: NDArray[np.float32], params: Optional[dict]
    ) -> Tuple[float, float, NDArray[np.float64], NDArray[np.float64]]:
        """
        Estime la variance du bruit par une approche robuste basée sur
        l'analyse en composantes principales (PCA) de patchs, inspirée des
        travaux de Pyatykh et al. et Chen et al. sur l'estimation du niveau
        de bruit par patchs faiblement texturés.

        Principe : dans une zone homogène, la matrice de covariance des
        patchs est dominée par le bruit ; ses valeurs propres les plus
        faibles convergent alors vers la variance du bruit.

        Sélection des patchs : le sous-ensemble de patchs "faiblement
        texturés" utilisé pour la covariance est choisi **une seule fois**,
        selon un critère de texture (norme du gradient local, comme pour
        `flat_patches`) totalement **indépendant** de la variance ensuite
        calculée. Une première version de cette méthode sélectionnait les
        patchs par un seuil sur leur propre variance, réévalué à chaque
        itération à partir de l'estimation courante du bruit : ce schéma
        auto-référencé réintroduit le biais de tri décrit pour
        `flat_patches`, et l'amplifie à chaque itération (chaque tour ne
        retient que les patchs dont le bruit s'est annulé par hasard, ce qui
        abaisse l'estimation, qui abaisse à son tour le seuil du tour
        suivant). Vérifié empiriquement sur une image de bruit blanc pur
        (variance réelle connue) : l'ancienne version dérivait de -6 % après
        1 itération à -70 % après 5 itérations. La sélection en un seul
        passage, sur un critère indépendant, élimine cette dérive.

        Correction de biais (auto-étalonnage) : même sans dérive itérative,
        prendre la médiane de la moitié inférieure des valeurs propres d'une
        covariance empirique sous-estime systématiquement la vraie variance
        (étalement des valeurs propres en échantillon fini, phénomène
        apparenté à la loi de Marchenko-Pastur). Ce biais ne dépend, pour un
        bruit blanc, que des dimensions (d = taille du patch au carré, N =
        nombre de patchs retenus). Il est corrigé ici par simulation : un
        jeu de N vecteurs gaussiens i.i.d. de dimension d et de variance
        unité est généré, le même estimateur (médiane de la moitié
        inférieure des valeurs propres) y est appliqué, et le facteur
        correctif (1 / statistique obtenue) est appliqué à l'estimation
        brute. Sans cette correction, l'estimation restait biaisée d'environ
        -35 % même sur une image de bruit blanc pur, pourtant le cas le plus
        favorable possible.

        Extraction et détrend entièrement vectorisés (`sliding_window_view`
        et régression par patch en une seule multiplication matricielle) :
        la version initiale, avec boucle Python par patch, était nettement
        plus lente que les autres méthodes du module sans bénéfice de
        précision correspondant.

        Limite : contrairement aux autres méthodes, `pca_estimate` n'a pas
        de notion naturelle de "variance par patch individuel" (l'estimateur
        repose sur la covariance de l'ensemble des patchs retenus, pas sur
        une moyenne de variances individuelles) — les échantillons bootstrap
        retournés ici reflètent donc l'incertitude de l'estimateur global,
        pas une variabilité "par luminance". Une courbe bruit-luminance
        n'est donc pas disponible pour cette méthode ; elle apparaît comme
        un point unique (à la luminance moyenne des patchs retenus) plutôt
        que comme une courbe, si elle est superposée à celle des autres
        méthodes.

        Chaque patch retenu est détrendé (plan ajusté par moindres carrés
        soustrait) avant le calcul de la covariance : sans cela, un gradient
        radiométrique local (relief, éclairement) à l'intérieur d'un patch
        se répartirait sur toutes les directions de la covariance, y compris
        les valeurs propres les plus faibles, et biaiserait l'estimation de
        la variance du bruit vers le haut.

        Parameters
        ----------
        image : NDArray[np.float32]
            L'image 2D d'entrée (h, w).
        params : dict ou None
            Dictionnaire des paramètres (voir `_REQUIRED_PARAMS['pca_estimate']`).

        Returns
        -------
        Tuple[float, float, NDArray[np.float64]]
            Variance du bruit, moyenne du signal (sur les patchs retenus), et
            tableau des estimations obtenues par ré-échantillonnage bootstrap
            (avec remise) du sous-ensemble de patchs retenus — dispersion
            intra-image de l'estimateur, pas de comparaison entre images.
        """

        if params is None:
            params = {}

        patch_size = params.get('patch_size', 7)
        stride = params.get('stride', 2)
        percentile = params.get('percentile', 20.0)
        detrend = params.get('detrend', True)
        calibrate_bias = params.get('calibrate_bias', True)
        calibration_seed = params.get('calibration_seed', 0)
        n_bootstrap = params.get('n_bootstrap', 20)
        bootstrap_seed = params.get('bootstrap_seed', 1)

        d = patch_size * patch_size

        # --- Extraction vectorisée des patchs (fenêtre glissante + sous-échantillonnage) ---
        windows = sliding_window_view(image.astype(np.float64), (patch_size, patch_size))
        windows = windows[::stride, ::stride]  # (ny, nx, patch_size, patch_size)
        X_raw = windows.reshape(-1, d)  # (N, d)
        raw_means_arr = X_raw.mean(axis=1)

        # --- Critère de texture vectorisé : énergie de gradient local par patch,
        # calculée sur le patch brut (avant détrend), indépendante de la variance. ---
        dy = np.diff(windows, axis=2)
        dx = np.diff(windows, axis=3)
        texture_scores_arr = (
            np.mean(dy ** 2, axis=(2, 3)) + np.mean(dx ** 2, axis=(2, 3))
        ).reshape(-1)

        # --- Détrend vectorisé (plan par moindres carrés, même base pour tous les patchs) ---
        if detrend:
            yy, xx = np.mgrid[0:patch_size, 0:patch_size]
            A = np.column_stack([
                xx.ravel().astype(np.float64), yy.ravel().astype(np.float64), np.ones(d)
            ])
            A_pinv = np.linalg.pinv(A)  # (3, d)
            coeffs = X_raw @ A_pinv.T  # (N, 3)
            X = X_raw - coeffs @ A.T  # résidus (N, d)
        else:
            X = X_raw

        # Nombre de patchs à conserver : au moins 2*d pour une covariance stable,
        # sinon le pourcentage demandé si celui-ci en fournit davantage.
        n_keep = max(2 * d, int(len(X) * (percentile / 100.0)))
        n_keep = min(n_keep, len(X))
        if n_keep < d:
            raise ValueError(
                "Pas assez de patchs extraits pour estimer une covariance stable : "
                "réduire patch_size/stride, augmenter percentile, ou fournir une image plus grande."
            )

        idx_keep = np.argsort(texture_scores_arr)[:n_keep]
        Xs = X[idx_keep]

        mean_vec = Xs.mean(axis=0)
        Xc = Xs - mean_vec
        cov = (Xc.T @ Xc) / len(Xs)

        eigvals = np.linalg.eigvalsh(cov)  # ordre croissant
        # Médiane (plus robuste que la moyenne à quelques valeurs propres
        # encore contaminées par de la texture résiduelle) de la moitié
        # inférieure des valeurs propres : dans une zone homogène, ces
        # directions ne portent que du bruit.
        sigma2_raw = float(np.median(eigvals[: d // 2]))

        if calibrate_bias:
            rng = np.random.default_rng(calibration_seed)
            calib = rng.normal(0.0, 1.0, size=(len(Xs), d))
            calib_c = calib - calib.mean(axis=0)
            calib_cov = (calib_c.T @ calib_c) / len(calib)
            calib_eigvals = np.linalg.eigvalsh(calib_cov)
            calib_stat = float(np.median(calib_eigvals[: d // 2]))
            correction_factor = 1.0 / calib_stat if calib_stat > 0 else 1.0
            sigma2 = sigma2_raw * correction_factor
        else:
            correction_factor = 1.0
            sigma2 = sigma2_raw

        mean_signal = float(np.mean(raw_means_arr[idx_keep]))

        # Bootstrap intra-image : ré-échantillonnage avec remise du sous-ensemble
        # de patchs déjà retenu (même image, même sélection de texture), pour
        # obtenir une dispersion de l'estimateur sans comparer des images
        # différentes entre elles.
        rng_b = np.random.default_rng(bootstrap_seed)
        n_sel = len(Xs)
        samples = np.empty(n_bootstrap, dtype=np.float64)
        for b in range(n_bootstrap):
            idx_b = rng_b.integers(0, n_sel, size=n_sel)
            Xb = Xs[idx_b]
            mean_b = Xb.mean(axis=0)
            Xcb = Xb - mean_b
            cov_b = (Xcb.T @ Xcb) / n_sel
            eig_b = np.linalg.eigvalsh(cov_b)
            stat_b = float(np.median(eig_b[: d // 2]))
            samples[b] = stat_b * correction_factor

        # Un seul point de luminance (la moyenne des patchs retenus), répété pour
        # chaque échantillon bootstrap -- pas une vraie courbe par luminance (voir
        # limite documentée ci-dessus).
        luminance_samples = np.full(n_bootstrap, mean_signal, dtype=np.float64)

        return (sigma2, mean_signal, samples, luminance_samples)

## ----------------------------------------------------------------------
# Méthode Publique d'Application d'un estimateur
# -----------------------------------------------------------------------

    def estimate_noise(
        self,
        image: NDArray[np.float32],
        estimator: NoiseMeasurementMethods,
        params: Optional[dict] = None,
        return_samples: bool = False,
    ) -> Union[
        Tuple[Union[float, List[float]], Union[float, List[float]]],
        Tuple[Union[float, List[float]], Union[float, List[float]], Any, Any],
    ]:
        """
        Applique la méthode d'estimation de bruit spécifiée à l'image.

        Parameters
        ----------
        image : NDArray[np.float32]
            L'image 2D d'entrée (H, W).
        estimator : {'flat_patches', 'wavelets', 'skimage_estimate_sigma', 'ponomarenko', 'adaptive_segmentation', 'pca_estimate'}
            Méthode d'estimation à appliquer.
        params : dict ou None
            Dictionnaire des paramètres requis pour la méthode choisie.
        return_samples : bool
            Si True, retourne en plus (1) les mesures élémentaires de
            variance (patchs, blocs ou segments individuels, ou répétitions
            par tuile/bootstrap selon la méthode) ayant servi à construire
            l'estimation finale, et (2) la luminance moyenne associée à
            chaque échantillon (même ordre), pour tracer une courbe
            bruit-luminance par méthode. Ces échantillons, obtenus au sein
            d'une seule et même image, permettent aussi de construire une
            barre d'erreur représentative de la dispersion réelle de la
            mesure — contrairement à une comparaison entre plusieurs images
            de scènes différentes, qui ne mesure pas la même grandeur. Pour
            `ponomarenko` avec `num_bins > 1`, une array est retournée par
            bin de luminance (liste de listes). Pour `pca_estimate`, la
            luminance retournée est une valeur unique répétée (voir limite
            documentée dans `_estimate_noise_var_pca`).

        Returns
        -------
        Tuple
            Sans `return_samples` : couple (variance du bruit, moyenne du
            signal), scalaires pour toute l'image ou listes par bin/segment
            selon la méthode. Avec `return_samples=True` : quadruplet
            incluant en 3e position les échantillons de variance et en 4e
            position la luminance associée à chacun.
        """

        # Vérification de estimator
        if estimator not in self._estimate_noise_functions:
            raise ValueError(
                f"Methode non supporté: '{estimator}'."
            )

        # Appel de la fonction d'estimation du bruit appropriée
        noise_variance, image_mean, var_samples, luminance_samples = self._estimate_noise_functions[estimator](image, params)

        if return_samples:
            return (noise_variance, image_mean, var_samples, luminance_samples)
        return (noise_variance, image_mean)


if __name__ == '__main__':
    pass

# EOF
