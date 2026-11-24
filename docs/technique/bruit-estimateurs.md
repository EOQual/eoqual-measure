# Variance du bruit : estimateurs sur scène

Module `eoqual/measure/noise/variance.py` : estimation de la variance du bruit radiométrique
dans des images 2D (télédétection métrique/infra-métrique), sur des zones
supposées homogènes détectées automatiquement selon plusieurs stratégies.
L'estimation par le variogramme, qui diagnostique sa propre fiabilité, a son
document : [variogramme](bruit-variogramme.md).

## Sommaire

- [Principe général](#principe-général)
- [Méthodes disponibles](#méthodes-disponibles)
  - [flat_patches](#flat_patches)
  - [wavelets](#wavelets)
  - [skimage_estimate_sigma](#skimage_estimate_sigma)
  - [ponomarenko](#ponomarenko)
  - [adaptive_segmentation](#adaptive_segmentation)
  - [pca_estimate](#pca_estimate)
- [Biais résiduel des méthodes par sélection de patchs](#biais-résiduel-des-méthodes-par-sélection-de-patchs)
- [Barres d'erreur et images multiples](#barres-derreur-et-images-multiples)
- [Taille des zones homogènes : un paramètre réglable](#taille-des-zones-homogènes--un-paramètre-réglable)
- [Deux modes bien séparés : validation synthétique vs mesure sur images réelles](#deux-modes-bien-séparés--validation-synthétique-vs-mesure-sur-images-réelles)
- [Non-linéarité aux extrêmes de luminance](#non-linéarité-aux-extrêmes-de-luminance)
- [Toutes les méthodes superposées sur la courbe bruit-luminance](#toutes-les-méthodes-superposées-sur-la-courbe-bruit-luminance)
- [Utilisation](#utilisation)
- [Script d'intercomparaison (`examples/noise_intercomparison.py`)](#script-dintercomparaison-examplesnoise_intercomparisonpy)
- [Mode Bayer : mesure par bande (R, G, B)](#mode-bayer--mesure-par-bande-r-g-b)
- [Recommandations de choix de méthode](#recommandations-de-choix-de-méthode)
- [Limites connues et pistes non retenues](#limites-connues-et-pistes-non-retenues)

## Principe général

Toutes les méthodes reposent sur la même idée : repérer des zones de l'image
où le signal radiométrique varie peu (pas de contours, pas de texture fine),
puis attribuer l'essentiel de la variabilité résiduelle mesurée dans ces
zones au bruit du capteur plutôt qu'au terrain.

Deux biais doivent être maîtrisés :

1. **Biais de sélection** : si le critère qui sert à choisir les zones
   "homogènes" est la variance elle-même (ex. trier des patchs par variance
   croissante et garder les plus faibles), on favorise statistiquement les
   patchs dont le bruit s'est annulé par hasard. La variance du bruit est
   alors **sous-estimée**, d'autant plus que les patchs sont petits.
2. **Confusion signal/bruit** : une zone homogène en télédétection n'est
   presque jamais parfaitement plate (relief, éclairement, BRDF). Un
   gradient basse fréquence non retiré ("détrend") gonfle artificiellement
   la variance mesurée et peut biaiser à la fois la sélection des zones et
   le calcul final.

Les méthodes ci-dessous adoptent des stratégies différentes pour limiter ces
deux biais.

## Méthodes disponibles

### `flat_patches`

Découpe l'image en patchs carrés non chevauchants, sélectionne les plus
"plats" selon un critère de flatness **indépendant** de la variance calculée
(norme du gradient local par défaut), puis calcule la variance sur le
**résidu** après ajustement d'un plan par moindres carrés (détrend).

| Paramètre | Type | Défaut | Description |
|---|---|---|---|
| `patch_size` | `int` | `8` | Taille des patchs carrés (px). |
| `percentile` | `float` | `10.0` | % des patchs les plus plats conservés. |
| `detrend` | `bool` | `True` | Soustraire un plan avant calcul de variance. |
| `flatness_criterion` | `str` | `'gradient'` | `'gradient'` ou `'variance'`. |

**Avantages** : simple, rapide.
**Limites** : grille rigide (peut couper des objets en résolution métrique) ;
reste sensible au biais de sélection si `flatness_criterion='variance'`.

### `wavelets`

Estimateur de Donoho-Johnstone : la variance est déduite de la médiane des
coefficients d'ondelettes de détail (sous-bande haute fréquence, niveau 1,
ondelette `db1`), robuste car peu sensible aux contours forts (médiane).

Pas de paramètre requis. Rapide, donne une variance **globale unique** pour
toute l'image (ne capture pas un bruit dépendant du signal).

### `skimage_estimate_sigma`

Même principe (Donoho-Johnstone) via l'implémentation de référence
`skimage.restoration.estimate_sigma`. Sert de méthode de comparaison /
validation croisée avec `wavelets`.

### `ponomarenko`

Analyse l'énergie des coefficients DCT hautes fréquences de blocs se
chevauchant, en ne conservant par bin de luminance que les blocs les moins
texturés (percentile bas de l'énergie basses fréquences `V_low`). Permet de
caractériser un **bruit dépendant du signal** (Poisson + lecture) en
découpant l'image en `num_bins` classes de luminance.

Structure de la méthode, d'après la littérature : DCT 2D de blocs, le
bruit blanc se répartissant uniformément sur tous les coefficients alors
que le signal d'une scène naturelle se concentre en basses fréquences
(Ponomarenko et al., 2007) ; partition du plan DCT par la diagonale
`i + j` (basses fréquences `0 < i + j ≤ T`, hautes fréquences `i + j > T`,
ici `T = block_size`), sélection de la fraction `percentil` des blocs de
plus faible énergie basse fréquence, répétition par classe de luminance
pour obtenir une courbe de bruit (Colom & Buades, 2013). Seule
l'agrégation est propre à cette implémentation : moyenne de l'énergie
haute fréquence par bloc, puis médiane sur les blocs retenus (voir la
docstring de `NoiseVariance._estimate_noise_ponomarenko`).

- N. Ponomarenko, V. Lukin, M. Zriakhov, A. Kaarna, J. Astola, « An
  automatic approach to lossy compression of AVIRIS images », *Proc.
  IEEE IGARSS*, 2007.
- M. Colom, A. Buades, « Analysis and Extension of the Ponomarenko et al.
  Method, Estimating a Noise Curve from a Single Image », *Image
  Processing On Line*, 3, pp. 173–197, 2013.

| Paramètre | Type | Défaut | Description |
|---|---|---|---|
| `num_bins` | `int` | `1` | Nombre de classes de luminance. |
| `block_size` | `int` | `8` | Taille des blocs carrés (px). |
| `overlap` | `int` | `8` | Pas (px) entre deux blocs successifs (`overlap = block_size` ⇒ pavage sans recouvrement). |
| `percentil` | `float` | `0.05` | Fraction des blocs les moins texturés gardés par bin. |

**Classes de luminance** : les blocs sont répartis en `num_bins` classes
d'**effectifs égaux** (quantiles de la luminance moyenne des blocs), comme
dans la variante de Colom & Buades (IPOL 2013) : chaque classe dispose du
même nombre de blocs candidats, donc d'une précision comparable. La DCT
est calculée en float64.

**Sortie** : listes (une valeur par bin) plutôt qu'un scalaire unique.
Utilisez `num_bins > 1` puis ajustez une régression `variance ~ f(moyenne)`
sur les points obtenus pour caractériser la relation bruit-signal.

### `adaptive_segmentation`

Segmente l'image en superpixels via **SLIC** (`skimage.segmentation.slic`)
plutôt que d'utiliser une grille régulière : les zones d'analyse épousent
les contours radiométriques réels, ce qui limite le mélange de plusieurs
matériaux dans une même zone — problème fréquent en résolution métrique où
un bloc 8×8 px peut déjà couvrir plusieurs objets au sol.

Un plan est ajusté et soustrait dans chaque segment (détrend), puis seuls
les segments les plus homogènes (variance résiduelle la plus faible) sont
conservés pour le calcul final.

| Paramètre | Type | Défaut | Description |
|---|---|---|---|
| `n_segments` | `int` | `200` | Nombre de superpixels visés. |
| `compactness` | `float` | `0.1` | Compacité SLIC (faible = épouse mieux les contours). |
| `detrend` | `bool` | `True` | Soustraire un plan par segment. |
| `min_segment_size` | `int` | `30` | Taille minimale (px) d'un segment retenu. |
| `homogeneity_percentile` | `float` | `20.0` | % des segments les plus homogènes conservés. |

**Avantages** : forme des zones adaptative, robuste au contenu de la scène.
**Limites** : plus coûteux en calcul ; dépend de la qualité de la
segmentation SLIC (paramètres `n_segments`/`compactness` à ajuster selon la
résolution et la texture de la scène).

### `pca_estimate`

Estimateur robuste inspiré des méthodes par patchs faiblement texturés
(Pyatykh et al. ; Chen et al., *"An Efficient Statistical Method for Image
Noise Level Estimation"*). Principe :

1. Extraction de patchs carrés (fenêtre glissante), détrendés (plan
   soustrait) pour ne pas confondre gradient radiométrique et bruit.
2. Sélection **en un seul passage** des patchs les moins texturés selon la
   norme du gradient local (même critère que `flat_patches`), calculée sur
   le patch brut, donc indépendante de la variance ensuite calculée.
3. Calcul de la matrice de covariance sur ce sous-ensemble fixe ; dans une
   zone homogène, ses valeurs propres les plus faibles convergent vers la
   variance du bruit — la médiane de la moitié inférieure des valeurs
   propres est utilisée comme estimateur robuste.

| Paramètre | Type | Défaut | Description |
|---|---|---|---|
| `patch_size` | `int` | `7` | Taille des patchs carrés utilisés pour la covariance. |
| `stride` | `int` | `2` | Pas d'échantillonnage des patchs. |
| `percentile` | `float` | `20.0` | % des patchs les moins texturés conservés pour la covariance. |
| `detrend` | `bool` | `True` | Soustraire un plan par moindres carrés à chaque patch avant covariance. |

**Sélection en un seul passage.** Sélectionner les patchs par un seuil sur
leur propre variance, réévalué à chaque itération à partir de l'estimation
courante du bruit, reproduit en l'amplifiant le biais de tri décrit pour
`flat_patches` : sur une image de bruit blanc pur, l'estimation dérive de
−6 % après une itération à −70 % après cinq. Les patchs sont donc
sélectionnés en un seul passage, sur un critère indépendant (voir aussi
[Biais résiduel](#biais-résiduel-des-méthodes-par-sélection-de-patchs),
qui s'applique à cette méthode comme à `flat_patches`).

**Avantages** : pas de seuil de platitude fixé a priori sur la variance ;
robuste à la texture fine résiduelle grâce à la médiane.
**Limites** : implémentation simplifiée (pas de reproduction stricte de
l'article) ; sensible au choix de `patch_size` (un patch trop petit rend la
covariance instable) ; coût de calcul non négligeable (gradient + covariance
par patch).

## Biais résiduel des méthodes par sélection de patchs

Le biais de tri décrit en introduction (sélectionner sur la variance
elle-même) est évité dans `flat_patches`, `adaptive_segmentation` et
`pca_estimate` : leur critère de sélection (gradient local, ou variance
résiduelle après détrend pour la segmentation) est calculé indépendamment de
la grandeur ensuite moyennée.

Un **biais résiduel plus subtil** demeure néanmoins, propre à toute méthode
qui *sélectionne* des patchs/segments avant de moyenner : sur une zone
**réellement dépourvue de texture** (aucune variation de signal, seulement du
bruit), le critère de sélection lui-même (norme du gradient local) n'a plus
rien de réel à discriminer — il ne mesure plus que la rugosité de la
réalisation de bruit locale. Choisir les patchs au gradient le plus faible
revient alors, indirectement, à choisir les réalisations de bruit les plus
"calmes", ce qui sous-estime la variance.

Vérifié empiriquement : bruit blanc gaussien de variance connue = 25, image
512 × 512, moyenne de 5 tirages (`seed` 0 à 4), paramètres par défaut. Scène
mixte : damier de pavés de 64 px alternativement plats (120 DN) et texturés ;
texture : bruit blanc gaussien lissé (`gaussian_filter`, σ = 1 px) ramené à un
écart-type de 30 DN autour de 120 DN (`default_rng(3)`).

| Scène | `flat_patches` | `pca_estimate` | `ponomarenko` | `wavelets` | `skimage_estimate_sigma` | `adaptive_segmentation` |
|---|---|---|---|---|---|---|
| 100 % plate (aucune texture réelle nulle part) | 17.9 (−29 %) | 17.0 (−32 %, avec calibration) | 25.1 (0 %) | 25.0 (0 %) | 24.9 (−1 %) | 23.8 (−5 %) |
| Damier zones plates / zones texturées (cas réaliste) | 19.0 (−24 %) | 20.0 (−20 %, avec calibration) | 24.5 (−2 %) | 41.2 (+65 %) | 33.6 (+34 %) | 144.0 (+476 %) |
| 100 % texturée | 334.6 (+1238 %) | 31.4 (+25 %) | 32.1 (+29 %) | 73.7 (+195 %) | 45.1 (+80 %) | 829.9 (+3219 %) |

Dans le deuxième cas — plus proche d'une scène réelle où existent de vraies
zones homogènes et de vraies zones texturées — le biais des méthodes par
patchs reste significatif (~-20 à -25 %) mais les méthodes globales par
ondelettes (`wavelets`, `skimage_estimate_sigma`) surestiment au contraire
nettement : leur hypothèse implicite (les coefficients de détail sont
majoritairement du bruit) s'effondre dès qu'une fraction significative de
l'image est texturée, d'autant plus que la texture est forte et fine.
`adaptive_segmentation` est la plus fragile dès que la scène est texturée.
`ponomarenko`, grâce à la séparation fréquentielle DCT basses/hautes
fréquences par bloc, reste le plus robuste dans les deux premiers cas. Sur
une scène **entièrement** texturée, aucune méthode n'est fiable (au mieux
+25 %) : il faut des zones réellement homogènes quelque part dans l'image
(ou dans une autre image du même capteur).

**Implication pratique** :

- Pour une estimation fiable, privilégiez `ponomarenko` quand la scène
  contient un mélange de zones homogènes et texturées (cas typique en
  télédétection).
- Si vous utilisez `flat_patches`/`pca_estimate`/`adaptive_segmentation`,
  validez toujours sur un échantillon où la vraie variance est connue (bruit
  synthétique injecté, cf. `examples/noise_intercomparison.py`), et envisagez d'augmenter `patch_size`
  (réduit le poids relatif du bruit dans le calcul du gradient) ou de
  diminuer `percentile`/`homogeneity_percentile` (ne garder que les patchs
  les plus confidemment plats, quitte à réduire le nombre d'échantillons).
- Ce biais résiduel n'a pas de solution universelle simple ; une correction
  rigoureuse nécessiterait un facteur de correction calibré (type
  Marchenko-Pastur pour `pca_estimate`) qui dépasse le cadre de cette
  implémentation.

### Cas particulier : faut-il conserver `pca_estimate` ?

`pca_estimate` sélectionne ses patchs en un seul passage (voir sa
description) et est entièrement vectorisée (`sliding_window_view`, détrend
et gradient par patch en opérations matricielles), pour un coût comparable
à celui des autres méthodes. Une correction de biais par auto-étalonnage
Monte-Carlo (`calibrate_bias`, activée par défaut) réduit légèrement le
biais résiduel (−35 % → −32 % sur bruit pur), mais ne le corrige pas
complètement : la part dominante de ce biais vient de la sélection
elle-même (le critère « gradient faible » reste partiellement corrélé à la
variance sur une zone dépourvue de vraie texture), pas seulement de
l'échantillonnage fini de la covariance.

**Verdict** : la méthode n'apporte pas d'avantage de précision par rapport
à `ponomarenko` ou `flat_patches` dans les scénarios testés. Elle est
conservée comme option exploratoire/expérimentale (utile pour comparer des
approches, ou si son comportement s'avère meilleur sur vos données réelles
spécifiques — la scène joue un rôle, cf. tableau), mais **`ponomarenko` reste
la méthode recommandée par défaut** quand la précision compte.

## Barres d'erreur et images multiples

**Principe** : construire une barre d'erreur en comparant plusieurs images
n'a de sens statistique que si ces images proviennent de la **même
statistique de bruit** — typiquement le même système d'acquisition/capteur.
Dans ce cas, fournir plusieurs images n'est pas une répétition de mesure au
sens classique, mais un moyen d'**enrichir l'échantillon** : plus d'images
donnent plus de chances de trouver des zones homogènes, et plus de zones
homogènes donnent une estimation plus précise et une barre d'erreur plus
fiable.

Les mesures élémentaires (patchs, blocs ou segments homogènes détectés
individuellement par chaque méthode, ou répétitions par tuile pour les
méthodes globales) sont donc **cumulées sur toutes les images fournies**
avant calcul de la variance finale et de sa barre d'erreur — un seul
rapport, pas un par image :

| Méthode | Échantillons utilisés (cumulés sur toutes les images) |
|---|---|
| `flat_patches` | Variance résiduelle de chaque patch plat individuellement retenu. |
| `adaptive_segmentation` | Variance résiduelle de chaque segment homogène individuellement retenu. |
| `ponomarenko` | Estimation individuelle par bloc, par bin de luminance — cumulée sur toutes les images pour ce bin (les bornes de bin sont calculées par image, donc supposées comparables d'une image à l'autre : hypothèse raisonnable pour un même capteur). |
| `pca_estimate` | Ré-échantillonnage bootstrap du sous-ensemble de patchs faiblement texturés, par image, puis échantillons cumulés. |
| `wavelets`, `skimage_estimate_sigma` | Estimateur répété sur une grille de tuiles de chaque image, échantillons cumulés. |

Le **nombre de zones homogènes détectées** (cumulé sur toutes les images)
est rapporté pour chaque méthode dans le tableau récapitulatif, et affiché
sur chaque barre du graphique (`n=...`), ainsi que pour chaque bin de la
courbe bruit-luminance.

```python
variance, mean, samples = nv.estimate_noise(image, 'flat_patches', return_samples=True)
# Pour plusieurs images du même capteur : cumuler les échantillons avant
# de recalculer variance/écart-type (voir run_comparison de examples/noise_intercomparison.py).
```

## Taille des zones homogènes : un paramètre réglable

La taille de zone recherchée est un paramètre explicite pour la plupart
des méthodes par patchs ou blocs :

| Méthode | Paramètre contrôlant la taille de zone | Contrôle |
|---|---|---|
| `flat_patches` | `patch_size` (px, côté du carré) | Direct |
| `ponomarenko` | `block_size` (px, côté du carré) | Direct |
| `pca_estimate` | `patch_size` (px, côté du carré) | Direct |
| `adaptive_segmentation` | `target_zone_size` (aire visée en px²) | Direct — calcule automatiquement `n_segments = aire_image / target_zone_size` |

Pour `adaptive_segmentation`, `target_zone_size` (en pixels²) fixe
`n_segments` = aire de l'image / `target_zone_size`, et prend le pas sur
`n_segments` si les deux sont fournis.

```python
# Vise des zones d'environ 20x20 px (400 px²)
nv.estimate_noise(image, 'adaptive_segmentation', {'target_zone_size': 400})
```

**Compromis à connaître** : une zone plus grande donne une variance plus
stable (moins bruitée par l'échantillonnage) mais risque davantage de
mélanger deux matériaux/objets différents (surtout en résolution métrique) ;
une zone plus petite est plus sûre de rester homogène mais donne une
variance individuelle plus dispersée (voir la discussion sur le biais
résiduel plus haut, qui s'aggrave aussi avec des patchs trop petits).

## Deux modes bien séparés : validation synthétique vs mesure sur images réelles

**Le point de vigilance** : injecter du bruit synthétique sur une image
réelle qui contient déjà du bruit capteur inconnu fausse la comparaison à
la référence (la variance mesurée = bruit préexistant + bruit injecté, mais
la référence affichée ne représenterait que le bruit injecté). `examples/noise_intercomparison.py`
distingue donc structurellement deux modes, sans état intermédiaire ambigu :

1. **Sans `--images`** : une image synthétique "propre" est générée en
   mémoire (aucun bruit préexistant possible par construction) et un bruit
   de variance connue y est injecté à la volée — pas besoin d'écrire quoi
   que ce soit sur disque, l'image de départ étant garantie propre à chaque
   exécution.
2. **Avec `--images chemin1 chemin2 ...`** : **aucun bruit n'est ajouté par
   défaut**. Le script mesure directement le bruit déjà présent dans vos
   images. Si vous avez un besoin explicite d'ajouter un bruit de test
   par-dessus (ex. évaluer la sensibilité des méthodes à un niveau de bruit
   connu, en plus du bruit capteur inconnu), passez `--sigma2` ou `--a`/`--b`
   explicitement : un avertissement s'affiche alors pour rappeler que la
   variance mesurée inclut les deux composantes.

```bash
# Validation sur image synthétique (bruit connu, injecté à la volée)
python examples/noise_intercomparison.py --sigma2 25.0 --a 5.0 --b 0.08

# Mesure directe sur vos images (aucun bruit ajouté, comportement par défaut)
python examples/noise_intercomparison.py --images photo1.tif photo2.tif

# Vos images + bruit de test explicite par-dessus (avertissement affiché)
python examples/noise_intercomparison.py --images photo1.tif --sigma2 25.0
```

`--bit_depth` (ex. 8, 16) reste disponible en mode validation synthétique
pour quantifier/écrêter l'image bruitée générée, et reproduire ainsi les
effets de saturation/quantification d'un capteur réel de façon contrôlée
(voir section suivante).

## Non-linéarité aux extrêmes de luminance

**Constat** : la relation
bruit-luminance σ²(I) = a + b·I n'est généralement valide que sur une plage
intermédiaire. En basse luminance, le pas de quantification du capteur
devient non négligeable devant l'écart-type du bruit ; en haute luminance,
l'écrêtage/saturation **comprime artificiellement** la variance mesurée (les
valeurs saturées sont plafonnées à une constante, donc leur variance
résiduelle chute).

`--bit_depth 8` (ou 16) sur `examples/noise_intercomparison.py` reproduit directement ce phénomène en
quantifiant et écrêtant l'image bruitée générée sur `[0, 2^bit_depth - 1]`,
comme un capteur réel (mode validation synthétique uniquement). L'image propre
par défaut du script (`generate_clean_image`, 85 à 180 DN) n'atteint pas la
saturation : pour l'observer, il faut une scène couvrant toute la dynamique.
Exemple : rampe horizontale de 5 à 255 DN (512 × 512, plus une ondulation
verticale de 3 DN), a = 5 et b = 0,3 injectés, capteur 8 bits, `ponomarenko` avec
`num_bins=12`, moyenne de 5 tirages :

```python
yy, xx = np.indices((512, 512))
rampe = (5 + 250 * xx / 511 + 3 * np.sin(yy / 40)).astype(np.float32)
est = estimate_noise(add_noise(rampe, a=5.0, b=0.3, seed=0, bit_depth=8),
                     "ponomarenko", num_bins=12)
```

| Luminance | Variance mesurée | Modèle linéaire attendu |
|---|---|---|
| ~201 | 64.3 | 65.3 (cohérent) |
| ~222 | 67.0 | 71.6 (déjà en retrait) |
| ~252 (proche saturation) | **25.5** | 80.5 (écart net — écrêtage) |

**Deux mécanismes pour en tenir compte**, disponibles dans `examples/noise_intercomparison.py` :

1. **`--trim_low_pct` / `--trim_high_pct`** : excluent les bins de luminance
   les plus extrêmes (en % de la plage observée) du calcul de la régression
   affine, tout en les affichant sur le graphique (marqueurs gris distincts)
   pour visualiser l'écart. La droite ajustée reflète alors la plage "bien
   comportée" du capteur, sans être tirée par les artefacts de saturation.
   Sur l'exemple ci-dessus, un ajustement sur les 12 bins donne a = 13,9 et
   b = 0,19 (faussé par le bin saturé) ; en excluant le premier et le
   dernier bin, a = 6,5 et b = 0,28 ; les deux premiers et les deux
   derniers, a = 5,8 et b = 0,29 (pour 5 et 0,3 injectés).
2. **`--fit_degree N`** (N=2 quadratique, N=3 cubique, etc.) : ajoute une
   régression polynomiale de degré N (calculée sur tous les bins, y compris
   les extrêmes) superposée à la droite affine, pour visualiser explicitement
   la courbure plutôt que de simplement l'exclure. Non affichée par défaut
   (N=1, affine seule).

```bash
python examples/noise_intercomparison.py --bit_depth 8 --a 5.0 --b 0.3 --num_bins 12 --trim_low_pct 15 --trim_high_pct 15 --fit_degree 2
```

**Recommandation pratique** : sur des images réelles, repérez d'abord la
plage de luminance où le capteur n'est ni proche du plancher de bruit
(quelques valeurs au-dessus du bruit de lecture) ni proche de la saturation
(typiquement les 5-15 % supérieurs de la dynamique), et limitez le
`trim_low_pct`/`trim_high_pct` (ou directement la plage d'entrée) à cette
zone pour obtenir un modèle a/b fiable, tout en gardant les bins extrêmes
affichés pour documenter le comportement réel du capteur sur toute sa
dynamique.

## Toutes les méthodes superposées sur la courbe bruit-luminance

La courbe bruit-luminance superpose toutes les méthodes qui produisent des estimations élémentaires par patch/segment/
tuile (`flat_patches`, `adaptive_segmentation`, `wavelets`,
`skimage_estimate_sigma` et `ponomarenko`) : chacune expose la
luminance moyenne associée à chaque échantillon. Ces échantillons sont
regroupés dans des **bins de luminance communs** (calculés sur la
distribution des pixels des images fournies, percentiles 1-99) et
superposés sur le même graphique, avec une couleur par méthode.

Intérêt pratique : si les points des différentes méthodes suivent la même
tendance générale (même si chacune a son propre biais de niveau, déjà
documenté plus haut), c'est un indice que la relation bruit-luminance
observée est réelle et pas un artefact d'une seule méthode. Un fort
désaccord entre méthodes à une luminance donnée doit au contraire inciter
à la prudence à cette luminance précise.

`pca_estimate` est une exception : son estimateur repose sur la covariance
de l'ensemble des patchs retenus, pas sur une variance individuelle par
patch — il n'y a donc pas de notion naturelle de "variance à telle
luminance" pour cette méthode. Elle apparaît comme un point unique (à la
luminance moyenne de ses patchs retenus) plutôt que comme une courbe.

La régression (affine, et polynôme de degré supérieur si `--fit_degree` >
1) n'est calculée et tracée que sur une **méthode de référence**
(`ponomarenko` par défaut — la plus adaptée à ce type d'analyse car elle
stratifie nativement par luminance), pour ne pas surcharger le graphique de
plusieurs droites. Les autres méthodes servent de comparaison visuelle.

## Utilisation

API fonctionnelle (résultat nommé, recommandée) :

```python
from eoqual.measure.noise import estimate_noise, list_noise_methods

list_noise_methods()                        # méthodes et paramètres
est = estimate_noise(image, "ponomarenko", num_bins=8)
est.variance, est.mean, est.sigma           # une valeur par classe de luminance
est.samples, est.luminances                 # mesures élémentaires (barres d'erreur)
```

API par classe (tuples) :

```python
import numpy as np
from eoqual.measure.noise import NoiseVariance

image = ...  # NDArray[np.float32], image 2D en niveaux de gris

nv = NoiseVariance(image)

# Liste des méthodes et de leurs paramètres
print(nv.get_available_methods())

# Estimation avec une méthode donnée
variance, mean = nv.estimate_noise(
    image,
    estimator='adaptive_segmentation',
    params={'n_segments': 300, 'homogeneity_percentile': 15.0},
)
```

## Script d'intercomparaison (`examples/noise_intercomparison.py`)

Le script `examples/noise_intercomparison.py` fonctionne en deux modes (voir section
[Deux modes bien séparés](#deux-modes-bien-séparés--validation-synthétique-vs-mesure-sur-images-réelles)
ci-dessus). Dans les deux cas, quand plusieurs images sont utilisées, les
échantillons sont **cumulés** dans un seul tableau de bord `plotly` (voir
section [Barres d'erreur et images multiples](#barres-derreur-et-images-multiples)) :

1. **Panneau de comparaison globale** : toutes les méthodes de
   `NoiseVariance` sont appliquées à chaque image, avec **barres d'erreur**
   et **nombre de zones homogènes détectées** cumulés.
2. **Courbe bruit-luminance** σ²(I) = `a` + `b`·I : la méthode
   `ponomarenko` est appliquée à chaque image avec `--num_bins` classes de
   luminance ; les blocs de chaque bin sont cumulés, avec barre d'erreur et
   nombre de zones par bin, et une régression affine pondérée — avec
   exclusion optionnelle des extrêmes de luminance (voir section
   [Non-linéarité aux extrêmes de luminance](#non-linéarité-aux-extrêmes-de-luminance)).

```bash
# Validation sur image synthétique
python examples/noise_intercomparison.py --sigma2 25.0 --a 5.0 --b 0.08 --num_bins 10 --trim_low_pct 10 --trim_high_pct 10

# Mesure directe sur vos images (aucun bruit ajouté)
python examples/noise_intercomparison.py --images mon_image.tif --num_bins 10
```

> Note : le mode validation reste une injection de bruit synthétique sur
> image synthétique (pas de multi-temporel, pas de site de calibration). Sur des images réelles en mesure directe, il n'y a pas de
> "vérité terrain" à comparer — l'intérêt est de mesurer le bruit réellement
> présent, avec barres d'erreur, plutôt que de valider une précision absolue.

## Mode Bayer : mesure par bande (R, G, B)

**Motivation** : sur une image issue directement du capteur (mosaïque
Bayer non démosaïquée), le bruit n'est en général **pas identique** sur
les trois composantes couleur — gains analogiques différents par canal,
sensibilité spectrale du filtre, etc. Mesurer la variance sur l'image
mosaïquée telle quelle (en la traitant comme une image en niveaux de
gris) mélangerait les statistiques des trois canaux et fausserait le
résultat. Le mode `--bayer` sépare d'abord l'image par canal, puis
applique **indépendamment** le pipeline complet d'intercomparaison à
chaque bande.

**Principe** :

1. L'image mosaïquée (un seul plan, motif 2×2 répété) est séparée en
   trois plans R, G, B par sous-échantillonnage selon leur position dans
   le motif — chaque plan résultant a une résolution moitié (H/2, W/2) de
   l'image d'origine. Les deux pixels verts du motif (G1, G2) sont
   combinés par simple moyenne en une seule bande G.
2. Chaque bande est ensuite traitée exactement comme une image en niveaux
   de gris par le pipeline existant : `run_comparison` (toutes les
   méthodes de `NoiseVariance`) et `run_noise_vs_luminance` (courbe
   bruit-luminance, méthode de référence `ponomarenko`).
3. Un **SNR par bande** est calculé à partir de l'intensité moyenne du
   signal et de l'écart-type de bruit donné par la méthode de référence :
   `SNR = moyenne(signal) / écart-type(bruit)`, reporté en linéaire et en
   décibels (`20·log10(SNR)`).

Quatre motifs de mosaïque sont supportés via `--bayer_pattern` : `RGGB`
(défaut), `BGGR`, `GRBG`, `GBRG`.

```bash
# Images Bayer mosaïquées (un seul plan par fichier, motif RGGB)
python examples/noise_intercomparison.py --images capteur1.tif capteur2.tif --bayer

# Autre motif de mosaïque, plus de bins pour la courbe bruit-luminance
python examples/noise_intercomparison.py --images raw.tif --bayer --bayer_pattern GRBG --num_bins 12
```

**Sorties** : un fichier HTML par bande (`comparaison_bruit_R.html`,
`_G.html`, `_B.html` avec le préfixe par défaut, ou `{--output}_R.html`
etc. si `--output` est fourni), chacun identique en structure au tableau
de bord du mode niveaux de gris, avec en plus le SNR de la bande affiché
dans le titre. Un tableau récapitulatif du SNR des trois bandes est
également imprimé en console à la fin de l'exécution.

**Limitations actuelles** :

- Le mode `--bayer` nécessite `--images` : il n'existe pas (encore)
  d'équivalent Bayer du mode validation synthétique (génération d'une
  mosaïque propre + injection de bruit connu par bande). Si `--sigma2`,
  `--a` ou `--b` sont fournis en même temps que `--bayer`, ils sont
  ignorés et un avertissement le rappelle — seule la mesure directe du
  bruit déjà présent est effectuée.
- L'entrée doit être le plan mosaïqué brut (un seul canal), pas une image
  déjà démosaïquée. Si un fichier à 3 canaux est fourni en mode `--bayer`,
  un avertissement s'affiche et seul le premier canal est utilisé — le
  motif Bayer d'origine ne peut alors plus être reconstitué correctement.
- G1 et G2 sont moyennés directement pixel à pixel malgré leur léger
  décalage spatial (une demi-diagonale de pixel) : approximation
  raisonnable pour une statistique de bruit sur zones homogènes, mais qui
  ne conserverait pas une information de position fine si un usage futur
  en avait besoin (voir pistes d'amélioration).

## Pistes d'amélioration supplémentaires

- **Correction de biais analytique plutôt que Monte-Carlo** pour
  `pca_estimate` : la loi de Marchenko-Pastur permet en théorie un facteur
  de correction en forme close (fonction de d/N), ce qui éviterait la
  simulation à chaque appel — gain de temps marginal mais plus élégant.
- **Régression robuste** pour la courbe bruit-luminance (Huber ou moindres
  carrés pondérés) plutôt qu'une régression linéaire simple : les points par
  bin n'ont pas tous la même incertitude (bins avec peu de blocs), ce
  qu'une régression pondérée par le nombre d'échantillons par bin
  représenterait mieux.
- **Barres d'erreur sur la courbe bruit-luminance**, en alignant les bins
  entre répétitions (bins de luminance fixes communs à toutes les images/
  recadrages plutôt que des bins recalculés à chaque appel de `ponomarenko`).
- **Parallélisation** (`joblib`/`multiprocessing`) : avec plusieurs
  images, chaque répétition est indépendante et pourrait être traitée en
  parallèle pour accélérer le protocole complet.
- **Cross-canal**, si vos images sont multispectrales : exploiter la
  redondance entre bandes voisines pourrait encore réduire le biais résiduel
  des méthodes par patchs, en augmentant le nombre d'échantillons
  disponibles sans dupliquer le même bruit spatial.
- **Mode Bayer synthétique** : étendre le mode validation (bruit de
  variance connue injecté) à une mosaïque Bayer générée en mémoire, pour
  pouvoir valider la précision du SNR par bande comme c'est déjà possible
  pour le mode niveaux de gris.
- **G1/G2 séparés** en option pour le mode Bayer, plutôt que moyennés
  systématiquement, pour les cas où le très léger décalage spatial entre
  les deux pixels verts du motif importerait (ex. étude fine d'un
  gradient de gain intra-canal G1 vs G2).
- **Validation sur données réelles de capteur** (hors sites de calibration
  au sens classique) : si vous disposez d'images de trame noire ("dark
  frames") ou de plusieurs vignettes du même capteur à luminance contrôlée
  en laboratoire, elles fourniraient une vérité terrain plus réaliste que le
  bruit synthétique pour recalibrer les méthodes sur votre capteur
  spécifique.

## Recommandations de choix de méthode

| Besoin | Méthode conseillée |
|---|---|
| Estimation rapide, image globalement homogène | `wavelets` ou `skimage_estimate_sigma` |
| Caractériser un bruit dépendant du signal (σ² = a + b·I) | `ponomarenko` avec `num_bins > 1` |
| Scène très hétérogène, résolution métrique, objets nombreux | `adaptive_segmentation` |
| Robustesse sans réglage fin de seuil de platitude | `pca_estimate` |
| Référence simple et rapide | `flat_patches` |
| Image brute, valeur avec diagnostic de fiabilité, rayures | `variogram` ([variogramme](bruit-variogramme.md)) |

En pratique, il est recommandé de **croiser au moins deux méthodes** de
familles différentes (ex. `ponomarenko` + `adaptive_segmentation`) : un
écart important entre elles est souvent le signe d'un bruit hétéroscédastique
mal capturé par les méthodes globales, ou d'une scène trop texturée pour
les méthodes par patchs fixes.

## Limites connues et pistes non retenues

- **Pas de multi-temporel** : aucune méthode ne suppose la disponibilité de
  deux acquisitions de la même zone (approche pourtant classique et souvent
  la plus fiable en calibration radiométrique satellite).
- **Pas de sites de calibration** : aucune méthode ne s'appuie sur des zones
  réputées homogènes connues a priori (désert, glace, océan).
- **Cross-canal** non implémenté : pour
  une image multispectrale, la redondance entre bandes voisines pourrait
  encore réduire le biais d'estimation ; à envisager en évolution future si
  les données le permettent.
- **Pas d'injection de bruit synthétique en mode Bayer** : le mode
  `--bayer` ne mesure que le bruit déjà présent dans vos images (voir
  section [Mode Bayer](#mode-bayer--mesure-par-bande-r-g-b)) ; il n'existe
  pas encore d'équivalent Bayer du mode validation synthétique du reste du
  script.
