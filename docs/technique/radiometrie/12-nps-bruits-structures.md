# 12. Spectre de puissance du bruit et bruits structurés

## 12.1 Objet

Spectre de puissance 2D du bruit, ou NPS (`noise_power_spectrum`), puis,
à partir de ce spectre, détection des motifs périodiques et mesure du
bruit de ligne et de colonne (`structured_noise`).

La variance ne dit pas comment le bruit se répartit entre les fréquences.
Le NPS le dit, et il sert à trois choses :

- **bruit corrélé** : un traitement (dématriçage, rééchantillonnage,
  compression, débruitage) colore le bruit. Le NPS en donne la forme
  σ²·|H(f)|², et donc l'effet du traitement sur le bruit ;
- **motifs périodiques** : une interférence apparaît comme un pic, dont on
  veut la fréquence, l'orientation, l'amplitude et la nature (fixe ou
  temporelle) ;
- **bruit de ligne et de colonne** : il forme une crête sur un axe du NPS.
  Il se mesure mieux par la séquence des moyennes de lignes, dont le
  spectre 1D donne l'exposant β du bruit en 1/f.

**Statut** : validées contre le simulateur S3 (`tests/test_noise_spectrum.py`,
17 tests ; robustes sur 30 graines). Taux de détection et de fausses
alarmes mesurés sur 270 analyses (§12.4). Module :
`eoqual.measure.noise.spectrum`.

## 12.2 Principe

### 12.2.1 Définition et normalisation

Pour un bruit stationnaire de covariance C(Δ), le NPS est la transformée
de Fourier de C : NPS(f) = Σ_Δ C(Δ) e^{−2iπ f·Δ}, en DN²·pixel². Son
intégrale sur [−½, ½]² vaut C(0) = σ². Un bruit blanc a un NPS plat de
niveau σ².

**Estimateur.** Pour une imagette de P × P pixels x, avec la fenêtre w :

NPS(f) = |FFT(w·(x − plan))|² / Σw².

La normalisation par Σw² préserve le niveau d'un bruit blanc, quelle que
soit la fenêtre. Sans fenêtre (w = 1), Σw² = P², et l'égalité de Parseval
donne Σ_f NPS / P² = var(x).

**Retrait d'un plan.** Il supprime les variations lentes de la scène.
Il annule la composante continue et déforme les deux ou trois premiers
anneaux de fréquence (|f| < 3/P), qu'il faut exclure de l'interprétation.

**Fenêtre de Hann.** Elle réduit les fuites spectrales, si bien qu'un motif
périodique reste concentré sur environ 3 × 3 fréquences, et rend les
imagettes recouvrantes peu corrélées.

**Décomposition périodique + lisse** (Moisan, 2011 ; `window="perdecomp"`,
via `eoqual.core.fourier.perdecomp`). Conçue pour supprimer la « croix »
d'artefacts de bord du spectre d'une *image*, elle n'est pas adaptée au
bruit. Sur un bruit blanc, la composante lisse retirée est elle-même du
bruit, ce qui donne un niveau trop faible de 2,5 % et un excès aux basses
fréquences sur les axes : sur un bruit blanc σ² = 4 (512 × 512), variance
de 3,97, niveau médian de 3,91 (−2,5 %), premier anneau à 12,4 au lieu de 4.
Elle reste disponible pour
comparaison ; la fenêtre de Hann est recommandée.

### 12.2.2 Moyenne de Welch et dispersion

On moyenne le NPS sur n imagettes recouvrantes à 50 % (Welch, 1967). Chaque
valeur d'un périodogramme suit approximativement σ²(f)·χ²₂/2, et leur
moyenne suit une loi Gamma(ν)/ν, avec ν ≈ n. Avec la fenêtre de Hann et
50 % de recouvrement, le facteur théorique en 1D vaut 1/(1 + 2c²) = 0,94,
avec c = 0,177 la corrélation de deux segments décalés de P/2. Mesuré en
2D sur 200 tirages, il vaut 1,05 ± 0,10 : les imagettes recouvrantes sont
pratiquement indépendantes. L'incertitude-type u_nps = sd/√n est donc
juste.

**Mode « différence ».** Sur une pile, (x_{2k} − x_{2k+1})/√2 annule tout
motif fixe (DSNU, PRNU, motif périodique synchronisé). Son NPS est celui
du bruit **temporel** seul. Les paires sont disjointes, donc indépendantes.

### 12.2.3 Détection des pics

Un motif A·cos(2πf₀·x + φ) ajoute à NPS(±f₀) une variance A²/4 de chaque
côté, étalée par la fenêtre. On l'intègre sur un carré de 5 × 5 fréquences
autour du maximum, après retrait du fond, et on double le résultat pour
compter le pic symétrique. La variance du motif vaut A²/2.

**Fond.** Le fond est une médiane locale 2D (9 × 9) hors des axes. Le long
de chacune des trois lignes de fréquence de chaque bande d'axe
(|f_y| ≤ 1/P et |f_x| ≤ 1/P), c'est une médiane 1D de 19 valeurs. Ces
bandes portent les crêtes des bruits de ligne et de colonne, élargies par
le lobe principal de la fenêtre de Hann. Une médiane 2D les manquerait, et
leurs voisines passeraient pour des pics.

**Seuil.** Le rapport R = NPS/fond suit à peu près Gamma(ν)/médiane. ν est
estimé par le rapport des quantiles 25 % / 50 % de R, qui est invariant
d'échelle et insensible aux pics (à moins d'un quart des fréquences
touchées) : ν = 1,0 ; 3,2 ; 29 ; 324 pour des valeurs vraies de 1, 3, 30
et 300 (2 000 tirages). Le seuil correspond à une probabilité de fausse
alarme p_FA pour tout le demi-plan (Bonferroni, q = p_FA / (P²/2)) :

R > Γ⁻¹(1 − q ; ν) / médiane(Gamma(ν)).

**Trois régions, trois statistiques.** Hors axes, ν est grand (moyenne de
toutes les imagettes et images). Sur une bande d'axe, un motif *fixe* le
long de l'axe (profil de colonnes pour f_y ≈ 0, de lignes pour f_x ≈ 0)
est identique dans toutes les imagettes alignées. Le nombre de moyennes
indépendantes y tombe alors au nombre de positions en travers de l'axe
multiplié par le nombre d'images, soit environ 7 au lieu de 45 pour une
image de 256 × 256. L'estimation par quantiles sur une seule ligne, d'une
soixantaine de valeurs, ne le capte pas, car la médiane 1D suit les
fluctuations de la ligne. On retient donc, sur chaque ligne des bandes,

ν_ligne = min(ν̂_ligne, ν_hors axes, 0,95 × images × positions),

une borne prudente qui a supprimé toutes les fausses alarmes dues aux
motifs de ligne et de colonne (§12.4). Les basses fréquences (|f| < 3/P),
déformées par le retrait du plan et par la pente d'un bruit en 1/f, sont
exclues de la recherche.

### 12.2.4 Motifs fixes et temporels

Sur une pile, mélanger dans un même NPS le bruit temporel (moyenné sur
toutes les poses, ν grand) et les motifs fixes (même périodogramme dans
toutes les poses, ν petit) alourdit la queue de R et fausse le seuil. On
fait donc deux recherches :

- **temporelle** : NPS des différences de poses ;
- **fixe** : NPS de l'image moyenne. Un motif temporel y laisse un résidu,
  la moyenne de K phases aléatoires, de variance moyenne var/K mais
  distribuée de façon ≈ exponentielle. Son excès prévu, (NPS_temporel −
  fond)/K, est ajouté au fond. Un pic fixe qui coïncide avec un pic
  temporel n'est gardé que s'il dépasse (var_temporel/K)·ln(1/p_FA),
  quantile 1 − p_FA de la loi exponentielle.

Chaque pic porte sa nature (`kind`). Sur une image seule, la nature est
indéterminée (`"image"`).

### 12.2.5 Bruit de ligne et de colonne

Sur une pile, on retire la moyenne temporelle, puis on forme la séquence
des moyennes de lignes R_{k,i} (pose k, ligne i), dans l'ordre de lecture.
Pour W colonnes et un bruit pixel de variance σ_p² :

var(R) = σ_ligne² + σ_p²/W,

d'où σ_ligne² = var(R) − σ_p²/W, avec σ_p² estimé sur les résidus
(x − R), corrigés des degrés de liberté perdus (W/(W − 1), K/(K − 1)).

**Spectre et exposant.** La densité spectrale unilatérale de la séquence
est S(f) = 2σ_p²/W + c·f^{−β}. On la moyenne par bandes logarithmiques
(24 bandes), on retire le plancher blanc connu 2σ_p²/W, et on ajuste
log(S − plancher) = log c − β·log f sur les bandes où S dépasse 3 fois le
plancher. Le poids de chaque bande vaut n_bande·((S − plancher)/S)², ce
qui correspond à une erreur relative d'une moyenne de n_bande valeurs de
périodogramme, propagée au travers du retrait du plancher.

**Colonnes.** Même calcul sur les moyennes de colonnes, après retrait de
la moyenne de **chaque pose**. Sans ce retrait, la part du bruit de ligne
commune à une pose (sa moyenne sur les H lignes) passe pour du bruit de
colonne : 0,48 DN mesuré à tort pour 1 DN de bruit de ligne en 1/f. Un
motif temporel de fréquence f_y = 0 (constant le long des colonnes) est,
lui, un vrai bruit de colonne par pose, et il y est compté : σ = A/√2.

## 12.3 Mise en œuvre

```
eoqual/measure/noise/
├── __init__.py
└── spectrum.py   noise_power_spectrum → NPSResult
                  find_structured_noise → StructuredNoiseResult
                    (PeakSearch par nature, StructuredPeak, LineNoise rows / cols)
```

**`noise_power_spectrum(data, *, patch_size=64, step=P/2, detrend="plane", window="hann", mode="image", mask=None)`**

1. Mode « différence » : paires disjointes de poses / √2.
2. Imagettes de P × P au pas `step`, entièrement dans `mask` si fourni.
3. Retrait du plan (moindres carrés, pseudo-inverse précalculée), fenêtre,
   |FFT|²/Σw², cumul de la moyenne et du carré.
4. Sorties centrées (`fftshift`), u_nps, variance = mean(NPS), positions
   distinctes d'imagettes (pour la borne du §12.2.3).

Méthodes : `horizontal` (f_y = 0), `vertical` (f_x = 0), `radial()`
(anneaux de largeur 1/P).

**`find_structured_noise(data, *, patch_size=64, false_alarm=1e-3, background_size=9, min_variance_fraction=0, line_noise=True)`**

1. Pile : NPS « différence » et NPS de l'image moyenne ; image : NPS de
   l'image.
2. Pour chacun : fond (§12.2.3), ν et seuils par région, maxima locaux 3 × 3
   au-dessus du seuil dans le demi-plan, intégration 5 × 5.
3. Fuite temporelle dans le NPS fixe et règle de coïncidence (§12.2.4).
4. Bruit de ligne et de colonne (§12.2.5).

`min_variance_fraction` ajoute un seuil d'importance pratique (fraction
de la variance du NPS), nul par défaut : le test statistique suffit (§12.4).

### Exemple et recommandations

```python
import numpy as np
from eoqual.measure.synthetic import (MatrixSensor, SensorParams, PeriodicPattern,
                                  apply_kernel, gaussian_kernel, kernel_power_transfer)
from eoqual.measure.noise import noise_power_spectrum, find_structured_noise

sensor = MatrixSensor(SensorParams(shape=(256, 256), seed=3, row_noise=1.0, row_noise_beta=1.0,
                                   periodic=(PeriodicPattern(0.5, fx=0.125, fy=0.0625),
                                             PeriodicPattern(0.4, fx=0.2, fy=0.1, temporal=False))))
frames = sensor.dark(n_frames=16, seed=1).frames

# Bruits structurés
st = find_structured_noise(frames)
for q in st.peaks:
    print(f"{q.kind:8s} f = ({q.fx:+.4f}, {q.fy:+.4f}) cycle/px, A = {q.amplitude:.3f} DN, "
          f"{q.variance_fraction:.1%} de la variance")
print(f"bruit de ligne σ = {st.rows.sigma:.3f} ± {st.rows.u_sigma:.3f} DN, "
      f"β = {st.rows.beta:.2f} ± {st.rows.u_beta:.2f}")

# NPS du bruit temporel, puis effet d'un traitement connu
nps = noise_power_spectrum(frames, mode="difference")
f, radial = nps.radial()
k = gaussian_kernel(0.8)
treated = apply_kernel(frames.astype(float), k)
nps_t = noise_power_spectrum(treated, mode="difference")
print(f"variance du bruit : {nps.variance:.3f} → {nps_t.variance:.3f} DN² après traitement")
```

Recommandations :

- **une pile** plutôt qu'une image : elle sépare motifs fixes et
  temporels, et donne le bruit de ligne temporel ;
- **zones uniformes** : obscurité, champ uniforme, ou `mask` sur les zones
  homogènes d'une scène ;
- **P = 64** : bon compromis entre la résolution en fréquence (1/64) et le
  nombre de moyennes. Pour séparer deux raies proches, augmenter P ;
- ignorer le NPS pour |f| < 3/P ; la fenêtre de Hann reste le défaut.

## 12.4 Validation

30 capteurs (graines 0 à 29) de 256 × 256 pixels, 16 poses d'obscurité,
p_FA = 10⁻³ par spectre, sans seuil d'importance pratique
(`examples/structured_noise_detection_study.py`) :

| Cas | Analyses avec fausse alarme | Motif manqué | Mauvaise nature | Amplitude mesurée |
|---|---|---|---|---|
| sans effet | 0/30 | — | — | — |
| bruit de ligne blanc, 1 DN | 0/30 | — | — | — |
| bruit de ligne 1/f, 1 DN | 0/30 | — | — | — |
| colonnage fixe, 2 DN | 0/30 | — | — | — |
| lignes fixes, 2 DN | 0/30 | — | — | — |
| motif temporel 0,3 DN (0,125 ; 0,0625) | 0/30 | 0 | 0 | 0,299 ± 0,035 |
| motif fixe 0,3 DN (0,125 ; 0,0625) | 0/30 | 0 | 0 | 0,299 ± 0,003 |
| motif temporel 0,5 DN sur l'axe (0,25 ; 0) | 1/30 | 0 | 0 | 0,499 ± 0,060 |
| motif temporel 0,5 DN près de l'axe (1/64 ; 0,2) | 1/30 | 0 | 0 | 0,497 ± 0,059 |

Les deux fausses alarmes restantes viennent du même capteur (graine 27),
qui sert à deux des cas : un pic d'amplitude 0,039 DN dans le FPN pixel de
son image moyenne, compatible avec le hasard. Au total, 2 analyses sur
270 (0,7 %), issues d'une seule réalisation de FPN, pour un nominal de
0,2 % par analyse (deux recherches par pile). L'amplitude d'un motif
temporel est 10 fois plus dispersée que celle d'un motif fixe : seules 8
paires de poses le mesurent, et la différence de deux phases aléatoires
fait varier sa puissance d'une paire à l'autre.

Liste des tests : §12.4.1.

### 12.4.1 Tests

| Test | Vérifie |
|---|---|
| `test_white_noise_level` (Hann, sans fenêtre) | niveau σ² à 2 % ; variance ; u_nps/NPS = 1/√n à 20 % |
| `test_filtered_noise_follows_transfer_function` | NPS = σ²·\|H\|² : rapport moyen à 2 %, 99 % des fréquences à 5 u |
| `test_difference_mode_removes_fixed_pattern` | DSNU de 3 DN : variance du mode « différence » = a à 3 % |
| `test_radial_and_cuts`, `test_mask_selects_patches` | moyenne radiale plate ; masque respecté |
| `test_perdecomp_option` | option Moisan disponible, variance à 3 % |
| `test_temporal_pattern`, `test_fixed_pattern` | fréquence, nature, amplitude (0,3 DN) et orientation |
| `test_no_false_alarm` (sans effet, colonnage 2 DN, lignes fixes 2 DN, ligne 1/f) | aucun pic à p_FA = 10⁻⁵ |
| `test_line_noise` (β = 0, 1) | σ_ligne = vrai (offsets réalisés) à 5 u ; β à 5 u + 0,05 ; pas de bruit de colonne |
| `test_single_image_mode` | image seule : nature « image », amplitude à 10 % |

## 12.5 Limites

- *Basses fréquences* : aucune détection pour |f| < 3/P.
- *Pics proches* : deux raies à moins de 3/P l'une de l'autre fusionnent,
  et un pic doit occuper au plus quelques pour cent des fréquences pour que
  l'estimation de ν reste valable.
- *Raies non sinusoïdales* (créneaux, impulsions) : leurs harmoniques sont
  détectées comme des pics distincts.
- *Bruit de ligne* : l'incertitude de σ_ligne suppose une séquence de
  valeurs indépendantes ; pour un bruit en 1/f, elle est approximative
  (le test passe à 5 u sur 30 graines). β n'est estimé que si le bruit de
  ligne dépasse 3 fois le plancher sur au moins 4 bandes.
- *Mode Bayer* (NPS par canal, corrélation entre canaux) : non traité.

## 12.6 Variantes écartées

Le seuil de détection a été construit en mesurant le taux de fausses
alarmes sur des simulations sans motif. Variantes écartées :

1. *Seuil gaussien global* (MAD du rapport) : fausses alarmes
   systématiques sur l'axe f_y = 0, causées par le colonnage fixe (DSNU
   colonne de 0,3 DN du capteur simulé par défaut), dont le périodogramme
   ne se moyenne pas d'une pose à l'autre. D'où la loi Gamma et des seuils
   par région.
2. *Moyennes de colonnes sans retrait de la moyenne de chaque pose* : la
   part d'un bruit de ligne commune à une pose y passe et crée un bruit de
   colonne fantôme (0,48 DN mesuré pour un bruit de ligne en 1/f seul).
3. *NPS mixte* (bruit temporel et motifs fixes dans un même spectre) :
   queue du rapport plus lourde qu'une loi Gamma, avec 20 à 37 %
   d'analyses en fausse alarme. D'où la séparation des recherches
   temporelle et fixe.
4. *ν des bandes d'axes estimé sur une seule ligne* : valeurs aberrantes
   (1 000 à 3 200 au lieu d'environ 400), puis surestimation sur les axes
   à motif fixe. ν y est borné par le ν hors axes et par le nombre de
   positions d'imagettes : 0,7 % d'analyses en fausse alarme au lieu de
   3 %.

## 12.7 Pistes

- Mode Bayer : NPS par canal et corrélation entre canaux d'un produit
  couleur.
- Raies non sinusoïdales : regroupement des harmoniques.
- Recherche dans le NPS 1D de la séquence de lecture (fréquences
  temporelles d'interférence), complémentaire du NPS 2D.
- Sélection automatique des zones homogènes d'une scène (critère de
  gradient, comme `noise_variance`) pour le `mask`.
