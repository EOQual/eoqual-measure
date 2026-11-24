# 10. Bruit temporel d'un push broom et SNR

## 10.1 Objet

Caractérisation **en vol** du bruit temporel d'une barrette push broom, à
partir d'images de scènes peu texturées (`pushbroom_temporal_noise`), puis
SNR aux niveaux spécifiés à partir de tout modèle σ² = a + b·μ
(`snr_model`).

En vol, on ne dispose ni de paires de poses identiques ni de champs
uniformes parfaits : le bruit temporel doit se mesurer dans les images
elles-mêmes, sur des scènes aussi uniformes que possible. La difficulté
est de **séparer le bruit de la texture de la scène**, qui gonfle la
variance. C'est le même problème que celui de `noise_variance`
d'eoqual-measure sur une image quelconque, mais la géométrie du push broom
l'allège :

- le long de la trace, chaque colonne est vue par un seul détecteur. Son
  offset et son gain sont constants, et le bruit change à chaque ligne ;
- en travers, la scène est continue d'un détecteur au voisin, alors que le
  bruit est indépendant.

Objectifs :

- estimer a et b du modèle σ² = a + b·μ, avec des incertitudes ;
- **détecter** quand la scène viole les hypothèses, plutôt que de rendre
  silencieusement une valeur biaisée ;
- en déduire le SNR à des niveaux spécifiés (L_min, L_ref, L_max…), avec
  son incertitude.

**Statut** : validées contre le simulateur S2
(`tests/test_pushbroom_noise.py`, `tests/test_snr.py`, 17 tests ; robustes
sur 30 graines) et étude de biais reproductible
(`examples/pushbroom_texture_bias_study.py`). Module :
`eoqual.measure.radiometry.{pushbroom, snr}`.

## 10.2 Principe

### 10.2.1 Filtre de différence séparable

Soit x(i, j) l'image corrigée de l'obscurité (i le long de la trace, j en
travers). On applique un noyau de différence d'ordre p_∥ le long de la
trace et d'ordre p_⊥ en travers, sur des groupes **disjoints** de
détecteurs voisins (paires pour p_⊥ = 1, triplets pour p_⊥ = 2) :

| Ordre | Noyau normé (Σk² = 1) | Annule |
|---|---|---|
| 0 | [1] | — |
| 1 | [1, −1]/√2 | constante |
| 2 | [1, −2, 1]/√6 | constante et pente |

- **Le long de la trace**, l'ordre 2 élimine l'offset du détecteur (une
  constante) ainsi que le niveau et la pente locaux de la scène sur le
  segment.
- **En travers**, la différence élimine la part de la scène commune aux
  détecteurs du groupe. Le résidu de gain (g_j − g_{j+1})·scène est
  d'ordre PRNU × texture, négligeable.
- Le noyau étant normé, un bruit blanc de variance σ² garde la variance
  σ² au travers du filtre. La variance mesurée correspond à la moyenne des
  variances des détecteurs du groupe, pondérée par k². Comme σ²(μ) est
  affine, elle vaut a + b·μ̄, avec μ̄ le niveau moyen du groupe.

Les groupes étant disjoints, leurs sorties sont indépendantes. Le long de
la trace, en revanche, les valeurs filtrées successives sont corrélées.

### 10.2.2 Statistique d'un segment

Sur un segment de L lignes, le filtre d'ordre p_∥ donne n = L − p_∥
valeurs f_t. On estime σ² par s² = mean(f_t²), sans retirer de moyenne :
celle-ci est nulle par construction sous l'hypothèse de bruit pur.

Les f_t forment un processus à moyenne mobile, d'autocorrélation
ρ_k = Σ_i k_i k_{i+k}. Par exemple, pour [1, −2, 1]/√6 :
ρ₁ = −2/3, ρ₂ = 1/6. Pour un processus gaussien :

Var(s²) = (2σ⁴/n) · Σ_{k=−∞}^{+∞} ρ_k²,  d'où ν = n / Σ ρ_k²

où ν est le nombre de degrés de liberté effectifs : ν = n/1,5 pour
l'ordre 1, n/(1 + 2(4/9 + 1/36)) = n/1,944 pour l'ordre 2. C'est ν qui
fixe les poids de l'ajustement. Le test `test_null_statistics` le vérifie
sur la dispersion des s² simulés.

### 10.2.3 Pourquoi la texture passe, et le rôle de l'optique

La variance filtrée vaut σ²_bruit + ∫ P_scène(f)·|H(f)|² df, où H est la
réponse du filtre : |H|² = 4 sin²(πf)/2 pour l'ordre 1, et
16 sin⁴(πf)/6 pour l'ordre 2, dans chaque direction. Le filtre est un
passe-haut. Il atténue la scène d'autant mieux que celle-ci concentre sa
puissance aux basses fréquences et que l'ordre est élevé.

- Une texture fBm de spectre 2D en 1/f^β garde, pour β ≈ 2, une puissance
  notable jusqu'à Nyquist. Une partie de la scène est alors
  **indiscernable du bruit** dans une seule image, quel que soit le
  filtre.
- L'**optique** sauve la mesure sur un imageur réel : la scène est
  multipliée par FTM², alors que le bruit, ajouté après l'optique, ne
  l'est pas. Avec FTM(Nyquist) ≈ 0,17 (σ_psf = 0,6 px), la puissance de
  la scène à Nyquist est divisée par ~35.

### 10.2.4 Diagnostic de cohérence

Sur les mêmes segments et les mêmes groupes de détecteurs, on calcule
aussi la variance avec l'autre ordre le long de la trace (1 ↔ 2). Sous
l'hypothèse de bruit pur, les deux estimateurs ont la même espérance σ².
Une texture résiduelle passe différemment par les deux filtres (|H|²
différents), d'où un rapport

R = mean(s²_contrôle) / mean(s²_choisi)

qui vaut 1 pour du bruit pur. Son incertitude se calcule par la méthode
delta sur les couples (x_s, y_s) des segments :

u(R) = sd(x − R·y) / (√n · ȳ).

Les deux estimateurs étant calculés sur les mêmes données, ils sont très
corrélés, et u(R) est petit (≈ 0,07 % ici). L'indicateur
`texture_suspected` est levé quand |R − 1| > 5·u(R). Le filtre de
contrôle garde volontairement la même pondération des détecteurs que le
filtre choisi : sinon, des variances inégales entre détecteurs écartent R
de 1 sans aucune texture (−0,3 % mesuré pour des variances de 1 à 2,25).

### 10.2.5 Autocorrélation et rejet de segments (option)

Sous bruit pur, l'autocorrélation empirique au retard 1 des f_t a pour
espérance ≈ ρ₁ et pour variance la formule de Bartlett,
w₁₁/n = Σ_{k≥1} (ρ_{k+1} + ρ_{k−1} − 2ρ₁ρ_k)² / n (écart-type 0,100 pour
l'ordre 2 et n = 30, conforme à la simulation). Une texture la décale vers
le haut. On pourrait donc rejeter les segments dont l'autocorrélation
dépasse ρ₁ + 3 écarts-types.

Ce rejet est **désactivé par défaut**, pour deux raisons :

1. sur un bruit filtré (corrélé), l'autocorrélation et s² ne sont pas
   indépendantes (corrélation ≈ −0,45 mesurée). L'argument de Basu,
   valable pour un échantillon i.i.d., ne s'applique pas. Le rejet garde
   préférentiellement les segments à forte variance et biaise σ² de
   +0,5 % sur bruit pur ;
2. une texture assez faible pour passer le test, mais assez forte pour
   biaiser σ², est la situation courante : l'autocorrélation a peu de
   puissance de détection, contrairement au diagnostic de cohérence.

### 10.2.6 Ajustement σ²(μ)

Chaque segment donne un couple (μ_s, s²_s) de variance 2σ⁴(μ_s)/ν. On
ajuste σ² = a + b·μ par moindres carrés pondérés itératifs, avec des
**poids tirés du modèle**, w_s = ν / (2·(a + b·μ_s)²). Pondérer par la
variance mesurée, 1/s⁴, favoriserait les segments de faible s² et biaiserait
le résultat vers le bas. Cinq itérations suffisent ; un plancher (5 % de
la médiane des s²) protège les itérations initiales d'un modèle négatif.
La covariance (XᵀWX)⁻¹ donne u(a) et u(b). Le χ² réduit contrôle le
modèle.

a est mal contraint par des scènes de 200 à 3 000 DN (±15 %). Si une
obscurité est disponible, on l'impose (`a=dark.temporal_variance`) et
seul b est ajusté.

### 10.2.7 SNR et propagation

SNR(μ) = μ / σ(μ), σ(μ) = √(a + b·μ). La jacobienne de σ par rapport à
(a, b) vaut J = (1/(2σ), μ/(2σ)), d'où u(σ)² = J·cov·Jᵀ et
u(SNR) = SNR · u(σ)/σ. La covariance négative entre a et b (ordonnée à
l'origine et pente) réduit u(σ) par rapport à une propagation qui
l'ignorerait. Si a est imposé, il est tenu pour exact (cov 1 × 1 sur b).

## 10.3 Mise en œuvre

```
eoqual/measure/radiometry/
├── pushbroom.py   measure_pushbroom_noise → PushbroomNoiseResult
│                  KERNELS, kernel_autocorrelation, effective_dof, rho1_null
└── snr.py         snr_at_levels → SNRResult ; protocole NoiseModel (a, b, cov)
```

`snr_at_levels` ne dépend d'aucune mesure. Il accepte tout objet qui porte
`a`, `b` et `cov` : `PTCResult`, `PushbroomNoiseResult`, ou un modèle
externe.

### 10.3.1 `measure_pushbroom_noise(images, *, dark, segment_length=32, along_order=2, across_order=2, rho_threshold=None, a=None)`

1. Contrôle des images (M, W), même W ; offset par détecteur tiré de
   `dark` (scalaire, profil (W,) ou (1, W), `DarkFrameResult`).
2. Pour chaque image : soustraction de l'offset, groupement des
   détecteurs (W // g groupes de g = 1, 2 ou 3), découpage en segments de
   L lignes, filtre en travers puis le long de la trace. Par segment ×
   groupe : niveau moyen, s² choisi, s² de contrôle, autocorrélation.
3. Sélection : signal > 0 (et autocorrélation sous le seuil si
   demandé).
4. Ajustement pondéré itératif (§10.2.6), avec ν = `effective_dof(n, p_∥)`.
5. Rapport de cohérence et son incertitude (§10.2.4).

Coût : quelques opérations vectorisées par image ; 5 images de
1 024 × 513 en ≈ 20 ms.

### 10.3.2 `snr_at_levels(model, levels, *, gain=None)`

Niveaux en DN (ou en e⁻ avec `gain`, qui est alors tenu pour exact) ;
refus des niveaux ≤ 0 et d'un modèle négatif.

### Exemple et recommandations

```python
from eoqual.measure.synthetic import PushbroomSensor, SensorParams, natural_scene, pattern_rng
from eoqual.measure.radiometry import (
    measure_dark_frame, measure_pushbroom_noise, snr_at_levels)

pb = PushbroomSensor(SensorParams(shape=(1, 513), seed=21, dsnu_pixel=0.0, dsnu_col=0.8,
                                  prnu_pixel=0.0, prnu_col=0.01))
t = pb.truth
dark = measure_dark_frame(pb.dark(n_lines=1000, seed=1).image, layout="pushbroom")

images = []
for i, level in enumerate([200, 600, 1200, 2000, 3000]):
    scene = natural_scene((1024, 513), t.photons_for_signal(level), pattern_rng(i, "scene"),
                          contrast=0.03, beta=3.0, psf_sigma=0.6,
                          gradient_along=0.1, gradient_across=0.05)
    images.append(pb.expose(scene.photons, seed=100 + i).image)

r = measure_pushbroom_noise(images, dark=dark, b=dark.temporal_variance)
print(f"a = {r.a:.5f} ± {r.u_a:.5f} (vrai {t.a:.5f}), K = {1 / r.a:.3f} e⁻/DN")
print(f"cohérence {r.consistency:.4f} ± {r.u_consistency:.4f}, texture suspectée : {r.texture_suspected}")
mu, var, u = r.binned(8)              # points de tracé σ²(μ)

snr = snr_at_levels(r, [100.0, 1000.0, 3000.0])
for L, s, us in zip(snr.levels, snr.snr, snr.u_snr):
    print(f"SNR({L:.0f} DN) = {s:.2f} ± {us:.2f}")
```

Recommandations :

- **choisir des zones peu texturées** (désert, neige, océan) et vérifier
  `texture_suspected` avant toute exploitation ;
- **imposer b** à partir d'une obscurité dès qu'elle existe (lignes de
  calibration, océan de nuit) ;
- garder le filtre par défaut [2, 2]. [2, 1] est un peu plus précis sur une
  scène lisse (groupes plus nombreux) mais plus sensible à la texture.
  [2, 0] (détecteur par détecteur) n'est sûr que sur une scène uniforme ;
- plusieurs images à des niveaux différents sont nécessaires pour
  contraindre b. Le niveau de chaque segment suffit, sans avoir besoin de
  niveaux annoncés.

## 10.4 Validation

**Domaine de validité** (`examples/pushbroom_texture_bias_study.py`) : biais de b
en unités de u(b) (≈ 0,2 % de b), pour 5 images de 1 024 × 513 à 200–3 000
DN, gradient 10 % le long et 5 % en travers, PRNU entre détecteurs de 1 %.

| β | c | σ_psf | (b − 1/K)/u [2,2] | (b − 1/K)/u [2,1] | (b − 1/K)/u [1,1] | (b − 1/K)/u [2,0] | cohérence [2,2] |
|---|---|---|---|---|---|---|---|
| 3.0 | 0% | 0.0 | -0.3 | -0.1 | -0.3 | -0.1 | -1.4 u |
| 3.0 | 3% | 0.0 | -1.2 | +0.5 | +1.3 | +1.9 | +2.4 u |
| 3.0 | 3% | 0.6 | +0.7 | -0.4 | -0.5 | -0.1 | -0.6 u |
| 3.0 | 10% | 0.6 | -0.7 | +0.4 | +2.6 | +6.8 | +3.5 u |
| 2.5 | 10% | 0.6 | +7.6 | +18.1 | +42.4 | +80.5 | +25.6 u ⚠ |
| 2.0 | 3% | 0.6 | +6.2 | +14.7 | +31.5 | +58.1 | +22.2 u ⚠ |
| 2.0 | 3% | 1.0 | -0.3 | +0.6 | +5.5 | +13.2 | +3.4 u |
| 2.0 | 10% | 0.6 | +64.3 | +124.5 | +221.8 | +347.9 | +130.1 u ⚠ |
| 2.0 | 10% | 0.0 | +320.4 | +400.9 | +475.7 | +596.7 | +94.9 u ⚠ |

Lecture :

- le filtre [2, 2] est le plus robuste : il reste sans biais pour une
  texture lisse (β = 3) jusqu'à 10 % de contraste, et pour une texture
  rugueuse (β = 2) de 3 % si l'optique est assez floue (σ_psf = 1 px) ;
- **dans tous les cas où [2, 2] est biaisé de plus de 5 u, l'alerte est
  levée** (⚠), et elle ne l'est jamais à tort dans ce tableau ;
- un biais résiduel est toujours **positif** : la texture ajoute de la
  variance.

Liste des tests : §10.4.1.

### 10.4.1 Tests

| Test | Vérifie |
|---|---|
| `test_null_statistics` (ordres 1 et 2) | écart-type de ρ₁ = Bartlett à 5 % ; Var(s²)·ν/(2σ⁴) = 1 à 5 % ; E[s²] = σ² |
| `test_uniform_scene` | a et b à 5 u ; χ² réduit à 0,05 de 1 ; pas d'alerte ; aucun rejet |
| `test_a_from_dark` | a imposé : u_a = 0, cov 1 × 1, b à 5 u |
| `test_smooth_blurred_texture` | β = 3, c = 3 %, σ_psf = 0,6 : b à 5 u, pas d'alerte |
| `test_rough_texture_is_flagged` | β = 2, c = 3 %, σ_psf = 0,6 : b surestimé et alerte levée |
| `test_other_filters_on_uniform_scene` | filtres [1,1], [2,0], [2,1] sur scène uniforme |
| `test_rejection_option`, `test_binned_summary`, `test_input_checks` | option de rejet, résumé par classes, contrôles d'entrée |
| `test_snr_value` | SNR = vrai à 5 u à 30, 300, 3 000 DN |
| `test_uncertainty_by_monte_carlo` | sur 150 PTC indépendantes, sd(SNR) = u(SNR) à 5/√(2(n − 1)) + 5 % |
| `test_levels_in_electrons`, `test_fixed_a_model`, `test_input_checks` | unités, a imposé, contrôles |

## 10.5 Limites

- *Texture rugueuse non floutée* : la mesure est biaisée (§10.2.3), ce que
  l'alerte signale sans le corriger.
- *Bruit de ligne* (commun à tous les détecteurs d'une ligne, S3) : la
  différence en travers l'élimine. C'est donc le bruit **non corrélé**
  entre détecteurs qui est mesuré, et le bruit de ligne demandera une
  mesure propre (`structured_noise`).
- *Bruit corrélé entre détecteurs voisins* (diaphotie, électronique
  partagée) : il réduit la variance de la différence. La mesure le sous-
  estime alors sans le détecter.
- *Saturation* : les segments proches de la saturation ne sont pas exclus
  automatiquement (contrairement à la PTC). Garder des scènes sous ~70 %
  de la saturation.
- *Incertitudes* : elles reposent sur l'hypothèse gaussienne
  (Var(s²) = 2σ⁴/ν), vérifiée à 5 % près sur le simulateur.

## 10.6 Variantes écartées

1. *Tendance linéaire par segment et détecteur, rejet sur
   l'autocorrélation* : biais de +3 u dès 1 % de texture, et de +39 u à
   3 %. La texture résiduelle est trop faible pour l'autocorrélation et
   suffit à gonfler σ². (L'espérance de l'autocorrélation des résidus
   d'une régression linéaire vaut ≈ −2/L.)
2. *Différence entre détecteurs voisins* : forte réduction du biais sur
   texture lisse, mais échec sur texture rugueuse ; d'où le filtre
   passe-haut dans les deux directions (§10.2.1). Sans optique, une
   texture fBm a d'ailleurs trop de puissance à Nyquist pour qu'aucune
   mesure en une image ne la sépare du bruit, situation irréaliste pour un
   imageur : le simulateur applique donc une PSF optique à ses scènes
   ([simulateur, §2.4](../simulateur/modeles.md#24-scènes-naturelles-et-optique-s2)).
3. *Rejet sur l'autocorrélation par défaut* : il biaise σ² de +0,5 % sur
   bruit pur (corrélation entre autocorrélation et s² d'un bruit filtré)
   et le rapport de cohérence de −0,5 %. Il reste une option
   (`rho_threshold`, §10.2.5).

## 10.7 Pistes

- Exclusion automatique des segments proches de la saturation (code
  maximal dans le segment).
- Masque de zones homogènes fourni en entrée (sélection de sites).
- Estimation de la puissance de texture résiduelle à partir du couple
  (s²_choisi, s²_contrôle) et des |H|² connus, pour **corriger** le biais
  au lieu de seulement le signaler.
- Whisk broom : les mêmes filtres, appliqués par détecteur selon le sens
  du balayage.
