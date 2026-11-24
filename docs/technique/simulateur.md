# Simulateur de capteur

Simulateur à vérité connue (`eoqual.measure.synthetic`), qui produit les
données de validation des [mesures radiométriques](radiometrie/index.md). Le
modèle physique du pixel et les sources de bruit (modèle EMVA 1288, bruit de
photons, de lecture, d'obscurité, quantification, écrêtages,
non-uniformités) sont exposés dans leurs
[fondements communs](radiometrie/02-fondements.md) ; ce document décrit leur
simulation et les effets propres au simulateur. Les renvois « §n » désignent
ses propres sections.

Production d'images de capteur à **vérité connue** : chaque grandeur que
les mesures doivent retrouver (offset, DSNU, PRNU, gain, bruit de lecture,
a, b…) est connue exactement, carte par carte.

Les effets sont regroupés en cinq familles, qui servent de repères dans
le code et dans ce document :

| Famille | Effets |
|---|---|
| S0 | capteur matriciel linéaire : photons, PRNU, puits, lecture, gain, offset et DSNU, convertisseur |
| S1 | courant d'obscurité et sa DSNU, loi en température, pixels chauds, lueur d'amplificateur, overscan, dérive d'offset |
| S2 | barrette push broom avec TDI ; scènes naturelles (fBm, gradient lent) et PSF optique |
| S3 | bruit de ligne en 1/f, motifs périodiques, rayons cosmiques, traitements à fonction de transfert connue |
| S4 | transmission optique (vignettage, poussières), non-linéarité, pixels morts et bloqués, pixels télégraphiques, convertisseur à pas inégaux, whisk broom |

**Statut** : chaque famille est vérifiée contre sa propre vérité par les
sept fichiers `tests/test_synthetic_*.py` (62 tests, robustes sur 30
graines). La QE(λ) n'est pas simulée (§7).

## Sommaire

1. Objet
2. Modèles
3. Architecture logicielle
4. Implémentation
5. Guide d'utilisation
6. Validation
7. Limites et choix de modélisation
8. Formulaire
9. Bibliographie

---

## 1. Objet

Une mesure de bruit ne se valide pas sur des images réelles, dont on ne
connaît pas la vérité. Les mesures radiométriques se valident donc sur
un simulateur, comme la MTF d'eoqual-measure se valide sur
`SyntheticSlantedEdge` et le Strehl sur `SyntheticPSF`. Le `synthetic.noise`
d'eoqual-measure ne fait qu'ajouter un bruit σ²(I) = a + b·I à une image :
il ne produit ni piles temporelles, ni motifs fixes, ni capteur dont on
puisse séparer les composantes.

Exigences :

- **physique** : chaque effet est simulé là où il naît (électrons, tension,
  convertisseur), dans l'ordre de la chaîne réelle ;
- **vérité complète** : toutes les cartes et tous les paramètres sont
  exposés, décomposés comme les mesures les décomposent ;
- **reproductible** : graines explicites, motifs fixes séparés du bruit
  temporel ;
- **NumPy seul** (SciPy est permis dans les mesures).

## 2. Modèles

### 2.1 Chaîne S0 et motifs fixes

Chaque pixel suit le modèle linéaire EMVA 1288 ([fondements, §2.1–§2.5](radiometrie/02-fondements.md)) : photons,
rendement quantique et PRNU, bruit de photons poissonnien en électrons, écrêtage au puits, bruit de
lecture, conversion par K, offset et DSNU, quantification et écrêtage du convertisseur. DSNU et
PRNU sont la somme de composantes pixel, ligne et colonne, centrées, tirées chacune dans un flux
aléatoire propre ; la mise en œuvre est détaillée au §4.

### 2.2 Bruits colorés et synthèse spectrale

Un bruit stationnaire se caractérise par sa loi, sa corrélation et sa
**densité spectrale de puissance** P(f). Une famille utile est celle des
lois de puissance, P(f) ∝ 1/f^β : β = 0 pour le bruit blanc, 1 pour le
bruit rose (1/f), 2 pour le bruit brownien (marche aléatoire).

On en tire un échantillon par **synthèse spectrale** : on prend un bruit
blanc gaussien, on multiplie sa transformée de Fourier par |f|^(−β/2),
on annule la composante continue et on revient dans l'espace direct.
Comme la transformée de Fourier d'un bruit blanc gaussien est encore un
bruit blanc gaussien complexe à phases aléatoires, le résultat est un
champ gaussien de spectre exactement 1/f^β en moyenne (Downey, *Think
DSP*, chap. 4). En 2D, f est la fréquence radiale √(f_x² + f_y²) : on
obtient un champ fractal (fBm) isotrope, proche des textures naturelles
et du bruit de Perlin par octaves.

Usages prévus :

- bruit de ligne temporel en 1/f (S3) ;
- scènes peu texturées pour le push broom (S2) ;
- bruit corrélé de spectre connu, pour valider le NPS (S3).

La génération par cumul de pas aléatoires donne bien un spectre brownien,
mais biaisé vers les basses fréquences par l'effet de bord (pente mesurée
d'environ −1,8 au lieu de −2, Downey). La synthèse spectrale n'a pas ce
défaut.

### 2.3 Push broom et TDI (S2)

Une barrette **push broom** compte W détecteurs alignés en travers de la
trace. Le défilement du satellite balaie la scène, et chaque instant
d'intégration produit une ligne de l'image. Dans une image (M, W) :

- une **colonne** est vue par un seul détecteur : elle porte son offset
  (DSNU du détecteur) et son gain (PRNU), constants le long de la trace.
  Une dispersion entre détecteurs produit un **colonnage** ;
- une **ligne** est une acquisition : le bruit temporel est indépendant
  d'une ligne à l'autre. Le bruit temporel se mesure donc le long des
  colonnes, et le bruit spatial fixe le long des lignes.

Le simulateur réutilise le modèle matriciel avec une matrice d'une seule
ligne (1, W) : chaque ligne de l'image est une « pose » de cette matrice.

**TDI** (*Time Delay Integration*) : N lignes de pixels voient
successivement le même point de la scène, et la charge est transférée au
rythme du défilement, ce qui l'accumule sur N étages avant une lecture
unique. Pour Φ photons par étage, μ_e = N·η·Φ·g. Le bruit de photons suit
le signal accumulé (variance μ_e) alors que le bruit de lecture n'est
compté qu'une fois. Le modèle σ² = a + b·μ est donc inchangé, avec
a = (σ_lect/K)² + 1/12 et b = 1/K, mais le SNR à flux égal gagne un facteur ≈ √N
quand le bruit de photons domine.

### 2.4 Scènes naturelles et optique (S2)

Les mesures en vol (push broom) s'appuient sur des scènes peu texturées :
désert, neige, océan de nuit. Leur modèle est

Φ(i, j) = Φ₀ · (1 + g_∥·u_i + g_⊥·v_j) · (1 + c·T(i, j))

- le gradient lent (g_∥, g_⊥) représente la variation d'éclairement le
  long et en travers de la fauchée (angle zénithal solaire, effets
  directionnels) ;
- T est une texture fBm (§2.2) d'écart-type 1, d'exposant β, et c son
  contraste relatif. β ≈ 3 donne une texture lisse, β ≈ 2 une texture
  rugueuse jusqu'à l'échelle du pixel.

**Optique.** La scène atteint le capteur au travers de l'optique, dont la
FTM atténue les hautes fréquences, alors que le bruit du capteur naît après
elle. Le simulateur applique une PSF gaussienne d'écart-type σ_psf,
FTM(f) = exp(−2π²σ_psf²f²) : FTM(Nyquist) ≈ 0,17 pour σ_psf = 0,6 px,
valeur typique d'un imageur spatial. Ce flou est essentiel au réalisme
des mesures de bruit sur scène : sans lui, une texture fBm garde de la
puissance jusqu'à Nyquist, ce qui la rend indiscernable du bruit dans une
seule image ([push broom, §10.2.3](radiometrie/10-pushbroom-snr.md)).

La texture est tirée sur un champ deux fois plus grand puis recadrée, pour
qu'elle ne soit pas périodique (la synthèse par FFT l'est).

### 2.5 Bruits temporels structurés et traitements (S3)

Le bruit des §2.1 à §2.4 est **non corrélé** entre pixels. Une
électronique réelle ajoute des composantes corrélées, qui ne relèvent
pas du modèle σ² = a + b·μ :

- **bruit de ligne** : un offset commun à tous les pixels d'une ligne,
  tiré à chaque lecture (référence ou alimentation de l'amplificateur,
  horloges). Dans une matrice, les lignes sont lues successivement, et la
  séquence des offsets est une série temporelle, souvent en 1/f à basse
  fréquence. Le simulateur tire une **seule** séquence de K·H valeurs pour
  une pile, de spectre 1/f^β, si bien qu'elle est continue d'une pose à
  l'autre comme un vrai bruit en 1/f. En push broom, l'offset est commun à
  une ligne acquise (bruit commun aux détecteurs) ;
- **motifs périodiques** : interférence (horloge, alimentation à
  découpage) ou couplage, A·cos(2π(f_x·j + f_y·i) + φ). Si l'interférence
  n'est pas synchronisée avec la lecture, φ change à chaque pose et le
  motif est **temporel**. Si elle l'est, φ est fixe et le motif est un
  **bruit spatial fixe**. Sa variance vaut A²/2 ;
- **rayons cosmiques** : particules qui déposent une charge (quelques
  milliers d'e⁻) le long d'une trace courte. La charge est ajoutée avant
  l'écrêtage du puits, qui peut donc saturer.

**Traitement à fonction de transfert connue.** Un traitement linéaire de
l'image (rééchantillonnage, dématriçage, débruitage) corrèle le bruit :
un bruit blanc σ² filtré par h a pour NPS σ²·|H(f)|², et pour variance
σ²·Σh². Le module `processing` applique h par convolution **circulaire**,
ce qui rend la relation exacte. Il fournit |H|² sur la grille FFT, qui sert
de vérité au NPS ([chapitre 12](radiometrie/12-nps-bruits-structures.md)).

### 2.6 Courant d'obscurité, offset et overscan (S1)

**Carte du courant.** Le courant d'obscurité varie fortement d'un pixel
à l'autre (défauts du silicium, contraintes). Sa distribution est
asymétrique, avec une queue vers les forts courants. On le modélise par
I_d(i, j) = I_d·g(i, j), avec g log-normale de moyenne 1 :
ln g ~ N(−s²/2, s²), d'où une DSNU relative du courant √(exp(s²) − 1)
(0,307 pour s = 0,3). Une fraction de pixels **chauds** reçoit en plus un
facteur tiré de façon log-uniforme dans [f_min, f_max] (10 à 1 000 par
défaut).

**Loi en température.** Le courant de génération dans la zone de
déplétion suit I_d ∝ T³·exp(−E_g/(2kT)), avec E_g ≈ 1,12 eV pour le
silicium. Le doublement exact, solution de I(T + ΔT) = 2·I(T), vaut
8,29 K à 20 °C. L'approximation au premier ordre, ln 2 / (d ln I/dT),
donne 8,07 K : elle sous-estime ΔT d'environ 2,6 %, et un pas de 8,07 K
ne multiplie le courant que par 1,965. Les « 7 à 8 °C » des fiches
techniques ([fondements, §2.2](radiometrie/02-fondements.md)) en sont l'ordre de grandeur.

**Lueur d'amplificateur.** L'amplificateur de sortie dissipe de la
chaleur et émet de la lumière. Il en résulte une charge additionnelle,
linéaire en temps de pose, maximale au coin (0, 0), où il est placé, et
qui décroît comme exp(−d/λ) avec la distance d. Elle ne suit pas la loi
en température.

**Variations d'offset.** L'offset peut varier d'une pose à l'autre de
façon commune à toute l'image : gigue (tirage gaussien par pose) et dérive
linéaire (thermique). Ce sont les défauts que contrôle la stationnarité
de `dark_frame` ([chapitre 3, §3.2.2](radiometrie/03-obscurite-ptc.md)).

**Overscan.** Ce sont des colonnes lues sans être éclairées. Elles portent
l'offset de ligne, le bruit de lecture, le courant d'obscurité moyen, le
bruit de ligne et les variations d'offset de la pose, mais ni la DSNU
pixel, ni la PRNU, ni les pixels chauds. Leur moyenne par pose suit donc
la variation d'offset et permet de la corriger.

**Charge d'obscurité dans la chaîne.** La charge moyenne (I_d(T)·g +
lueur)·t s'ajoute au signal moyen **avant** le tirage de Poisson. Comme une
somme de variables de Poisson est poissonnienne, le bruit d'obscurité est
juste (variance d_e), et sans courant d'obscurité les tirages sont
inchangés.

### 2.7 Transmission optique, non-linéarité, défauts de réponse (S4)

**Transmission optique.** Elle multiplie la *lumière*, avant la PRNU :
μ_e = η·Φ·T·g.

- Vignettage : T_v = 1 − v·ρ², avec ρ la distance au centre normalisée à
  1 au coin, soit une baisse relative v aux coins.
- Poussières : l'ombre défocalisée d'une poussière sur un filtre ou une
  fenêtre forme un anneau. On le modélise par une gaussienne de rayon R et
  de largeur R/4, et de profondeur d. Les anneaux qui se chevauchent
  multiplient leurs atténuations.

Dans un champ uniforme, T est indiscernable de la PRNU ; elle s'en
sépare par son échelle spatiale ([chapitre 5](radiometrie/05-prnu.md)).

**Non-linéarité.** Elle est appliquée aux électrons collectés, après
l'écrêtage du puits : e' = e·(1 − β·e/FW), avec β = β₀ + δβ(i, j).
β > 0 comprime le signal, β < 0 le dilate.

- Moyenne : pour e poissonnien, E[e'] = μ − β(μ² + μ)/FW, exactement.
- Variance, au premier ordre : Var(e') ≈ (1 − 2βμ/FW)²·μ. La PTC
  s'infléchit, comme sur un capteur réel.
- En DN, avec x le signal linéaire : y ≈ x − c·x², où c = βK/FW. La
  courbure κ = q/s² d'un ajustement y = s·x + q·x² vaut donc −βK/FW.

**Défauts de réponse.** Les **pixels morts** ne répondent pas à la
lumière, mais gardent l'offset, le bruit de lecture et le courant
d'obscurité. Les **pixels bloqués** sortent un code constant, tiré
uniformément sur [0, 2ⁿ − 1], sans bruit. Les deux ensembles sont
disjoints, et distincts des pixels chauds (S1).

### 2.8 Pixels télégraphiques, convertisseur, whisk broom (S4)

**Pixels télégraphiques (RTS).** Un piège de charge près du transistor de
lecture capture et libère un électron au hasard, et la sortie bascule
entre deux niveaux séparés de A. L'état suit une chaîne de Markov à deux
états d'une pose à l'autre (d'une ligne à l'autre en push broom) : on
passe du niveau bas au haut avec la probabilité p_↑, du haut au bas avec
p_↓. Les durées moyennes de séjour valent 1/p_↑ et 1/p_↓, et la fraction de
temps passée à l'état haut vaut p_↑/(p_↑ + p_↓). L'état initial suit la loi
stationnaire. A est tiré uniformément dans une plage (DN), ainsi que p_↑
et p_↓, séparément.

**Convertisseur à pas inégaux.** Le code c correspond aux tensions de
[t_{c−1}, t_c[. Un convertisseur idéal a t_c = c + ½ ; ici
t_c = c + ½ + δ_c, avec δ_c gaussien d'écart-type s_DNL, rendu monotone.
Un code est forcé manquant en posant t_m = t_{m−1}. On en déduit :

- largeur w_c = t_c − t_{c−1} ;
- DNL_c = w_c − 1 (LSB) ;
- INL_c = t_c − (c + ½).

Deux codes voisins ont des DNL anticorrélées, puisqu'un seuil déplacé
élargit l'un et rétrécit l'autre d'autant. La somme des largeurs de codes
contigus se télescope.

**Whisk broom.** k détecteurs balaient ensemble k lignes de l'image, une
par détecteur, et les balayages alternent l'aller et le retour. Chaque
détecteur a son offset et son gain, d'où des **bandes de période k
lignes**. Un balayage retour peut avoir un offset et un gain propres
(hystérésis de l'électronique ou du mécanisme), d'où une structure de
période 2k. Le bruit temporel se mesure le long d'une ligne, dont les
pixels sont des acquisitions successives d'un même détecteur.

## 3. Architecture logicielle

```
eoqual/measure/synthetic/
├── __init__.py      API publique
├── patterns.py      générateurs : pattern_rng, structured_map, power_law_noise
├── scene.py         Scene, natural_scene (S2)
├── temporal_effects.py  PeriodicPattern, row_offsets, periodic_field, cosmic_rays (S3) ;
│                    RTSMap, rts_map, rts_states (S4)
├── processing.py    apply_kernel, kernel_power_transfer, gaussian_kernel (S3)
├── dark.py          dark_current_map, temperature_factor, doubling_temperature (S1)
├── response.py      transmission_map, defect_maps, adc_transfer (S4)
└── sensor.py        SensorParams, SensorTruth, MatrixSensor, Acquisition,
                     PushbroomSensor, PushbroomAcquisition (S2),
                     WhiskbroomSensor, WhiskbroomAcquisition (S4)
```

| Objet | Rôle |
|---|---|
| `SensorParams` | paramètres d'un capteur, figés (`frozen`), validés à la construction |
| `SensorTruth` | vérité : cartes, décompositions, a, b, seuils ; méthodes `expected_mean`, `expected_variance` |
| `MatrixSensor` | tire les motifs fixes à la construction, produit des piles avec `expose` / `dark` |
| `Acquisition` | résultat d'une pose : pile `uint16` (K, H, W), photons, graine, vérité |
| `StructuredMap` | carte décomposée `total`, `pixel`, `row`, `col` |
| `PushbroomSensor` | barrette (1, W), `tdi_stages` ; `expose(scene ou niveau, n_lines=, seed=)` |
| `PushbroomAcquisition` | image `uint16` (M, W), photons, graine, vérité |
| `Scene` / `natural_scene` | scène en photons et ses composantes (éclairement, texture) |

Principes :

- **la vérité est construite en même temps que le capteur**. Une mesure
  reçoit `acq.frames` et le test compare son résultat à `acq.truth` ;
- **résultats typés** : dataclasses `slots=True`, sur le modèle de
  `NoiseEstimate` d'eoqual-measure ;
- **un effet = un motif tiré dans son propre flux aléatoire**, nommé par
  une étiquette (§4.3).

## 4. Implémentation

### 4.1 Construction d'un capteur (`MatrixSensor.__init__`)

1. DSNU : `structured_map(shape, dsnu_pixel, dsnu_row, dsnu_col,
   pattern_rng(seed, "dsnu"))`, en DN.
2. PRNU : de même, avec l'étiquette `"prnu"`, sans unité. Le capteur est
   refusé si 1 + PRNU ≤ 0 sur un pixel.
3. Vérité : `offset_map = offset + DSNU`, `gain_map = 1 + PRNU`,
   a = (σ_lect/K)² + 1/12, b = 1/K, `saturation_dn` = 2ⁿ − 1.

`structured_map` tire les trois composantes gaussiennes et **centre**
chacune d'elles (moyenne spatiale nulle). L'offset moyen et le gain moyen
sont donc exactement ceux demandés, et les composantes sont celles que
les mesures doivent retrouver.

### 4.2 Acquisition (`MatrixSensor.expose`)

Pour une pile de forme (K, H, W) :

```text
mu_e = qe * photons * gain_map                 # e⁻, carte (H, W)
e = rng.poisson(mu_e, size=(K, H, W))          # bruit de photons
e = minimum(e, full_well)                      # puits
e = e + rng.normal(0, read_noise, size)        # bruit de lecture
y = e / K + offset_map                         # DN analogique
dn = clip(floor(y + 0.5), 0, 2**n - 1)         # convertisseur, uint16
```

`photons` est un scalaire (champ uniforme) ou une carte (H, W) (scène).
Il est positif, en photons moyens par pixel pendant la pose.
`dark(n_frames, seed)` équivaut à `expose(0, ...)`.

Mémoire : la pile est calculée en `float64` avant conversion, soit
8·K·H·W octets (32 Mo pour 64 poses de 256 × 256).

### 4.3 Graines et flux aléatoires

| Flux | Entropie du générateur | Rôle |
|---|---|---|
| motif `tag` | `[seed, crc32(tag)]` | DSNU (`"dsnu"`), PRNU (`"prnu"`), plus tard pixels chauds, défauts… |
| bruit temporel | `[seed, crc32("temporal"), seed_pose]` | Poisson et bruit de lecture d'une pile |

Conséquences :

- même `SensorParams` → même capteur, au bit près ;
- changer un paramètre de PRNU ne change pas la DSNU (test
  `test_fixed_pattern_does_not_depend_on_other_patterns`) ;
- ajouter un effet en S1 ne modifiera pas les cartes S0 d'un capteur
  existant ;
- deux poses de même `seed_pose` sont identiques, deux graines
  différentes donnent deux réalisations indépendantes du même capteur.

La graine de pose est obligatoire (argument nommé sans valeur par défaut) :
aucune simulation ne dépend d'un état global.

### 4.4 Bruit coloré (`power_law_noise`)

```text
white = rng.standard_normal(shape)
F = fftn(white) * |f|^(−β/2)      # f radiale, f[0] remplacée par 1
F[0] = 0                          # moyenne nulle
x = real(ifftn(F)); x /= x.std()  # écart-type 1
```

Le champ obtenu est **périodique** (synthèse par FFT) : ses bords opposés
se raccordent. Pour une scène non périodique, tirer un champ plus grand et
en extraire une partie, ce que fait `natural_scene`.

### 4.5 Barrette push broom (`PushbroomSensor`)

Les deux capteurs partagent la même chaîne temporelle (`_digitize` :
Poisson → puits → lecture → ÷K + offset → convertisseur) et la même
construction de la vérité (`_build_truth`).

- `params.shape` vaut (1, W). La composante ligne des motifs est nulle par
  construction, et la dispersion entre détecteurs se règle par les
  composantes `*_col` ou `*_pixel`.
- `expose(photons, n_lines=M, seed=)` : `photons` est un scalaire (scène
  uniforme, `n_lines` obligatoire) ou une scène (M, W). Le signal moyen
  N·η·Φ·g est diffusé sur les lignes, et le bruit tiré pour chaque pixel
  de l'image.
- `truth.detector_offset` et `truth.detector_gain` (W,) donnent l'offset
  et le gain de chaque détecteur ; `truth.tdi_stages` vaut N.

### 4.6 Scènes (`natural_scene`)

1. Texture : `power_law_noise((2M, 2W), β)` ; si σ_psf > 0, produit par
   la FTM gaussienne dans le domaine de Fourier ; recadrage à (M, W) ;
   recentrage et normalisation à un écart-type de 1, si bien que `contrast`
   est le contraste *après* optique.
2. Éclairement : 1 + g_∥·u + g_⊥·v, avec u, v les coordonnées réduites des
   centres de pixels dans [−½, ½].
3. Φ = Φ₀ · éclairement · (1 + c·T). La fonction refuse toute scène
   négative ou nulle.

### 4.7 Bruits structurés et traitements (S3)

Paramètres de `SensorParams`, tous nuls par défaut, de sorte que les
capteurs S0 et S2 sont inchangés au bit près :

- `row_noise` (DN), `row_noise_beta` ;
- `periodic` : tuple de `PeriodicPattern(amplitude, fx, fy, temporal,
  phase)` ;
- `cosmic_rate` (traces par pose), `cosmic_charge` (e⁻ par pixel),
  `cosmic_max_length` (pixels).

Chaque effet tire dans son propre flux, `[seed, crc32(étiquette),
graine_de_pose]` (étiquettes `"row_noise"`, `"periodic"`, `"cosmic"`). Les
charges cosmiques entrent dans la chaîne commune avant le puits ; le bruit
de ligne et les motifs, après la conversion en DN. Les réalisations sont
renvoyées pour la vérité : `acq.row_offsets` (K, H), `acq.periodic`
(K, H, W), `acq.cosmic_mask` (K, H, W), et en push broom
`line_offsets` (M,), `periodic` (M, W), `cosmic_mask` (M, W).
`truth.correlated_variance` donne la variance par pixel due aux effets
temporels corrélés, σ_ligne² + Σ A²/2.

Note : la séquence de bruit de ligne est normalisée à un écart-type
réalisé exactement égal à `row_noise`. En 1/f, la variance *temporelle*
par pixel (d'une pose à l'autre) est plus petite que σ_ligne², car la part
basse fréquence varie peu entre poses voisines. Le test compare à la
variance réalisée par ligne.

### 4.8 Courant d'obscurité et overscan (S1)

Paramètres de `SensorParams`, nuls par défaut : `dark_current` (e⁻/s à
`temperature_ref`), `dark_nonuniformity` (s), `hot_pixel_fraction`,
`hot_pixel_range`, `temperature_ref` (°C), `bandgap` (eV), `glow_rate`
(e⁻/s), `glow_scale` (pixels), `bias_jitter` (DN), `bias_drift` (DN par
pose), `overscan_columns`.

- `expose(..., exposure_time=1.0, temperature=None)` et `dark(...)` :
  temps de pose (s) et température (°C). `photons` reste la quantité
  intégrée sur la pose.
- Vérité : `truth.dark_current` (`DarkCurrentMap` : `current`,
  `relative`, `hot`, `hot_factor`, `glow`), `truth.hot_pixels`,
  `truth.dark_electrons(t, T)`, `truth.expected_mean(photons, t, T)`,
  `truth.overscan_offset` (H, n) = offset + composante ligne de la DSNU.
- Réalisations : `acq.bias_offsets` (K,), `acq.overscan` (K, H, n),
  `acq.exposure_time`, `acq.temperature`.
- Flux aléatoires propres : `"dark_current"`, `"hot_pixels"` (motifs
  fixes), `"bias"`, `"overscan"` (temporels).
- Push broom : courant par détecteur, temps d'intégration d'une ligne ;
  gigue d'offset tirée une fois par image et dérive par ligne. Pas
  d'overscan (refusé).

### 4.9 Transmission, non-linéarité, défauts de réponse (S4)

Paramètres : `vignetting`, `n_dust`, `dust_depth`, `dust_radius`,
`nonlinearity` (β₀), `nonlinearity_dispersion` (écart-type de δβ),
`dead_pixel_fraction`, `stuck_pixel_fraction`. Flux propres :
`"transmission"`, `"nonlinearity"`, `"defects"`.

- Vérité : `truth.transmission` (`Transmission` : `total`, `vignetting`,
  `dust`, centres et rayons), `truth.nonlinearity_map` (β par pixel),
  `truth.defects` (`Defects` : `dead`, `stuck`, `stuck_value`).
  `signal_electrons` inclut T et annule les pixels morts ;
  `expected_mean` et `expected_variance` incluent la non-linéarité (terme
  de Poisson du second ordre pour la moyenne, compression au premier ordre
  pour la variance) et les pixels bloqués.
- Chaîne : la non-linéarité s'applique dans `_digitize` après le puits ;
  le code des pixels bloqués remplace la sortie du convertisseur.
- Les cartes (H, W) ou (1, W) sont diffusées, si bien que les effets
  valent aussi en push broom. Les colonnes d'overscan, pixels virtuels,
  n'en reçoivent aucun.

### 4.10 Pixels télégraphiques, convertisseur, whisk broom (S4)

- `rts_fraction`, `rts_amplitude` (DN), `rts_rate` (p_↑, p_↓ par pose) ;
  flux `"rts"` (carte) et `"rts_state"` (chaîne de Markov par pose ou par
  ligne). Vérité : `truth.rts` (`RTSMap` : `index`, `amplitude`, `p_up`,
  `p_down`, `high_fraction`) ; réalisation `acq.rts_states` (K, n_rts).
  Le saut s'ajoute en DN.
- `adc_dnl` (LSB), `adc_missing_codes` ; flux `"adc"`. Vérité :
  `truth.adc` (`ADCTransfer` : `thresholds`, `widths`, `missing`, `dnl`,
  `inl`). `_digitize` quantifie alors par `searchsorted` sur les seuils au
  lieu de l'arrondi.
- `WhiskbroomSensor(params, scan_offset=, scan_gain=)` : `params.shape` =
  (1, k). `expose(photons, n_scans=, width=, seed=)` renvoie l'image
  (n_scans·k, W), le détecteur et le sens de chaque ligne, et le profil
  attendu. Il ne simule ni les effets S3, ni les pixels télégraphiques,
  ni l'overscan.

## 5. Guide d'utilisation

Installation : celle d'eoqual-measure (`README.md`) ; le simulateur
n'ajoute aucune dépendance (NumPy seul). Tests :

```bash
python -m pytest tests/test_synthetic_*.py
```

**Un capteur, une pile d'obscurité, une pile de champ uniforme :**

```python
from eoqual.measure.synthetic import MatrixSensor, SensorParams

sensor = MatrixSensor(SensorParams(seed=1))
truth = sensor.truth

dark = sensor.dark(n_frames=32, seed=10)               # (32, 256, 256) uint16
photons = truth.photons_for_signal(1000.0)             # 1 000 DN au-dessus de l'offset
flat = sensor.expose(photons, n_frames=2, seed=11)

print(truth.a, truth.b)                 # 1/K et σ²(0) en DN²
print(truth.dsnu_rms, truth.prnu_rms)   # DSNU (DN) et PRNU (relative)
```

**Capteur à défauts de colonnes marqués** (pour la décomposition
ligne/colonne) :

```python
p = SensorParams(seed=2, dsnu_col=2.0, prnu_col=0.01, dsnu_row=0.5)
sensor = MatrixSensor(p)
sensor.truth.dsnu.col      # composante colonne vraie (W,)
```

**Valider une mesure contre la vérité** (modèle d'un test) :

```python
import numpy as np

acq = sensor.dark(n_frames=64, seed=3)
offset_estimate = acq.frames.mean(axis=0)               # mesure « naïve »
error = offset_estimate - acq.truth.offset_map
assert abs(error.std() - np.sqrt(acq.truth.b / 64)) < 0.05
```

**Série pour une PTC :**

```python
levels = [100, 400, 1000, 2000, 3000]                   # DN
pairs = [sensor.expose(truth.photons_for_signal(s), n_frames=2, seed=100 + i)
         for i, s in enumerate(levels)]
```

**Saturation par le puits plutôt que par le convertisseur :** choisir
`full_well < K·(2ⁿ − 1 − offset)`, par exemple `full_well=6000` avec les
valeurs par défaut.

**Barrette push broom et scène naturelle :**

```python
from eoqual.measure.synthetic import PushbroomSensor, SensorParams, natural_scene, pattern_rng

pb = PushbroomSensor(SensorParams(shape=(1, 1024), seed=3, dsnu_pixel=0.0, dsnu_col=0.8,
                                  prnu_pixel=0.0, prnu_col=0.01), tdi_stages=4)
t = pb.truth
scene = natural_scene((2000, 1024), t.photons_for_signal(1500.0), pattern_rng(0, "scene"),
                      contrast=0.02, beta=3.0, psf_sigma=0.6,
                      gradient_along=0.1, gradient_across=0.05)
img = pb.expose(scene.photons, seed=4).image             # (2000, 1024) uint16
dark = pb.dark(n_lines=500, seed=5).image                # bloc d'obscurité
t.detector_offset, t.detector_gain                       # vérité par détecteur
```

**Bruits structurés (S3) et traitement :**

```python
from eoqual.measure.synthetic import (MatrixSensor, SensorParams, PeriodicPattern,
                                  apply_kernel, gaussian_kernel, kernel_power_transfer)

s3 = MatrixSensor(SensorParams(seed=7, row_noise=1.0, row_noise_beta=1.0,
                               periodic=(PeriodicPattern(0.5, fx=0.125, fy=0.0625),
                                         PeriodicPattern(0.3, fx=0.2, fy=0.0, temporal=False)),
                               cosmic_rate=3.0))
acq = s3.dark(n_frames=16, seed=1)
acq.row_offsets, acq.periodic, acq.cosmic_mask           # réalisations (vérité)

k = gaussian_kernel(1.0)
processed = apply_kernel(acq.frames.astype(float), k)    # bruit corrélé
h2 = kernel_power_transfer(k, (64, 64))                  # |H|² : NPS attendu = σ²·|H|²
```

**Courant d'obscurité, température, overscan (S1) :**

```python
s1 = MatrixSensor(SensorParams(seed=8, dark_current=2.0, dark_nonuniformity=0.3,
                               hot_pixel_fraction=1e-3, glow_rate=20.0,
                               bias_jitter=0.5, overscan_columns=16))
cold = s1.dark(n_frames=16, seed=1, exposure_time=60.0, temperature=10.0)
warm = s1.dark(n_frames=16, seed=2, exposure_time=60.0, temperature=30.0)
s1.truth.dark_electrons(60.0, temperature=30.0)          # charge vraie par pixel (e⁻)
cold.overscan.mean(axis=(1, 2))                          # suit cold.bias_offsets
```

**Robustesse des tolérances** : la variable `EOQUAL_TEST_SEED` décale
les graines des capteurs de test.

```bash
for s in $(seq 0 29); do EOQUAL_TEST_SEED=$s .venv/bin/python -m pytest -q | tail -1; done
```

### Valeurs par défaut (`SensorParams()`)

| Paramètre | Valeur | Unité |
|---|---|---|
| `shape` | (256, 256) | pixels |
| `qe` | 0,7 | e⁻/photon |
| `conversion_gain` (K) | 3,0 | e⁻/DN |
| `read_noise` | 5,0 | e⁻ |
| `full_well` | 15 000 | e⁻ |
| `bit_depth` | 12 | bits |
| `offset` | 100 | DN |
| `dsnu_pixel` / `dsnu_row` / `dsnu_col` | 0,5 / 0 / 0,3 | DN |
| `prnu_pixel` / `prnu_row` / `prnu_col` | 0,5 % / 0 / 0,2 % | — |

Grandeurs dérivées : a = 2,86 DN² (σ_lect = 1,67 DN) ; b = 0,333 ;
saturation du convertisseur à 4 095 DN, avant le puits (5 100 DN).

## 6. Validation

Le simulateur est vérifié contre sa propre vérité. Chaque tolérance est
déduite de l'incertitude statistique de l'estimateur ([fondements, §2.6](radiometrie/02-fondements.md)), en général à
5 σ, et non fixée au jugé.

| Test | Vérifie |
|---|---|
| `test_same_seed_same_sensor`, `test_other_seed_other_sensor` | identité du capteur par sa graine |
| `test_temporal_seed` | reproductibilité et indépendance des poses |
| `test_fixed_pattern_does_not_depend_on_other_patterns` | séparation des flux aléatoires |
| `test_maps_are_centred` | offset moyen et gain moyen exacts |
| `test_component_amplitudes` | σ_p, σ_r, σ_c de la DSNU et de la PRNU, à 5/√(2(n−1)) près |
| `test_dark_stack_mean_and_variance` | moyenne temporelle = carte d'offset, à √(a/K) ; variance = a |
| `test_flat_mean_and_variance` (50, 500, 2 500 DN) | E[y] et Var[y] = a + b·μ |
| `test_photon_transfer_slope_is_one_over_gain` | pente de la PTC par paires = 1/K à 2 % |
| `test_adc_saturation`, `test_full_well_clipping` | plafonds ; variance réduite à a au plateau du puits |
| `test_output_format`, `test_scene_map_input` | type `uint16`, forme, contrôles d'entrée |
| `test_power_law_1d_slope`, `test_power_law_2d_radial_slope` | pente spectrale −β à 0,1 près |
| `test_power_law_normalised` | moyenne 0, écart-type 1 |
| `test_detector_statistics` (push broom) | moyenne par détecteur = offset + μ·g ; variance = a + b·μ |
| `test_lines_are_independent` | autocorrélation le long de la trace nulle à 5/√n près |
| `test_tdi` (N = 1, 4, 16) | signal ×N à flux par étage égal ; variance = a + b·μ |
| `test_natural_scene_components`, `test_scene_not_periodic` | texture normée, gradient exact, texture non périodique |
| `test_scene_psf_attenuates_high_frequencies` | spectre 2D de la texture × FTM² exactement (champ périodique) |
| `test_effects_off_by_default` | S3 désactivé par défaut |
| `test_row_noise` (β = 0 et 1) | moyennes de lignes = offsets réalisés à √(a/W) près ; variance temporelle = a + variance par ligne des offsets |
| `test_row_noise_spectrum` | pente −1 à 0,15 près pour β = 1 |
| `test_periodic_temporal_and_fixed` | amplitude projetée exacte, phase variable (temporel) ou constante (fixe) |
| `test_cosmic_rays` | charge/K sur les pixels touchés ; écrêtage au puits |
| `test_pushbroom_line_noise` | bruit commun à une ligne acquise |
| `test_apply_kernel_nps` | NPS moyen = σ²·\|H\|² ; variance σ²·Σh² ; noyau centré |
| `test_s3_params_validation` | contrôles d'entrée |
| `test_off_by_default` (S1) | sans courant, le temps de pose est sans effet |
| `test_dark_mean_and_poisson_variance` | moyenne = offset + d_e/K ; variance = a + d_e/K² |
| `test_dark_linear_in_exposure_time` | pente = courant moyen / K à 1 % |
| `test_temperature_law` | doublement exact 8,29 K à 20 °C |
| `test_lognormal_map_and_hot_pixels` | moyenne 1, DSNU √(e^{s²} − 1), nombre et facteurs (log-uniformes) des pixels chauds |
| `test_amplifier_glow` | lueur au coin, décroissance exp(−d/λ) |
| `test_bias_and_overscan_track_each_other` | l'overscan suit la variation d'offset ; correction de l'offset actif |
| `test_pushbroom_dark_and_drift`, `test_s1_validation` | push broom ; contrôles d'entrée |
| `test_transmission` (S4) | vignettage 1 − v aux coins ; poussières dans [(1 − d)ⁿ, 1 − 0,07] ; flat moyen = T·g |
| `test_nonlinearity_mean_and_variance` (500, 3 000 DN) | moyenne (terme de Poisson) et variance comprimée ; compression visible ; dispersion de β |
| `test_dead_and_stuck_pixels` | nombres exacts, ensembles disjoints ; morts à l'offset avec variance a ; bloqués constants |
| `test_pushbroom_response_effects`, `test_s4_validation` | effets en push broom ; contrôles d'entrée |
| `test_rts_levels_and_rates` | écart des niveaux = amplitude ; fréquences de bascule à 5/√n ; fraction haute stationnaire |
| `test_rts_pushbroom_along_lines` | chaîne de Markov le long de la trace |
| `test_ideal_adc_equals_rounding` | sans DNL, quantification = arrondi |
| `test_unequal_steps_and_missing_codes` | codes manquants jamais produits ; occurrences ∝ largeur sur une rampe |
| `test_whiskbroom_bands_and_scan_direction` | profil des lignes = attendu ; différence aller/retour |
| `test_s4_rest_validation` | contrôles d'entrée |
| `test_shapes_and_format`, `test_scene_input` | formes, type, contrôles d'entrée |

Robustesse : les 62 tests passent pour 30 décalages de graine
(`EOQUAL_TEST_SEED` de 0 à 29).

## 7. Limites et choix de modélisation

**Limites du modèle**

- *Quantification* : le terme 1/12 de a suppose σ_lect/K ≳ 0,5 DN. Pour
  un capteur à très faible bruit (σ < 0,5 DN), la variance réelle dépend
  de la position de l'offset entre deux codes. Le simulateur reste exact
  (il arrondit vraiment), mais `truth.a` n'est plus qu'une approximation.
- *Saturation* : `expected_mean` et `expected_variance` ignorent les
  écrêtages. Près du plafond, comparer aux images et non à ces formules.
- *Puits* : pas de débordement vers les voisins (*blooming*), pas de
  perte d'efficacité de transfert (CTE), réponse parfaitement linéaire
  jusqu'au plafond.
- *PRNU gaussienne* : non bornée. Le capteur est refusé si un gain
  devient négatif ou nul, ce qui n'arrive que pour des PRNU irréalistes
  (plus de 20 % environ).
- *Pas de dépendance spectrale* : η est un scalaire. La QE(λ) relève de
  l'étalonnage spectral et absolu, hors du périmètre des mesures
  contrôlées, et aucune mesure n'en dépend.
- *Push broom* : un gain par détecteur. En TDI, la dispersion propre des
  N étages (moyennée par le TDI) et le filé (défaut de synchronisation du
  transfert) ne sont pas simulés.
- *Scènes* : texture gaussienne isotrope, sans structure (routes,
  parcelles, nuages) ni réflectance bornée. PSF gaussienne seulement.
- *S3* : pas de bruit de colonne temporel, pas de diaphotie ; rayons
  cosmiques en segments droits à dépôt uniforme (ni diffusion de charge, ni
  distribution de Landau).
- *S1* : un seul mécanisme de courant (génération en zone de déplétion,
  E_g/2) pour tous les pixels, pixels chauds compris (ils ont souvent une
  énergie d'activation différente) ; lueur purement exponentielle ; pas de
  persistance d'image (rémanence) ni de courant d'obscurité dépendant du
  signal.
- *S4, transmission et réponse* : vignettage quadratique isotrope ; non-linéarité
  quadratique seulement (ni « deux pentes », ni coude près de la
  saturation) ; pixels bloqués à un code uniforme (souvent, en réalité, le
  code maximal ou zéro).
- *S4, RTS, convertisseur, whisk broom* : RTS à deux niveaux seulement (pas de pièges multiples) ;
  DNL gaussienne non corrélée d'un seuil à l'autre (pas de motif lié aux
  bits de poids fort) ; whisk broom sans dépendance temporelle à
  l'intérieur d'un balayage.

**Choix de modélisation**

- *Capacité du puits par défaut de 15 000 e⁻* : avec 12 000 e⁻, le plateau
  du puits (4 000 + 100 DN) tombe au niveau du code maximal (4 095 DN), et
  les deux saturations se mêlent selon la DSNU du pixel. À 15 000 e⁻, elles
  restent distinctes.
- *PSF optique appliquée aux scènes* : sans optique, une texture fBm a trop
  de puissance à Nyquist pour qu'aucune mesure en une image ne la sépare du
  bruit, situation irréaliste pour un imageur.
- *Température de doublement exacte* : `doubling_temperature` résout
  I(T + ΔT) = 2·I(T) (8,29 K à 20 °C) ; l'écart au premier ordre (8,07 K)
  ne multiplie le courant que par 1,965.
- *Push broom* : la non-linéarité et les défauts s'appliquent aussi à une
  carte (1, W) diffusée sur M lignes ; l'overscan en est exclu
  explicitement.

**Pièges courants des simulateurs de bruit, évités ici**

| Piège | Conséquence | Choix retenu |
|---|---|---|
| Poisson du courant d'obscurité tiré en DN (`poisson(I_d·t/K)`) | variance fausse d'un facteur K | tout tirage de Poisson en électrons |
| Arrondi des électrons après application de η, et du bruit de lecture | discrétisation parasite avant le convertisseur | Poisson(η·μ_p) direct, seul arrondi au convertisseur |
| Conversion par `astype(int16)` | troncature au lieu d'arrondi, repli des valeurs négatives | `floor(y + 0.5)` puis écrêtage à [0, 2ⁿ − 1], `uint16` |
| Deux gains incohérents (5,88 DN/e⁻ et 1 e⁻/DN) | modèle de bruit inexploitable | un seul K (e⁻/DN), b = 1/K |
| Mauvaises colonnes multipliant l'image finale en DN | l'offset est lui aussi multiplié | colonne = offset (DSNU) ou gain du signal (PRNU) |
| Motifs fixes et bruit temporel tirés du même générateur | ajouter un effet change tous les tirages | un flux par motif, nommé par étiquette |

## 8. Formulaire

| Grandeur | Expression | Unité |
|---|---|---|
| Signal en électrons | μ_e = η·μ_p·(1 + PRNU) | e⁻ |
| Signal en DN au-dessus de l'offset | μ = μ_e / K | DN |
| Moyenne d'un pixel | E[y] = offset + DSNU + μ | DN |
| Variance temporelle | σ²(μ) = a + b·μ | DN² |
| Variance à signal nul (plancher) | a = (σ_lect² + σ_d²)/K² + 1/12 | DN² |
| Pente | b = 1/K = K_E | DN/DN |
| Photons pour un signal s | μ_p = s·K/η | photons |
| Bruit temporel par paire | σ_temp² = var(x₁ − x₂)/2 | DN² |
| Bruit spatial corrigé | σ_spatial² = s²(moyenne de K poses) − σ_temp²/K | DN² |
| Variance des moyennes de lignes | σ_r² + σ_p²/W | DN² |
| Incertitude relative d'une variance | √(2/(n − 1)) | — |
| SNR | μ / √(a + b·μ) | — |
| Courant d'obscurité (S1) | d_e = I_d(T)·t_exp, I_d ∝ T³·exp(−E_g/2kT) | e⁻ |
| TDI à N étages | μ_e = N·η·Φ·g ; σ² = a + b·μ inchangé | e⁻ |
| FTM de la PSF gaussienne | exp(−2π²σ_psf²f²) | — |
| Scène naturelle | Φ₀·(1 + g_∥u + g_⊥v)·(1 + c·T) | photons |
| Motif périodique | A·cos(2π(f_x j + f_y i) + φ), variance A²/2 | DN |
| Bruit filtré par h | NPS = σ²·\|H(f)\|², variance σ²·Σh² | DN² |
| Carte de courant | I_d·g, ln g ~ N(−s²/2, s²) ; DSNU relative √(e^{s²} − 1) | e⁻/s |
| Doublement à 20 °C (E_g = 1,12 eV) | 8,29 K (exact) | K |
| Vignettage | T_v = 1 − v·ρ² | — |
| Non-linéarité | e' = e(1 − βe/FW) ; E[e'] = μ − β(μ² + μ)/FW ; κ = −βK/FW | e⁻, DN⁻¹ |
| RTS | séjours moyens 1/p_↑, 1/p_↓ ; fraction haute p_↑/(p_↑ + p_↓) | poses |
| Convertisseur | w_c = t_c − t_{c−1} ; DNL_c = w_c − 1 ; INL_c = t_c − (c + ½) | LSB |

## 9. Bibliographie

- EMVA, *Standard 1288 — Standard for Characterization of Image Sensors
  and Cameras*, European Machine Vision Association (modèle
  mathématique : chap. 2 et 3 ; non-uniformités : chap. 8).
- J. R. Janesick, *Photon Transfer: DN → λ*, SPIE Press, PM170, 2007.
- J. R. Janesick, *Scientific Charge-Coupled Devices*, SPIE Press, PM83,
  2001.
- M. Craig et al., *CCD Data Reduction Guide*, Astropy Project
  (image artificielle, sensibilité non uniforme, overscan, courant
  d'obscurité).
- K. Douglass, « Modeling noise for image simulations », billet de blog,
  kmdouglass.github.io.
- A. B. Downey, *Think DSP — Digital Signal Processing in Python*, Green
  Tea Press, chap. 4 « Noise ».
- P. D. Welch, « The use of fast Fourier transform for the estimation of
  power spectra », *IEEE Trans. Audio Electroacoust.* 15(2), 1967
  (pour le NPS, S3).
