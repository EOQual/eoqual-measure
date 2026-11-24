# MTF : choix de conception et pièges évités

## 9. Choix de conception et pièges évités

Problèmes de méthode qui expliquent des choix d'implémentation peu
intuitifs au premier abord : ce que fait le pipeline, ce que ferait une
implémentation naïve, et l'effet mesuré.

### 9.1 Seuillage par percentile sur magnitude de gradient

`np.percentile(magnitude, p)` dégénère à 0 dès que plus de `100-p` % des
pixels sont exactement plats (bord fin non flouté) : il sélectionne alors
toute l'image au lieu du seul bord. Le percentile est donc calculé sur les
seules valeurs strictement non nulles
(`EdgeOrientationDetector._robust_threshold`).

### 9.2 Biais PCA sur image bruitée (centroïde par ligne)

Le centroïde pondéré par ligne (PCA) moyenne *tous* les pixels au-dessus
du seuil de gradient. Sur une zone bruitée, le bruit crée plus de pixels
au-dessus du seuil que le vrai bord, ce qui biaise la position de façon
*systématique*, et pas seulement aléatoire. Un filtrage par MAD ne suffit
pas (les aberrants sont majoritaires) : les points sont filtrés par RANSAC
([§3.4](estimateurs.md#34-orientationpcapy--pcaorientationdetector)). L'angle est ajusté sur les mêmes inliers que les positions, pour
que l'angle et les positions restent cohérents.

### 9.3 Bruit de mesure près de la fréquence de Nyquist

Phénomène **normal** : la MTF vraie décroît vers 0 alors que le bruit de
mesure ne décroît pas au même rythme. Au-delà d'une certaine fréquence, le
rapport signal/bruit de l'estimation se dégrade et la courbe reflète le
plancher de bruit, quelle que soit la méthode.

### 9.4 Projection géométrique de l'ESF (référence fixe)

La distance de chaque pixel au bord est calculée par rapport à une
référence *fixe*, la droite ajustée évaluée à la ligne centrale ; la
position de bord par ligne ne sert qu'à choisir la fenêtre de colonnes.
Utiliser directement la position *par ligne* (bruitée) dans le calcul de
distance ne se simplifie pas correctement (dérivation analytique) et
réinjecte quasiment l'indice de ligne brut dans la distance projetée
(plage observée de ±39 au lieu des ±20 attendus).

La distance point-droite vaut `(col - edge_pos_fit)·cos(θ)`. La
différence brute `col - edge_pos_fit` n'est la projection orthogonale que
pour un angle faible ; au-delà, elle surestime la largeur effective de
l'ESF.

### 9.5 Sur-lissage de l'ESF (Savitzky-Golay)

Le lissage Savitzky-Golay de l'ESF (fenêtre adaptative `min(31, n/5)`)
dégrade nettement la MTF haute fréquence (0,32 → 0,066 à f = 0,25 c/px,
vérifié numériquement). Il est désactivé par défaut
(`esf_smoothing=False`).

### 9.6 Biais de forme du modèle sigmoïde (ESF)

La dérivée d'une sigmoïde (`sech²`) a des queues spectrales qui décroissent
plus lentement que celles d'une PSF gaussienne : la MTF haute fréquence est
surestimée de façon systématique si la vraie PSF n'a pas une forme en tanh,
*même sans bruit* (0,153 contre 0,077 théorique à f = 0,3 c/px, PSF
gaussienne connue, σ = 1,2 px). `pchip`, qui n'impose aucune forme, ne
montre pas ce biais (0,098 dans le même test) : c'est pourquoi c'est le
modèle par défaut.

La correction du biais de dérivation numérique (`MTFCorrections.derivative`,
[§1.6](theorie.md#16-biais-de-dérivation-numérique)) ne s'applique qu'à `esf_model="numeric"` : les dérivées analytiques
(`sigmoid`, `pchip`) n'en ont pas besoin. Un biais de forme ne se corrige
pas non plus par une correction sinc : réintroduire une correction pour
compenser une surestimation aux hautes fréquences masquerait le symptôme
sans traiter sa cause, le modèle.

### 9.7 Fenêtrage de la LSF (corrélation spectrale)

Sur une image bruitée, sans fenêtrage de la LSF avant FFT, la fuite
spectrale déstabilise la corrélation de phase et sa normalisation par la
valeur en f = 0 : la MTF dépasse 1 d'environ 4 %. Le fenêtrage de Hann
ramène ce dépassement sous 0,6 % (5 tirages bruités).

### 9.8 Différenciation avant corrélation (corrélation spectrale)

Corréler le spectre du bord *brut* (ESF, non différencié) plutôt que
celui de la LSF hérite de la décroissance en 1/k du spectre d'un échelon :
la MTF est sous-évaluée d'un facteur 30 environ (vérifié contre une MTF
théorique connue). Chaque ligne est donc différenciée avant corrélation.

### 9.9 Convention de correction d'angle

Diviser la MTF par `cos(angle)`, plutôt que corriger la fréquence, n'est
pas conforme à la convention ISO 12233 suivie dans tout le pipeline
([§1.3](theorie.md#13-correction-dangle)).

### 9.10 FWHM : première traversée de part et d'autre du pic

`compute_fwhm` cherche la première traversée du demi-maximum de part et
d'autre du pic, avec interpolation sous-pixel : c'est la définition
standard. Prendre le premier et le dernier point au-dessus du demi-maximum
sur tout le domaine de la LSF rend la mesure sensible à un rebond de bruit
isolé loin du pic : avec `esf_model="tikhonov"` sur un cas très bruité, un
unique point parasite à +21 px du pic donne une FWHM de 22 px au lieu des
2 à 3 px attendus.

Au bruit nul, `"tikhonov"` donne une FWHM de 2,820 px pour 2,826 px
théorique (PSF gaussienne σ = 1,2 px). `"numeric"` et `"pchip"` donnent au
contraire des FWHM anormalement petites (0,3 à 0,4 px) sur le même cas :
leur LSF dérivée brute porte un pic à l'échelle d'un seul bin, non
représentatif de la largeur physique.

### 9.11 Régularisation de Tikhonov et sélection de λ

`esf_model="tikhonov"` ([§4.4](estimateurs.md#44-estimatorsesf_lsfpy--estimatorsesf_modelspy--esflsfmtfestimator)) reprend le principe de
[[Xie2018]](references.md#10-bibliographie) par une formulation de Tikhonov standard
construite indépendamment. Sur bord synthétique (PSF gaussienne
σ = 1,2 px), il est compétitif avec `"pchip"` et `"numeric"` en MTF à tous
les niveaux de bruit testés, et nettement meilleur sur la FWHM au bruit
nul (§9.10).

`fit_esf_tikhonov` (`tikhonov_lambda="auto"`) sélectionne λ au « coin » de
la courbe en L, repéré par la courbure de `log(résidu pondéré)` en fonction
de `log(rugosité de f)`. Ce calcul dérive *deux fois de suite* une courbe
échantillonnée : c'est une instabilité différente de celle du [§1.6](theorie.md#16-biais-de-dérivation-numérique) (biais
*systématique* en `sinc` d'une dérivation simple). Une double dérivée
numérique d'une courbe faiblement échantillonnée amplifie la moindre
irrégularité, au point de déplacer le maximum de courbure détecté : sur
40 valeurs de λ, deux bords à angles très proches (−4° et −5°, même PSF,
sans bruit) donnent des λ dans un rapport 24 environ (14 contre 345), et au
premier angle une LSF dentelée de FWHM 0,58 px au lieu de 2,826 px.

La sélection utilise donc une grille densifiée (80 valeurs log-espacées
entre 1e-6 et 1e3) et lisse `log_res`/`log_rough` par Savitzky-Golay
(fenêtre impaire ≤ 11, ordre 3) *avant* de calculer la courbure. Sur le cas
−4°/−5°, le second angle retrouve une FWHM de 2,829 px, mais le premier
reste sous-régularisé (0,58 px) : le lissage stabilise la détection sans la
rendre fiable pour tout angle.

**Limite connue, non résolue.** Quand la courbe en L est lisse, sans
coin net (cas rencontré en pratique, pas rare), la détection par
courbure peut encore sélectionner un λ insuffisant. Vérifié
numériquement : sur des bords synthétiques à angles proches (-1° à
-20°, PSF gaussienne identique), la FWHM récupérée en mode `"auto"`
varie entre ~0.5 et ~2.9 px pour une valeur théorique de 2.826 px — y
compris avec un λ **fixe** (essayé 50.0), certains angles restent mal
régularisés : la fragilité n'est donc pas uniquement une question de
grille ou de lissage de la sélection automatique. Des pistes plus
robustes (critère de discordance de Morozov, validation croisée
généralisée — GCV) n'ont pas été implémentées. **En pratique** : si un
résultat `esf_model="tikhonov"` semble implausible, inspecter
`details["fwhm"]` et essayer plusieurs valeurs fixes de
`tikhonov_lambda` plutôt que de faire confiance à `"auto"` sans
vérification.

### 9.12 Normalisation de la MTF (ESF/LSF)

La MTF de l'estimateur ESF/LSF est normalisée par sa composante continue
(`eoqual.core.edge.lsf_to_mtf`). La normaliser par son maximum sur les
seules fréquences positives revient à la normaliser par sa valeur au
premier bin `1/(N·dx)` : la courbe est alors surestimée de 0,7 à 1,3 % sur
les mires de test, et la dispersion bootstrap est plus forte (la
normalisation par la composante continue la réduit de 25 à 37 %), la
valeur au premier bin étant plus bruitée que le saut de l'ESF. Les
estimateurs spatial, spectral et corrélation spectrale ne sont pas
concernés.

### 9.13 MTF du système ou de l'optique seule (`esf_lsf`)

Les quatre estimateurs renvoient la **MTF du système** avant
échantillonnage, pixel compris (optique × pixel × diffusion × mouvements) :
celle que mesure l'ISO 12233 et que publient les opérateurs. Le retrait du
pixel (`MTFCorrections.pixel_aperture`, [§1.5](theorie.md#15-correction-douverture-pixel-sinc)) est une option de `esf_lsf`
(`remove_pixel_aperture=True`, CLI `--esf-lsf-remove-pixel-aperture`), pour
l'usage de laboratoire où l'on caractérise un objectif indépendamment du
capteur. Mélanger les deux conventions dans un comparateur ferait passer
pour une différence de méthode un écart de `1/sinc(f)` : +11 % à
0,25 cycle/pixel, +57 % au Nyquist.

Le simulateur `SyntheticSlantedEdge` intègre par défaut le bord et la PSF
sur le pixel (`pixel_integration=True` : calcul sur une grille
`supersample` fois plus fine, puis moyenne sur chaque pixel), et sa vérité
`mtf_theoretical` inclut `|sinc(f)|`. L'échantillonnage ponctuel
(`pixel_integration=False`) replie les bords fins et fausse la MTF près du
Nyquist : pour un bord fin (σ = 0,6 px), la MTF mesurée au Nyquist double
(0,31 pour 0,17).

**Justesse mesurée** (bord intégré sur le pixel, σ = 0,8 px, 128 × 128, 5°,
bruit 0,002, 12 tirages) :

| `esf_model` | MTF(0,25) mesurée | vérité | écart-type d'une mesure |
|---|---|---|---|
| `numeric` | 0,4058 | 0,4088 | 0,0009 |
| `pchip` | 0,4137 | 0,4088 | 0,0008 |

Restent deux biais de méthode, petits devant le facteur `1/sinc` corrigé
mais supérieurs à l'incertitude statistique : `numeric` sous-estime
d'environ 0,003 (≈ 3 u), `pchip` surestime d'environ 0,005 (≈ 6 u). Sans
bruit et avec σ = 0,6 px, `numeric` donne 0,571 et `pchip` 0,579 pour une
vérité de 0,577 ; `tikhonov` (λ automatique) sous-estime nettement (0,499)
et `sigmoid` surestime au Nyquist (0,160 pour 0,108). Sur le simulateur à
échantillonnage ponctuel, un bord fin (σ = 0,6 px) est replié et la MTF
mesurée au Nyquist double (0,31 pour 0,17) : d'où l'intégration sur le pixel
par défaut.

Tests (`tests/test_mtf.py`) : vérité du simulateur intégré, relation
exacte entre les deux options, et choix entre les hypothèses « système » et
« optique » sur 12 tirages, stable sur les 30 valeurs de
`EOQUAL_TEST_SEED`.

### 9.14 Binning adaptatif à l'angle

[[Wu2022]](references.md#10-bibliographie), qui s'appuie sur le jeu de données de
comparaison de [[ViallefontRobinet2018]](references.md#10-bibliographie), montre
numériquement que deux choix affectent sensiblement la qualité de la MTF
aux moyennes et hautes fréquences :
- la largeur de bin de l'ESF, fixe (`"omni"`) ou réduite par `cos(theta)`
  (`"iso_cos"`, meilleure convergence d'après cette étude) ;
- le suréchantillonnage, fixe ou adaptatif à l'angle mesuré
  (`OSR_optimal = 1/tan(theta)`, borné à [2, 10]).

Les deux sont des options de `GeometricESFBuilder` et `EsfLsfMTFEstimator`
(`binning_mode`, `oversample="auto"`, [§4.4](estimateurs.md#44-estimatorsesf_lsfpy--estimatorsesf_modelspy--esflsfmtfestimator)) ; `"omni"` et un `oversample`
fixe restent le défaut. Sur bord synthétique (PSF gaussienne σ = 1,2 px,
angles −3° et −15°), `"iso_cos"` réduit légèrement la surestimation de la
MTF aux hautes fréquences par rapport à `"omni"`, dans le sens attendu par
Wu et al., mais l'écart reste modeste : à confirmer sur une gamme d'angles
et de bruits plus large ([§8.12](guide.md#812-études-de-sensibilité)).

### 9.15 RER et FWHM, pas d'« overshoot H »

Le "H" de la GIQE
[[Harrington2015]](references.md#10-bibliographie) mesure le ringing introduit par
un filtre de *sharpening* appliqué en traitement sol -- ce n'est pas
une propriété du bord lui-même, mais un artefact de post-traitement.
Ce pipeline ne modélise aucune étape de sharpening ; calculer "H" sur
une image brute non traitée serait trompeur (toujours proche de sa
valeur neutre, sans rapport avec la définition GIQE). Non ajouté pour
cette raison, contrairement à RER et FWHM ([§4.4](estimateurs.md#44-estimatorsesf_lsfpy--estimatorsesf_modelspy--esflsfmtfestimator)) qui sont eux des
propriétés directes du bord mesuré.
