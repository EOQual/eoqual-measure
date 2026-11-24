# Simulateur : objet et modèles

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

Chaque pixel suit le modèle linéaire EMVA 1288 ([fondements, §2.1–§2.5](../radiometrie/02-fondements.md)) : photons,
rendement quantique et PRNU, bruit de photons poissonnien en électrons, écrêtage au puits, bruit de
lecture, conversion par K, offset et DSNU, quantification et écrêtage du convertisseur. DSNU et
PRNU sont la somme de composantes pixel, ligne et colonne, centrées, tirées chacune dans un flux
aléatoire propre ; la mise en œuvre est détaillée au [§4](implementation.md#4-implémentation).

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
seule image ([push broom, §10.2.3](../radiometrie/10-pushbroom-snr.md)).

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
de vérité au NPS ([chapitre 12](../radiometrie/12-nps-bruits-structures.md)).

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
techniques ([fondements, §2.2](../radiometrie/02-fondements.md)) en sont l'ordre de grandeur.

**Lueur d'amplificateur.** L'amplificateur de sortie dissipe de la
chaleur et émet de la lumière. Il en résulte une charge additionnelle,
linéaire en temps de pose, maximale au coin (0, 0), où il est placé, et
qui décroît comme exp(−d/λ) avec la distance d. Elle ne suit pas la loi
en température.

**Variations d'offset.** L'offset peut varier d'une pose à l'autre de
façon commune à toute l'image : gigue (tirage gaussien par pose) et dérive
linéaire (thermique). Ce sont les défauts que contrôle la stationnarité
de `dark_frame` ([chapitre 3, §3.2.2](../radiometrie/03-obscurite-ptc.md)).

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
sépare par son échelle spatiale ([chapitre 5](../radiometrie/05-prnu.md)).

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
