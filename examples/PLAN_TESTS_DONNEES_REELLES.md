# Plan de tests sur données réelles

Application des 16 métriques à des acquisitions réelles, sous forme de
notebooks d'exemple (`examples/`), en complément de la validation sur
simulation (`REFERENCE_TECHNIQUE.md`, partie IV, annexe B).

## 1. Principes

- **Pas de vérité** : sur données réelles, on ne peut pas comparer une
  mesure à sa valeur vraie. On vérifie donc des **cohérences** :
  - entre métriques qui mesurent la même grandeur par des voies
    différentes, par exemple le gain de la PTC et le gain tiré du
    caractère poissonnien de l'obscurité ;
  - entre sessions ou entre acquisitions répétées ;
  - avec la fiche technique du capteur ;
  - avec les indicateurs internes des métriques : χ² réduit, rapport de
    stationnarité, `texture_suspected`, incertitudes.
- **Accord à l'incertitude près** : deux estimations x₁ ± u₁ et x₂ ± u₂ de la
  même grandeur sont jugées cohérentes si |x₁ − x₂| < 3·√(u₁² + u₂²).
  Au-delà, l'écart est un **résultat** à analyser : effet non modélisé,
  hypothèse violée ou anomalie de la métrique. Il n'est pas à masquer.
- **Données hors du projet** : les notebooks lisent les images à partir
  de chemins définis dans une cellule de configuration. Rien n'est copié
  ni versionné, et la reproductibilité n'est pas recherchée.
- **Pas de réglage a posteriori** : les paramètres par défaut des
  métriques sont utilisés d'abord. Tout changement (taille d'imagette,
  coupure BF/HF, seuils) est noté et justifié dans le notebook.

## 2. Organisation des notebooks

Un notebook par groupe de métriques, dans `examples/`, avec des noms de
fichiers en anglais et un contenu en français :

| Notebook | Métriques | Données |
|---|---|---|
| `real_matrix_dark.ipynb` | `dark_frame`, `dark_current`, `defective_pixels` (chauds) | obscurités |
| `real_matrix_ptc.ipynb` | `temporal_noise`, `photon_transfer_curve`, `snr_model` | champs uniformes + obscurité |
| `real_matrix_flat.ipynb` | `prnu`, `linearity`, `defective_pixels`, `spatial_noise` (pile) | champs uniformes, expositions connues |
| `real_matrix_structured.ipynb` | `noise_power_spectrum`, `structured_noise`, `rts_pixels` | piles d'obscurité et de champs uniformes |
| `real_adc.ipynb` | `adc_histogram` | rampe ou fond lisse |
| `real_pushbroom.ipynb` | `dark_frame` (push broom), `pushbroom_temporal_noise`, `snr_model`, `spatial_noise`, `equalization_noise` | images push broom |
| `real_whiskbroom.ipynb` | `spatial_noise` (whisk broom) | images whisk broom |
| `real_products.ipynb` | `noise_power_spectrum` (produit traité), `cnr_target` | produits traités, mires |

Structure commune :

1. **Configuration** : chemins, description du capteur (format, nombre de
   bits, gain annoncé, température), version d'eoqual-measure.
2. **Lecture** : fonction de lecture propre au format (FITS, TIFF, RAW…),
   renvoyant des tableaux (K, H, W) ou (M, W) en DN, ainsi que les
   métadonnées utiles (temps de pose, température, niveau).
3. **Contrôles des données** : saturation, codes min/max, stationnarité
   (`dark_frame`), pixels défectueux à exclure.
4. **Mesures**, avec leurs incertitudes.
5. **Contrôles de cohérence** (§4) : tableau « grandeur, voie 1, voie 2,
   écart / u combinée, verdict ».
6. **Conclusion** : anomalies relevées, hypothèses violées, suites à
   donner (ticket, évolution du simulateur, correction de méthode).

## 3. Données à rassembler

| Code | Acquisition | Quantité indicative | Sert à |
|---|---|---|---|
| D1 | Obscurité, temps de pose courant | ≥ 32 poses | `dark_frame`, b, DSNU, stationnarité, `structured_noise` |
| D2 | Obscurité à 4–6 temps de pose (de t_min à > σ_lect/I_d) | ≥ 16 poses par temps | `dark_current`, gain poissonnien |
| D3 | Obscurité à 2–4 températures (si le plan focal est régulé) | 2 temps de pose par température | loi en température |
| D4 | Longue pile d'obscurité | 500 à 2 000 poses | `rts_pixels` |
| D5 | Obscurité ou bias à t ≈ 0 (et overscan s'il existe) | ≥ 16 poses | offset, gigue d'offset |
| F1 | Paires de champs uniformes à 10–15 niveaux, jusqu'au-delà de la saturation | 2 poses par niveau | `temporal_noise`, PTC, saturation |
| F2 | Piles de champs uniformes à 2–3 niveaux | ≥ 16 poses par niveau | `prnu`, `spatial_noise`, `defective_pixels`, NPS |
| F3 | Source stable à 10–15 temps de pose | ≥ 8 poses par temps | `linearity` |
| A1 | Rampe ou fond lisse couvrant la plage de codes, bruit ≥ quelques DN | ≥ 10⁶ échantillons, idéalement 10⁷ | `adc_histogram` |
| P1 | Push broom : lignes d'obscurité (calibration interne, océan de nuit) | ≥ 500 lignes | `dark_frame` (push broom), b |
| P2 | Push broom : scènes peu texturées (désert, neige, océan) à plusieurs niveaux | 3–5 sites ou dates, ≥ 1 000 lignes chacun | `pushbroom_temporal_noise`, `snr_model` |
| P3 | Push broom : zone uniforme avant et après égalisation (diffuseur, calibration) | ≥ 500 lignes | `spatial_noise`, `equalization_noise` |
| W1 | Whisk broom : zone uniforme, nombre de détecteurs k connu | ≥ 40 balayages | `spatial_noise` (bandes, aller/retour) |
| T1 | Mire à deux plages (contraste connu si possible) | ≥ 10 poses | `cnr_target` |
| T2 | Produit traité (dématriçage, rééchantillonnage, compression) sur zone uniforme, et le brut correspondant | ≥ 4 imagettes de 64 × 64 | `noise_power_spectrum` |

Les données peuvent venir d'un même capteur, ce qui permet le plus de
recoupements, ou de capteurs différents selon ce qui est disponible.
Noter pour chaque jeu : capteur, mode, gain électronique, température,
temps de pose et date.

## 4. Fiches de test par métrique

Chaque fiche donne les données, la procédure, les contrôles de cohérence
(C), les critères indicatifs et les pièges connus. Les critères sont
indicatifs : un écart est d'abord analysé.

### 4.1 `dark_frame` (D1, P1)

- **Procédure** : `measure_dark_frame(D1)` ; en push broom,
  `layout="pushbroom"` sur P1.
- **C1** : `stationarity_ratio` ≈ 1 ± 3·√(2/(K − 1)) et `drift_per_frame`
  compatible avec 0. Sinon, la dérive thermique ou un bruit commun est à
  documenter avant toute autre mesure.
- **C2** : b de deux piles D1 acquises à des moments différents, à 3 u près.
- **C3** : DSNU et ses composantes ligne et colonne, répétables entre deux
  piles. Un colonnage marqué est attendu sur les CMOS à convertisseurs de
  colonne.
- **C4** : σ₀ = √b comparé au bruit de lecture annoncé, converti en DN
  avec le gain de la PTC.
- **Pièges** : pixels chauds qui gonflent la DSNU (les exclure via
  `defective_pixels`), bruit de ligne qui fausse le rapport de
  stationnarité (vérifier avec `structured_noise`).

### 4.2 `dark_current` (D2, D3)

- **Procédure** : `measure_dark_current(D2, temps, saturation_dn=…)`,
  avec `temperatures=` si D3 est disponible.
- **C1** : **gain poissonnien** comparé au gain K de la PTC (§4.4), à
  3 u près. C'est le recoupement principal du gain.
- **C2** : courant moyen comparé à la fiche technique (même température).
- **C3** : loi en température. La loi ajustée s'écrit en E_g/2 :
  - courant de génération : E_g ajusté ≈ 1,1 eV, doublement ≈ 8 K vers 20 °C ;
  - courant de diffusion, en exp(−E_g/kT) : E_g ajusté ≈ 2,2 eV,
    doublement ≈ 4 à 5 K.

  Une valeur intermédiaire indique un mélange des deux mécanismes, que la
  loi à un seul terme ne décrit pas (`REFERENCE_TECHNIQUE.md`, partie IV, §4.5).
- **C4** : pixels chauds de deux sessions, avec un recouvrement attendu
  ≥ 90 %, puisqu'ils sont stables.
- **Pièges** : temps de pose trop courts (courant noyé dans le bruit de
  lecture : t_max > σ_lect / I_d), dérive de température entre piles,
  lueur d'amplificateur comptée comme pixels chauds (la masquer).

### 4.3 `temporal_noise` (F1, F2)

- **C1** : σ² par paire et σ² par pile au même niveau, à 3 u près.
- **C2** : au niveau zéro (D1), σ² = b de `dark_frame`.
- **Pièges** : fluctuation de la source entre poses (biais
  ε²·var_spatiale/2, `REFERENCE_TECHNIQUE.md`, partie IV, §3.2.3), qu'on contrôle
  par le rapport des moyennes des deux poses.

### 4.4 `photon_transfer_curve` (F1 + D1)

- **C1** : χ² réduit ≈ 1 ± 3·√(2/ν). Une valeur élevée signale une
  non-linéarité ou un bruit non modélisé.
- **C2** : gain K comparé au gain poissonnien (§4.2) et à la fiche
  technique.
- **C3** : bruit de lecture en e⁻ comparé à la fiche technique ; b de la
  PTC comparé à b de `dark_frame`.
- **C4** : saturation en e⁻ comparée à la capacité du puits annoncée,
  attendue un peu sous FW (pic de variance vers FW − 2,8√FW).
- **C5** : erreur de linéarité de la PTC comparée à celle de `linearity`
  (§4.6).
- **Pièges** : niveaux trop espacés près du plafond (saturation
  imprécise) ; gain électronique changé entre niveaux.

### 4.5 `prnu` (F2 + D1, exclusion par `defective_pixels`)

- **C1** : PRNU à 2–3 niveaux. Sans non-linéarité, elle est constante ;
  avec une non-linéarité commune, elle suit le facteur
  (1 − 2cx)/(1 − cx) (`REFERENCE_TECHNIQUE.md`, partie IV, §5.2), à rapprocher de
  `linearity`.
- **C2** : PRNU EMVA (plusieurs niveaux) comparée à la PRNU à un niveau.
- **C3** : basses fréquences (vignettage) cohérentes avec l'optique
  (symétrie, amplitude) ; hautes fréquences comparées à la fiche
  technique.
- **C4** : avec et sans exclusion des pixels défectueux : l'effet de
  l'exclusion doit être faible si les défauts sont rares.
- **Pièges** : éclairement non uniforme de la source (confondu avec le
  vignettage), poussières (restent dans la HF).

### 4.6 `linearity` (F3 + D1)

- **C1** : erreur de linéarité EMVA comparée à la fiche technique.
- **C2** : signe et ordre de grandeur de la courbure κ, cohérents avec
  l'inflexion de la PTC.
- **C3** : stabilité de la source, par une exposition de référence
  répétée en début, milieu et fin de série (dérive < u de la moyenne).
- **Pièges** : courant d'obscurité non négligeable aux longues poses (à
  retirer, ou utiliser des poses courtes) ; dérive de la source.

### 4.7 `defective_pixels` (D1, F2, résultat de `dark_current`)

- **C1** : pixels bloqués et morts identiques d'une session à l'autre.
- **C2** : pixels chauds cohérents avec `dark_current` (même liste).
- **C3** : taux de « bruyants » comparé au nominal (p_FA × nombre de
  pixels). Un excès signale des pixels télégraphiques (§4.8) ou un bruit
  non gaussien.
- **Pièges** : grappes ou colonnes défectueuses, non traitées comme telles
  (médiane 5 × 5).

### 4.8 `rts_pixels` (D4)

- **C1** : liste des pixels télégraphiques recoupée avec la classe
  « bruyant » de `defective_pixels`.
- **C2** : deux longues piles : mêmes pixels, amplitudes à 3 u près,
  probabilités de bascule du même ordre (elles dépendent de la
  température).
- **C3** : amplitudes en e⁻ (avec le gain) de l'ordre de quelques
  électrons, ordre de grandeur des pièges de lecture.
- **Pièges** : saut < 2,5 σ non détecté ; bascules sous-estimées sous 4 σ.

### 4.9 `adc_histogram` (A1)

- **C1** : DNL de deux acquisitions corrélée (coefficient de corrélation
  ≫ 0 si la DNL dépasse le bruit), et mêmes codes manquants.
- **C2** : mode `"uniform"` sur une rampe et mode `"local"` sur un fond
  lisse : DNL compatibles sur les codes communs.
- **C3** : motif lié aux bits de poids fort (codes 2ⁿ), recherché sur la
  DNL, puisqu'il est typique des convertisseurs à approximations
  successives.
- **Pièges** : densité d'entrée non lisse (bruit trop faible devant le
  pas), écrêtage massif, trop peu d'occurrences par code.

### 4.10 `pushbroom_temporal_noise` et `snr_model` (P1, P2)

- **C1** : `texture_suspected` faux sur les sites retenus. Sinon, changer
  de site ou de zone plutôt que d'interpréter la mesure.
- **C2** : pente a cohérente entre sites et dates, à 3 u près.
- **C3** : avec b tiré de P1, a = 1/K comparé au gain de la PTC au sol,
  si l'on dispose de la caractérisation au sol du même instrument.
- **C4** : SNR aux niveaux spécifiés (L_min, L_ref, L_max) comparé aux
  spécifications de l'instrument.
- **C5** : χ² réduit ≈ 1.
- **Pièges** : scène rugueuse à l'échelle du pixel (biais positif, signalé
  par l'indicateur), bruit corrélé entre détecteurs voisins (biais
  négatif, non signalé), saturation partielle.

### 4.11 `spatial_noise` et `equalization_noise` (F2, P3, W1)

- **C1** : push broom, FPN avant égalisation ≫ FPN après ; FPN résiduel
  comparé à l'exigence de l'instrument.
- **C2** : décomposition BF / HF. La BF doit refléter les défauts
  attendus (vignettage, raccords de barrettes), la HF le bruit des
  coefficients de calibration ; vérifier u(HF) avant d'interpréter.
- **C3** : spectre de la ligne moyenne, avec des pics à des fréquences
  explicables (périodicité des barrettes, multiplexage).
- **C4** : whisk broom. Bandes de période k = nombre de détecteurs, et
  différence aller/retour à comparer à l'exigence ; elle doit être stable
  d'une image à l'autre.
- **Pièges** : variation de scène en travers confondue avec un FPN BF ;
  pixels défectueux qui dominent la HF.

### 4.12 `noise_power_spectrum` et `structured_noise` (D1, F2, T2)

- **C1** : variance intégrée du NPS en mode « différence » égale à σ²
  temporel (`temporal_noise`).
- **C2** : produit traité. Rapport NPS(traité) / NPS(brut) comparé à
  |H|² du traitement s'il est connu (noyau de rééchantillonnage,
  dématriçage).
- **C3** : pics stables d'une session à l'autre, avec des fréquences
  explicables (horloges, alimentations). Leur nature (fixe ou temporelle)
  doit être cohérente avec l'électronique.
- **C4** : bruit de ligne. σ_ligne et β comparés entre sessions ;
  cohérence avec le rapport de stationnarité de `dark_frame`.
- **Pièges** : basses fréquences (< 3/P) non interprétables ; motifs non
  sinusoïdaux détectés comme plusieurs pics (harmoniques).

### 4.13 `cnr_target` (T1)

- **C1** : CNR mesuré comparé à la prédiction tirée du modèle de bruit,
  (μ_c − μ_s)/√(σ²(μ_c) + σ²(μ_s) + PRNU²·(μ_c² + μ_s²) + 2·DSNU²), avec
  a, b de la PTC, la PRNU et la DSNU (dans une seule image, le bruit d'une
  plage comprend le bruit spatial fixe).
- **C2** : CNR répétable d'une pose à l'autre (dispersion ≈ u).
- **Pièges** : plages trop petites ou non uniformes (structure de la
  mire), transition mal érodée.

## 5. Recoupements transverses

| Grandeur | Voie 1 | Voie 2 | Voie 3 |
|---|---|---|---|
| Gain K (e⁻/DN) | PTC | gain poissonnien de `dark_current` | a = 1/K de `pushbroom_temporal_noise`, fiche technique |
| Bruit à signal nul b (DN²) | `dark_frame` | ordonnée de la PTC | b de `pushbroom_temporal_noise` |
| Bruit de lecture (e⁻) | PTC | √b × K (`dark_frame` + gain) | fiche technique |
| Linéarité | erreur de linéarité de la PTC | `linearity` | évolution de la PRNU avec le niveau |
| Pixels chauds | `dark_current` | `defective_pixels` | deux sessions |
| Pixels télégraphiques | `rts_pixels` | classe « bruyant » de `defective_pixels` | deux sessions |
| Bruit temporel σ² | `temporal_noise` | NPS intégré (mode « différence ») | `dark_frame` à μ = 0 |

## 6. Phasage

1. **Capteur matriciel au sol** (D1, D2, F1, F2, F3) : `real_matrix_dark`,
   `real_matrix_ptc`, `real_matrix_flat`. Ce sont les recoupements les
   plus nombreux (gain, b, bruit de lecture), et ils servent de référence
   aux autres phases.
2. **Bruits structurés et défauts** (D4, F2) : `real_matrix_structured`.
3. **Push broom** (P1–P3) : `real_pushbroom`.
4. **Convertisseur, whisk broom, produits, mires** selon les données
   disponibles : `real_adc`, `real_whiskbroom`, `real_products`.

## 7. Suites à donner

Un écart non expliqué conduit à l'une de ces suites :

- **effet physique non simulé** : l'ajouter au simulateur, puis vérifier
  que la métrique le traite ou le signale (nouveau test sur simulation) ;
- **hypothèse violée** (stationnarité, scène texturée, source instable) :
  la documenter dans les limites de la métrique, et ajouter si possible un
  indicateur ;
- **anomalie de la métrique** : la corriger, ajouter un test de
  non-régression sur simulation, et la consigner dans l'annexe A de la
  partie IV de `REFERENCE_TECHNIQUE.md`.

Chaque notebook se termine par un tableau récapitulatif des contrôles,
« contrôle, résultat, verdict (cohérent / écart / non testable), suite ».
Ces tableaux réunis forment le bilan de qualification sur données
réelles.
