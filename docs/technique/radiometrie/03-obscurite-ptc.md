# 3. Obscurité, bruit temporel et courbe de transfert photonique

## 3.1 Objet

Chaîne de caractérisation au sol d'un capteur matriciel : offset et DSNU
sur l'obscurité (`dark_frame`), bruit temporel sur champs uniformes
(`temporal_noise`), puis gain, bruit de lecture, saturation et linéarité
par la courbe de transfert photonique (`photon_transfer_curve`).

Ces trois mesures partagent le même principe : **séparer le temporel du spatial** sur des acquisitions
contrôlées (obscurité, champs uniformes) en exploitant la répétition des
poses, au lieu d'estimer le bruit dans une seule image comme le fait
`noise_variance` ([estimateurs sur scène](../bruit-estimateurs.md)).

Chaque résultat est **typé** (dataclass) et porte ses **incertitudes-types**,
calculées analytiquement. Elles servent à la fois de barres d'erreur et de
tolérances de validation.

| Mesure | Entrée | Sorties principales |
|---|---|---|
| `measure_dark_frame` | pile d'obscurité (K, H, W), ou bloc push broom (M, W) | carte d'offset, a = σ²(0), DSNU et décomposition, stationnarité |
| `measure_temporal_noise` | paire ou pile de champs uniformes | σ²_temp, signal corrigé de l'obscurité |
| `measure_ptc` | une paire par niveau + obscurité | K (e⁻/DN), σ_lect (e⁻), saturation, linéarité, χ² |

**Statut** : validées contre le simulateur S0 (`tests/test_dark_frame.py`,
`tests/test_temporal_noise.py`, `tests/test_ptc.py`, 28 tests ; robustes
sur 50 graines). Module : `eoqual.measure.radiometry.{dark,
temporal, ptc}`.

## 3.2 Principe

La décomposition ligne / colonne / pixel de la DSNU utilise l'outil commun du §2.7.

### 3.2.1 Pile d'obscurité

Une pose d'obscurité y_k(i, j), k = 1…K, contient l'offset fixe o_ij
(offset moyen + DSNU) et le bruit temporel de variance a (§2.2) :

y_k = o + e_k,  E[e_k] = 0,  Var[e_k] = a.

- La moyenne temporelle ô = (1/K) Σ y_k estime la carte o, avec un résidu
  de variance a/K par pixel.
- La variance temporelle (1/(K − 1)) Σ (y_k − ô)², moyennée sur les
  N pixels, estime a avec une incertitude relative de √(2/((K − 1)·N)).
- La variance spatiale de ô vaut s²(o) + a/K. La DSNU s'en déduit par
  DSNU² = var_spatiale(ô) − â/K (EMVA 1288, chap. 8).

**Incertitude de la DSNU.** On estime la dispersion de la carte
*réalisée* o, et non celle d'une population de capteurs. Notons
D = var(ô) − v̂, avec v = b/K. Les termes aléatoires sont la covariance
croisée 2·cov(o, e), de variance 4·S·v/N où S = var(o), la variance
propre du résidu (2v²/N) et l'erreur sur v̂ (2v²/((K − 1)N)). D'où

Var(D) ≈ [4·S·v + 2v²·(1 + 1/(K − 1))] / N,  u(DSNU) = √Var(D) / (2·DSNU).

### 3.2.2 Stationnarité

La moyenne spatiale de la pose k, f_k = moyenne_ij y_k, a pour variance
b/N si le bruit est indépendant d'un pixel à l'autre : les motifs fixes
sont les mêmes dans toutes les poses et disparaissent des écarts entre
f_k. On en tire deux contrôles :

- le **rapport de stationnarité** var(f) / (â/N), proche de 1 à
  √(2/(K − 1)) près pour une pile stationnaire. Il croît en cas de dérive
  thermique, de bruit commun à toute l'image ou de bruit de ligne, ces
  deux derniers étant corrélés entre pixels ;
- la **pente de dérive**, par régression linéaire de f_k sur k. Son
  incertitude-type vaut √(b/N / Σ(k − k̄)²).

Ce contrôle signale une dérive thermique pendant l'acquisition, qui fausserait la DSNU.

### 3.2.3 Bruit temporel sur champ uniforme

- **Paire** : x₁ − x₂ ne contient plus aucun motif fixe (offset, DSNU,
  PRNU) et a pour variance 2σ²_temp. D'où σ²_temp = var(x₁ − x₂)/2, avec
  une incertitude relative de √(2/(N − 1)). Retirer la moyenne de la
  différence rend l'estimateur insensible à un écart de niveau **additif**
  entre les deux poses (dérive d'offset).
- **Pile** (K ≥ 3) : moyenne spatiale des variances temporelles par
  pixel, d'incertitude relative √(2/((K − 1)·N)).

Une fluctuation de la source entre poses, **proportionnelle** au signal
(x₂ = (1 + ε)·x₁), laisse un résidu ε·x₁. Une fois la moyenne de la
différence retirée, il n'en reste que la partie spatiale (PRNU, scène) :
var(x₁ − x₂)/2 ≈ σ²_temp + ε²·var_spatiale(x₁)/2. À 3 000 DN avec 1 % de
PRNU (var_spatiale ≈ 1 900 DN²) et ε = 1 %, le biais vaut environ 0,1 DN²,
soit 0,01 % de σ² ≈ 1 000 DN² : il est négligeable. Il devient sensible
pour une scène non uniforme ou une forte PRNU : avec ε = 5 % et 3 % de
PRNU, il dépasse 1 % de σ².

### 3.2.4 Courbe de transfert photonique

À chaque niveau i, on calcule μ_i = moyenne − offset moyen et σ_i² par
paire. Sur la plage linéaire, σ²(μ) = a + b·μ (§2.2), d'où :

- **gain** K = 1/b (e⁻/DN), avec u(K) = u(b)/b² ;
- **bruit de lecture** σ_lect = K·√(a − 1/12) (e⁻), où le terme de
  quantification est retiré (option). Son incertitude se propage avec la
  covariance de (a, b) : J = (1/(2b·√(a − q)), −√(a − q)/b²),
  u² = J·cov·Jᵀ.

**Ajustement pondéré.** On pondère par w_i = 1/u_i², avec
u_i = σ_i²·√(2/(N − 1)) (paire). Les incertitudes croissent avec le
niveau : un ajustement non pondéré serait dominé par les forts signaux et
contraindrait mal a. Le point d'obscurité (μ = 0, σ² = â) est ajouté par
défaut, car c'est lui qui fixe a. Le χ² réduit, de loi χ²_ν/ν avec
ν = n − 2, vaut 1 ± √(2/ν) si le modèle affine et les incertitudes sont
corrects. Il sert à contrôler le modèle.

**Saturation.** EMVA 1288 définit le point de saturation comme le signal
où la variance est maximale. L'écrêtage (puits ou convertisseur) réduit
la variance *avant* que la moyenne n'atteigne le plafond. Pour un signal
poissonnien ≈ normal écrêté à FW, la variance de min(X, FW) culmine vers
μ_e ≈ FW − 2,8·√FW (5 784 e⁻ pour FW = 6 000 e⁻), et son pic est plat : elle
reste à 1 % du maximum de FW − 3,8√FW à FW − 2,2√FW. Le point mesuré est
donc intrinsèquement imprécis, d'environ ±1σ. Une fois la saturation
détectée (maximum de variance avant le dernier niveau), l'ajustement se
limite aux signaux ≤ 70 % de ce point, comme le recommande EMVA 1288. Si le
code de saturation est connu, un niveau qui contient plus de 10⁻⁴
d'échantillons à ce code est aussi exclu.

**Linéarité.** Si l'exposition x_i de chaque niveau est connue (temps de
pose, photons), on ajuste μ = s·x + c sur les niveaux retenus. On calcule
les écarts relatifs δ_i = (μ_i − ajustement)/ajustement, en %, puis
l'erreur de linéarité (max δ − min δ)/2. C'est la définition d'EMVA 1288,
sur la plage de l'ajustement de la PTC (simplification : EMVA utilise la
plage de 5 à 95 % de la saturation).

## 3.3 Mise en œuvre

```
eoqual/measure/radiometry/
├── __init__.py        API publique
├── _stack.py          contrôle et mise en forme des piles (K, H, W), blocs push broom
├── decomposition.py   decompose_spatial → SpatialDecomposition (réutilisé par spatial_noise, prnu)
├── dark.py            measure_dark_frame → DarkFrameResult
├── temporal.py        measure_temporal_noise → TemporalNoiseResult
└── ptc.py             measure_ptc → PTCResult (s'appuie sur dark et temporal)
```

Dépendances internes : `ptc` → `temporal` → `dark` → `decomposition`. Les
mesures ne connaissent pas le simulateur : elles prennent des tableaux, et
c'est le test qui confronte le résultat à `acq.truth`.

Dans le registre (`config.py`), la PTC s'appuie sur `temporal_noise`
(paires d'images) et `dark_frame`, et non sur `noise_variance`.

### 3.3.1 `measure_dark_frame(frames, *, layout="stack")`

1. Mise en forme : pile (K, H, W), K ≥ 2. En `layout="pushbroom"`, un bloc
   (M, W) devient (M, 1, W) : chaque ligne est une pose, chaque colonne un
   détecteur.
2. `offset_map` = moyenne temporelle, `temporal_variance_map` = variance
   en 1/(K − 1), â = moyenne de la carte de variance.
3. `decompose_spatial(offset_map, noise_variance=â/K)` donne la DSNU et sa
   décomposition, avec u(DSNU) par la formule du §3.2.1.
4. Stationnarité : moyennes des poses, pente par moindres carrés sur
   l'indice centré, rapport var(f)/(â/N).

### 3.3.2 `measure_temporal_noise(frames, *, dark=None, method="auto")`

- `"auto"` choisit la paire pour K = 2 et la pile au-delà.
- `dark` (scalaire, carte (H, W) ou `DarkFrameResult`) ne sert qu'au
  niveau : signal = moyenne − offset moyen. La variance n'est pas
  corrigée de a, qui fait partie de σ²(μ).
- Une pile passée à la place de `dark` est refusée, pour éviter la
  confusion avec une carte d'offset.

### 3.3.3 `measure_ptc(flats, dark, *, exposures, saturation_dn, ...)`

1. `dark` : pile (mesurée par `measure_dark_frame`) ou résultat déjà
   calculé.
2. Pour chaque niveau : `measure_temporal_noise(..., dark=dark_result)`,
   puis la fraction d'échantillons au code de saturation.
3. Tri par signal, détection du maximum de variance, sélection des niveaux
   (≤ `fit_fraction` × saturation, fraction saturée ≤
   `max_saturated_fraction`).
4. Ajout du point d'obscurité (`include_dark=True`).
5. Moindres carrés pondérés y = a + b·x : covariance (XᵀWX)⁻¹ non
   renormalisée par le χ², car les incertitudes sont connues. Le χ²
   réduit est rendu séparément.
6. Grandeurs dérivées (§3.2.4) et erreur de linéarité si `exposures` est
   fourni.

Au moins trois points retenus sont exigés (obscurité comprise).

### Exemple et recommandations

**Chaîne complète sur le simulateur :**

```python
import numpy as np
from eoqual.measure.synthetic import MatrixSensor, SensorParams
from eoqual.measure.radiometry import (
    measure_dark_frame, measure_temporal_noise, measure_ptc)

sensor = MatrixSensor(SensorParams(seed=1))
truth = sensor.truth

# 1. Obscurité : 32 poses
dark = measure_dark_frame(sensor.dark(n_frames=32, seed=1).frames)
print(f"offset {dark.mean_offset:.3f} ± {dark.u_mean_offset:.3f} DN")
print(f"b      {dark.temporal_variance:.3f} ± {dark.u_temporal_variance:.3f} DN²  (vrai {truth.b:.3f})")
print(f"DSNU   {dark.dsnu:.3f} ± {dark.u_dsnu:.3f} DN  (vrai {truth.dsnu_rms:.3f})")
print("colonnes", dark.decomposition.col, "lignes", dark.decomposition.row)
print("stationnarité", dark.stationarity_ratio, "dérive", dark.drift_per_frame)

# 2. Bruit temporel à un niveau
pair = sensor.expose(truth.photons_for_signal(1000.0), n_frames=2, seed=2).frames
tn = measure_temporal_noise(pair, dark=dark)
print(f"signal {tn.signal:.1f} DN, σ² {tn.temporal_variance:.1f} ± {tn.u_temporal_variance:.1f} DN²")

# 3. PTC : une paire par niveau, jusqu'au-delà de la saturation
levels = [50, 150, 400, 800, 1500, 2200, 3000, 3600, 3900, 4100, 4500]
photons = [truth.photons_for_signal(s) for s in levels]
flats = [sensor.expose(p, n_frames=2, seed=10 + i).frames for i, p in enumerate(photons)]
ptc = measure_ptc(flats, dark, exposures=photons, saturation_dn=truth.saturation_dn)
print(f"K = {ptc.gain:.4f} ± {ptc.u_gain:.4f} e⁻/DN (vrai {truth.gain})")
print(f"σ_lect = {ptc.read_noise_e:.3f} ± {ptc.u_read_noise_e:.3f} e⁻")
print(f"saturation {ptc.saturation_signal_dn:.0f} DN = {ptc.saturation_e:.0f} e⁻")
print(f"χ² réduit {ptc.reduced_chi2:.2f}, linéarité {ptc.linearity_error:.3f} %")
print("niveaux retenus", ptc.used)
```

**Données réelles** : mêmes appels sur des tableaux (K, H, W) lus par
ailleurs (FITS, TIFF…). Recommandations tirées du §2 :

- assez de poses d'obscurité pour que â/K soit petit devant la DSNU
  (K ≥ 16) ;
- vérifier `stationarity_ratio` avant d'interpréter la DSNU ;
- pour la PTC, une dizaine de niveaux répartis sur toute la plage, dont
  quelques-uns au-delà de la saturation (pour la détecter), et des
  niveaux resserrés près du plafond si le point de saturation importe ;
- `exposures` = temps de pose si la source est stable ;
- un `reduced_chi2` loin de 1 signale une non-linéarité ou un bruit non
  modélisé (bruit de ligne, source instable).

**Push broom** : `measure_dark_frame(lines, layout="pushbroom")` avec
`lines` de forme (M, W) ; `offset_map` est alors la ligne moyenne (1, W).

## 3.4 Validation

Toutes les tolérances découlent des incertitudes-types rendues par les
mesures (5 u), plus une marge de 0,5 % de a pour la validité du terme de
quantification 1/12. Les capteurs de test changent avec
`EOQUAL_TEST_SEED` : la série passe pour 50 décalages de graine.

Résultat type (graine 7, DSNU marquée, 64 poses) :

| Grandeur | Mesurée | Vraie |
|---|---|---|
| a (DN²) | 2,8586 ± 0,0020 | 2,8611 |
| DSNU (DN) | 1,3761 ± 0,0008 | 1,3755 |
| σ_r / σ_c / σ_p (DN) | 0,425 / 0,841 / 1,003 | 0,427 / 0,838 / 1,003 |
| K (e⁻/DN), PTC à 11 niveaux | 2,9997 ± 0,0070 | 3 |
| σ_lect (e⁻) | 4,997 ± 0,012 | 5 |

L'écart sur a (−0,09 %, soit 1,2 u) reste dans l'approximation du terme
de quantification.

Liste des tests : §3.4.1.

### 3.4.1 Tests

| Test | Vérifie |
|---|---|
| `test_offset_map` | carte d'offset à √(b/K) près ; offset moyen à 5 u |
| `test_temporal_variance_is_a` | â = a vrai |
| `test_dsnu_total` | DSNU = écart-type de la carte vraie, à 5 u(DSNU) |
| `test_dsnu_decomposition` | σ_r, σ_c (5 u, formule du §2.7), σ_p à 1 %, profil des colonnes |
| `test_stationary_stack`, `test_drift_is_detected` | rapport ≈ 1 et pente nulle ; dérive injectée de 0,02 DN/pose retrouvée, rapport > 10 |
| `test_pushbroom_block` | bloc (400, 512) : DSNU entre détecteurs et a |
| `test_decomposition_exact_map` | carte sans bruit : composantes et total exacts |
| `test_pair` (20, 1 000, 3 000 DN), `test_stack` | σ² = a + b·μ ; signal corrigé |
| `test_pair_removes_fixed_pattern` | PRNU 3 % : variance spatiale > 5 σ², paire exacte |
| `test_pair_insensitive_to_level_shift` | écart additif de 7 DN entre poses : σ² inchangé |
| `test_gain`, `test_read_noise` | K à 5 u et à 2 % ; σ_lect = 5 e⁻ à 5 u + 1 % ; a |
| `test_fit_quality` | χ² réduit à 5·√(2/ν) de 1 |
| `test_saturation_excluded` | saturation du convertisseur détectée, niveaux > 70 % exclus |
| `test_full_well_saturation` | puits à 6 000 e⁻ : point mesuré sur le plateau théorique de la variance tronquée |
| `test_linearity` | erreur de linéarité < 0,2 % (capteur linéaire) |
| `test_saturation_not_reached`, `test_dark_result_accepted`, `test_input_checks` | cas limites et contrôles d'entrée |

## 3.5 Limites

- *Bruit corrélé entre pixels* (bruit de ligne, S3) : les incertitudes
  supposent des pixels indépendants et sont alors sous-estimées. Le
  rapport de stationnarité le signale pour l'obscurité.
- *Pixels chauds* (S1) : ils gonflent la DSNU et a. `dark_frame` n'a pas
  encore d'option robuste (§3.6) : mesurer sur une zone qui les exclut,
  ou les repérer d'abord avec `defective_pixels` (chapitre 7).
- *Source instable* : la paire n'est pas corrigée d'une fluctuation
  multiplicative ε entre poses. Le biais ε²·var_spatiale/2 est négligeable
  sur champ uniforme, sauf forte PRNU (§3.2.3, vérifié par
  `test_pair_source_fluctuation_bias`).
- *Saturation* : point imprécis d'environ ±√FW par nature (§3.2.4) ; la
  précision dépend de l'espacement des niveaux près du plafond.
- *Linéarité* : estimée sur la plage de l'ajustement de la PTC, non sur
  5–95 % de la saturation comme dans EMVA 1288.
- *Quantification* : la correction de 1/12 n'est valable que si
  σ₀ ≳ 0,5 DN ([simulateur, §7](../simulateur.md#7-limites-et-choix-de-modélisation)).

## 3.6 Pistes

- **Obscurité** : statistiques robustes en option de `dark_frame` (médiane,
  écrêtage sigma), pour que les pixels chauds ne gonflent pas la DSNU et a.
- **PTC** : bootstrap sur les pixels comme alternative aux incertitudes
  analytiques ; correction de la fluctuation de source (normalisation de
  la paire par le rapport de ses moyennes, EMVA 1288) ; linéarité sur
  5–95 % de la saturation.
