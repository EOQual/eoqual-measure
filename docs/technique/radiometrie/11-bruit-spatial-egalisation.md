# 11. Bruit spatial fixe et bruit d'égalisation

## 11.1 Objet

Mesure du bruit spatial fixe (FPN) d'un champ uniforme, avec sa
décomposition et ses éventuelles bandes périodiques (`spatial_noise`).
Mesure du FPN **résiduel** après égalisation d'un push broom, séparé en
basses et hautes fréquences (`equalization_noise`).

- **Avant égalisation**, le bruit spatial fixe mesure la non-uniformité
  du capteur : DSNU et PRNU (§2.5), colonnage d'un push broom,
  bandes d'un whisk broom (période égale au nombre de détecteurs balayés)
  ou d'une lecture multiplexée.
- **Après égalisation** (correction des offsets et des gains de chaque
  détecteur), il reste un résidu qui qualifie la correction elle-même.
  Il comprend le bruit des coefficients, issu du bruit temporel des
  acquisitions de calibration, et les erreurs systématiques (vignettage
  mal corrigé, dérive des gains). Les deux se distinguent par leur échelle
  spatiale : HF détecteur à détecteur pour le premier, BF pour les
  secondes.

Les deux mesures reposent sur le même principe que `dark_frame`. La carte
moyenne (ou la ligne moyenne) contient un résidu temporel connu, qu'il
faut retirer **composante par composante**.

**Statut** : validées contre les simulateurs S0 et S2
(`tests/test_spatial_noise.py`, `tests/test_equalization_noise.py`,
17 tests ; robustes sur 30 graines ; incertitudes vérifiées par
Monte-Carlo). Module : `eoqual.measure.radiometry.{spatial,
equalization}`.

## 11.2 Principe

### 11.2.1 Bruit spatial d'une pile ou d'une zone push broom

- **Pile** (K, H, W) : carte moyenne m̄, résidu temporel v = σ²_temp/K
  par pixel ; σ²_spatial = var(m̄) − v. La décomposition ligne / colonne /
  pixel est celle du §2.7.
- **Zone push broom** (M, W) : ligne moyenne sur M lignes (un détecteur
  par colonne), v = σ²_temp/M. Le bruit temporel est estimé par la
  différence de lignes successives, σ²_temp = mean((x_{i+1} − x_i)²)/2.
  Une variation lente de la scène le long de la trace n'y contribue qu'au
  second ordre, contrairement à la variance le long de la colonne.
- **Obscurité retirée** : soustraire une carte d'offset estimée sur K_d
  poses ajoute son propre résidu b/K_d, qui est propagé dans v. Le bruit
  spatial mesuré est alors celui du signal seul, μ·PRNU (PRNU apparente =
  σ_spatial/μ).

L'incertitude, par rapport à la carte réalisée, suit la formule du
§3.2.1.

### 11.2.2 Bandes périodiques

Pour un profil p_j = m + q_{j mod k} + r_j (bandes q de période k, résidu
r de variance σ_r², n = W/k colonnes par phase) :

- moyennes de phase Q_φ = moyenne des p_j, j ≡ φ (mod k) ;
- E[var(Q)] = σ_q² + (σ_r²/n)·(k − 1)/k ;
- σ_r² s'estime sans biais par la variance intra-phase,
  Σ (p_j − Q_φ)² / (k·(n − 1)).

d'où σ_q² = var(Q) − (s²_intra/n)·(k − 1)/k. Le résidu r contient la
dispersion propre des détecteurs (et le bruit temporel moyenné). Le σ_q
mesuré est donc celui de la structure *périodique* : les moyennes de
phase de la PRNU réalisée en font partie, puisqu'elles sont fixes.

### 11.2.3 Whisk broom

Avec `layout="whiskbroom"` et `period=k` (nombre de détecteurs), l'image
(n_balayages·k, W) est analysée par lignes :

- profil des moyennes de lignes ;
- bruit temporel le long des lignes, σ²_temp = mean((x_{j+1} − x_j)²)/2 ;
- bruit spatial du profil corrigé de σ²_temp/W ;
- bandes de période k sur le profil (§11.2.2) ;
- **différence aller/retour**, la moyenne des lignes retour moins celle
  des lignes aller. La dispersion entre détecteurs s'y annule, puisque les
  mêmes détecteurs servent dans les deux sens. Son incertitude est tirée
  de la dispersion résiduelle du profil, une fois retirés la moyenne de
  chaque détecteur et l'effet aller/retour.

Validation (`tests/test_p3_measures.py::test_whiskbroom_spatial_noise`,
10 détecteurs, 60 balayages, offset retour de 0,8 DN et gain retour de
0,2 %) : différence mesurée 3,87 ± 0,08 DN pour 3,80, bandes 14,76 pour
14,74 DN, bruit spatial 14,89 pour 14,87 DN.

### 11.2.4 Séparation BF / HF d'un profil égalisé

La ligne moyenne p (W,) est lissée par une gaussienne d'écart-type σ_g
(bords en miroir) : BF = h ∗ p, HF = p − BF. La coupure f_c est la
fréquence où le gain de h vaut 1/2 :

exp(−2π²σ_g²f_c²) = 1/2  ⇔  σ_g = √(ln 2 / 2) / (π·f_c),

soit σ_g = 6,0 détecteurs pour f_c = 1/32 (défaut).

Le résidu temporel ε de la ligne moyenne est blanc, de variance v. Un
filtre de réponse impulsionnelle g en transmet la variance v·Σg². D'où :

- σ²_BF = var(BF) − v·Σh² ;
- σ²_HF = var(HF) − v·Σ(δ − h)².

Σh² + Σ(δ − h)² < 1 : la différence correspond à la covariance négative
des deux sorties. Pour σ_g = 6, Σh² ≈ 0,047 et Σ(δ − h)² ≈ 0,914.

### 11.2.5 Incertitude d'une composante filtrée

Pour une composante estimée par S = var(g ∗ p) − v·Σg², avec un motif de
variance S dans la bande de g :

Var(Ŝ) ≈ [4·S·v + 2v²·Σ_k (g⋆g)_k² + 2(v·Σg²)²/M] / W

- le **terme croisé** 4·S·v/W ne dépend pas de la transmission du filtre :
  c'est la covariance entre le motif et le bruit, dont seule compte la
  part dans la bande du motif ;
- le **terme propre** du bruit filtré tient compte de sa corrélation par
  l'autocorrélation de g. Traiter les W valeurs filtrées comme
  indépendantes sous-estime u(BF) d'un facteur ~6 (§11.6) ;
- le dernier terme vient de l'erreur sur v̂.

Pour g = δ, on retrouve la formule de la DSNU (§3.2.1).

### 11.2.6 Spectre de la ligne moyenne

On calcule le périodogramme unilatéral de la ligne moyenne centrée,
normalisé pour que Σ S(f)·Δf = var(p) (égalité de Parseval, exacte, et
fréquence de Nyquist comptée une fois). Un bruit blanc de variance σ² a
un spectre plat de niveau 2σ². Une erreur de gain sinusoïdale apparaît
comme un pic à sa fréquence. C'est la version 1D du NPS (fiche
`noise_power_spectrum`).

## 11.3 Mise en œuvre

```
eoqual/measure/radiometry/
├── spatial.py        measure_spatial_noise → SpatialNoiseResult ; band_structure → BandStructure
└── equalization.py   measure_equalization_noise → EqualizationNoiseResult ; profile_spectrum
```

**`measure_spatial_noise(data, *, layout="stack"|"pushbroom", dark=None, period=None)`**

1. Carte moyenne et σ²_temp (variance temporelle pour une pile,
   différences de lignes pour un push broom).
2. Obscurité : niveau, carte, profil ou `DarkFrameResult`. Dans ce dernier
   cas, son résidu b/K_d est ajouté à v.
3. σ_spatial, u, PRNU apparente ; décomposition (pile) ; bandes sur le
   profil des colonnes si `period` est donné.

**`measure_equalization_noise(image, *, cutoff=1/32)`**

1. Ligne moyenne, σ²_temp par différences de lignes, v = σ²_temp/M.
2. BF par `scipy.ndimage.gaussian_filter1d(mode="reflect")`, HF par
   différence ; fractions Σh², Σ(δ − h)² calculées sur la réponse
   impulsionnelle au centre du profil.
3. FPN, BF, HF corrigés et leurs incertitudes (§11.2.5) ; spectre.

L'égalisation elle-même n'est pas une mesure : elle relève de la chaîne
de traitement (eoqual-destriping). Les tests la réalisent explicitement
sur le simulateur.

### Exemple et recommandations

```python
import numpy as np
from eoqual.measure.synthetic import PushbroomSensor, SensorParams
from eoqual.measure.radiometry import (
    measure_dark_frame, measure_spatial_noise, measure_equalization_noise)

W = 1024
pb = PushbroomSensor(SensorParams(shape=(1, W), seed=40, dsnu_pixel=0.0, dsnu_col=0.8,
                                  prnu_pixel=0.0, prnu_col=0.01))
t = pb.truth

# Avant égalisation : non-uniformité du capteur
dark = measure_dark_frame(pb.dark(n_lines=1000, seed=1).image, layout="pushbroom")
raw = pb.expose(t.photons_for_signal(1500.0), n_lines=500, seed=2).image
sn = measure_spatial_noise(raw, layout="pushbroom", dark=dark, period=4)
print(f"FPN {sn.spatial_sigma:.2f} ± {sn.u_spatial_sigma:.2f} DN, PRNU apparente {sn.relative:.3%}")
print("bandes de période 4 :", sn.bands.sigma)

# Égalisation à partir d'une calibration (obscurité + champ uniforme)
cal_dark = pb.dark(n_lines=200, seed=10).image.mean(axis=0)
cal_flat = pb.expose(t.photons_for_signal(2000.0), n_lines=200, seed=11).image.mean(axis=0)
gain = (cal_flat - cal_dark) / np.mean(cal_flat - cal_dark)
x = pb.expose(t.photons_for_signal(1500.0), n_lines=1000, seed=12).image
equalized = (x - cal_dark) / gain

eq = measure_equalization_noise(equalized, cutoff=1 / 32)
print(f"FPN résiduel {eq.fpn:.3f} ± {eq.u_fpn:.3f} DN ({eq.fpn_relative:.4%})")
print(f"BF {eq.lf:.3f} ± {eq.u_lf:.3f}  HF {eq.hf:.3f} ± {eq.u_hf:.3f} DN")
# eq.frequency, eq.spectrum : spectre de la ligne moyenne (tracé log-log)
```

Recommandations :

- choisir M assez grand pour que v = σ²_temp/M soit petit devant le FPN
  recherché. Le FPN HF après une bonne égalisation est souvent du même
  ordre que v, et u(HF) le dit ;
- la coupure BF/HF est un choix d'analyse : la fixer selon l'échelle des
  défauts recherchés (vignettage : quelques centaines de détecteurs ;
  raccords de barrettes : quelques dizaines).

## 11.4 Validation

**Monte-Carlo** (200 réalisations de l'image égalisée, calibration de
200 lignes et erreur BF de 0,4 %) :

| | FPN | BF | HF |
|---|---|---|---|
| biais moyen (DN) | +0,0003 | −0,0000 | +0,0013 |
| écart-type de l'erreur (DN) | 0,0226 | 0,0228 | 0,0222 |
| incertitude u moyenne (DN) | 0,0223 | 0,0222 | 0,0236 |

Liste des tests : §11.4.1.

### 11.4.1 Tests

| Test | Vérifie |
|---|---|
| `test_stack_with_dark` | σ_spatial = écart-type de μ·g vrai à 5 u ; PRNU apparente = PRNU vraie à 1 % ; v propagé |
| `test_stack_decomposition` | composantes = décomposition exacte de la carte vraie |
| `test_stack_without_dark` | σ_spatial = écart-type de offset + DSNU + μ·g |
| `test_pushbroom` | σ entre détecteurs = écart-type de μ·g_j à 5 u |
| `test_pushbroom_temporal_insensitive_to_along_track_ramp` | rampe de ±20 DN le long de la trace : σ²_temp à 0,2 %, σ_spatial inchangé |
| `test_bands`, `test_band_structure_exact` | σ et profil des bandes = ceux du profil vrai complet |
| `test_fpn`, `test_lf_hf` (calibration 200 lignes + erreur BF de 0,4 %, et 2 000 lignes sans erreur) | FPN, BF, HF = résidu vrai à 5 u |
| `test_noise_fractions_on_white_noise` | var(BF), var(HF) d'un bruit blanc = v·Σh², v·Σ(δ − h)² ; aucun FPN détecté |
| `test_spectrum_normalisation_and_white_level` | Parseval exact (W pair et impair) ; niveau 2σ² |
| `test_lf_error_seen_in_spectrum` | pic du spectre à 3/W pour une erreur de gain à 3 cycles |
| `test_cutoff_convention`, `test_input_checks` | gain 1/2 à la coupure ; contrôles d'entrée |

## 11.5 Limites

- *Zone uniforme supposée* : une variation de la scène **en travers** de
  la trace est indiscernable d'un FPN BF. Une scène peu texturée suffit
  pour le HF ; pour le BF, il faut un champ réellement uniforme
  (calibration interne, diffuseur solaire) ou une scène de moyenne
  connue.
- *Bandes* : période k connue a priori.
- *Whisk broom* (`layout="whiskbroom"`) : voir §11.2.3.
- *Bords* : le lissage en miroir modifie légèrement BF et HF sur ~3σ_g
  détecteurs de chaque bord, et les fractions de bruit sont celles du
  centre.
- *Pixels défectueux* : ils dominent le HF s'ils ne sont pas masqués au
  préalable (`defective_pixels`, P2).

## 11.6 Variantes écartées

- *Incertitudes BF et HF avec des valeurs filtrées indépendantes* : les W
  valeurs filtrées sont corrélées, et les traiter comme indépendantes
  sous-estime u(BF) d'un facteur 6 environ. La formule générale du
  §11.2.5 en tient compte ; elle est vérifiée par Monte-Carlo à 6 % près.

## 11.7 Pistes

- Masque de détecteurs défectueux en entrée.
- Correction de la variation de scène en travers par une référence
  externe (scène de moyenne connue), pour le BF en vol.
