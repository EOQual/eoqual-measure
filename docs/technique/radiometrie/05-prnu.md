# 5. Non-uniformité de réponse (PRNU)

## 5.1 Objet

Carte de réponse relative d'un capteur matriciel, PRNU totale, séparation
basses / hautes fréquences (vignettage et PRNU propre) et décomposition
ligne / colonne / pixel.

**Statut** : validée contre les simulateurs S0 et S4
(`tests/test_prnu_linearity_defects.py`, 3 tests de PRNU et un test
commun avec la linéarité ; robustes sur 30 graines). Module :
`eoqual.measure.radiometry.nonuniformity:measure_prnu`.

## 5.2 Principe

**Carte de réponse.** Pour une pile de K champs uniformes et une carte
d'obscurité estimée sur K_d poses :

m(p) = moyenne_temporelle(flat) − ô(p) = μ·T(p)·g(p) + ε(p),

où T est la transmission, g le gain relatif et ε un résidu de variance
v = σ²_temp/K + b/K_d. La réponse relative vaut r = m/μ̄. Avec plusieurs
niveaux, on combine les cartes relatives avec des poids μ²/v, inverses de
leur variance relative, et le résidu combiné a pour variance relative
1/Σ(μ²/v).

**PRNU.** PRNU² = var(r) − v_rel, avec l'incertitude du §3.2.1. C'est la dispersion de T·g, transmission comprise. Méthode EMVA
1288, avec au moins deux niveaux : σ²_s(μ) = var(m) − v = PRNU²·μ², ajusté
en passant par l'origine (moindres carrés pondérés par 1/u(σ²_s)²).

**Basses et hautes fréquences.** On ajuste à r une surface lisse S, polynôme
de Legendre 2D de degré total d (4 par défaut), ce qui donne :

- BF = écart-type relatif de S : vignettage, éclairement non uniforme ;
- HF = √(var(r/S) − v_rel/S̄²) : PRNU propre, avec les poussières, dont
  l'échelle (quelques pixels) dépasse ce que peut suivre une surface
  lisse ;
- décomposition ligne / colonne / pixel **de r/S** (§2.7). Faite
  sur r, elle compterait le vignettage dans les profils moyens : la
  composante colonne mesurée valait 0,016 au lieu de 0,0042 (§5.6).

La surface absorbe la part de basse fréquence des profils aléatoires de
lignes et de colonnes, soit environ (d + 1)/W de leur variance : environ
−2 % mesuré sur la composante colonne pour d = 4 et W = 256.

**Non-linéarité commune.** Elle comprime aussi le contraste entre pixels :
pour e' = e(1 − βe/FW), la PRNU apparente au signal linéaire x vaut
PRNU₀·(1 − 2cx)/(1 − cx), avec c = βK/FW. Le test le vérifie : rapport
mesuré 0,919 pour 0,919 prédit, entre 500 et 3 500 DN. En revanche,
l'égalisation relative (§6.2) n'en garde qu'un résidu du second
ordre.

## 5.3 Mise en œuvre

**`measure_prnu(flats, dark, *, exclude=None, smooth_degree=4)`** :
`flats` est une pile ou une séquence de piles, `dark` une pile ou un
`DarkFrameResult`. `exclude` masque des pixels (par exemple
`map_defective_pixels(...).any`) ; ils valent NaN dans `response_map` et
sont ignorés dans tous les calculs.

```python
from eoqual.measure.synthetic import MatrixSensor, SensorParams
from eoqual.measure.radiometry import measure_dark_frame, measure_prnu

s = MatrixSensor(SensorParams(seed=3, prnu_pixel=0.01, prnu_row=0.002, prnu_col=0.004, vignetting=0.1))
t = s.truth
dark = measure_dark_frame(s.dark(n_frames=32, seed=1).frames)
flats = [s.expose(t.photons_for_signal(L), n_frames=16, seed=10 + i).frames
         for i, L in enumerate([500, 1500, 3000])]
r = measure_prnu(flats, dark)
print(f"PRNU {r.prnu:.4%} (EMVA {r.prnu_emva:.4%}), BF {r.low_frequency:.3%}, HF {r.high_frequency:.3%}")
print("colonnes", r.decomposition.col, "lignes", r.decomposition.row, "pixels", r.decomposition.pixel)
```

## 5.4 Validation

Résultat type (PRNU pixel 1 %, ligne 0,2 %, colonne 0,4 %, vignettage
10 %, 3 niveaux de 16 poses) :

| Grandeur | Mesurée | Vraie |
|---|---|---|
| PRNU totale | 0,024572 ± 0,000008 | 0,024578 |
| PRNU EMVA | 0,024573 | 0,024578 |
| BF (vignettage) | 0,021973 | 0,021982 |
| HF | 0,010994 ± 0,000008 | 0,010990 |
| colonne / ligne / pixel | 0,00414 / 0,00215 / 0,009956 | 0,00422 / 0,00215 / 0,009960 |

Liste des tests : §5.4.1.

### 5.4.1 Tests

| Test | Vérifie |
|---|---|
| `test_prnu_total` | PRNU = dispersion vraie de T·g à 5 u ; EMVA et par niveau à 1 % ; erreur de la carte = √v_rel à 5 % |
| `test_prnu_low_and_high_frequency` | BF = dispersion de la surface ajustée sur la vérité à 0,5 % ; HF à 5 u ; décomposition pixel à 1 %, colonne à 8 %, ligne à 15 % |
| `test_prnu_exclusion_and_checks` | masque d'exclusion ; contrôles d'entrée |
| `test_relative_equalization_misses_common_nonlinearity` | compression de la PRNU par (1 − 2cx)/(1 − cx) ; résidu d'égalisation du second ordre (< 0,1 %) |

## 5.5 Limites

- La surface polynomiale ne suit ni les poussières ni les structures de
  quelques pixels, qui restent dans la HF. Un éclairement non uniforme de
  la source est indiscernable du vignettage.

## 5.6 Variantes écartées

- *Décomposer la réponse totale r plutôt que r/S* : les profils moyens
  contiennent alors le vignettage (composante colonne mesurée à 0,016 pour
  0,0042 vraie). La décomposition ligne / colonne / pixel porte donc sur
  r/S (§5.2).
