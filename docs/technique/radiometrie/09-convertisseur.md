# 9. Convertisseur : DNL, INL, codes manquants

## 9.1 Objet

Non-linéarité différentielle (DNL), non-linéarité intégrale (INL) et
codes manquants, par la méthode de l'histogramme.

**Statut** : validée contre le simulateur S4 (`tests/test_p3_measures.py`,
4 tests ; robustes sur 30 graines). Module :
`eoqual.measure.radiometry.adc:adc_histogram`.

## 9.2 Principe

Si la tension d'entrée a une densité f lisse à l'échelle du pas, le
nombre d'occurrences du code c vaut n_c ≈ N·f(c)·w_c. D'où
DNL_c = n_c/n̂_c − 1, où n̂_c est la densité attendue, d'incertitude de
Poisson 1/√n̂_c. Deux façons d'estimer n̂ :

- **`density="uniform"`** (IEEE Std 1241) : entrée uniformément répartie,
  par exemple une rampe beaucoup plus large que le bruit. Sur le plateau
  de l'histogramme, f est constante et n̂ = (Σ n_c)/(nombre de codes),
  exactement, puisque Σ w = nombre de codes. Le plateau est rogné d'une
  largeur de transition de chaque côté, mesurée de 10 % à 90 % du
  plateau. L'**INL** est alors exacte : somme cumulée de la DNL,
  rapportée à la droite des extrémités. Son incertitude est un pont
  brownien d'écart-type ≤ √(n/(4n̂)) au milieu de la plage.
- **`density="local"`** : densité lisse quelconque. n̂_c est estimé par une
  **régression locale quadratique** de l'histogramme, avec des poids
  gaussiens d'écart-type σ (16 codes) et **sans le code ni ses voisins
  immédiats**. Une moyenne pondérée simple serait biaisée de σ²·f″/2 là
  où la densité est courbée. Les codes où les régressions de largeurs σ
  et σ/2 diffèrent de plus de 3 fois leur bruit (densité trop courbée,
  queues) sont écartés de la DNL. Un bord écrêté (code extrême
  anormalement peuplé) est écarté sur 4σ codes. Incertitude :
  u² = (1 + Σk²)/n̂, avec k le noyau de la régression. La DNL des voisins
  n'y contribue presque pas, car la somme des largeurs de codes contigus
  se télescope. L'INL n'est connue qu'à l'échelle de σ : ses variations
  plus lentes sont absorbées par la densité estimée.

**Codes manquants** : aucune occurrence pour n̂ ≥ 50, alors que la
probabilité d'un zéro fortuit vaut e^{−n̂} ≈ 2·10⁻²². En mode local, ils
sont cherchés sur tous les codes peuplés, y compris ceux écartés de la
DNL.

## 9.3 Mise en œuvre

```python
import numpy as np
from eoqual.measure.synthetic import MatrixSensor, SensorParams
from eoqual.measure.radiometry import adc_histogram

s = MatrixSensor(SensorParams(shape=(256, 256), seed=2, bit_depth=10, adc_dnl=0.15, adc_missing_codes=4,
                              read_noise=30.0, conversion_gain=1.0, offset=50.0,
                              dsnu_pixel=0.0, dsnu_col=0.0, prnu_pixel=0.0, prnu_col=0.0))
t = s.truth
ramp = np.linspace(50, 900, 256)[None, :].repeat(256, axis=0)       # rampe : densité uniforme
frames = s.expose(t.photons_for_signal(1.0) * ramp, n_frames=64, seed=1).frames
r = adc_histogram(frames, bit_depth=10, density="uniform")
print(f"DNL efficace {r.dnl_rms:.3f} LSB, max {r.dnl_max:.2f} ; codes manquants {r.missing.tolist()}")
print(f"INL de {r.inl.min():.2f} à {r.inl.max():.2f} LSB sur les codes {r.codes.min()}–{r.codes.max()}")
```

## 9.4 Validation

Résultats types (10 bits, s_DNL = 0,15, 4 codes manquants, bruit de
30 DN) :

| Mode | Codes | z : moyenne / écart-type | DNL efficace mesurée / vraie | INL |
|---|---|---|---|---|
| local, sans écrêtage (16 poses) | 91–927 | −0,003 / 1,018 | 0,219 / 0,218 | — |
| local, avec écrêtage | 141–958 | 0,010 / 1,073 | 0,219 / 0,220 | — |
| uniforme (128 poses) | 230–763 | 0,011 / 0,968 | — | erreur max 0,14 LSB pour une amplitude de 1,48 |

Liste des tests : §9.4.1.

### 9.4.1 Tests

| Test | Vérifie |
|---|---|
| `test_adc_dnl_local` (sans et avec écrêtage) | z = (DNL − vraie)/u de moyenne nulle et d'écart-type 1 ± 0,15 ; codes manquants de la plage retrouvés exactement |
| `test_adc_uniform_ramp_gives_inl` | z calibré ; INL à 4 écarts-types du pont brownien, et INL vraie résolue |
| `test_adc_checks` | contrôles d'entrée |

## 9.5 Limites

- *Densité* : le mode `"local"` suppose une densité d'entrée lisse à l'échelle de la
  régression ; il ne donne pas l'INL au-delà de cette échelle. L'INL demande une
  entrée uniforme (`"uniform"`).
- *Écrêtage* : un bord écrêté est écarté automatiquement, mais un écrêtage massif
  réduit la plage analysable.
- *Quantité de données* : l'incertitude de l'INL croît comme √(n/n̂) ; il faut des
  centaines à des milliers d'occurrences par code.

## 9.6 Variantes écartées

La méthode a été construite en contrôlant la calibration de z (écart à la
vérité rapporté à l'incertitude annoncée). Variantes écartées :

1. *Lissage gaussien de l'histogramme* : z d'écart-type 2,0, et INL fausse
   de 6 LSB. La somme cumulée intègre les erreurs lentes de la densité
   estimée : c'est une limite de principe, d'où le mode `uniform` pour
   l'INL.
2. *Exclusion du code, puis de ses voisins, de l'estimation de la densité* :
   sans effet ; l'hypothèse d'anticorrélation ne tient pas ici.
3. *Mesure séparée du centre et des bords* : z vaut 0,62 au centre
   (incertitude trop large, terme de DNL des voisins à retirer) et 3,3 aux
   bords (biais de courbure).
4. *Régression quadratique seule* : centre calibré, bords encore à 3,5.
   Une régression sur ln n pondérée par les comptes donne un biais de
   −1,3 u, les poids étant corrélés aux données.
5. *Rogner deux largeurs de transition en mode uniforme* : 316 codes
   gardés au lieu de 534 avec une seule largeur, sans gain de calibration.

La méthode retenue (§9.2 : régression locale quadratique, test de
cohérence entre deux largeurs, retrait d'un bord écrêté) est calibrée
partout (écart-type de z de 1,02 à 1,07).
