# 4. Courant d'obscurité

## 4.1 Objet

Mesure du courant d'obscurité par pixel à partir de piles d'obscurité à
plusieurs temps de pose : moyenne, dispersion (DSNU du courant), pixels
chauds, gain déduit du caractère poissonnien et, avec plusieurs
températures, loi en température.

**Statut** : validée contre le simulateur S1 (`tests/test_dark_current.py`,
7 tests ; robustes sur 30 graines). Module :
`eoqual.measure.radiometry.dark:measure_dark_current`.

## 4.2 Principe

### 4.2.1 Régression par pixel

Pour une pile de K_i poses au temps t_i, la moyenne m_i(p) d'un pixel p
vaut o(p) + I(p)·t_i, avec un bruit de variance v_i(p)/K_i. La moyenne
temporelle de la pose ne contient pas que le bruit de lecture : v_i(p) =
a + I(p)·t_i/K, car le bruit d'obscurité est poissonnien. Par moindres
carrés ordinaires sur les n_t temps :

Î = Σ (t_i − t̄)(m_i − m̄) / S_xx,  S_xx = Σ (t_i − t̄)²,

Var(Î) = Σ (t_i − t̄)²·v_i/K_i / S_xx².

Les moindres carrés ordinaires restent sans biais quand les variances
sont inégales, et la formule de Var(Î) tient compte de ces différences.

**Pixels saturés.** Une pose où le pixel atteint le code de saturation est
retirée de *son* ajustement (moindres carrés pondérés par un masque,
vectorisés). Un pixel ajusté sur au moins deux temps distincts reste
valide, et un pixel qui sature dans l'obscurité est chaud par définition.

### 4.2.2 Courant moyen, dispersion, pixels chauds

- **Pixels chauds** : Î > f_hot × médiane et Î − médiane > z_hot·u(Î)
  (défauts : 10 et 5), ou saturation dans au moins une pose. La médiane,
  calculée sur les pixels non saturés, est insensible aux pixels chauds.
- **Pixels ordinaires** : valides et non chauds. Courant moyen Ī et
  incertitude √(Σ Var(Î))/N ; dispersion corrigée du bruit de la
  régression, DSNU_I² = var(Î) − mean(Var(Î)), d'incertitude donnée par
  la formule du §3.2.1, avec v = mean(Var(Î)).

### 4.2.3 Gain par le caractère poissonnien

Sur les pixels ordinaires, la variance temporelle moyenne vaut, en DN²,
v(t) = a + (Ī_e·t)/K², avec Ī_e = K·Ī le courant en e⁻/s. Sa pente en
fonction de t vaut donc Ī/K, et

K = Ī / (dv/dt).

C'est une PTC d'obscurité : le gain est mesuré **sans lumière**, ce qui
contrôle à la fois le caractère poissonnien du courant et la PTC
(chapitre 3). L'ajustement de v(t) est pondéré par u(v_i) =
v_i·√(2/((K_i − 1)N)), et l'incertitude de K combine celles de Ī et de la
pente.

### 4.2.4 Loi en température

Pour chaque température qui compte au moins deux temps de pose distincts,
on calcule le courant moyen des pixels ordinaires I(T). La loi
I = I₀·(T/T₀)³·exp(−(E_g/2k)(1/T − 1/T₀)) se linéarise :

ln I − 3·ln(T/T₀) = ln I₀ − (E_g/2k)·(1/T − 1/T₀).

On fait une régression pondérée par (I/u_I)², avec T₀ la température
médiane, puis on résout exactement I(T₀ + ΔT) = 2·I(T₀) pour obtenir la
température de doublement.

## 4.3 Mise en œuvre

**`measure_dark_current(stacks, exposure_times, *, temperatures=None, gain=None, saturation_dn=None, hot_factor=10, hot_sigma=5)`**

1. Groupement par température. La température de référence est la plus
   représentée (à égalité, la plus basse) parmi celles qui ont au moins
   deux temps de pose distincts.
2. Pour chaque groupe, `_fit_group` : moyennes et variances par pixel et
   par pose, masque de non-saturation, régression masquée, Var(Î),
   médiane, pixels chauds.
3. Référence : courant moyen, médiane, DSNU, gain poissonnien.
4. Plusieurs groupes : `TemperatureFit` (E_g, I₀, ΔT de doublement).

Résultat : `DarkCurrentResult`. `current_map` (DN/s), `current_noise`,
`offset_map` (ordonnée à l'origine) et `valid` sont des cartes ;
`current_map` vaut NaN pour les pixels non ajustables.

### Exemple et recommandations

```python
from eoqual.measure.synthetic import MatrixSensor, SensorParams
from eoqual.measure.radiometry import measure_dark_current

s = MatrixSensor(SensorParams(seed=1, dark_current=2.0, dark_nonuniformity=0.3,
                              hot_pixel_fraction=1e-3))
t = s.truth
times = [1.0, 10.0, 30.0, 60.0, 100.0]
temps = [20.0] * 5 + [10.0, 10.0, 30.0, 30.0]
all_times = times + [10.0, 100.0, 10.0, 100.0]
stacks = [s.dark(n_frames=16, seed=i, exposure_time=te, temperature=T).frames
          for i, (te, T) in enumerate(zip(all_times, temps))]

r = measure_dark_current(stacks, all_times, temperatures=temps, saturation_dn=t.saturation_dn)
print(f"courant {r.mean_current:.5f} ± {r.u_mean_current:.5f} DN/s = {r.mean_current_e:.3f} e⁻/s")
print(f"DSNU du courant {r.dsnu_relative:.1%}, {r.n_hot} pixels chauds")
print(f"gain poissonnien {r.poisson_gain:.4f} ± {r.u_poisson_gain:.4f} e⁻/DN")
f = r.temperature_fit
print(f"E_g = {f.bandgap:.4f} ± {f.u_bandgap:.4f} eV, doublement {f.doubling_temperature:.2f} K")
```

Recommandations, tirées du guide CCD d'astropy et de la théorie :

- **temps de pose** : le plus long doit donner une charge d'obscurité
  supérieure au bruit de lecture, soit t > σ_lect/I_d. Sinon la carte
  mesure le bruit et non le courant. Plusieurs poses par temps réduisent
  u(Î) en 1/√K ;
- **plage de temps étendue** : S_xx croît avec la dispersion des t_i, et
  u(Î) diminue en conséquence ;
- **température stable** pendant chaque pile (contrôle :
  `measure_dark_frame(...).drift_per_frame`) ;
- la **lueur d'amplificateur** produit une zone de forts courants près de
  son coin, comptée parmi les pixels chauds : la masquer si l'on veut ne
  compter que les défauts ponctuels.

## 4.4 Validation

Résultat type (256 × 256, I_d = 2 e⁻/s, s = 0,3, 0,1 % de pixels chauds,
16 poses à 1, 10, 30, 60, 100 s, et 10 / 30 °C) :

| Grandeur | Mesurée | Vraie |
|---|---|---|
| courant moyen (DN/s) | 0,665388 ± 0,000049 | 0,665429 |
| DSNU (DN/s) | 0,205590 ± 0,000049 | 0,205632 |
| pixels chauds | 66 détectés, 0 faux | 66 |
| gain poissonnien (e⁻/DN) | 2,9986 ± 0,0033 | 3 |
| E_g (eV) | 1,11990 ± 0,00010 | 1,12 |
| doublement à 20 °C (K) | 8,2887 | 8,2880 |

Liste des tests : §4.4.1.

### 4.4.1 Tests

| Test | Vérifie |
|---|---|
| `test_mean_and_dsnu` | courant moyen et DSNU des pixels ordinaires vrais, à 5 u ; DSNU relative √(e^{0,09} − 1) à 5 % |
| `test_current_map_and_offset` | erreur de la carte = u(Î) à 5 % ; offset à t = 0 sans biais |
| `test_hot_pixels` | aucune fausse détection, ≥ 98 % de détection, courant des chauds (même saturés aux longues poses) à 1 % |
| `test_poisson_gain` | gain poissonnien = K vrai à 5 u |
| `test_temperature_law` (0, 10, 20, 30 °C) | E_g = 1,12 eV à 5 u ; doublement à 1 % ; I(T) à 1 % |
| `test_glow_appears_in_current_map` | courant au coin = (I + lueur)/K |
| `test_input_checks` | contrôles d'entrée |

## 4.5 Limites

- *Loi en température* : un seul mécanisme (E_g/2). Un courant de
  diffusion (E_g) ou des pixels chauds d'énergie d'activation différente
  demandent un modèle à deux termes, et l'ajustement porte sur le courant
  moyen, pas pixel par pixel.
- *Linéarité en t* supposée : un courant qui dépend du signal accumulé,
  ou une rémanence, n'est pas modélisé.
- *Variations d'offset entre piles* : l'ordonnée à l'origine absorbe un
  offset constant, mais une dérive entre les piles de temps différents
  biaise la pente. Corriger par l'overscan, ou intercaler les poses.
- *Pixels chauds* : le critère multiplicatif suppose une médiane > 0. À
  courant quasi nul (capteur cryogénique), seul le critère statistique
  (z_hot·u) a un sens ; le régler en conséquence.

## 4.6 Variantes écartées

- *Exclure de l'ajustement tout pixel saturé à la pose la plus longue* :
  44 des 66 pixels chauds simulés sont alors manqués, et ce sont les plus
  chauds (facteurs effectifs de 60 à 1 360). La régression masquée
  (§4.2.1), qui n'écarte que les poses saturées de chaque pixel, les
  détecte tous et mesure leur courant à 0,1 % près (médiane).
