# 7. Pixels défectueux

## 7.1 Objet

Classification des pixels défectueux : bloqués, morts, chauds, bruyants,
à réponse aberrante. Le résultat est une carte de codes réutilisable
(exclusion dans les autres mesures, correction par interpolation).

**Statut** : validée contre les simulateurs S1 et S4
(`tests/test_prnu_linearity_defects.py`, 3 tests ; robustes sur 30
graines). Module :
`eoqual.measure.radiometry.defects:map_defective_pixels`.

## 7.2 Principe

| Classe (bit) | Critère | Fausse alarme |
|---|---|---|
| bloqué (1) | s² < χ²_{K−1}⁻¹(p_FA)/(K − 1) × σ²_réf dans **toutes** les piles | p_FA par pile |
| mort (2) | réponse / médiane locale 5 × 5 < 0,1 | — |
| chaud (4) | repris de `measure_dark_current` (critère multiplicatif et statistique sur la pente, pixels saturés) | chapitre 4 |
| bruyant (8) | s²_obscurité > χ²_{K−1}⁻¹(1 − p_FA)/(K − 1) × variance attendue | p_FA par pixel |
| réponse (16) | \|réponse / médiane locale − 1\| > 6 écarts-types robustes (MAD) | ≈ 2·10⁻⁹ par pixel |

**Exclusions.** Un pixel chaud n'est ni bloqué, ni mort, ni bruyant, ni à
réponse aberrante. Sur une pose longue, il sature et sort un code
constant, avec une réponse apparente « flat − obscurité » nulle : 34 faux
bloqués et 1 faux mort sans cette règle.

**Variance attendue.** Un pixel ordinaire dont le courant d'obscurité est
élevé (queue log-normale) a une variance poissonnienne plus grande ;
comparé à la médiane des variances, il passerait pour bruyant (16 cas
sur 65 536 sans correction). Avec `dark_current` et `dark_exposure_time`,
la variance attendue est donc a + b·I(p)·t, avec b = 1/K poissonnien. a
s'obtient en résolvant médiane(s²_p / (a + b·d_p)) = médiane(χ²_{K−1})/
(K − 1), chaque pixel normalisé par sa propre variance attendue.

**Biais de la médiane.** La médiane d'une variance empirique vaut
médiane(χ²_{K−1})/(K − 1) fois sa moyenne, soit 0,956 pour K = 16. Une
référence prise à la médiane brute est trop basse de 4,4 %, ce qui, au
quantile 10⁻⁶, multiplie les fausses alarmes par environ 3,5. Les
références sont corrigées de ce facteur.

**Médiane locale.** Une fenêtre de 5 × 5 suit le vignettage et les
anneaux de poussière, ce qui ne retient que des pixels isolés. Avec une
fenêtre de 15, 7 pixels d'anneaux de poussière étaient signalés à tort.

## 7.3 Mise en œuvre

**`map_defective_pixels(dark, flats, *, dark_current=None, dark_exposure_time=None, false_alarm=1e-6, dead_fraction=0.1, response_sigma=6, window=5)`**.
Il faut au moins 3 poses par pile ; la réponse est calculée sur le flat le
plus lumineux.

```python
from eoqual.measure.synthetic import MatrixSensor, SensorParams
from eoqual.measure.radiometry import map_defective_pixels, measure_dark_current

s = MatrixSensor(SensorParams(seed=5, dark_current=2.0, dark_nonuniformity=0.3, hot_pixel_fraction=1e-3,
                              dead_pixel_fraction=1e-3, stuck_pixel_fraction=5e-4, vignetting=0.15, n_dust=4))
t = s.truth
dark = s.dark(n_frames=16, seed=1, exposure_time=30.0).frames
flats = [s.expose(t.photons_for_signal(L), n_frames=16, seed=10 + i, exposure_time=30.0).frames
         for i, L in enumerate([1000.0, 3000.0])]
dc = measure_dark_current([s.dark(n_frames=16, seed=20 + i, exposure_time=te).frames
                           for i, te in enumerate([1.0, 30.0, 100.0])], [1.0, 30.0, 100.0],
                          saturation_dn=t.saturation_dn)
m = map_defective_pixels(dark, flats, dark_current=dc, dark_exposure_time=30.0)
print(m.counts())                 # {'stuck': …, 'dead': …, 'hot': …, 'noisy': …, 'response': …}
mask = m.any                      # à passer en `exclude` à measure_prnu, measure_linearity
```

## 7.4 Validation

Sur 3 capteurs (0,1 % de chauds, 0,1 % de morts, 0,05 % de bloqués,
vignettage 15 %, 4 poussières) : bloqués 99/99, morts 198/198 et chauds
197/198 détectés, 0 fausse détection dans toutes les classes. Le pixel
chaud manqué a un facteur effectif sous le critère de 10 × la médiane.
Taux de fausses alarmes « bruyant » au défaut p_FA = 10⁻⁶ : 5 sur 40
cartes, pour 2,6 nominales.

Liste des tests : §7.4.1.

### 7.4.1 Tests

| Test | Vérifie |
|---|---|
| `test_defect_classes` | bloqués, morts, chauds : aucune fausse détection, ≥ 97 % de détection ; aucun bruyant ni réponse aberrante (p_FA = 10⁻⁸) |
| `test_injected_noisy_and_response_outliers` | 20 pixels de bruit télégraphique (σ = 15 DN) et 20 réponses à −30 % injectés : retrouvés exactement |
| `test_defect_checks` | contrôles d'entrée |

## 7.5 Limites

- *Pixels télégraphiques* : détectés comme bruyants s'ils augmentent la
  variance, mais sans analyse de leurs niveaux (fiche `rts_pixels`, P3).
- *Grappes de défauts* : la médiane 5 × 5 suppose des défauts isolés ; une
  colonne ou une grappe défectueuse de plus de quelques pixels n'est pas
  classée comme telle.
