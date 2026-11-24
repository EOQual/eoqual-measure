# 8. Pixels télégraphiques

## 8.1 Objet

Détection des pixels à bruit télégraphique (RTS) dans une longue pile
temporelle, et mesure de leurs niveaux, de l'amplitude du saut et des
probabilités de bascule.

**Statut** : validée contre le simulateur S4 (`tests/test_p3_measures.py`,
3 tests ; robustes sur 30 graines). Module :
`eoqual.measure.radiometry.defects:find_rts_pixels`.

## 8.2 Principe

1. **Présélection** : variance temporelle supérieure au quantile 1 − p du
   χ²_{K−1}/(K − 1) de la variance de référence (médiane corrigée de son
   biais, chapitre 7). Un saut A ajoute environ A²·π(1 − π) à la variance,
   π étant la fraction de temps à l'état haut.
2. **Deux niveaux ou un seul** : par pixel candidat, mélange de deux
   gaussiennes de même variance, ajusté par EM (vectorisé), comparé à une
   gaussienne seule par ΔBIC = 2(ln L₂ − ln L₁) − 2·ln K, puisque le
   mélange a deux paramètres de plus. Un pixel est retenu si ΔBIC > 10,
   soit une « preuve forte ».
3. **États et bascules** : décodage de Viterbi d'une chaîne de Markov à
   deux états gaussiens, puis réestimation des probabilités de bascule à
   partir des états décodés (apprentissage de Viterbi, 5 itérations,
   initialisé par seuillage au milieu des niveaux). Un simple seuillage
   compterait comme bascule chaque franchissement du seuil par le bruit.

## 8.3 Mise en œuvre

```python
from eoqual.measure.synthetic import MatrixSensor, SensorParams
from eoqual.measure.radiometry import find_rts_pixels

s = MatrixSensor(SensorParams(shape=(128, 128), seed=1, rts_fraction=3e-3,
                              rts_amplitude=(2.0, 10.0), rts_rate=(0.02, 0.2)))
r = find_rts_pixels(s.dark(n_frames=2000, seed=1).frames)
print(r.index.size, "pixels ;", r.amplitude[:3], r.p_up[:3], r.p_down[:3])
```

Recommandations : une pile de quelques centaines à quelques milliers de
poses, assez longue pour observer des dizaines de bascules par pixel ;
exclure au préalable les pixels chauds et bloqués (`exclude`).

## 8.4 Validation

Résultat type (128 × 128, 0,3 % de pixels RTS, A de 2 à 10 DN, soit 1,2 à
5,9 σ, p de 0,02 à 0,2, 2 000 poses) : 45 détectés sur 49, aucun faux. Les
4 manqués ont un saut de 1,3 à 1,7 σ. L'erreur d'amplitude est d'au plus
0,31 DN. L'erreur relative sur p_↑ est de 5,6 % (médiane) pour A ≥ 4 σ ;
pour A < 4 σ, p_↑ est **sous-estimé** de 14,5 %, car le décodage de Viterbi
lisse les séjours courts que le bruit rend ambigus.

Liste des tests : §8.4.1.

### 8.4.1 Tests

| Test | Vérifie |
|---|---|
| `test_rts_detection` | aucune fausse détection ; tous les sauts ≥ 2,5 σ détectés |
| `test_rts_parameters` | amplitude à 5·σ·√(1/n_bas + 1/n_haut) + 5 % ; p_↑ à 5/√n + 5 % pour les sauts ≥ 4 σ |
| `test_no_rts_on_clean_sensor` | aucune détection sans pixel télégraphique |

## 8.5 Limites

- Deux niveaux seulement ; un pixel à plusieurs pièges (trois niveaux ou
  plus) est vu comme un mélange à deux niveaux mal ajusté.
- Bascules sous-estimées pour les sauts faibles (§8.4) ; durées de séjour
  supposées géométriques (Markov).
