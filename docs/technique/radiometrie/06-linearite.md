# 6. Linéarité

## 6.1 Objet

Erreur de linéarité EMVA 1288, courbure commune de la réponse et
dispersion de cette courbure entre pixels, à partir d'expositions connues.

**Statut** : validée contre le simulateur S4
(`tests/test_prnu_linearity_defects.py`, 4 tests ; robustes sur 30
graines). Module :
`eoqual.measure.radiometry.linearity:measure_linearity`.

## 6.2 Principe

**Pourquoi une référence d'exposition.** Une égalisation relative (gains
calculés à un niveau, appliqués à un autre, la moyenne des pixels servant
de cible) rapporte chaque pixel à ses voisins. Une non-linéarité
**commune** à tous les pixels déforme la cible comme les pixels, et
disparaît au premier ordre. Seule une exposition connue (temps de pose
avec une source stable, niveaux de référence) la révèle. Le test le
chiffre : 8,75 % de compression au plus fort niveau, mais un résidu
d'égalisation inférieur à 0,1 % (§5.2).

**Erreur de linéarité (EMVA 1288).** Sur les expositions dont le signal
moyen est compris entre 5 % et 95 % du maximum non saturé, on ajuste la
droite μ = s·x + c (pondérée par 1/u(μ)²) ; δ_i = (μ_i − ajustement)/
ajustement, et LE = (max δ − min δ)/2, en %.

**Courbure commune.** On ajuste μ = c + s·x + q·x² sur la même plage, et
on exprime la courbure par κ = q/s² (DN⁻¹), indépendante de l'unité
d'exposition. Pour e' = e(1 − βe/FW), κ = −βK/FW exactement : la moyenne
E[e'] = μ − β(μ² + μ)/FW est quadratique.

**Courbure par pixel.** Même ajustement par pixel (projection par
moindres carrés, vectorisée), κ_p = q_p/s_p², indépendante du gain du
pixel. La dispersion se corrige du bruit par la méthode delta, avec la
covariance de (s, q) :

Var(κ) = Var(q)/s⁴ + 4q²·Var(s)/s⁶ − 4q·Cov(q, s)/s⁵,

où Cov(s, q) = P·diag(v_i/K_i)·Pᵀ, et P est la matrice de projection. Le
terme croisé change de signe avec q (§6.6).

## 6.3 Mise en œuvre

**`measure_linearity(stacks, exposures, dark, *, saturation_dn=None, fit_range=(0.05, 0.95), exclude=None)`**.
Au moins trois expositions distinctes. Une exposition est exclue si plus
de 10⁻⁴ de ses échantillons sont saturés.

```python
import numpy as np
from eoqual.measure.synthetic import MatrixSensor, SensorParams
from eoqual.measure.radiometry import measure_dark_frame, measure_linearity

s = MatrixSensor(SensorParams(seed=4, nonlinearity=0.1, nonlinearity_dispersion=0.02, full_well=12000.0))
t = s.truth
dark = measure_dark_frame(s.dark(n_frames=16, seed=1).frames)
expo = np.linspace(0.05, 1.1, 14)                         # temps de pose relatifs
stacks = [s.expose(e * t.photons_for_signal(3800.0), n_frames=8, seed=10 + i).frames
          for i, e in enumerate(expo)]
r = measure_linearity(stacks, expo, dark, saturation_dn=t.saturation_dn)
print(f"LE {r.linearity_error:.2f} %, κ = {r.curvature:.3e} ± {r.u_curvature:.1e} DN⁻¹ "
      f"(vrai {-0.1 * t.gain / 12000:.3e}), dispersion {r.curvature_dispersion:.3e}")
```

## 6.4 Validation

Résultats types (14 expositions de 8 poses, FW = 12 000 e⁻) :

| β, s_β | LE | κ mesuré / vrai (DN⁻¹) | dispersion mesurée / vraie |
|---|---|---|---|
| 0, 0 | 0,003 % | 8·10⁻⁹ ± 1·10⁻⁸ / 0 | 0 / 0 |
| 0,1, 0,02 | 5,15 % | −2,498·10⁻⁵ / −2,500·10⁻⁵ | 5,005·10⁻⁶ / 4,997·10⁻⁶ |
| −0,05, 0,01 | 2,02 % | 1,252·10⁻⁵ / 1,250·10⁻⁵ | 2,497·10⁻⁶ / 2,499·10⁻⁶ |

Liste des tests : §6.4.1.

### 6.4.1 Tests

| Test | Vérifie |
|---|---|
| `test_linearity_curvature` (β = 0,1 et −0,05) | κ = −βK/FW à 5 u ; dispersion = vraie à 5 u ; LE > 1 % ; exposition saturée exclue |
| `test_linear_sensor` | LE < 0,05 % ; κ et dispersion nuls à 5 u |
| `test_relative_equalization_misses_common_nonlinearity` | voir le chapitre 5 |
| `test_linearity_checks` | contrôles d'entrée |

## 6.5 Limites

- Modèle quadratique : une réponse à deux pentes ou un coude près de la
  saturation n'est décrit que par la LE.
- La source doit être stable : une dérive de la source entre expositions
  se confond avec une non-linéarité. L'intercaler ou la surveiller.

## 6.6 Variantes écartées

- *Corriger la dispersion du bruit par le seul terme Var(q)/s⁴* : la
  correction est fausse de +42 % ou de −15 % selon le signe de la courbure
  (dispersion mesurée à −3 % et +13 % de la vraie). Avec la méthode delta
  complète (§6.2), les écarts restent sous 1,6 u.
